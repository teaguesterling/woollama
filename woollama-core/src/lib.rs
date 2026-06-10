//! woollama-core (Rust) — the embeddable model-management core, the first slice
//! of the woollama v1.0 Rust port.
//!
//! Slice 1 scope (callback-free, fully serves lackpy): the built-in inferencer
//! registry + `complete` (async) / `complete_sync` (HTTP inference, incl.
//! ollama-native num_ctx routing and per-call api_key/base_url overrides).
//! Behavior mirrors `woollama.core.complete` in Python; the Python hermetic suite
//! is the conformance oracle.
//!
//! Deferred (later slices): `complete_stream` (SSE), config-file
//! (`inferencers.toml`) loading, an explicit `ModelRegistry`, structured
//! `InferenceError` fields (kind/status/payload), and the recipe loop + the Python
//! `ToolProvider`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyStopAsyncIteration};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use serde_json::{json, Value};
use tokio::sync::Mutex;

create_exception!(woollama_core, InferenceError, PyException);

/// A resolved inference backend (OpenAI-compatible endpoint).
#[derive(Clone)]
struct Inferencer {
    name: String,
    base_url: String,
    api_key_env: Option<String>,
}

/// The built-in providers — same set/URLs as `woollama.core.inferencers`. ollama's
/// base honors `$WOOLLAMA_OLLAMA_URL`. (Config-file providers are a later slice.)
fn get_inferencer(provider: &str) -> Option<Inferencer> {
    let owned = |n: &str, b: &str, k: Option<&str>| Inferencer {
        name: n.to_string(),
        base_url: b.to_string(),
        api_key_env: k.map(str::to_string),
    };
    match provider {
        "ollama" => Some(Inferencer {
            name: "ollama".to_string(),
            base_url: std::env::var("WOOLLAMA_OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434/v1".to_string()),
            api_key_env: None,
        }),
        "anthropic" => Some(owned("anthropic", "https://api.anthropic.com/v1", Some("ANTHROPIC_API_KEY"))),
        "openai" => Some(owned("openai", "https://api.openai.com/v1", Some("OPENAI_API_KEY"))),
        "groq" => Some(owned("groq", "https://api.groq.com/openai/v1", Some("GROQ_API_KEY"))),
        "together" => Some(owned("together", "https://api.together.ai/v1", Some("TOGETHER_API_KEY"))),
        "openrouter" => Some(owned("openrouter", "https://openrouter.ai/api/v1", Some("OPENROUTER_API_KEY"))),
        _ => None,
    }
}

fn known_providers() -> &'static [&'static str] {
    &["ollama", "anthropic", "openai", "groq", "together", "openrouter"]
}

fn build_headers(inf: &Inferencer, api_key: Option<&str>) -> Result<HashMap<String, String>, String> {
    if let Some(k) = api_key {
        return Ok(HashMap::from([("Authorization".to_string(), format!("Bearer {k}"))]));
    }
    match &inf.api_key_env {
        None => Ok(HashMap::new()),
        Some(env) => match std::env::var(env) {
            Ok(v) if !v.is_empty() => {
                Ok(HashMap::from([("Authorization".to_string(), format!("Bearer {v}"))]))
            }
            _ => Err(format!("inferencer '{}' requires ${} to be set", inf.name, env)),
        },
    }
}

fn split_model(model: &str) -> (String, String) {
    match model.split_once('/') {
        Some((p, m)) => (p.to_string(), m.to_string()),
        None => (model.to_string(), String::new()),
    }
}

fn obj(v: &Option<Value>) -> Option<&serde_json::Map<String, Value>> {
    v.as_ref().and_then(Value::as_object)
}

/// Everything needed to issue one request — all `Send`, so it can move into the
/// async block. Building it is synchronous and can raise (unknown provider /
/// missing key), which we do BEFORE going async.
struct Request {
    url: String,
    body: Value,
    headers: HashMap<String, String>,
    timeout: u64,
    native: bool,
}

fn build_request(
    model: &str,
    msgs: Value,
    opts: Option<Value>,
    prms: Option<Value>,
    api_key: Option<&str>,
    base_url: Option<String>,
    stream: bool,
) -> PyResult<Request> {
    let (provider, bare) = split_model(model);
    let inf = get_inferencer(&provider).ok_or_else(|| {
        InferenceError::new_err(format!(
            "unknown model namespace: '{model}'. Use 'woollama/<recipe>' or \
             '<provider>/<model>' for a known inferencer ({}).",
            known_providers().join(", ")
        ))
    })?;
    let base = base_url
        .unwrap_or_else(|| inf.base_url.clone())
        .trim_end_matches('/')
        .to_string();
    let headers = build_headers(&inf, api_key).map_err(InferenceError::new_err)?;

    // ollama-native num_ctx routing is non-stream only (matches Python).
    let native = !stream
        && provider == "ollama"
        && obj(&opts).and_then(|o| o.get("num_ctx")).map_or(false, |v| !v.is_null());

    let (url, body, timeout) = if native {
        let mut native_opts = opts.clone().unwrap_or_else(|| json!({}));
        if let (Some(no), Some(po)) = (native_opts.as_object_mut(), obj(&prms)) {
            for (k, v) in po {
                no.insert(k.clone(), v.clone());
            }
        }
        let root = base.strip_suffix("/v1").unwrap_or(&base);
        (
            format!("{root}/api/chat"),
            json!({"model": bare, "messages": msgs, "options": native_opts, "stream": false}),
            600,
        )
    } else {
        let mut body = json!({"model": bare, "messages": msgs, "stream": stream});
        if let Some(o) = &opts {
            body["options"] = o.clone();
        }
        if let Some(po) = obj(&prms) {
            for (k, v) in po {
                body[k] = v.clone(); // top-level OpenAI fields (temperature, …)
            }
        }
        (format!("{base}/chat/completions"), body, 180)
    };
    Ok(Request { url, body, headers, timeout, native })
}

/// Pull the assistant text out of the (native or /v1) response, or `Err(msg)` on
/// the upstream-error case. `Send` error so it works inside the async block.
fn parse_response(data: &Value, native: bool) -> Result<String, String> {
    let content = if native {
        data.get("message").and_then(|m| m.get("content")).and_then(Value::as_str)
    } else {
        data.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
    };
    let present = if native { data.get("message").is_some() } else { data.get("choices").is_some() };
    match content {
        Some(s) => Ok(s.to_string()),
        None if present => Ok(String::new()), // content present-but-null → ""
        None => Err(format!("inferencer error: {data}")),
    }
}

fn depy(v: &Option<Bound<'_, PyAny>>) -> PyResult<Option<Value>> {
    v.as_ref().map(pythonize::depythonize).transpose().map_err(Into::into)
}

/// Run one stateless turn against `<provider>/<model>` and return the assistant
/// text — an awaitable (so `await complete(...)` works for async embedders).
#[pyfunction]
#[pyo3(signature = (model, messages, *, options=None, params=None, api_key=None, base_url=None))]
fn complete<'py>(
    py: Python<'py>,
    model: String,
    messages: Bound<'py, PyAny>,
    options: Option<Bound<'py, PyAny>>,
    params: Option<Bound<'py, PyAny>>,
    api_key: Option<String>,
    base_url: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let msgs: Value = pythonize::depythonize(&messages)?;
    let req = build_request(&model, msgs, depy(&options)?, depy(&params)?, api_key.as_deref(), base_url, false)?;
    future_into_py(py, async move {
        let out: Result<String, String> = async move {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(req.timeout))
                .build()
                .map_err(|e| e.to_string())?;
            let mut rb = client.post(&req.url).json(&req.body);
            for (k, v) in &req.headers {
                rb = rb.header(k, v);
            }
            let resp = rb.send().await.map_err(|e| e.to_string())?;
            let data: Value = resp.json().await.map_err(|e| e.to_string())?;
            parse_response(&data, req.native)
        }
        .await;
        out.map_err(InferenceError::new_err)
    })
}

/// Synchronous variant — for non-async embedders. HTTP runs off the GIL.
#[pyfunction]
#[pyo3(signature = (model, messages, *, options=None, params=None, api_key=None, base_url=None))]
fn complete_sync(
    py: Python<'_>,
    model: String,
    messages: Bound<'_, PyAny>,
    options: Option<Bound<'_, PyAny>>,
    params: Option<Bound<'_, PyAny>>,
    api_key: Option<String>,
    base_url: Option<String>,
) -> PyResult<String> {
    let msgs: Value = pythonize::depythonize(&messages)?;
    let req = build_request(&model, msgs, depy(&options)?, depy(&params)?, api_key.as_deref(), base_url, false)?;
    py.allow_threads(move || -> Result<String, String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(req.timeout))
            .build()
            .map_err(|e| e.to_string())?;
        let mut rb = client.post(&req.url).json(&req.body);
        for (k, v) in &req.headers {
            rb = rb.header(k, v);
        }
        let resp = rb.send().map_err(|e| e.to_string())?;
        let data = resp.json::<Value>().map_err(|e| e.to_string())?;
        parse_response(&data, req.native)
    })
    .map_err(InferenceError::new_err)
}

/// The built-in provider names (introspection / parity with `inferencers.names()`).
#[pyfunction]
fn provider_names() -> Vec<String> {
    known_providers().iter().map(|s| s.to_string()).collect()
}

// --- streaming ---------------------------------------------------------------

enum StreamState {
    Pending(Request),                                 // not yet POSTed
    Active { resp: reqwest::Response, buf: String },  // reading SSE
    Done,
}

enum Delta {
    Yield(String),
    NeedMore,
    Done,
}

/// Pull the next non-empty `choices[0].delta.content` out of complete SSE lines in
/// `buf` (consuming them); leaves any trailing partial line. `[DONE]` → `Done`.
fn next_delta(buf: &mut String) -> Delta {
    while let Some(nl) = buf.find('\n') {
        let line: String = buf.drain(..=nl).collect();
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if data == "[DONE]" {
                return Delta::Done;
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(data) {
                let d = chunk
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(Value::as_str);
                if let Some(piece) = d {
                    if !piece.is_empty() {
                        return Delta::Yield(piece.to_string());
                    }
                }
            }
        }
    }
    Delta::NeedMore
}

/// An async iterator over assistant text deltas — `async for d in complete_stream(...)`.
#[pyclass]
struct DeltaStream {
    state: Arc<Mutex<StreamState>>,
}

#[pymethods]
impl DeltaStream {
    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state = self.state.clone();
        future_into_py(py, async move {
            let mut s = state.lock().await;
            loop {
                // 1. Lazily POST on the first pull (so a setup error raises here,
                //    matching the Python generator's "before first yield").
                if matches!(&*s, StreamState::Pending(_)) {
                    let req = match std::mem::replace(&mut *s, StreamState::Done) {
                        StreamState::Pending(r) => r,
                        _ => unreachable!(),
                    };
                    let client = reqwest::Client::builder()
                        .timeout(Duration::from_secs(req.timeout))
                        .build()
                        .map_err(|e| InferenceError::new_err(e.to_string()))?;
                    let mut rb = client.post(&req.url).json(&req.body);
                    for (k, v) in &req.headers {
                        rb = rb.header(k, v);
                    }
                    let resp = rb
                        .send()
                        .await
                        .map_err(|e| InferenceError::new_err(e.to_string()))?;
                    if resp.status().as_u16() >= 400 {
                        let body = resp.text().await.unwrap_or_default();
                        return Err(InferenceError::new_err(format!("inferencer error: {body}")));
                    }
                    *s = StreamState::Active { resp, buf: String::new() };
                }
                // 2. Serve a buffered delta, or read more bytes.
                match &mut *s {
                    StreamState::Done => return Err(PyStopAsyncIteration::new_err(())),
                    StreamState::Pending(_) => unreachable!(),
                    StreamState::Active { resp, buf } => match next_delta(buf) {
                        Delta::Yield(piece) => return Ok(piece),
                        Delta::Done => {
                            *s = StreamState::Done;
                            return Err(PyStopAsyncIteration::new_err(()));
                        }
                        Delta::NeedMore => match resp
                            .chunk()
                            .await
                            .map_err(|e| InferenceError::new_err(e.to_string()))?
                        {
                            Some(bytes) => buf.push_str(&String::from_utf8_lossy(&bytes)),
                            None => {
                                *s = StreamState::Done;
                                return Err(PyStopAsyncIteration::new_err(()));
                            }
                        },
                    },
                }
            }
        })
    }
}

/// Stream one stateless turn as assistant text deltas (the inferencer's `/v1` SSE;
/// num_ctx-native routing is non-stream only). Returns an async iterator.
#[pyfunction]
#[pyo3(signature = (model, messages, *, options=None, params=None, api_key=None, base_url=None))]
fn complete_stream(
    model: String,
    messages: Bound<'_, PyAny>,
    options: Option<Bound<'_, PyAny>>,
    params: Option<Bound<'_, PyAny>>,
    api_key: Option<String>,
    base_url: Option<String>,
) -> PyResult<DeltaStream> {
    let msgs: Value = pythonize::depythonize(&messages)?;
    let req = build_request(&model, msgs, depy(&options)?, depy(&params)?, api_key.as_deref(), base_url, true)?;
    Ok(DeltaStream { state: Arc::new(Mutex::new(StreamState::Pending(req))) })
}

#[pymodule]
fn woollama_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("InferenceError", m.py().get_type::<InferenceError>())?;
    m.add_function(wrap_pyfunction!(complete, m)?)?;
    m.add_function(wrap_pyfunction!(complete_sync, m)?)?;
    m.add_function(wrap_pyfunction!(complete_stream, m)?)?;
    m.add_function(wrap_pyfunction!(provider_names, m)?)?;
    m.add_class::<DeltaStream>()?;
    Ok(())
}
