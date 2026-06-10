//! woollama-core (Rust) — the embeddable model-management core, the first slice
//! of the woollama v1.0 Rust port.
//!
//! Slice 1 (callback-free, fully serves lackpy): the built-in inferencer registry
//! + `complete` / `complete_sync` / `complete_stream` (HTTP inference, incl.
//! ollama-native num_ctx routing and per-call api_key/base_url overrides).
//!
//! Slice 2 (the recipe↔tool loop): `orchestrate_events` — the single source of
//! truth, an async iterator of `tool_call`/`tool_result`/`final` events that runs a
//! recipe against an inferencer, dispatching its allow-listed tools through a Python
//! `ToolProvider` (the callback seam). `orchestrate` is the drainer over it (returns
//! the final OpenAI dict). Non-streaming turns only so far (no `delta` events).
//!
//! Behavior mirrors `woollama.core` in Python; its hermetic suite is the
//! conformance oracle. Deferred (later slices): streamed turns (`stream=True`'s
//! `delta` events + SSE per-turn tool_call reassembly); config-file
//! (`inferencers.toml`) loading + an explicit `ModelRegistry`; structured
//! `InferenceError` fields (kind/status/payload).

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_stream::stream;
use futures::stream::{Stream, StreamExt};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyStopAsyncIteration};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use serde_json::{json, Value};
use tokio::sync::Mutex;

create_exception!(core, InferenceError, PyException);

/// A resolved inference backend (OpenAI-compatible endpoint).
#[derive(Clone)]
struct Inferencer {
    name: String,
    base_url: String,
    api_key_env: Option<String>,
    /// Provider-specific request fields merged into each ORCHESTRATION request
    /// (NOT into `complete` — there the caller owns the body). Mirrors
    /// `inferencers.Inferencer.extra_body`: ollama keeps its native `options`,
    /// anthropic gets a sane max_tokens, clouds get temperature=0 for determinism.
    extra_body: Value,
}

/// The built-in providers — same set/URLs/extra_body as `woollama.core.inferencers`.
/// ollama's base honors `$WOOLLAMA_OLLAMA_URL`. (Config-file providers are a later slice.)
fn get_inferencer(provider: &str) -> Option<Inferencer> {
    let cloud = |n: &str, b: &str, k: &str| Inferencer {
        name: n.to_string(),
        base_url: b.to_string(),
        api_key_env: Some(k.to_string()),
        extra_body: json!({"temperature": 0}),
    };
    match provider {
        "ollama" => Some(Inferencer {
            name: "ollama".to_string(),
            base_url: std::env::var("WOOLLAMA_OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434/v1".to_string()),
            api_key_env: None,
            extra_body: json!({"options": {"temperature": 0}}),
        }),
        "anthropic" => Some(Inferencer {
            name: "anthropic".to_string(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            api_key_env: Some("ANTHROPIC_API_KEY".to_string()),
            extra_body: json!({"temperature": 0, "max_tokens": 4096}),
        }),
        "openai" => Some(cloud("openai", "https://api.openai.com/v1", "OPENAI_API_KEY")),
        "groq" => Some(cloud("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY")),
        "together" => Some(cloud("together", "https://api.together.ai/v1", "TOGETHER_API_KEY")),
        "openrouter" => Some(cloud("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")),
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

// --- orchestrate (the recipe↔tool loop) --------------------------------------

/// `"{TypeName}: {message}"` for a caught Python error — matches the Python loop's
/// `f"{type(e).__name__}: {e}"` when a dispatch raises (orchestrate.py:141).
fn pyerr_brief(e: &PyErr) -> String {
    Python::with_gil(|py| {
        let tn = e
            .get_type(py)
            .name()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "Error".to_string());
        let val = e.value(py).str().map(|s| s.to_string()).unwrap_or_default();
        format!("{tn}: {val}")
    })
}

/// Render a Python `ToolResult` (duck-typed: `.blocks` list, `.is_error` bool; each
/// block has `.text` and/or `.model_dump()`) into `(content, is_error)` — the `tool`
/// message content per `tooling.render_tool_result` for text-only `DEFAULT_CAPS`,
/// plus the `is_error` flag (the loop's `ok = !is_error`). Reimplemented in Rust
/// (reading attrs) so the core never imports Python woollama.
fn render_tool_result_rs(result: &Bound<'_, PyAny>) -> PyResult<(String, bool)> {
    let is_error: bool = result
        .getattr("is_error")
        .and_then(|v| v.extract())
        .unwrap_or(false);
    let blocks = result.getattr("blocks")?;
    let mut text_parts: Vec<String> = Vec::new();
    let mut dumped: Vec<Value> = Vec::new(); // the JSON fallback (used only if no text block)
    let mut any_block = false;
    for b in blocks.try_iter()? {
        let b = b?;
        any_block = true;
        // text block: `.text` that is a str → goes to the join (matches `_block_text`)
        if let Ok(t) = b.getattr("text") {
            if let Ok(s) = t.extract::<String>() {
                text_parts.push(s);
                continue;
            }
        }
        // non-text: `_block_dump` = model_dump() if callable, else the block itself
        let v: Value = match b.getattr("model_dump") {
            Ok(md) if md.is_callable() => md
                .call0()
                .ok()
                .and_then(|d| pythonize::depythonize(&d).ok())
                .unwrap_or(Value::Null),
            _ => pythonize::depythonize(&b).unwrap_or(Value::Null),
        };
        dumped.push(v);
    }
    let mut body = if !text_parts.is_empty() {
        text_parts.join("\n")
    } else if any_block {
        serde_json::to_string(&dumped).unwrap_or_default()
    } else {
        String::new()
    };
    if is_error {
        body = if body.is_empty() { "[tool error]".to_string() } else { format!("[tool error] {body}") };
    }
    Ok((body, is_error))
}

// --- the recipe↔tool loop: one source of truth (orchestrate_events) + drainer --

/// One progress event from the loop. Converted to the OpenAI-ish event dict
/// (`{"type": …}`) only at the Python boundary (`event_to_py`); the loop stays pure
/// Rust. `Delta` is streaming-only (deferred — non-stream turns emit none).
enum Event {
    #[allow(dead_code)]
    Delta(String),
    ToolCall { turn: u32, name: String, args: Value },
    ToolResult { turn: u32, name: String, ok: bool },
    Final(Value),
}

fn pyval(v: &Value) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| Ok(pythonize::pythonize(py, v)?.unbind()))
}

fn event_to_py(ev: &Event) -> PyResult<Py<PyAny>> {
    let v = match ev {
        Event::Delta(c) => json!({"type": "delta", "content": c}),
        Event::ToolCall { turn, name, args } =>
            json!({"type": "tool_call", "turn": turn, "name": name, "args": args}),
        Event::ToolResult { turn, name, ok } =>
            json!({"type": "tool_result", "turn": turn, "name": name, "ok": ok}),
        Event::Final(resp) => json!({"type": "final", "response": resp}),
    };
    pyval(&v)
}

/// Everything the loop needs, resolved EAGERLY (under the GIL) before any async work
/// — so an unsupported inferencer / missing key / bad `tools_for` raises on the call,
/// not lazily on first iteration. A deliberate divergence from the Python generator's
/// lazy timing (matches the committed `orchestrate` behavior, and keeps the sync
/// fail-fast tests green).
struct Setup {
    url: String,
    headers: HashMap<String, String>,
    model: String,
    schemas: Vec<Value>,
    allowed: std::collections::HashSet<String>,
    sorted_allowed: Vec<String>,
    messages: Vec<Value>,
    extra_body: Value,
    params: Option<Value>,
    tools: Py<PyAny>,
}

fn build_setup(
    py: Python<'_>,
    recipe: &Bound<'_, PyAny>,
    user_msgs: &Bound<'_, PyAny>,
    tools: &Bound<'_, PyAny>,
    api_key: Option<String>,
    base_url: Option<String>,
) -> PyResult<Setup> {
    let rv: Value = pythonize::depythonize(recipe)?;
    let inferencer = rv
        .get("inferencer")
        .and_then(Value::as_str)
        .ok_or_else(|| InferenceError::new_err("recipe missing 'inferencer'"))?
        .to_string();
    let system = rv.get("system").and_then(Value::as_str).unwrap_or("").to_string();
    let tool_names: Vec<String> = rv
        .get("tools")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    let params: Option<Value> = rv.get("params").filter(|v| !v.is_null()).cloned();

    let (provider, model) = split_model(&inferencer);
    let inf = get_inferencer(&provider).ok_or_else(|| {
        InferenceError::new_err(format!(
            "unsupported inferencer '{inferencer}' (supported providers: {}, claude-code)",
            known_providers().join(", ")
        ))
    })?;
    let headers = build_headers(&inf, api_key.as_deref()).map_err(InferenceError::new_err)?;
    let base = base_url
        .unwrap_or_else(|| inf.base_url.clone())
        .trim_end_matches('/')
        .to_string();
    let url = format!("{base}/chat/completions");

    // tools_for(allow) → the model-facing schemas (lossless seam; we read only .schema).
    let allow_list = pyo3::types::PyList::new(py, &tool_names)?;
    let specs = tools.call_method1("tools_for", (allow_list,))?;
    let mut schemas: Vec<Value> = Vec::new();
    for s in specs.try_iter()? {
        schemas.push(pythonize::depythonize(&s?.getattr("schema")?)?);
    }
    // The allow-list is the BOUNDARY (only these may be dispatched); the schema's
    // function name IS the namespaced allow-list name, so membership matches directly.
    let allowed: std::collections::HashSet<String> = tool_names.iter().cloned().collect();
    let mut sorted_allowed: Vec<String> = tool_names.clone();
    sorted_allowed.sort();

    // messages = [system] + user_msgs
    let user_list: Value = pythonize::depythonize(user_msgs)?;
    let mut messages: Vec<Value> = vec![json!({"role": "system", "content": system})];
    if let Some(arr) = user_list.as_array() {
        messages.extend(arr.iter().cloned());
    }

    Ok(Setup {
        url,
        headers,
        model,
        schemas,
        allowed,
        sorted_allowed,
        messages,
        extra_body: inf.extra_body.clone(),
        params,
        tools: tools.clone().unbind(),
    })
}

/// Dispatch ONE tool call through the Python `ToolProvider` and render the result.
/// Build the awaitable + future under the GIL, await it off-GIL. A `dispatch` that
/// raises is CAUGHT and returned as `(ERROR: …, false)` — never propagated
/// (orchestrate.py:139-141). Returns `(rendered, ok)` with `ok = !is_error`.
async fn dispatch_and_render(tools: &Py<PyAny>, name: &str, args: &Value) -> (String, bool) {
    let built = Python::with_gil(|py| -> PyResult<_> {
        let args_py = pythonize::pythonize(py, args)?;
        let awaitable = tools.bind(py).call_method1("dispatch", (name, args_py))?;
        pyo3_async_runtimes::tokio::into_future(awaitable)
    });
    let fut = match built {
        Ok(f) => f,
        Err(e) => return (format!("ERROR: {}", pyerr_brief(&e)), false),
    };
    match fut.await {
        Err(e) => (format!("ERROR: {}", pyerr_brief(&e)), false),
        Ok(tr) => match Python::with_gil(|py| render_tool_result_rs(tr.bind(py))) {
            Ok((body, is_error)) => (body, !is_error),
            Err(e) => (format!("ERROR: {}", pyerr_brief(&e)), false),
        },
    }
}

/// The recipe↔tool loop as a stream of `Event`s — the SINGLE source of truth (the
/// drainer + the `EventIter` both consume it; do NOT reimplement). Non-streaming
/// turns only for now (no `Delta` events; streamed turns are a later slice). Offers
/// the allow-listed tools, dispatches the ones the model calls, feeds results back,
/// repeats (≤8 turns). The allow-list is a BOUNDARY: an off-list call is refused
/// WITHOUT dispatching and the refusal is fed back so the loop recovers. Yields
/// `ToolCall`/`ToolResult` progress and a terminal `Final`, or an `Err` (inferencer
/// error / max turns).
fn events_stream(setup: Setup) -> impl Stream<Item = PyResult<Event>> + Send {
    let Setup {
        url, headers, model, schemas, allowed, sorted_allowed, messages, extra_body, params, tools,
    } = setup;
    stream! {
        let mut messages = messages;
        let client = match reqwest::Client::builder().timeout(Duration::from_secs(180)).build() {
            Ok(c) => c,
            Err(e) => { yield Err(InferenceError::new_err(e.to_string())); return; }
        };
        for turn in 1..=8u32 {
            // body = {model, messages, tools, stream:false, **extra_body, **params}
            let mut body = json!({"model": model, "messages": messages, "tools": schemas, "stream": false});
            if let Some(o) = extra_body.as_object() { for (k, v) in o { body[k] = v.clone(); } }
            if let Some(o) = params.as_ref().and_then(Value::as_object) { for (k, v) in o { body[k] = v.clone(); } }

            let mut rb = client.post(&url).json(&body);
            for (k, v) in &headers { rb = rb.header(k, v); }
            let resp = match rb.send().await {
                Ok(r) => r,
                Err(e) => { yield Err(InferenceError::new_err(e.to_string())); return; }
            };
            let data: Value = match resp.json().await {
                Ok(d) => d,
                Err(e) => { yield Err(InferenceError::new_err(e.to_string())); return; }
            };
            if data.get("choices").is_none() {
                yield Err(InferenceError::new_err(format!("inferencer error: {data}")));
                return;
            }
            let msg = &data["choices"][0]["message"];
            let calls = msg.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default();
            let content = msg.get("content").and_then(Value::as_str).unwrap_or("").to_string();

            if calls.is_empty() {
                // no tool calls → this turn's whole response IS the final answer
                yield Ok(Event::Final(data));
                return;
            }
            // Owned per-call metadata so we never borrow `calls` across the awaits below.
            let metas: Vec<(String, Value, String)> = calls.iter().map(|call| {
                let func = call.get("function");
                let name = func.and_then(|f| f.get("name")).and_then(Value::as_str).unwrap_or("").to_string();
                // OpenAI gives `arguments` as a JSON STRING; parse before dispatch.
                let args = match func.and_then(|f| f.get("arguments")) {
                    Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
                    Some(v) => v.clone(),
                    None => json!({}),
                };
                let id = call.get("id").and_then(Value::as_str).map(str::to_string)
                    .unwrap_or_else(|| format!("call_{turn}_{name}"));
                (name, args, id)
            }).collect();
            messages.push(json!({"role": "assistant", "content": content, "tool_calls": calls}));

            for (name, args, call_id) in &metas {
                yield Ok(Event::ToolCall { turn, name: name.clone(), args: args.clone() });
                let (result, ok) = if !allowed.contains(name) {
                    // refuse, but feed the refusal back (every tool_call needs a tool message)
                    (format!("ERROR: tool '{name}' is not permitted by this recipe (allowed: {sorted_allowed:?})"), false)
                } else {
                    dispatch_and_render(&tools, name, args).await
                };
                yield Ok(Event::ToolResult { turn, name: name.clone(), ok });
                messages.push(json!({"role": "tool", "content": result, "tool_call_id": call_id}));
            }
        }
        yield Err(InferenceError::new_err("max turns (8) exceeded"));
    }
}

/// Run a recipe and return the final OpenAI response dict — the drainer over
/// `events_stream` (Python's `core.orchestrate`). Setup is eager (raises on the
/// call); the loop runs in the awaitable, and progress events are discarded.
#[pyfunction]
#[pyo3(signature = (recipe, user_msgs, tools, *, api_key=None, base_url=None))]
fn orchestrate<'py>(
    py: Python<'py>,
    recipe: Bound<'py, PyAny>,
    user_msgs: Bound<'py, PyAny>,
    tools: Bound<'py, PyAny>,
    api_key: Option<String>,
    base_url: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let setup = build_setup(py, &recipe, &user_msgs, &tools, api_key, base_url)?;
    future_into_py(py, async move {
        let mut s = Box::pin(events_stream(setup));
        while let Some(item) = s.next().await {
            if let Event::Final(resp) = item? {
                return Python::with_gil(|py| Ok(pythonize::pythonize(py, &resp)?.unbind()));
            }
        }
        Err(InferenceError::new_err("orchestrate: loop ended without a final response"))
    })
}

/// An async iterator over the recipe loop's progress events — `async for ev in
/// orchestrate_events(...)`. Each `ev` is a dict: `tool_call` / `tool_result`
/// progress and a terminal `final` (`{"type": "final", "response": <openai dict>}`);
/// raises `InferenceError` on inferencer error / max turns.
#[pyclass]
struct EventIter {
    inner: Arc<Mutex<Pin<Box<dyn Stream<Item = PyResult<Event>> + Send>>>>,
}

#[pymethods]
impl EventIter {
    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let mut s = inner.lock().await;
            match s.next().await {
                Some(Ok(ev)) => event_to_py(&ev),
                Some(Err(e)) => Err(e),
                None => Err(PyStopAsyncIteration::new_err(())),
            }
        })
    }
}

/// The recipe loop as an async iterator of progress + `final` events (the server's
/// surface). Setup is eager (raises on the call). `stream=True` (delta events from
/// streamed turns) is a later slice — passing it raises.
#[pyfunction]
#[pyo3(signature = (recipe, user_msgs, tools, *, api_key=None, base_url=None, stream=false))]
fn orchestrate_events<'py>(
    py: Python<'py>,
    recipe: Bound<'py, PyAny>,
    user_msgs: Bound<'py, PyAny>,
    tools: Bound<'py, PyAny>,
    api_key: Option<String>,
    base_url: Option<String>,
    stream: bool,
) -> PyResult<EventIter> {
    if stream {
        return Err(InferenceError::new_err(
            "streamed orchestration (delta events) is not yet implemented in the Rust core",
        ));
    }
    let setup = build_setup(py, &recipe, &user_msgs, &tools, api_key, base_url)?;
    Ok(EventIter { inner: Arc::new(Mutex::new(Box::pin(events_stream(setup)))) })
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

// `module-name = "woollama.core"` (pyproject) → the init symbol must be `PyInit_core`,
// so this fn is named `core` (the last dotted component); `woollama` is the PEP 420 namespace.
#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // `create_exception!` can't take a dotted module ident, so its `__module__` is the bare
    // `core`; relabel it to the real import path for clean reprs / pickling.
    let exc = m.py().get_type::<InferenceError>();
    exc.setattr("__module__", "woollama.core")?;
    m.add("InferenceError", exc)?;
    m.add_function(wrap_pyfunction!(complete, m)?)?;
    m.add_function(wrap_pyfunction!(complete_sync, m)?)?;
    m.add_function(wrap_pyfunction!(complete_stream, m)?)?;
    m.add_function(wrap_pyfunction!(provider_names, m)?)?;
    m.add_function(wrap_pyfunction!(orchestrate, m)?)?;
    m.add_function(wrap_pyfunction!(orchestrate_events, m)?)?;
    m.add_class::<EventIter>()?;
    m.add_class::<DeltaStream>()?;
    Ok(())
}
