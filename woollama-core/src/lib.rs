//! woollama-core (Rust) — the embeddable model-management core, the first slice
//! of the woollama v1.0 Rust port.
//!
//! Slice 1 scope (callback-free, fully serves lackpy): the built-in inferencer
//! registry + `complete` (HTTP inference, incl. ollama-native num_ctx routing and
//! per-call api_key/base_url overrides). Behavior mirrors `woollama.core.complete`
//! in Python; the Python hermetic suite is the conformance oracle.
//!
//! Deferred (later slices): config-file (`inferencers.toml`) loading, an explicit
//! `ModelRegistry` object, `complete_stream` (SSE), structured `InferenceError`
//! fields (kind/status/payload), and the recipe loop + Python `ToolProvider`.

use std::collections::HashMap;
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use serde_json::{json, Value};

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

/// Auth headers; a per-call `api_key` overrides the env lookup (fail-fast with a
/// clear message if the configured key env is unset).
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

fn as_object(v: &Option<Value>) -> Option<&serde_json::Map<String, Value>> {
    v.as_ref().and_then(Value::as_object)
}

/// Run one stateless turn against `<provider>/<model>` and return the assistant
/// text. `options` carries ollama-native knobs (e.g. `num_ctx` → native /api/chat);
/// `params` are top-level OpenAI request fields (temperature, …).
#[pyfunction]
#[pyo3(signature = (model, messages, *, options=None, params=None, api_key=None, base_url=None))]
fn complete(
    py: Python<'_>,
    model: String,
    messages: Bound<'_, PyAny>,
    options: Option<Bound<'_, PyAny>>,
    params: Option<Bound<'_, PyAny>>,
    api_key: Option<String>,
    base_url: Option<String>,
) -> PyResult<String> {
    let msgs: Value = pythonize::depythonize(&messages)?;
    let opts: Option<Value> = options.map(|o| pythonize::depythonize(&o)).transpose()?;
    let prms: Option<Value> = params.map(|p| pythonize::depythonize(&p)).transpose()?;

    let (provider, bare) = split_model(&model);
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
    let headers = build_headers(&inf, api_key.as_deref()).map_err(InferenceError::new_err)?;

    let native = provider == "ollama"
        && as_object(&opts)
            .and_then(|o| o.get("num_ctx"))
            .map_or(false, |v| !v.is_null());

    let (url, body, timeout) = if native {
        // On the native path temperature et al. live inside `options`.
        let mut native_opts = opts.clone().unwrap_or_else(|| json!({}));
        if let (Some(no), Some(po)) = (native_opts.as_object_mut(), as_object(&prms)) {
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
        let mut body = json!({"model": bare, "messages": msgs, "stream": false});
        if let Some(o) = &opts {
            body["options"] = o.clone();
        }
        if let Some(po) = as_object(&prms) {
            for (k, v) in po {
                body[k] = v.clone(); // top-level OpenAI fields (temperature, …)
            }
        }
        (format!("{base}/chat/completions"), body, 180)
    };

    // Blocking HTTP off the GIL so Python stays responsive.
    let data: Value = py
        .allow_threads(move || -> Result<Value, String> {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(timeout))
                .build()
                .map_err(|e| e.to_string())?;
            let mut req = client.post(&url).json(&body);
            for (k, v) in &headers {
                req = req.header(k, v);
            }
            let resp = req.send().map_err(|e| e.to_string())?;
            resp.json::<Value>().map_err(|e| e.to_string())
        })
        .map_err(|e| InferenceError::new_err(format!("inferencer error: {e}")))?;

    let content = if native {
        data.get("message").and_then(|m| m.get("content")).and_then(Value::as_str)
    } else {
        data.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
    };
    match content {
        Some(s) => Ok(s.to_string()),
        // Python returns "" when content is present-but-null; an absent
        // message/choices is the upstream-error case.
        None if (native && data.get("message").is_some())
            || (!native && data.get("choices").is_some()) =>
        {
            Ok(String::new())
        }
        None => Err(InferenceError::new_err(format!("inferencer error: {data}"))),
    }
}

/// The built-in provider names (introspection / parity with `inferencers.names()`).
#[pyfunction]
fn provider_names() -> Vec<String> {
    known_providers().iter().map(|s| s.to_string()).collect()
}

#[pymodule]
fn woollama_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("InferenceError", m.py().get_type::<InferenceError>())?;
    m.add_function(wrap_pyfunction!(complete, m)?)?;
    m.add_function(wrap_pyfunction!(provider_names, m)?)?;
    Ok(())
}
