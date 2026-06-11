//! woollama-engine — the pure-Rust woollama engine (NO PyO3).
//!
//! This is the reusable heart of the router: the built-in/config inferencer
//! registry, stateless `complete` (+ blocking + streaming), and the recipe↔tool
//! orchestrate loop (`build_setup` + `events_stream`). It speaks `serde_json::Value`
//! and a Rust `ToolProvider` trait at its edges — no Python types cross this
//! boundary. Two consumers wrap it:
//!   - `woollama-core` (cdylib): the PyO3 wheel — wraps `EngineError` as the
//!     `InferenceError` pyclass and bridges a Python `ToolProvider` callback.
//!   - `woollama-server` (bin): the native router — dispatches tools to downstream
//!     MCP servers (rmcp), from slice 4 onward.
//!
//! Behavior mirrors `woollama.core` in Python; the conformance suite pins it.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_stream::stream;
use futures::stream::Stream;
use serde_json::{json, Value};

// --- structured error ---------------------------------------------------------

/// Structured inference/orchestration error — the pure-Rust core of what the wheel
/// surfaces as `InferenceError(message, kind, status, payload)` and the server maps
/// to an HTTP/MCP error. `payload` carries the raw upstream response verbatim.
#[derive(Debug, Clone)]
pub struct EngineError {
    pub message: String,
    pub kind: Option<String>,
    pub status: Option<i64>,
    pub payload: Option<Value>,
}

impl EngineError {
    pub fn new(message: impl Into<String>, kind: &str, status: i64) -> Self {
        EngineError {
            message: message.into(),
            kind: Some(kind.to_string()),
            status: Some(status),
            payload: None,
        }
    }
    pub fn with_payload(message: impl Into<String>, kind: &str, status: i64, payload: Option<Value>) -> Self {
        EngineError {
            message: message.into(),
            kind: Some(kind.to_string()),
            status: Some(status),
            payload,
        }
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for EngineError {}

// --- the tool seam (replaces the Python `Py<PyAny>` ToolProvider) -------------

/// The recipe↔tool seam. The engine offers the recipe's allow-listed tool schemas
/// to the model and dispatches the calls the model makes. `tool_schemas` is resolved
/// eagerly during setup; `dispatch` runs one call and returns the already-rendered
/// `(content, ok)` (`ok = !is_error`) — a dispatch that errors is reported as
/// `(ERROR: …, false)`, never propagated.
#[async_trait::async_trait]
pub trait ToolProvider: Send + Sync {
    fn tool_schemas(&self, allow: &[String]) -> Result<Vec<Value>, EngineError>;
    async fn dispatch(&self, name: &str, args: &Value) -> (String, bool);
}

// --- inferencers --------------------------------------------------------------

/// A resolved inference backend (OpenAI-compatible endpoint).
#[derive(Clone)]
pub struct Inferencer {
    pub name: String,
    pub base_url: String,
    pub api_key_env: Option<String>,
    /// Provider-specific request fields merged into each ORCHESTRATION request
    /// (NOT into `complete`). ollama keeps native `options`, anthropic gets a sane
    /// max_tokens, clouds get temperature=0 for determinism.
    pub extra_body: Value,
}

/// The built-in providers — same set/URLs/extra_body as `woollama.core.inferencers`.
/// ollama's base honors `$WOOLLAMA_OLLAMA_URL`.
pub fn get_inferencer(provider: &str) -> Option<Inferencer> {
    let cloud = |n: &str, b: &str, k: &str| Inferencer {
        name: n.to_string(),
        base_url: b.to_string(),
        api_key_env: Some(k.to_string()),
        extra_body: json!({"temperature": 0}),
    };
    match provider {
        "ollama" => {
            let raw = std::env::var("WOOLLAMA_OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
            let root = raw.trim_end_matches('/');
            let root = root.strip_suffix("/v1").unwrap_or(root).trim_end_matches('/');
            Some(Inferencer {
                name: "ollama".to_string(),
                base_url: format!("{root}/v1"),
                api_key_env: None,
                extra_body: json!({"options": {"temperature": 0}}),
            })
        }
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

pub fn known_providers() -> &'static [&'static str] {
    &["ollama", "anthropic", "openai", "groq", "together", "openrouter"]
}

/// The built-in provider names (parity with `inferencers.names()`).
pub fn provider_names() -> Vec<String> {
    known_providers().iter().map(|s| s.to_string()).collect()
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

impl Inferencer {
    /// The OpenAI-compatible chat endpoint (`<base_url>/chat/completions`).
    pub fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }
    /// Auth headers from the configured `api_key_env` (empty if none); errors if a
    /// required key env is unset — mirrors Python `Inferencer.headers()`. Used by the
    /// server's passthrough.
    pub fn auth_headers(&self) -> Result<HashMap<String, String>, EngineError> {
        build_headers(self, None).map_err(|e| EngineError::new(e, "invalid_request_error", 400))
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

// --- stateless request build / run -------------------------------------------

/// Everything needed to issue one request. Built synchronously (can fail on
/// unknown provider / missing key) BEFORE going async. Opaque to callers.
pub struct Request {
    url: String,
    body: Value,
    headers: HashMap<String, String>,
    timeout: u64,
    native: bool,
}

pub fn build_request(
    model: &str,
    msgs: Value,
    opts: Option<Value>,
    prms: Option<Value>,
    api_key: Option<&str>,
    base_url: Option<String>,
    stream: bool,
) -> Result<Request, EngineError> {
    let (provider, bare) = split_model(model);
    let inf = get_inferencer(&provider).ok_or_else(|| {
        EngineError::new(
            format!(
                "unknown model namespace: '{model}'. Use 'woollama/<recipe>' or \
                 '<provider>/<model>' for a known inferencer ({}).",
                known_providers().join(", ")
            ),
            "invalid_request_error",
            400,
        )
    })?;
    let base = base_url
        .unwrap_or_else(|| inf.base_url.clone())
        .trim_end_matches('/')
        .to_string();
    let headers = build_headers(&inf, api_key).map_err(|e| EngineError::new(e, "invalid_request_error", 400))?;

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
                body[k] = v.clone();
            }
        }
        (format!("{base}/chat/completions"), body, 180)
    };
    Ok(Request { url, body, headers, timeout, native })
}

/// Pull the assistant text out of the (native or /v1) response, or `Err(msg)` on the
/// upstream-error case.
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
        None if present => Ok(String::new()),
        None => Err(format!("inferencer error: {data}")),
    }
}

/// Run one stateless turn and return the assistant text (async).
pub async fn run_complete(req: Request) -> Result<String, EngineError> {
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
    out.map_err(|e| EngineError::new(e, "server_error", 502))
}

/// Synchronous variant — for non-async embedders. Blocks the calling thread.
pub fn run_complete_blocking(req: Request) -> Result<String, EngineError> {
    let out: Result<String, String> = (|| {
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
    })();
    out.map_err(|e| EngineError::new(e, "server_error", 502))
}

// --- config-driven inferencers (inferencers.toml) + Registry ------------------

fn expanduser(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    } else if p == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return home;
        }
    }
    p.to_string()
}

/// `$config/woollama` — `$WOOLLAMA_CONFIG_DIR` → `$XDG_CONFIG_HOME/woollama` →
/// `~/.config/woollama` (mirrors `config.config_dir`).
pub fn config_dir() -> std::path::PathBuf {
    if let Ok(o) = std::env::var("WOOLLAMA_CONFIG_DIR") {
        if !o.is_empty() {
            return std::path::PathBuf::from(expanduser(&o));
        }
    }
    let base = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{}/.config", std::env::var("HOME").unwrap_or_default()));
    std::path::PathBuf::from(base).join("woollama")
}

/// Expand `${VAR}` from the environment (unset → empty), matching the braced form of
/// `os.path.expandvars`. (Braceless `$VAR` is left as-is — a minor divergence.) Public
/// so the server's `mcp.json` loader can reuse it.
pub fn expand_env(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find("${") {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + 2..];
        match after.find('}') {
            Some(end) => {
                out.push_str(&std::env::var(&after[..end]).unwrap_or_default());
                rest = &after[end + 1..];
            }
            None => {
                out.push_str(&rest[pos..]);
                return out;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Parse `$config/inferencers.toml` into `{name: <spec table>}`. Missing file → `{}`.
fn load_inferencers_toml() -> Result<HashMap<String, serde_json::Map<String, Value>>, EngineError> {
    let path = config_dir().join("inferencers.toml");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return Ok(HashMap::new()),
    };
    let v: Value = toml::from_str(&expand_env(&text)).map_err(|e| {
        EngineError::new(
            format!("inferencers.toml parse error in {}: {e}", path.display()),
            "invalid_request_error",
            400,
        )
    })?;
    let raw = match v.get("inferencers") {
        None => return Ok(HashMap::new()),
        Some(Value::Object(o)) => o,
        Some(_) => {
            return Err(EngineError::new(
                format!("inferencers.toml {}: 'inferencers' must be a table", path.display()),
                "invalid_request_error",
                400,
            ))
        }
    };
    let mut out = HashMap::new();
    for (name, entry) in raw {
        let obj = entry.as_object().ok_or_else(|| {
            EngineError::new(
                format!("inferencers.toml {}: '{name}' must be a table", path.display()),
                "invalid_request_error",
                400,
            )
        })?;
        out.insert(name.clone(), obj.clone());
    }
    Ok(out)
}

/// Built-ins overlaid by `inferencers.toml`, merged FIELD BY FIELD (mirrors
/// `inferencers._registry`): `base_url`/`extra_body` inherit on FALSY; `api_key_env`
/// inherits on ABSENCE. A new provider must supply `base_url`.
fn build_config_registry() -> Result<HashMap<String, Inferencer>, EngineError> {
    let mut reg: HashMap<String, Inferencer> = HashMap::new();
    for name in known_providers() {
        if let Some(inf) = get_inferencer(name) {
            reg.insert(name.to_string(), inf);
        }
    }
    for (name, spec) in load_inferencers_toml()? {
        let base = reg.get(&name).cloned();
        let base_url = spec
            .get("base_url")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| base.as_ref().map(|b| b.base_url.clone()))
            .ok_or_else(|| {
                EngineError::new(
                    format!(
                        "inferencer '{name}' has no base_url and is not a known built-in to \
                         extend — add base_url in inferencers.toml"
                    ),
                    "invalid_request_error",
                    400,
                )
            })?;
        let api_key_env = if spec.contains_key("api_key_env") {
            spec.get("api_key_env")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        } else {
            base.as_ref().and_then(|b| b.api_key_env.clone())
        };
        let extra_body = spec
            .get("extra_body")
            .filter(|v| v.as_object().is_some_and(|o| !o.is_empty()))
            .cloned()
            .unwrap_or_else(|| base.as_ref().map_or_else(|| json!({}), |b| b.extra_body.clone()));
        reg.insert(name.clone(), Inferencer { name, base_url, api_key_env, extra_body });
    }
    Ok(reg)
}

fn inferencer_to_json(inf: &Inferencer) -> Value {
    json!({
        "name": inf.name,
        "base_url": inf.base_url,
        "api_key_env": inf.api_key_env,
        "extra_body": inf.extra_body,
    })
}

/// An explicit, instance-scoped inferencer set — the embeddable alternative to the
/// built-in lookup. Mirrors `inferencers.ModelRegistry`.
#[derive(Default)]
pub struct Registry {
    infs: HashMap<String, Inferencer>,
}

impl Registry {
    pub fn new() -> Self {
        Registry { infs: HashMap::new() }
    }
    /// Built-ins overlaid by `inferencers.toml` — the same set the server resolves.
    pub fn from_config() -> Result<Registry, EngineError> {
        Ok(Registry { infs: build_config_registry()? })
    }
    pub fn add(&mut self, name: String, base_url: String, api_key_env: Option<String>, extra_body: Value) {
        self.infs.insert(name.clone(), Inferencer { name, base_url, api_key_env, extra_body });
    }
    /// The resolved inferencer as a JSON dict, or None.
    pub fn get_json(&self, provider: &str) -> Option<Value> {
        self.infs.get(provider).map(inferencer_to_json)
    }
    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<String> = self.infs.keys().cloned().collect();
        n.sort();
        n
    }
    pub fn all_json(&self) -> Value {
        let map: serde_json::Map<String, Value> =
            self.infs.iter().map(|(k, inf)| (k.clone(), inferencer_to_json(inf))).collect();
        Value::Object(map)
    }
    fn resolve(&self, provider: &str) -> Option<Inferencer> {
        self.infs.get(provider).cloned()
    }
}

// --- the recipe↔tool loop -----------------------------------------------------

/// One progress event from the loop. The wheel/server convert it to their surface
/// shape; the loop stays pure.
pub enum Event {
    Delta(String),
    ToolCall { turn: u32, name: String, args: Value },
    ToolResult { turn: u32, name: String, ok: bool },
    Final(Value),
}

/// Everything the loop needs, resolved EAGERLY before any async work — so an
/// unsupported inferencer / missing key / bad `tool_schemas` fails on the call, not
/// lazily on first iteration. Opaque to callers.
pub struct Setup {
    url: String,
    headers: HashMap<String, String>,
    model: String,
    schemas: Vec<Value>,
    allowed: std::collections::HashSet<String>,
    sorted_allowed: Vec<String>,
    messages: Vec<Value>,
    extra_body: Value,
    params: Option<Value>,
    tools: Arc<dyn ToolProvider>,
}

pub fn build_setup(
    recipe: &Value,
    user_msgs: &Value,
    tools: Arc<dyn ToolProvider>,
    api_key: Option<String>,
    base_url: Option<String>,
    registry: Option<&Registry>,
) -> Result<Setup, EngineError> {
    let inferencer = recipe
        .get("inferencer")
        .and_then(Value::as_str)
        .ok_or_else(|| EngineError::new("recipe missing 'inferencer'", "invalid_request_error", 400))?
        .to_string();
    let system = recipe.get("system").and_then(Value::as_str).unwrap_or("").to_string();
    let tool_names: Vec<String> = recipe
        .get("tools")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    let params: Option<Value> = recipe.get("params").filter(|v| !v.is_null()).cloned();

    let (provider, model) = split_model(&inferencer);
    let inf = match registry {
        Some(reg) => reg.resolve(&provider),
        None => get_inferencer(&provider),
    }
    .ok_or_else(|| {
        let known = match registry {
            Some(reg) => reg.names().join(", "),
            None => known_providers().join(", "),
        };
        EngineError::new(
            format!("unsupported inferencer '{inferencer}' (supported providers: {known}, claude-code)"),
            "not_implemented",
            501,
        )
    })?;
    let headers = build_headers(&inf, api_key.as_deref()).map_err(|e| EngineError::new(e, "invalid_request_error", 400))?;
    let base = base_url
        .unwrap_or_else(|| inf.base_url.clone())
        .trim_end_matches('/')
        .to_string();
    let url = format!("{base}/chat/completions");

    // tool_schemas(allow) → the model-facing schemas (the lossless seam).
    let schemas = tools.tool_schemas(&tool_names)?;
    // The allow-list is the BOUNDARY; the schema's function name IS the namespaced
    // allow-list name, so membership matches directly.
    let allowed: std::collections::HashSet<String> = tool_names.iter().cloned().collect();
    let mut sorted_allowed: Vec<String> = tool_names.clone();
    sorted_allowed.sort();

    // messages = [system] + user_msgs
    let mut messages: Vec<Value> = vec![json!({"role": "system", "content": system})];
    if let Some(arr) = user_msgs.as_array() {
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
        tools,
    })
}

/// One streamed tool call, reassembled across SSE chunks.
#[derive(Default)]
struct CallSlot {
    id: Option<String>,
    name: String,
    arguments: String,
}

fn accumulate_tool_calls(chunk: &Value, slots: &mut std::collections::BTreeMap<i64, CallSlot>) {
    let tcs = chunk
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("tool_calls"))
        .and_then(Value::as_array);
    let Some(tcs) = tcs else { return };
    for tc in tcs {
        let idx = tc.get("index").and_then(Value::as_i64).unwrap_or(0);
        let slot = slots.entry(idx).or_default();
        if let Some(id) = tc.get("id").and_then(Value::as_str) {
            if !id.is_empty() {
                slot.id = Some(id.to_string());
            }
        }
        if let Some(f) = tc.get("function") {
            if let Some(n) = f.get("name").and_then(Value::as_str) {
                if !n.is_empty() {
                    slot.name = n.to_string();
                }
            }
            if let Some(a) = f.get("arguments").and_then(Value::as_str) {
                slot.arguments.push_str(a);
            }
        }
    }
}

fn synthesize_calls(slots: std::collections::BTreeMap<i64, CallSlot>) -> Vec<Value> {
    slots
        .into_iter()
        .map(|(idx, s)| {
            json!({
                "id": s.id.unwrap_or_else(|| format!("call_{idx}")),
                "type": "function",
                "function": {"name": s.name, "arguments": s.arguments},
            })
        })
        .collect()
}

fn chunk_content(chunk: &Value) -> Option<&str> {
    chunk
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("content"))
        .and_then(Value::as_str)
}

/// Drain the next complete line (incl. `\n`) from a RAW BYTE buffer, decoding only
/// that complete line (UTF-8 safety across chunk boundaries). None if no full line yet.
fn take_line(buf: &mut Vec<u8>) -> Option<String> {
    let nl = buf.iter().position(|&b| b == b'\n')?;
    let line: Vec<u8> = buf.drain(..=nl).collect();
    Some(String::from_utf8_lossy(&line).into_owned())
}

/// The recipe↔tool loop as a stream of `Event`s — the SINGLE source of truth.
/// `stream_mode` switches the per-turn transport: SSE (emitting `Delta`s,
/// synthesizing the response, reassembling fragmented tool_calls) vs. one non-stream POST.
pub fn events_stream(setup: Setup, stream_mode: bool) -> impl Stream<Item = Result<Event, EngineError>> + Send {
    let Setup {
        url, headers, model, schemas, allowed, sorted_allowed, messages, extra_body, params, tools,
    } = setup;
    stream! {
        let mut messages = messages;
        let client = match reqwest::Client::builder().timeout(Duration::from_secs(180)).build() {
            Ok(c) => c,
            Err(e) => { yield Err(EngineError::new(e.to_string(), "server_error", 502)); return; }
        };
        for turn in 1..=8u32 {
            // Omit `tools` when the recipe allow-lists none (Anthropic rejects `tools: []`).
            let mut body = json!({"model": model, "messages": messages, "stream": stream_mode});
            if !schemas.is_empty() { body["tools"] = json!(schemas); }
            if let Some(o) = extra_body.as_object() { for (k, v) in o { body[k] = v.clone(); } }
            if let Some(o) = params.as_ref().and_then(Value::as_object) { for (k, v) in o { body[k] = v.clone(); } }

            let mut rb = client.post(&url).json(&body);
            for (k, v) in &headers { rb = rb.header(k, v); }
            let resp = match rb.send().await {
                Ok(r) => r,
                Err(e) => { yield Err(EngineError::new(e.to_string(), "server_error", 502)); return; }
            };

            let content: String;
            let calls: Vec<Value>;
            let response: Value;
            if stream_mode {
                if resp.status().as_u16() >= 400 {
                    let err_body = resp.text().await.unwrap_or_default();
                    yield Err(EngineError::with_payload("inferencer error", "server_error", 502,
                                                        serde_json::from_str::<Value>(&err_body).ok()));
                    return;
                }
                let mut resp = resp;
                let mut buf: Vec<u8> = Vec::new();
                let mut parts: Vec<String> = Vec::new();
                let mut slots: std::collections::BTreeMap<i64, CallSlot> = std::collections::BTreeMap::new();
                let mut flushed = false;
                'read: loop {
                    while let Some(line) = take_line(&mut buf) {
                        let line = line.trim();
                        let Some(payload) = line.strip_prefix("data:") else { continue };
                        let payload = payload.trim();
                        if payload == "[DONE]" { break 'read; }
                        let Ok(chunk) = serde_json::from_str::<Value>(payload) else { continue };
                        if let Some(piece) = chunk_content(&chunk) {
                            if !piece.is_empty() {
                                let piece = piece.to_string();
                                parts.push(piece.clone());
                                yield Ok(Event::Delta(piece));
                            }
                        }
                        accumulate_tool_calls(&chunk, &mut slots);
                    }
                    match resp.chunk().await {
                        Ok(Some(bytes)) => buf.extend_from_slice(&bytes),
                        Ok(None) => {
                            if !buf.is_empty() && !flushed { flushed = true; buf.push(b'\n'); continue; }
                            break 'read;
                        }
                        Err(e) => { yield Err(EngineError::new(e.to_string(), "server_error", 502)); return; }
                    }
                }
                content = parts.join("");
                calls = synthesize_calls(slots);
                response = json!({
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content,
                                    "tool_calls": if calls.is_empty() { Value::Null } else { Value::Array(calls.clone()) }},
                        "finish_reason": if calls.is_empty() { "stop" } else { "tool_calls" },
                    }],
                });
            } else {
                let data: Value = match resp.json().await {
                    Ok(d) => d,
                    Err(e) => { yield Err(EngineError::new(e.to_string(), "server_error", 502)); return; }
                };
                if data.get("choices").is_none() {
                    yield Err(EngineError::with_payload("inferencer error", "server_error", 502, Some(data.clone())));
                    return;
                }
                let msg = &data["choices"][0]["message"];
                calls = msg.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default();
                content = msg.get("content").and_then(Value::as_str).unwrap_or("").to_string();
                response = data;
            }

            if calls.is_empty() {
                yield Ok(Event::Final(response));
                return;
            }
            let metas: Vec<(String, Value, String)> = calls.iter().map(|call| {
                let func = call.get("function");
                let name = func.and_then(|f| f.get("name")).and_then(Value::as_str).unwrap_or("").to_string();
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
                    (format!("ERROR: tool '{name}' is not permitted by this recipe (allowed: {sorted_allowed:?})"), false)
                } else {
                    tools.dispatch(name, args).await
                };
                yield Ok(Event::ToolResult { turn, name: name.clone(), ok });
                messages.push(json!({"role": "tool", "content": result, "tool_call_id": call_id}));
            }
        }
        yield Err(EngineError::new("max turns (8) exceeded", "server_error", 500));
    }
}

// --- streaming (`complete_stream`) -------------------------------------------

enum Delta {
    Yield(String),
    NeedMore,
    Done,
}

/// Pull the next non-empty delta content from complete SSE lines in the RAW BYTE
/// `buf` (consuming them); leaves any trailing partial line. `[DONE]` → `Done`.
fn next_delta(buf: &mut Vec<u8>) -> Delta {
    while let Some(line) = take_line(buf) {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if data == "[DONE]" {
                return Delta::Done;
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(data) {
                if let Some(piece) = chunk_content(&chunk) {
                    if !piece.is_empty() {
                        return Delta::Yield(piece.to_string());
                    }
                }
            }
        }
    }
    Delta::NeedMore
}

/// Stream one stateless turn as assistant text deltas (the inferencer's `/v1` SSE).
/// Lazy: nothing is POSTed until the first poll (so a setup/4xx error surfaces on the
/// first `.next()`, matching the Python generator). A 4xx yields one Err (carrying the
/// real upstream status + payload) then ends.
pub fn complete_stream_events(req: Request) -> impl Stream<Item = Result<String, EngineError>> + Send {
    stream! {
        let client = match reqwest::Client::builder().timeout(Duration::from_secs(req.timeout)).build() {
            Ok(c) => c,
            Err(e) => { yield Err(EngineError::new(e.to_string(), "server_error", 502)); return; }
        };
        let mut rb = client.post(&req.url).json(&req.body);
        for (k, v) in &req.headers { rb = rb.header(k, v); }
        let resp = match rb.send().await {
            Ok(r) => r,
            Err(e) => { yield Err(EngineError::new(e.to_string(), "server_error", 502)); return; }
        };
        if resp.status().as_u16() >= 400 {
            let status = resp.status().as_u16() as i64;
            let body = resp.text().await.unwrap_or_default();
            yield Err(EngineError::with_payload("inferencer error", "server_error", status,
                                                serde_json::from_str::<Value>(&body).ok()));
            return;
        }
        let mut resp = resp;
        let mut buf: Vec<u8> = Vec::new();
        loop {
            match next_delta(&mut buf) {
                Delta::Yield(piece) => yield Ok(piece),
                Delta::Done => return,
                Delta::NeedMore => match resp.chunk().await {
                    Ok(Some(bytes)) => buf.extend_from_slice(&bytes),
                    Ok(None) => return,
                    Err(e) => { yield Err(EngineError::new(e.to_string(), "server_error", 502)); return; }
                },
            }
        }
    }
}
