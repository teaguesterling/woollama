//! The woollama router service (Rust) — the axum HTTP surface over `woollama-engine`.
//!
//! Surface (slices 2–8, see docs/rust-router-port.md):
//!   - `GET /v1/models` — inferencer discovery (static + live) + recipes (slice 8).
//!   - `POST /v1/chat/completions` — passthrough (`<provider>/<model>`, incl. native
//!     num_ctx + streaming) and `woollama/<recipe>` orchestration (incl. streaming),
//!     dispatching tools to the downstream MCP registry; claude-code recipes execute
//!     via the claude CLI.
//!   - `POST /v1/responses` — stateless (incl. streaming) and STATEFUL (claude-resume,
//!     store-backed, managed-agents) with the requires_action pause/resume.
//!   - `/v1/conversations` CRUD + `/items` — the durable handle table.
//!   - `/mcp` — woollama AS an MCP server (Streamable-HTTP), plus a `mcp` stdio subcommand.
//!
//! Remaining: the Unix-socket surface; the cutover (slice 9). Managed-agents' Anthropic
//! wire format is best-effort pending live reconciliation (see managed_agents.rs).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value};

use woollama_engine as engine;
use engine::EngineError;

mod claude_code;
mod config;
mod conversations;
mod managed_agents;
mod mcp_registry;
mod mcp_surface;
mod ollama_native;
mod responses;

use mcp_surface::WoollamaMcp;

pub use config::{load_mcp_servers, load_recipes};

/// Shared, process-lifetime server state: loaded recipes, the connected downstream MCP
/// registry, and the inferencer registry. Built once at startup, shared via axum state.
pub struct AppState {
    pub recipes: HashMap<String, config::Recipe>,
    pub registry: Arc<mcp_registry::McpRegistry>,
    pub inferencers: engine::Registry,
    /// The mcp.json specs (for claude-code delegation, which writes a per-recipe
    /// --mcp-config from the referenced subset).
    pub mcp_specs: HashMap<String, config::McpServerSpec>,
    /// The durable conversation handle table (stateful /v1/responses + /v1/conversations).
    pub conversations: Arc<conversations::Conversations>,
    /// An external conversation store (issue #2), wired from mcp.json's
    /// `conversationStore`. When present, non-claude models become stateful (store-backed).
    pub store: Option<Arc<dyn conversations::StoreProvider>>,
    /// The Anthropic Managed Agents backend (claude-agent/* models). Paid; errors at
    /// turn time if ANTHROPIC_API_KEY is unset.
    pub managed_agents: Arc<managed_agents::ManagedAgents>,
}

impl AppState {
    fn backend_for_model(&self, model: &str) -> Option<&'static str> {
        conversations::backend_for_model(model, self.store.is_some())
    }
}

/// Load config + connect the downstream MCP servers. Errors are logged and degraded to
/// empty (the router still starts) rather than fatal.
pub async fn build_state() -> AppState {
    // Resolve WOOLLAMA_EXAMPLES_DIR before any config load — the bundled mcp.json expands it.
    config::ensure_examples_dir();
    let recipes = config::load_recipes().unwrap_or_else(|e| {
        eprintln!("woollamad: recipes load error: {e}");
        HashMap::new()
    });
    let specs = config::load_mcp_servers().unwrap_or_else(|e| {
        eprintln!("woollamad: mcp.json load error: {e}");
        HashMap::new()
    });
    let registry = Arc::new(mcp_registry::McpRegistry::connect(specs.clone()).await);
    let inferencers = engine::Registry::from_config().unwrap_or_else(|e| {
        eprintln!("woollamad: inferencers load error: {e}");
        engine::Registry::new()
    });
    // Durable handle table at $WOOLLAMA_STATE_DIR/conversations.json (in-memory if unset).
    let state_path = std::env::var("WOOLLAMA_STATE_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|d| std::path::PathBuf::from(d).join("conversations.json"));
    let conversations = Arc::new(conversations::Conversations::new(state_path));
    // Optional external conversation store (makes non-claude models stateful).
    let store: Option<Arc<dyn conversations::StoreProvider>> = match config::load_conversation_store() {
        Ok(Some(config::ConvStoreConfig::Http { url })) => Some(Arc::new(conversations::HttpStoreProvider::new(&url))),
        Ok(Some(config::ConvStoreConfig::Mcp { server })) => {
            Some(Arc::new(conversations::McpStoreProvider::new(registry.clone(), server)))
        }
        Ok(None) => None,
        Err(e) => {
            eprintln!("woollamad: conversationStore config error: {e}");
            None
        }
    };
    let managed_agents = Arc::new(managed_agents::ManagedAgents::new());
    AppState { recipes, registry, inferencers, mcp_specs: specs, conversations, store, managed_agents }
}

/// The TCP host/port to bind — `$WOOLLAMA_ADDRESS=host[:port]`, else `127.0.0.1:0`.
pub fn resolve_tcp_target() -> (String, u16) {
    match std::env::var("WOOLLAMA_ADDRESS") {
        Ok(addr) if !addr.is_empty() => match addr.split_once(':') {
            Some((host, port)) => (
                if host.is_empty() { "127.0.0.1".to_string() } else { host.to_string() },
                port.parse().unwrap_or(0),
            ),
            None => (addr, 0),
        },
        _ => ("127.0.0.1".to_string(), 0),
    }
}

/// The axum app (shared by the binary and the integration tests). Mounts woollama's own
/// MCP surface at `/mcp` (Streamable-HTTP) on the same port — the per-session factory
/// shares the one `AppState` (and thus the one downstream registry).
pub fn router(state: Arc<AppState>) -> Router {
    use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
    use rmcp::transport::streamable_http_server::StreamableHttpService;

    let mcp_state = state.clone();
    let mcp_svc = StreamableHttpService::new(
        move || Ok(WoollamaMcp { state: mcp_state.clone() }),
        Arc::new(LocalSessionManager::default()),
        Default::default(),
    );
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses_create))
        .route("/v1/conversations", post(conversations_create).get(conversations_list))
        .route("/v1/conversations/{conv_id}", get(conversations_get).delete(conversations_delete))
        .route("/v1/conversations/{conv_id}/items", get(conversations_items))
        .nest_service("/mcp", mcp_svc)
        .with_state(state)
}

/// Serve woollama's MCP surface over stdio — the `woollamad mcp` subcommand (what
/// an MCP client puts in its mcp.json). stdout is the JSON-RPC channel; logs go to stderr.
pub async fn serve_mcp_stdio(state: Arc<AppState>) -> Result<(), Box<dyn std::error::Error>> {
    let running = rmcp::serve_server(WoollamaMcp { state }, (tokio::io::stdin(), tokio::io::stdout())).await?;
    running.waiting().await?;
    Ok(())
}

// --- error helpers ------------------------------------------------------------

fn err_response(status: StatusCode, message: impl Into<String>, kind: &str) -> Response {
    (status, Json(json!({"error": {"message": message.into(), "type": kind}}))).into_response()
}

fn engine_err_response(e: EngineError) -> Response {
    let status = StatusCode::from_u16(e.status.unwrap_or(500) as u16)
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let kind = e.kind.clone().unwrap_or_else(|| "server_error".to_string());
    match e.payload {
        Some(payload) => (status, Json(payload)).into_response(),
        None => err_response(status, e.message, &kind),
    }
}

async fn forward_post(
    url: String,
    body: &Value,
    headers: &HashMap<String, String>,
    timeout: u64,
) -> Result<reqwest::Response, Response> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout))
        .build()
        .map_err(|e| err_response(StatusCode::BAD_GATEWAY, e.to_string(), "server_error"))?;
    let mut rb = client.post(url).json(body);
    for (k, v) in headers {
        rb = rb.header(k, v);
    }
    rb.send()
        .await
        .map_err(|e| err_response(StatusCode::BAD_GATEWAY, e.to_string(), "server_error"))
}

async fn relay_json(resp: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let data: Value = resp.json().await.unwrap_or_else(|_| json!({}));
    (status, Json(data)).into_response()
}

// --- SSE helpers --------------------------------------------------------------

fn now_secs() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn chatcmpl_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4().simple())
}

/// Next complete `\n`-terminated line from a raw byte buffer (UTF-8-safe), or None.
fn take_line(buf: &mut Vec<u8>) -> Option<String> {
    let nl = buf.iter().position(|&b| b == b'\n')?;
    let line: Vec<u8> = buf.drain(..=nl).collect();
    Some(String::from_utf8_lossy(&line).into_owned())
}

/// One OpenAI `chat.completion.chunk` SSE frame.
fn chat_chunk(cid: &str, created: i64, model: &str, delta: Value, finish: Option<&str>) -> Bytes {
    let payload = json!({
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    });
    Bytes::from(format!("data: {}\n\n", serde_json::to_string(&payload).unwrap()))
}

fn sse_response(body: Body) -> Response {
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(body)
        .unwrap_or_else(|_| err_response(StatusCode::BAD_GATEWAY, "stream build failed", "server_error"))
}

/// num_ctx + stream → native `/api/chat` NDJSON, translated frame-by-frame to OpenAI SSE.
async fn native_stream(
    inf: &engine::Inferencer,
    fwd: &Value,
    headers: &HashMap<String, String>,
    model: &str,
) -> Response {
    let url = ollama_native::native_chat_url(&inf.base_url);
    let req = ollama_native::to_native_request(fwd);
    let resp = match forward_post(url, &req, headers, 600).await {
        Ok(r) => r,
        Err(e) => return e,
    };
    if resp.status().as_u16() >= 400 {
        return relay_json(resp).await;
    }
    let model = model.to_string();
    let body = Body::from_stream(stream! {
        let mut t = ollama_native::SseTranslator::new(&model);
        let mut buf: Vec<u8> = Vec::new();
        let mut bs = resp.bytes_stream();
        while let Some(chunk) = bs.next().await {
            let Ok(bytes) = chunk else { break };
            buf.extend_from_slice(&bytes);
            while let Some(line) = take_line(&mut buf) {
                for out in t.translate(&line) {
                    yield Ok::<Bytes, std::io::Error>(Bytes::from(out));
                }
            }
        }
    });
    sse_response(body)
}

/// woollama/<recipe> + stream → run the loop over SSE and emit `chat.completion.chunk`
/// frames: a role chunk, the content deltas, then exactly one stop terminator + [DONE].
/// Primed before returning so a setup/first-turn error maps to an HTTP status.
async fn orchestrate_stream(
    state: Arc<AppState>,
    recipe: config::Recipe,
    messages: Value,
    model: String,
) -> Response {
    // claude-code is non-streaming: run it, then surface the whole answer as one delta.
    if let Some(cc_model) = recipe.inferencer.strip_prefix("claude-code/") {
        let text = match run_claude_recipe(&state, &recipe, &messages, cc_model).await {
            Ok(resp) => resp["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string(),
            Err(e) => return engine_err_response(e),
        };
        let cid = chatcmpl_id();
        let created = now_secs();
        let body = Body::from_stream(stream! {
            yield Ok::<Bytes, std::io::Error>(chat_chunk(&cid, created, &model, json!({"role": "assistant"}), None));
            if !text.is_empty() {
                yield Ok(chat_chunk(&cid, created, &model, json!({"content": text}), None));
            }
            yield Ok(chat_chunk(&cid, created, &model, json!({}), Some("stop")));
            yield Ok(Bytes::from("data: [DONE]\n\n"));
        });
        return sse_response(body);
    }
    let recipe_val = recipe.to_value();
    let provider: Arc<dyn engine::ToolProvider> =
        Arc::new(mcp_registry::RegistryToolProvider { reg: state.registry.clone() });
    let setup = match engine::build_setup(&recipe_val, &messages, provider, None, None, Some(&state.inferencers)) {
        Ok(s) => s,
        Err(e) => return engine_err_response(e),
    };
    let mut s = Box::pin(engine::events_stream(setup, true));
    let first_ev = match s.next().await {
        Some(Err(e)) => return engine_err_response(e),
        Some(Ok(ev)) => Some(ev),
        None => None,
    };
    let cid = chatcmpl_id();
    let created = now_secs();
    let body = Body::from_stream(stream! {
        yield Ok::<Bytes, std::io::Error>(chat_chunk(&cid, created, &model, json!({"role": "assistant"}), None));
        if let Some(engine::Event::Delta(c)) = first_ev {
            yield Ok(chat_chunk(&cid, created, &model, json!({"content": c}), None));
        }
        while let Some(item) = s.next().await {
            match item {
                Ok(engine::Event::Delta(c)) => {
                    yield Ok(chat_chunk(&cid, created, &model, json!({"content": c}), None));
                }
                Ok(_) => {}
                Err(e) => {
                    let payload = e.payload.clone().unwrap_or_else(|| json!({"error": {"message": e.message, "type": e.kind}}));
                    yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&payload).unwrap())));
                    break;
                }
            }
        }
        yield Ok(chat_chunk(&cid, created, &model, json!({}), Some("stop")));
        yield Ok(Bytes::from("data: [DONE]\n\n"));
    });
    sse_response(body)
}

/// Stream a stateless /v1/responses turn as OpenAI Responses SSE (the canonical
/// created → output_item.added → content_part.added → output_text.delta* →
/// output_text.done → content_part.done → output_item.done → completed sequence).
/// Primed before returning so a setup error maps to an HTTP status.
async fn responses_stream(mut source: BoxStream<'static, Result<String, EngineError>>, model: String) -> Response {
    let first_delta = match source.next().await {
        Some(Err(e)) => return engine_err_response(e),
        Some(Ok(d)) => Some(d),
        None => None,
    };
    let resp_id = responses::new_id("resp");
    let item_id = responses::new_id("msg");
    let created = now_secs();
    let body = Body::from_stream(stream! {
        let mut seq = 0i64;
        yield Ok::<Bytes, std::io::Error>(responses::resp_ev("response.created", seq,
            json!({"response": responses::build_response_full(&resp_id, &model, "", "in_progress", created)}))); seq += 1;
        yield Ok(responses::resp_ev("response.output_item.added", seq,
            json!({"output_index": 0, "item": responses::msg_item(&item_id, "", false)}))); seq += 1;
        yield Ok(responses::resp_ev("response.content_part.added", seq,
            json!({"item_id": item_id, "output_index": 0, "content_index": 0,
                   "part": {"type": "output_text", "text": "", "annotations": []}}))); seq += 1;

        let mut chunks: Vec<String> = Vec::new();
        if let Some(d) = first_delta {
            chunks.push(d.clone());
            yield Ok(responses::resp_ev("response.output_text.delta", seq,
                json!({"item_id": item_id, "output_index": 0, "content_index": 0, "logprobs": [], "delta": d}))); seq += 1;
        }
        while let Some(item) = source.next().await {
            match item {
                Ok(piece) => {
                    chunks.push(piece.clone());
                    yield Ok(responses::resp_ev("response.output_text.delta", seq,
                        json!({"item_id": item_id, "output_index": 0, "content_index": 0, "logprobs": [], "delta": piece}))); seq += 1;
                }
                Err(e) => {
                    yield Ok(responses::resp_ev("error", seq, json!({"message": e.message, "code": e.kind}))); seq += 1;
                    break;
                }
            }
        }
        let full = chunks.concat();
        yield Ok(responses::resp_ev("response.output_text.done", seq,
            json!({"item_id": item_id, "output_index": 0, "content_index": 0, "logprobs": [], "text": full}))); seq += 1;
        yield Ok(responses::resp_ev("response.content_part.done", seq,
            json!({"item_id": item_id, "output_index": 0, "content_index": 0,
                   "part": {"type": "output_text", "text": full, "annotations": []}}))); seq += 1;
        yield Ok(responses::resp_ev("response.output_item.done", seq,
            json!({"output_index": 0, "item": responses::msg_item(&item_id, &full, true)}))); seq += 1;
        yield Ok(responses::resp_ev("response.completed", seq,
            json!({"response": responses::build_response_full(&resp_id, &model, &full, "completed", created)})));
    });
    sse_response(body)
}

// --- orchestration (shared by chat-completions + responses) -------------------

/// The mcp.json `{command, args}` for the servers a recipe's tools reference (the subset
/// claude-code delegation hands the child). Errors if a referenced server isn't configured.
fn referenced_mcp_servers(state: &AppState, tools: &[String]) -> Result<HashMap<String, Value>, EngineError> {
    let mut servers = HashMap::new();
    for t in tools {
        let server = t.split_once('.').map(|(s, _)| s).unwrap_or(t.as_str());
        if servers.contains_key(server) {
            continue;
        }
        let Some(spec) = state.mcp_specs.get(server) else {
            return Err(EngineError::new(
                format!("recipe references MCP server '{server}' not in mcp.json config"),
                "invalid_request_error",
                400,
            ));
        };
        servers.insert(server.to_string(), json!({"command": spec.command, "args": spec.args}));
    }
    Ok(servers)
}

/// Run a `claude-code/<model>` recipe: tool-less completion, or delegation when the
/// recipe allow-lists tools (Claude owns the loop). Returns an OpenAI dict.
async fn run_claude_recipe(
    state: &AppState,
    recipe: &config::Recipe,
    messages: &Value,
    model: &str,
) -> Result<Value, EngineError> {
    let cc_err = |e: claude_code::ClaudeCodeError| EngineError::new(format!("claude-code backend: {e}"), "server_error", 502);
    if recipe.tools.is_empty() {
        claude_code::run_completion(&recipe.system, messages, model).await.map_err(cc_err)
    } else {
        let servers = referenced_mcp_servers(state, &recipe.tools)?;
        claude_code::run_delegated(&recipe.system, messages, model, &recipe.tools, &servers, 8)
            .await
            .map_err(cc_err)
    }
}

/// Run a recipe to completion and return the final OpenAI response dict. A
/// `claude-code/<model>` recipe runs through the executor; otherwise tools are
/// dispatched to the downstream MCP registry via the engine loop. Shared by the HTTP
/// handlers and the MCP `chat` tool; each maps the `EngineError` to its own surface.
pub(crate) async fn orchestrate_recipe(
    state: &AppState,
    recipe: &config::Recipe,
    messages: &Value,
) -> Result<Value, EngineError> {
    if let Some(cc_model) = recipe.inferencer.strip_prefix("claude-code/") {
        return run_claude_recipe(state, recipe, messages, cc_model).await;
    }
    let recipe_val = recipe.to_value();
    let provider: Arc<dyn engine::ToolProvider> =
        Arc::new(mcp_registry::RegistryToolProvider { reg: state.registry.clone() });
    let setup = engine::build_setup(&recipe_val, messages, provider, None, None, Some(&state.inferencers))?;
    let mut s = Box::pin(engine::events_stream(setup, false));
    while let Some(item) = s.next().await {
        match item {
            Ok(engine::Event::Final(resp)) => return Ok(resp),
            Ok(_) => continue,
            Err(e) => return Err(e),
        }
    }
    Err(EngineError::new("orchestrate: loop ended without a final response", "server_error", 500))
}

// --- GET /v1/models -----------------------------------------------------------

/// `GET /v1/models` (slice 8): each inferencer's opted-in models (static `models` +
/// optional live `discover`, namespaced `provider/<id>`) plus `woollama/<recipe>`.
async fn list_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut data: Vec<Value> = Vec::new();
    for inf in state.inferencers.list() {
        let mut seen = std::collections::HashSet::new();
        let mut ids = inf.models.clone();
        if inf.discover {
            if let Ok(found) = discover_models(&inf).await {
                ids.extend(found);
            }
        }
        for id in ids {
            if seen.insert(id.clone()) {
                data.push(json!({"id": format!("{}/{id}", inf.name), "object": "model", "owned_by": inf.name}));
            }
        }
    }
    let mut recipe_names: Vec<String> = state.recipes.keys().cloned().collect();
    recipe_names.sort();
    for r in recipe_names {
        data.push(json!({"id": format!("woollama/{r}"), "object": "model", "owned_by": "woollama"}));
    }
    Json(json!({"object": "list", "data": data}))
}

/// Live-query a provider's own `/v1/models`, filtered by its `model_patterns` (empty =
/// all). Errors (missing key / unreachable / non-200) are the caller's cue to skip.
async fn discover_models(inf: &engine::Inferencer) -> Result<Vec<String>, ()> {
    let headers = inf.auth_headers().map_err(|_| ())?;
    let client = reqwest::Client::builder().timeout(Duration::from_secs(10)).build().map_err(|_| ())?;
    let mut rb = client.get(format!("{}/models", inf.base_url.trim_end_matches('/')));
    for (k, v) in &headers {
        rb = rb.header(k, v);
    }
    let r = rb.send().await.map_err(|_| ())?;
    if !r.status().is_success() {
        return Err(());
    }
    let v: Value = r.json().await.map_err(|_| ())?;
    let mut ids: Vec<String> = v
        .get("data")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|m| m.get("id").and_then(Value::as_str).map(String::from)).collect())
        .unwrap_or_default();
    if !inf.model_patterns.is_empty() {
        ids.retain(|id| inf.model_patterns.iter().any(|p| fnmatch(p, id)));
    }
    Ok(ids)
}

/// fnmatch-style glob (`*` any run, `?` one char) — mirrors Python's `fnmatch` filtering.
fn fnmatch(pattern: &str, name: &str) -> bool {
    fn m(p: &[u8], n: &[u8]) -> bool {
        match p.split_first() {
            None => n.is_empty(),
            Some((b'*', rest)) => m(rest, n) || (!n.is_empty() && m(p, &n[1..])),
            Some((b'?', rest)) => !n.is_empty() && m(rest, &n[1..]),
            Some((c, rest)) => !n.is_empty() && n[0] == *c && m(rest, &n[1..]),
        }
    }
    m(pattern.as_bytes(), name.as_bytes())
}

// --- POST /v1/chat/completions ------------------------------------------------

async fn chat_completions(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();

    if let Some(name) = model.strip_prefix("woollama/") {
        let Some(recipe) = state.recipes.get(name) else {
            return err_response(StatusCode::NOT_FOUND, format!("unknown recipe '{name}'"), "not_found");
        };
        let messages = body.get("messages").cloned().unwrap_or_else(|| json!([]));
        if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
            return orchestrate_stream(state.clone(), recipe.clone(), messages, model).await;
        }
        return match orchestrate_recipe(&state, recipe, &messages).await {
            Ok(resp) => Json(resp).into_response(),
            Err(e) => engine_err_response(e),
        };
    }

    let provider = model.split('/').next().unwrap_or("");
    let Some(inf) = engine::get_inferencer(provider) else {
        return err_response(
            StatusCode::BAD_REQUEST,
            format!(
                "unknown model namespace: '{model}'. Use 'woollama/<recipe>' or \
                 '<provider>/<model>' for a known inferencer ({}).",
                engine::provider_names().join(", ")
            ),
            "invalid_request_error",
        );
    };

    let headers = match inf.auth_headers() {
        Ok(h) => h,
        Err(e) => return engine_err_response(e),
    };

    let bare = model.splitn(2, '/').nth(1).unwrap_or("").to_string();
    let mut fwd = body.clone();
    fwd["model"] = json!(bare);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    if provider == "ollama" && ollama_native::wants_native(&fwd) {
        if stream {
            return native_stream(&inf, &fwd, &headers, &bare).await;
        }
        return passthrough_native(&inf, &fwd, &headers, &bare).await;
    }

    if stream {
        return passthrough_stream(&inf, &fwd, &headers).await;
    }

    fwd["stream"] = json!(false);
    match forward_post(inf.chat_url(), &fwd, &headers, 180).await {
        Ok(resp) => relay_json(resp).await,
        Err(e) => e,
    }
}

async fn passthrough_native(
    inf: &engine::Inferencer,
    fwd: &Value,
    headers: &HashMap<String, String>,
    model: &str,
) -> Response {
    let url = ollama_native::native_chat_url(&inf.base_url);
    let req = ollama_native::to_native_request(fwd);
    let resp = match forward_post(url, &req, headers, 600).await {
        Ok(r) => r,
        Err(e) => return e,
    };
    if resp.status().as_u16() >= 400 {
        return relay_json(resp).await;
    }
    match resp.json::<Value>().await {
        Ok(native) => Json(ollama_native::from_native_response(&native, model)).into_response(),
        Err(e) => err_response(StatusCode::BAD_GATEWAY, e.to_string(), "server_error"),
    }
}

async fn passthrough_stream(
    inf: &engine::Inferencer,
    fwd: &Value,
    headers: &HashMap<String, String>,
) -> Response {
    let resp = match forward_post(inf.chat_url(), fwd, headers, 180).await {
        Ok(r) => r,
        Err(e) => return e,
    };
    if resp.status().as_u16() >= 400 {
        return relay_json(resp).await;
    }
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(resp.bytes_stream()))
        .unwrap_or_else(|_| err_response(StatusCode::BAD_GATEWAY, "stream build failed", "server_error"))
}

// --- POST /v1/responses (stateless, non-stream) -------------------------------

async fn responses_create(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let input = body.get("input").cloned().unwrap_or_else(|| json!(""));
    let messages = match responses::parse_input(&input) {
        Ok(m) => m,
        Err(e) => return err_response(StatusCode::BAD_REQUEST, e, "invalid_request_error"),
    };

    let nonnull = |k: &str| body.get(k).map_or(false, |v| !v.is_null());
    let stateful = body.get("store").and_then(Value::as_bool).unwrap_or(false)
        || nonnull("conversation")
        || nonnull("previous_response_id")
        || nonnull("key");
    if stateful {
        if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
            return err_response(
                StatusCode::BAD_REQUEST,
                "streaming is not supported for STATEFUL /v1/responses conversations",
                "invalid_request_error",
            );
        }
        return responses_stateful(&state, &body, &model, &json!(messages)).await;
    }
    if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        let source: BoxStream<'static, Result<String, EngineError>> =
            if let Some(name) = model.strip_prefix("woollama/") {
                let Some(recipe) = state.recipes.get(name) else {
                    return err_response(StatusCode::NOT_FOUND, format!("unknown recipe '{name}'"), "not_found");
                };
                if let Some(cc_model) = recipe.inferencer.strip_prefix("claude-code/") {
                    // claude-code is non-streaming: the answer is one delta.
                    let text = match run_claude_recipe(&state, recipe, &json!(messages), cc_model).await {
                        Ok(resp) => resp["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string(),
                        Err(e) => return engine_err_response(e),
                    };
                    futures::stream::once(async move { Ok::<String, EngineError>(text) }).boxed()
                } else {
                    let recipe_val = recipe.to_value();
                    let provider: Arc<dyn engine::ToolProvider> =
                        Arc::new(mcp_registry::RegistryToolProvider { reg: state.registry.clone() });
                    let setup = match engine::build_setup(&recipe_val, &json!(messages), provider, None, None, Some(&state.inferencers)) {
                        Ok(s) => s,
                        Err(e) => return engine_err_response(e),
                    };
                    engine::events_stream(setup, true)
                        .filter_map(|item| async move {
                            match item {
                                Ok(engine::Event::Delta(c)) => Some(Ok(c)),
                                Ok(_) => None,
                                Err(e) => Some(Err(e)),
                            }
                        })
                        .boxed()
                }
            } else {
                let options = body.get("options").cloned();
                let req = match engine::build_request(&model, json!(messages), options, None, None, None, true) {
                    Ok(r) => r,
                    Err(e) => return engine_err_response(e),
                };
                engine::complete_stream_events(req).boxed()
            };
        return responses_stream(source, model).await;
    }

    let resp_id = responses::new_id("resp");

    // woollama/<recipe> → orchestrate; extract the final assistant text.
    if let Some(name) = model.strip_prefix("woollama/") {
        let Some(recipe) = state.recipes.get(name) else {
            return err_response(StatusCode::NOT_FOUND, format!("unknown recipe '{name}'"), "not_found");
        };
        return match orchestrate_recipe(&state, recipe, &json!(messages)).await {
            Ok(resp) => {
                let text = resp["choices"][0]["message"]["content"].as_str().unwrap_or("");
                Json(responses::build_response(&resp_id, &model, text)).into_response()
            }
            Err(e) => engine_err_response(e),
        };
    }

    // Stateless inferencer turn — the engine's complete handles native num_ctx.
    let options = body.get("options").cloned();
    let req = match engine::build_request(&model, json!(messages), options, None, None, None, false) {
        Ok(r) => r,
        Err(e) => return engine_err_response(e),
    };
    match engine::run_complete(req).await {
        Ok(text) => Json(responses::build_response(&resp_id, &model, &text)).into_response(),
        Err(e) => engine_err_response(e),
    }
}

// --- stateful conversations (slice 6a) ----------------------------------------

fn no_stateful_backend_msg(model: &str) -> String {
    format!(
        "no stateful backend for model '{model}': only claude-code (claude-resume) has one \
         in this build (managed-agents + store-backed are later slices). Use store:false \
         (the caller owns history)."
    )
}

/// Run one stateful /v1/responses turn: resolve/create the conversation handle, run the
/// turn on its backend under a per-conversation write lock, return the Responses object
/// carrying the conversation id. Slice 6a: the claude-resume backend only.
async fn responses_stateful(state: &AppState, body: &Value, model: &str, messages: &Value) -> Response {
    let conv_id_param = body.get("conversation").and_then(Value::as_str);
    let prev = body.get("previous_response_id").and_then(Value::as_str);
    let key = body.get("key").and_then(Value::as_str);

    // Resolve or create the conversation handle (explicit id wins, then prev, then key,
    // else a new one whose backend is derived from the model).
    let conv = {
        let mut t = state.conversations.table.lock().await;
        if let Some(cid) = conv_id_param {
            match t.get(cid) {
                Some(c) => c,
                None => return err_response(StatusCode::NOT_FOUND, format!("unknown conversation '{cid}'"), "not_found"),
            }
        } else if let Some(p) = prev {
            match t.by_response(p) {
                Some(c) => c,
                None => return err_response(StatusCode::NOT_FOUND, format!("unknown previous_response_id '{p}'"), "not_found"),
            }
        } else {
            let Some(backend) = state.backend_for_model(model) else {
                return err_response(StatusCode::NOT_IMPLEMENTED, no_stateful_backend_msg(model), "not_implemented");
            };
            match key {
                Some(k) => t.get_or_create_by_alias(k, backend, model),
                None => t.create(backend, model, json!({}), None, None),
            }
        }
    };

    // One writer per conversation: hold the per-conv lock across the turn (but NOT the
    // table lock, which only guards brief reads/writes — each backend turn does its own).
    let lock = state.conversations.conv_lock(&conv.id).await;
    let _guard = lock.lock().await;

    let options = body.get("options").cloned();
    let turn: Result<(String, Option<Value>), EngineError> = match conv.backend.as_str() {
        "claude-resume" => claude_resume_turn(state, &conv.id, &conv.model, messages).await.map(|t| (t, None)),
        "store-backed" => store_backed_turn(state, &conv.id, &conv.model, messages, options).await.map(|t| (t, None)),
        "managed-agents" => managed_agents_turn(state, &conv, messages).await,
        other => {
            return err_response(
                StatusCode::NOT_IMPLEMENTED,
                format!("the '{other}' backend is not in the Rust server"),
                "not_implemented",
            )
        }
    };
    let (text, required_action) = match turn {
        Ok(x) => x,
        Err(e) => return engine_err_response(e),
    };

    let resp_id = responses::new_id("resp");
    {
        let mut t = state.conversations.table.lock().await;
        t.record_response(&conv.id, &resp_id);
    }
    let status = if required_action.is_some() { "requires_action" } else { "completed" };
    Json(responses::build_response_stateful(&resp_id, &conv.model, &text, &conv.id, status, required_action))
        .into_response()
}

/// The latest user message text — for a stateful turn the backend already owns prior
/// history, so woollama sends only the new user input.
fn latest_user_text(messages: &Value) -> String {
    let Some(arr) = messages.as_array() else { return String::new() };
    for m in arr.iter().rev() {
        if m.get("role").and_then(Value::as_str) == Some("user") {
            return m.get("content").and_then(Value::as_str).unwrap_or("").to_string();
        }
    }
    arr.last().and_then(|m| m.get("content").and_then(Value::as_str)).unwrap_or("").to_string()
}

/// One managed-agents turn: resume a paused session with the answer (if awaiting input),
/// else run a fresh turn (creating the hosted session lazily). Returns the text + the
/// `required_action` payload when the agent paused on ask_user.
async fn managed_agents_turn(state: &AppState, conv: &conversations::Conversation, messages: &Value) -> Result<(String, Option<Value>), EngineError> {
    let ma = &state.managed_agents;
    let to_err = |e: managed_agents::ManagedAgentsError| EngineError::new(format!("managed-agents backend: {e}"), "server_error", 502);

    let mut native_id = conv.native_id.clone();
    let turn = if conv.status == "awaiting_input" && conv.pending_tool_use_id.is_some() {
        ma.answer_turn(
            native_id.as_deref().unwrap_or(""),
            conv.pending_tool_use_id.as_deref().unwrap(),
            &latest_user_text(messages),
        )
        .await
        .map_err(to_err)?
    } else {
        if native_id.is_none() {
            native_id = Some(ma.create_session(&conv.model, conv.title.as_deref(), &conv.metadata).await.map_err(to_err)?);
        }
        ma.run_turn(native_id.as_deref().unwrap(), &latest_user_text(messages)).await.map_err(to_err)?
    };

    let (status, required_action, pending_id) = match &turn.pending {
        Some(p) => (
            "awaiting_input".to_string(),
            Some(json!({"type": "ask_user", "question": p.input})),
            Some(p.id.clone()),
        ),
        None => ("idle".to_string(), None, None),
    };
    state.conversations.table.lock().await.set_managed(&conv.id, native_id, status, required_action.clone(), pending_id);
    Ok((turn.text, required_action))
}

/// One claude-resume turn: ensure a stable workdir, `--resume` the session, persist the
/// captured/echoed session_id.
async fn claude_resume_turn(state: &AppState, conv_id: &str, model: &str, messages: &Value) -> Result<String, EngineError> {
    let (mut native_id, mut workdir) = {
        let t = state.conversations.table.lock().await;
        let c = t.get(conv_id);
        (c.as_ref().and_then(|c| c.native_id.clone()), c.and_then(|c| c.workdir.clone()))
    };
    if workdir.is_none() {
        let dir = std::env::temp_dir().join(format!("woollama-conv-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).map_err(|e| EngineError::new(e.to_string(), "server_error", 500))?;
        workdir = Some(dir.to_string_lossy().to_string());
    }
    let cc_model = model.strip_prefix("claude-code/").unwrap_or("");
    let (resp, sid) = claude_code::run_resumable(messages, cc_model, native_id.as_deref(), workdir.as_deref().unwrap())
        .await
        .map_err(|e| EngineError::new(format!("claude-resume backend: {e}"), "server_error", 502))?;
    if sid.is_some() {
        native_id = sid;
    }
    let text = resp["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
    state.conversations.table.lock().await.set_native(conv_id, native_id, workdir);
    Ok(text)
}

/// One store-backed turn: the external store owns the transcript; woollama assembles
/// prior + new, runs STATELESS inference, and writes the turn back.
async fn store_backed_turn(state: &AppState, conv_id: &str, model: &str, messages: &Value, options: Option<Value>) -> Result<String, EngineError> {
    let store = state.store.clone().ok_or_else(|| EngineError::new("no conversation store configured", "server_error", 500))?;
    let mut native_id = {
        let t = state.conversations.table.lock().await;
        t.get(conv_id).and_then(|c| c.native_id.clone())
    };
    if native_id.is_none() {
        native_id = Some(store.create().await?); // the store mints the thread
    }
    let tid = native_id.clone().unwrap();
    let mut combined = store.get(&tid).await?; // bytes owned by the store
    combined.extend(messages.as_array().cloned().unwrap_or_default());
    let answer = complete_stateless(state, model, &json!(combined), options).await?;
    let mut to_append = messages.as_array().cloned().unwrap_or_default();
    to_append.push(json!({"role": "assistant", "content": answer}));
    store.append(&tid, &json!(to_append)).await?; // write the turn back
    state.conversations.table.lock().await.set_native(conv_id, native_id, None);
    Ok(answer)
}

/// Run one stateless turn and return the assistant text — routes by model exactly like
/// /v1/chat/completions (woollama/<recipe> → orchestrate; a known inferencer → complete,
/// honoring native num_ctx via options). The inference fn for store-backed turns.
async fn complete_stateless(state: &AppState, model: &str, messages: &Value, options: Option<Value>) -> Result<String, EngineError> {
    if let Some(name) = model.strip_prefix("woollama/") {
        let recipe = state
            .recipes
            .get(name)
            .ok_or_else(|| EngineError::new(format!("unknown recipe '{name}'"), "not_found", 404))?;
        let resp = orchestrate_recipe(state, recipe, messages).await?;
        Ok(resp["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
    } else {
        let req = engine::build_request(model, messages.clone(), options, None, None, None, false)?;
        engine::run_complete(req).await
    }
}

async fn conversations_create(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("");
    if model.is_empty() {
        return err_response(StatusCode::BAD_REQUEST, "`model` is required to create a conversation", "invalid_request_error");
    }
    let backend = body
        .get("backend")
        .and_then(Value::as_str)
        .map(String::from)
        .or_else(|| state.backend_for_model(model).map(String::from));
    let Some(backend) = backend.filter(|b| b == "claude-resume" || b == "store-backed" || b == "managed-agents") else {
        return err_response(StatusCode::NOT_IMPLEMENTED, no_stateful_backend_msg(model), "not_implemented");
    };
    let key = body.get("key").and_then(Value::as_str).map(String::from);
    let metadata = body.get("metadata").cloned().unwrap_or_else(|| json!({}));
    let title = body.get("title").and_then(Value::as_str).map(String::from);

    let mut t = state.conversations.table.lock().await;
    if let Some(k) = &key {
        if let Some(existing) = t.by_alias(k) {
            return (StatusCode::OK, Json(existing.to_object())).into_response();
        }
    }
    let conv = t.create(&backend, model, metadata, title, key);
    (StatusCode::CREATED, Json(conv.to_object())).into_response()
}

async fn conversations_list(State(state): State<Arc<AppState>>) -> Json<Value> {
    let t = state.conversations.table.lock().await;
    let data: Vec<Value> = t.list().iter().map(|c| c.to_object()).collect();
    Json(json!({"object": "list", "data": data}))
}

async fn conversations_get(State(state): State<Arc<AppState>>, Path(conv_id): Path<String>) -> Response {
    let t = state.conversations.table.lock().await;
    match t.get(&conv_id) {
        Some(c) => Json(c.to_object()).into_response(),
        None => err_response(StatusCode::NOT_FOUND, format!("unknown conversation '{conv_id}'"), "not_found"),
    }
}

async fn conversations_items(State(state): State<Arc<AppState>>, Path(conv_id): Path<String>) -> Response {
    let conv = {
        let t = state.conversations.table.lock().await;
        match t.get(&conv_id) {
            Some(c) => c,
            None => return err_response(StatusCode::NOT_FOUND, format!("unknown conversation '{conv_id}'"), "not_found"),
        }
    };
    // store-backed + managed-agents serve the transcript (from the store / Anthropic's
    // event log); claude-resume has no `history` (a later driver slice) → 501.
    if conv.backend == "store-backed" || conv.backend == "managed-agents" {
        let msgs: Vec<Value> = match conv.backend.as_str() {
            "store-backed" => {
                let Some(store) = state.store.clone() else {
                    return engine_err_response(EngineError::new("no conversation store configured", "server_error", 500));
                };
                match &conv.native_id {
                    Some(tid) => match store.get(tid).await {
                        Ok(m) => m,
                        Err(e) => return engine_err_response(e),
                    },
                    None => Vec::new(),
                }
            }
            _ => match &conv.native_id {
                Some(sid) => match state.managed_agents.history(sid).await {
                    Ok(m) => m,
                    Err(e) => return engine_err_response(EngineError::new(format!("managed-agents backend: {e}"), "server_error", 502)),
                },
                None => Vec::new(),
            },
        };
        let data: Vec<Value> = msgs.iter().map(responses::item_object).collect();
        let first_id = data.first().and_then(|x| x.get("id").cloned()).unwrap_or(Value::Null);
        let last_id = data.last().and_then(|x| x.get("id").cloned()).unwrap_or(Value::Null);
        return Json(json!({
            "object": "list", "data": data,
            "first_id": first_id, "last_id": last_id, "has_more": false,
        }))
        .into_response();
    }
    err_response(
        StatusCode::NOT_IMPLEMENTED,
        format!("conversation transcript items are not available for the '{}' backend yet", conv.backend),
        "not_implemented",
    )
}

async fn conversations_delete(State(state): State<Arc<AppState>>, Path(conv_id): Path<String>) -> Response {
    let conv = {
        let t = state.conversations.table.lock().await;
        t.get(&conv_id)
    };
    let Some(conv) = conv else {
        return err_response(StatusCode::NOT_FOUND, format!("unknown conversation '{conv_id}'"), "not_found");
    };
    // Backend teardown (best-effort): claude-resume removes the per-conversation workdir
    // (the on-disk Claude session is the user's data, left intact); store-backed tells the
    // external store to drop the thread.
    match conv.backend.as_str() {
        "claude-resume" => {
            if let Some(wd) = &conv.workdir {
                let _ = std::fs::remove_dir_all(wd);
            }
        }
        "store-backed" => {
            if let (Some(store), Some(tid)) = (state.store.clone(), conv.native_id.clone()) {
                let _ = store.delete(&tid).await;
            }
        }
        "managed-agents" => {
            if let Some(sid) = &conv.native_id {
                let _ = state.managed_agents.delete_session(sid).await;
            }
        }
        _ => {}
    }
    {
        let mut t = state.conversations.table.lock().await;
        t.remove(&conv_id);
    }
    Json(json!({"id": conv_id, "object": "conversation.deleted", "deleted": true})).into_response()
}

#[cfg(test)]
mod fnmatch_tests {
    use super::fnmatch;

    #[test]
    fn glob_star_and_question() {
        assert!(fnmatch("gpt-4*", "gpt-4o"));
        assert!(fnmatch("keep-*", "keep-this"));
        assert!(!fnmatch("keep-*", "drop-that"));
        assert!(fnmatch("*", "anything"));
        assert!(fnmatch("q?en", "qwen"));
        assert!(!fnmatch("q?en", "qween"));
        assert!(fnmatch("exact", "exact"));
        assert!(!fnmatch("exact", "exacto"));
        assert!(fnmatch("a*b*c", "axxbyyc"));
    }
}
