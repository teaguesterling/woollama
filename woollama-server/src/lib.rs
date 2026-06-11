//! The woollama router service (Rust) — the axum HTTP surface over `woollama-engine`.
//!
//! Slices landed here:
//!   2  — TCP binding, `GET /v1/models`, `POST /v1/chat/completions` passthrough.
//!   3  — native num_ctx passthrough (non-stream), streaming passthrough, stateless
//!        `POST /v1/responses` (non-stream).
//!   4a — `woollama/<recipe>` ORCHESTRATION: a downstream MCP registry (rmcp
//!        child-process clients) + `RegistryToolProvider` drives the engine recipe
//!        loop, on `/v1/chat/completions` and stateless `/v1/responses` (non-stream).
//!
//! Deferred (each a clear 501): native num_ctx STREAMING + Responses streaming/items
//! (3b); STREAMING orchestration + the woollama-as-MCP `/mcp` surface (4b); stateful
//! `/v1/responses` (6–7); `/v1/models` discovery (8); the Unix socket.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::body::Body;
use axum::extract::State;
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
}

/// Load config + connect the downstream MCP servers. Errors are logged and degraded to
/// empty (the router still starts) rather than fatal.
pub async fn build_state() -> AppState {
    let recipes = config::load_recipes().unwrap_or_else(|e| {
        eprintln!("woollama-server: recipes load error: {e}");
        HashMap::new()
    });
    let specs = config::load_mcp_servers().unwrap_or_else(|e| {
        eprintln!("woollama-server: mcp.json load error: {e}");
        HashMap::new()
    });
    let registry = Arc::new(mcp_registry::McpRegistry::connect(specs.clone()).await);
    let inferencers = engine::Registry::from_config().unwrap_or_else(|e| {
        eprintln!("woollama-server: inferencers load error: {e}");
        engine::Registry::new()
    });
    AppState { recipes, registry, inferencers, mcp_specs: specs }
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
        .nest_service("/mcp", mcp_svc)
        .with_state(state)
}

/// Serve woollama's MCP surface over stdio — the `woollama-server mcp` subcommand (what
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

async fn list_models() -> Json<Value> {
    Json(json!({"object": "list", "data": []}))
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
        return err_response(
            StatusCode::NOT_IMPLEMENTED,
            "stateful /v1/responses is not yet in the Rust server (slices 6-7)",
            "not_implemented",
        );
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
