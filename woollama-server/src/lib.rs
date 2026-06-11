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
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde_json::{json, Value};

use woollama_engine as engine;
use engine::EngineError;

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
    let registry = Arc::new(mcp_registry::McpRegistry::connect(specs).await);
    let inferencers = engine::Registry::from_config().unwrap_or_else(|e| {
        eprintln!("woollama-server: inferencers load error: {e}");
        engine::Registry::new()
    });
    AppState { recipes, registry, inferencers }
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

// --- orchestration (shared by chat-completions + responses) -------------------

/// Run a recipe to completion and return the final OpenAI response dict. Tools are
/// dispatched to the downstream MCP registry. Shared by the HTTP handlers and the MCP
/// `chat` tool; each maps the `EngineError` to its own surface.
pub(crate) async fn orchestrate_recipe(
    state: &AppState,
    recipe: &config::Recipe,
    messages: &Value,
) -> Result<Value, EngineError> {
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
        if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
            return err_response(
                StatusCode::NOT_IMPLEMENTED,
                "streaming orchestration is not yet in the Rust server (slice 4b)",
                "not_implemented",
            );
        }
        let messages = body.get("messages").cloned().unwrap_or_else(|| json!([]));
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
            return err_response(
                StatusCode::NOT_IMPLEMENTED,
                "native num_ctx streaming is not yet in the Rust server (slice 3b); \
                 omit stream, or drop num_ctx to stream on /v1",
                "not_implemented",
            );
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
        return err_response(
            StatusCode::NOT_IMPLEMENTED,
            "streaming /v1/responses is not yet in the Rust server (slice 3b)",
            "not_implemented",
        );
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
