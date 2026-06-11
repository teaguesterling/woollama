//! The woollama router service (Rust) — the axum HTTP surface over `woollama-engine`.
//!
//! Slices landed here:
//!   2  — TCP binding, `GET /v1/models`, `POST /v1/chat/completions` passthrough.
//!   3  — native num_ctx passthrough (non-stream), streaming passthrough (SSE relay),
//!        stateless `POST /v1/responses` (non-stream).
//!
//! Deferred (each returns a clear 501, not a silent gap):
//!   - native num_ctx STREAMING (NDJSON→SSE) + Responses streaming/items → slice 3b
//!   - `woollama/<recipe>` orchestration (needs the MCP registry)        → slice 4
//!   - stateful `/v1/responses` (conversation backends)                  → slices 6–7
//!   - `/v1/models` discovery (catalog/recipes)                          → slice 8
//!   - the Unix-socket surface                                           → later
//!
//! Behavior mirrors Python `router.py` / `ollama_native.py` / `responses.py`.

use std::collections::HashMap;
use std::time::Duration;

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use woollama_engine as engine;
use engine::EngineError;

mod ollama_native;
mod responses;

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

/// The axum app (shared by the binary and the integration tests).
pub fn router() -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses_create))
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

/// POST `body` to `url` with `headers` and a timeout; map transport failure to a 502.
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

/// Relay an upstream JSON response (status + body) verbatim.
async fn relay_json(resp: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let data: Value = resp.json().await.unwrap_or_else(|_| json!({}));
    (status, Json(data)).into_response()
}

// --- GET /v1/models -----------------------------------------------------------

/// Slice 2 shape; per-model discovery (ollama catalog, configured `models`, recipes)
/// is slice 8.
async fn list_models() -> Json<Value> {
    Json(json!({"object": "list", "data": []}))
}

// --- POST /v1/chat/completions ------------------------------------------------

async fn chat_completions(Json(body): Json<Value>) -> Response {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();

    if let Some(name) = model.strip_prefix("woollama/") {
        return err_response(
            StatusCode::NOT_IMPLEMENTED,
            format!("orchestration for 'woollama/{name}' is not yet in the Rust server (slice 4)"),
            "not_implemented",
        );
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

    // Forward body with the bare model.
    let bare = model.splitn(2, '/').nth(1).unwrap_or("").to_string();
    let mut fwd = body.clone();
    fwd["model"] = json!(bare);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    // ollama num_ctx → native /api/chat (which honors it). Streaming-native is slice 3b.
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

    // Plain non-stream /v1 passthrough.
    fwd["stream"] = json!(false);
    match forward_post(inf.chat_url(), &fwd, &headers, 180).await {
        Ok(resp) => relay_json(resp).await,
        Err(e) => e,
    }
}

/// num_ctx → native `/api/chat`: translate request to native shape, response back to
/// OpenAI `chat.completion` (non-stream).
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

/// Relay the upstream OpenAI SSE byte-for-byte (status checked first: a 4xx surfaces as
/// JSON, not an empty 200 stream).
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

async fn responses_create(Json(body): Json<Value>) -> Response {
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
    if let Some(name) = model.strip_prefix("woollama/") {
        return err_response(
            StatusCode::NOT_IMPLEMENTED,
            format!("orchestration for 'woollama/{name}' is not yet in the Rust server (slice 4)"),
            "not_implemented",
        );
    }

    // Stateless inferencer turn — the engine's complete handles native num_ctx via options.
    let options = body.get("options").cloned();
    let req = match engine::build_request(&model, json!(messages), options, None, None, None, false) {
        Ok(r) => r,
        Err(e) => return engine_err_response(e),
    };
    match engine::run_complete(req).await {
        Ok(text) => {
            Json(responses::build_response(&responses::new_id("resp"), &model, &text)).into_response()
        }
        Err(e) => engine_err_response(e),
    }
}
