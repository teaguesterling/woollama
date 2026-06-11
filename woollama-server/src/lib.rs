//! The woollama router service (Rust) — the axum HTTP surface over `woollama-engine`.
//!
//! Slice 2 (the skeleton): TCP binding (`WOOLLAMA_ADDRESS`), `GET /v1/models`, and
//! `POST /v1/chat/completions` **passthrough** for `<provider>/<model>`. Deferred to
//! later slices (each returns a clear 501 / empty result for now, not a silent gap):
//!   - native num_ctx routing + streaming passthrough + `/v1/responses` → slice 3
//!   - `woollama/<recipe>` orchestration (needs the MCP registry)        → slice 4
//!   - `/v1/models` discovery (ollama catalog, configured `models`, recipes) → slice 8
//!   - the Unix-socket surface                                            → later
//!
//! Behavior mirrors the Python `router.py`; once enough slices land, the live
//! integration suite repoints its process-spawn here as the differential oracle.

use std::time::Duration;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use woollama_engine as engine;
use engine::EngineError;

/// The TCP host/port to bind — `$WOOLLAMA_ADDRESS=host[:port]` (the only way to bind a
/// non-loopback host), else `127.0.0.1:0` (loopback, free port). Mirrors Python
/// `binding.resolve_tcp_target`.
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
}

/// An OpenAI-style error body `{error: {message, type}}` with an HTTP status.
fn err_response(status: StatusCode, message: impl Into<String>, kind: &str) -> Response {
    (status, Json(json!({"error": {"message": message.into(), "type": kind}}))).into_response()
}

/// Map a structured `EngineError` to its HTTP surface (payload verbatim when present).
fn engine_err_response(e: EngineError) -> Response {
    let status = StatusCode::from_u16(e.status.unwrap_or(500) as u16)
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let kind = e.kind.clone().unwrap_or_else(|| "server_error".to_string());
    match e.payload {
        Some(payload) => (status, Json(payload)).into_response(),
        None => err_response(status, e.message, &kind),
    }
}

/// `GET /v1/models`. Slice 2 serves the route + shape; per-model discovery (the ollama
/// catalog, configured `models`, `woollama/<recipe>` entries) lands in slice 8.
async fn list_models() -> Json<Value> {
    Json(json!({"object": "list", "data": []}))
}

/// `POST /v1/chat/completions`. Slice 2: passthrough for a known `<provider>/<model>`
/// (non-stream). `woollama/<recipe>` and streaming are explicit 501s until their slices.
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

    if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        return err_response(
            StatusCode::NOT_IMPLEMENTED,
            "streaming passthrough is not yet in the Rust server (slice 3)",
            "not_implemented",
        );
    }

    // Forward the body straight through, swapping the namespaced model for the bare
    // name and forcing non-stream (the caller owns the rest of the body).
    let bare = model.splitn(2, '/').nth(1).unwrap_or("");
    let mut fwd = body.clone();
    fwd["model"] = json!(bare);
    fwd["stream"] = json!(false);

    let headers = match inf.auth_headers() {
        Ok(h) => h,
        Err(e) => return engine_err_response(e),
    };
    let client = match reqwest::Client::builder().timeout(Duration::from_secs(180)).build() {
        Ok(c) => c,
        Err(e) => return err_response(StatusCode::BAD_GATEWAY, e.to_string(), "server_error"),
    };
    let mut rb = client.post(inf.chat_url()).json(&fwd);
    for (k, v) in &headers {
        rb = rb.header(k, v);
    }
    match rb.send().await {
        Ok(r) => {
            let status = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let data: Value = r.json().await.unwrap_or_else(|_| json!({}));
            (status, Json(data)).into_response()
        }
        Err(e) => err_response(StatusCode::BAD_GATEWAY, e.to_string(), "server_error"),
    }
}
