//! A pluggable `/w1/` pattern backend — the extension point that keeps backend-specific code
//! (fabric, future providers) OUT of `lib.rs`.
//!
//! Design:
//!   - The **native** path (recipes.toml + the `[patterns]` dir scan, dispatched through
//!     `woollama-engine`) is the BUILT-IN core, handled directly in `lib.rs`. It isn't a
//!     backend because it needs the engine/registry/inferencers; making it one would buy
//!     churn, not clarity. Stated asymmetry: native is the core, everything else is a plugin.
//!   - **Additional** backends implement this trait, are constructed from config in
//!     `build_state`, and live in `AppState.pattern_backends`. The `/w1/` handlers iterate
//!     them generically — no per-backend branches, no backend name literals in `lib.rs`.
//!
//! Dispatch order (deterministic): a native recipe WINS on a name collision; otherwise the
//! first backend (in registration order) whose `has()` is true serves the pattern.
//!
//! "Config-driven" here means **registration + location** (which backends are active,
//! managed-vs-url, address) — NOT protocol. A backend's wire format (endpoints, body shape,
//! SSE event types) is code, inside its own module; it cannot be expressed in config.

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Map, Value};

/// A discovery entry for `GET /w1/patterns`.
pub struct PatternInfo {
    pub name: String,
    /// Scanned `{{var}}` names, when the backend can supply them cheaply (else empty).
    pub variables: Vec<String>,
    /// Provenance label surfaced to clients (e.g. `"fabric"`).
    pub source: String,
}

#[async_trait::async_trait]
pub trait PatternBackend: Send + Sync {
    /// Stable backend id (e.g. `"fabric"`). Also the mount prefix for its transparent proxy.
    fn id(&self) -> &str;

    /// The patterns this backend offers (for `/w1/patterns` discovery + `/v1/models`).
    fn list(&self) -> Vec<PatternInfo>;

    /// Whether this backend serves `name`.
    fn has(&self, name: &str) -> bool;

    /// The rendered system text for `name` with `variables` substituted — woollama appends the
    /// user input. `None` if the backend doesn't know the pattern.
    async fn render(&self, name: &str, variables: &Map<String, Value>) -> Option<String>;

    /// Run `name` → an OpenAI chat-completion `Response` (or OpenAI SSE when the body sets
    /// `stream:true`). Only called after `has(name)` returned true.
    async fn run(&self, name: &str, body: &Value) -> Response;

    /// Whether this backend exposes a transparent reverse-proxy of its native API at `/{id}/*`.
    fn proxies(&self) -> bool {
        false
    }

    /// Transparent reverse-proxy of the backend's native API. `path_and_query` is the part
    /// after `/{id}`. Default: not supported.
    async fn proxy(
        &self,
        _method: axum::http::Method,
        _path_and_query: &str,
        _content_type: Option<&str>,
        _body: bytes::Bytes,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "this backend has no proxy surface").into_response()
    }

    /// Release resources on graceful shutdown (e.g. kill a supervised child). Default no-op.
    async fn shutdown(&self) {}
}

/// Stream a `reqwest::Response` straight back to the client (SSE-safe — body is never
/// buffered), preserving status + content-type. Shared by proxy implementations.
pub fn stream_reqwest(resp: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let ct = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).map(String::from);
    let mut builder = Response::builder().status(status);
    if let Some(ct) = ct {
        builder = builder.header("content-type", ct);
    }
    builder.body(Body::from_stream(resp.bytes_stream())).unwrap_or_else(|_| {
        (StatusCode::BAD_GATEWAY, Json(json!({"error": {"message": "proxy stream build failed", "type": "server_error"}})))
            .into_response()
    })
}
