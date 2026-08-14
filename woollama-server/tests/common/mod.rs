//! Shared test-support; the pluggable-protocol tests consume the rest.
#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use serde_json::{json, Value};

/// One inbound request as observed by a mock backend.
#[derive(Clone, Debug)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: String,
}

#[derive(Default)]
struct Inner {
    requests: Vec<RecordedRequest>,
    loaded: BTreeSet<String>,
    fail_start: bool,
    fail_stop: bool,
}

/// Handle to a spawned mock management backend.
#[derive(Clone)]
pub struct MockHandle {
    pub base_url: String,
    inner: Arc<Mutex<Inner>>,
}

impl MockHandle {
    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.inner.lock().unwrap().requests.clone()
    }

    pub fn requests_to(&self, path_substr: &str) -> Vec<RecordedRequest> {
        self.inner
            .lock()
            .unwrap()
            .requests
            .iter()
            .filter(|r| r.path.contains(path_substr))
            .cloned()
            .collect()
    }

    /// Currently "loaded" model ids, sorted.
    pub fn loaded(&self) -> Vec<String> {
        self.inner.lock().unwrap().loaded.iter().cloned().collect()
    }

    pub fn set_loaded(&self, ids: &[&str]) {
        self.inner.lock().unwrap().loaded = ids.iter().map(|s| s.to_string()).collect();
    }

    pub fn set_fail_start(&self, fail: bool) {
        self.inner.lock().unwrap().fail_start = fail;
    }

    pub fn set_fail_stop(&self, fail: bool) {
        self.inner.lock().unwrap().fail_stop = fail;
    }
}

// --- rest mock -----------------------------------------------------------

/// Shape of the "running" (list-loaded) response body.
pub enum RunningShape {
    /// `{ "<field>": ["id", ...] }`; `field == ""` means the array is the top-level body.
    Strings { field: String },
    /// `{ "<field>": [ { "<id_key>": "id" }, ... ] }`; `field == ""` means a top-level array.
    Objects { field: String, id_key: String },
}

/// Where the model id lives for `start`/`stop` requests.
pub enum IdLoc {
    /// The route contains an `{id}` segment; ids may contain slashes and are matched by
    /// the fixed prefix/suffix around `{id}` (not axum path-param routing).
    Path,
    /// The id is a field in the JSON request body.
    Body { field: String },
}

pub struct RestMockConfig {
    pub running_route: String,
    pub start_route: String,
    pub stop_route: String,
    pub running_shape: RunningShape,
    pub id_loc: IdLoc,
}

#[derive(Clone)]
struct RestState {
    inner: Arc<Mutex<Inner>>,
    cfg: Arc<RestMockConfig>,
}

/// Spawn a mock REST list/start/stop device on an ephemeral `127.0.0.1` port. The socket
/// is bound (and thus already accepting connections) before this function returns, so
/// `base_url` is live as soon as the handle comes back — no readiness race.
pub fn spawn_rest(cfg: RestMockConfig) -> MockHandle {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let addr = listener.local_addr().unwrap();
    let listener = tokio::net::TcpListener::from_std(listener).unwrap();

    let inner = Arc::new(Mutex::new(Inner::default()));
    let state = RestState { inner: inner.clone(), cfg: Arc::new(cfg) };
    let router: Router<()> = Router::new().fallback(rest_handler).with_state(state);
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    MockHandle { base_url: format!("http://{addr}"), inner }
}

/// A single fallback handler: records every inbound request, then dispatches by
/// method + path against the configured routes. Handling everything (including
/// slash-bearing ids) through one fallback keeps recording uniform and sidesteps
/// axum's segment-based path-param matching, which can't express ids containing `/`.
async fn rest_handler(
    State(state): State<RestState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let path = uri.path_and_query().map(|pq| pq.as_str().to_string()).unwrap_or_else(|| uri.path().to_string());
    let path_only = uri.path().to_string();
    let body_str = String::from_utf8_lossy(&body).to_string();
    let recorded_headers: HashMap<String, String> =
        headers.iter().map(|(k, v)| (k.as_str().to_lowercase(), v.to_str().unwrap_or("").to_string())).collect();

    let mut inner = state.inner.lock().unwrap();
    inner.requests.push(RecordedRequest {
        method: method.to_string(),
        path: path.clone(),
        headers: recorded_headers,
        body: body_str.clone(),
    });

    if method == Method::GET && path_only == state.cfg.running_route {
        return running_response(&inner, &state.cfg);
    }

    if method == Method::POST {
        if let Some(id) = resolve_id(&path_only, &body_str, &state.cfg.start_route, &state.cfg.id_loc) {
            if inner.fail_start {
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "start failed"}))).into_response();
            }
            inner.loaded.insert(id);
            return (StatusCode::OK, Json(json!({"ok": true}))).into_response();
        }
        if let Some(id) = resolve_id(&path_only, &body_str, &state.cfg.stop_route, &state.cfg.id_loc) {
            if inner.fail_stop {
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "stop failed"}))).into_response();
            }
            inner.loaded.remove(&id);
            return (StatusCode::OK, Json(json!({"ok": true}))).into_response();
        }
    }

    (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response()
}

fn running_response(inner: &Inner, cfg: &RestMockConfig) -> Response {
    let ids: Vec<String> = inner.loaded.iter().cloned().collect();
    let body = match &cfg.running_shape {
        RunningShape::Strings { field } => {
            let arr = Value::Array(ids.into_iter().map(Value::String).collect());
            if field.is_empty() {
                arr
            } else {
                json!({ field: arr })
            }
        }
        RunningShape::Objects { field, id_key } => {
            let arr: Vec<Value> = ids.into_iter().map(|id| json!({ id_key: id })).collect();
            if field.is_empty() {
                Value::Array(arr)
            } else {
                json!({ field: arr })
            }
        }
    };
    (StatusCode::OK, Json(body)).into_response()
}

/// Match `path`/`body` against a configured start/stop route, returning the id if it
/// matches. `Path` mode matches by fixed prefix/suffix around `{id}` (ids may contain
/// slashes); `Body` mode requires an exact path match and pulls the id from the JSON body.
fn resolve_id(path: &str, body: &str, route: &str, id_loc: &IdLoc) -> Option<String> {
    match id_loc {
        IdLoc::Path => match_path_id(path, route),
        IdLoc::Body { field } => {
            if path != route {
                return None;
            }
            let v: Value = serde_json::from_str(body).ok()?;
            v.get(field)?.as_str().map(|s| s.to_string())
        }
    }
}

fn match_path_id(path: &str, route: &str) -> Option<String> {
    let idx = route.find("{id}")?;
    let prefix = &route[..idx];
    let suffix = &route[idx + 4..];
    if path.len() < prefix.len() + suffix.len() || !path.starts_with(prefix) || !path.ends_with(suffix) {
        return None;
    }
    let id = &path[prefix.len()..path.len() - suffix.len()];
    if id.is_empty() {
        None
    } else {
        Some(id.to_string())
    }
}

// --- ollama mock -----------------------------------------------------------

#[derive(Clone)]
struct OllamaState {
    inner: Arc<Mutex<Inner>>,
}

/// Spawn a mock Ollama-shaped device: `GET /api/ps` lists loaded models, `POST
/// /api/generate` loads (or, with `keep_alive: 0`/`"0"`, unloads) a model — mirroring
/// how the future `ollama` adapter warms up / unloads.
pub fn spawn_ollama() -> MockHandle {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let addr = listener.local_addr().unwrap();
    let listener = tokio::net::TcpListener::from_std(listener).unwrap();

    let inner = Arc::new(Mutex::new(Inner::default()));
    let state = OllamaState { inner: inner.clone() };
    let router: Router<()> = Router::new().fallback(ollama_handler).with_state(state);
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    MockHandle { base_url: format!("http://{addr}"), inner }
}

async fn ollama_handler(
    State(state): State<OllamaState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let path = uri.path_and_query().map(|pq| pq.as_str().to_string()).unwrap_or_else(|| uri.path().to_string());
    let path_only = uri.path().to_string();
    let body_str = String::from_utf8_lossy(&body).to_string();
    let recorded_headers: HashMap<String, String> =
        headers.iter().map(|(k, v)| (k.as_str().to_lowercase(), v.to_str().unwrap_or("").to_string())).collect();

    let mut inner = state.inner.lock().unwrap();
    inner.requests.push(RecordedRequest {
        method: method.to_string(),
        path: path.clone(),
        headers: recorded_headers,
        body: body_str.clone(),
    });

    if method == Method::GET && path_only == "/api/ps" {
        let models: Vec<Value> = inner.loaded.iter().map(|id| json!({"name": id, "model": id})).collect();
        return (StatusCode::OK, Json(json!({"models": models}))).into_response();
    }

    if method == Method::POST && path_only == "/api/generate" {
        let Ok(parsed) = serde_json::from_str::<Value>(&body_str) else {
            return (StatusCode::BAD_REQUEST, Json(json!({"error": "bad json"}))).into_response();
        };
        let model = parsed.get("model").and_then(Value::as_str).unwrap_or("").to_string();
        if model.is_empty() {
            return (StatusCode::BAD_REQUEST, Json(json!({"error": "missing model"}))).into_response();
        }
        let unload = match parsed.get("keep_alive") {
            Some(Value::Number(n)) => n.as_f64() == Some(0.0),
            Some(Value::String(s)) => s == "0",
            _ => false,
        };
        if unload {
            inner.loaded.remove(&model);
        } else {
            inner.loaded.insert(model);
        }
        return (StatusCode::OK, Json(json!({"done": true}))).into_response();
    }

    (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response()
}
