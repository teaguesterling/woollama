//! A device can drop a model underneath the pool, and the pool must stop believing it is resident.
//!
//! The device evicts on its own and has other mutators (its UI, its CLI, a watchdog), so the pool's
//! `loaded` flag is a cache. A concrete model id never re-reads residency, so the only signal that
//! the cache went stale is the device refusing the request. Beta firmware refused with a 503.
//! tiiny_os 0.1.38 refuses with `404 model_not_found`, which `passthrough_pooled` did not treat as
//! that signal, and the embeddings route treated nothing as that signal. Every later request then
//! took the "already loaded" short-circuit and failed the same way until the daemon restarted.
//!
//! The fake device speaks the 0.1.38 shapes, as measured through dobby on 2026-11-01:
//!   GET  running -> {"object":"list","running":[],"pending":[],"instances":{"running":[],"pending":[]}}
//!   POST start   -> 200 {"message":"start loading ...","progress":0}, and the model appears later
//!   not loaded   -> 404 {"error":{"code":404,"message":"\"<id>\" is not loaded.","type":"model_not_found"}}
//!
//! Two things in it are ASSUMPTIONS, because the device was not probed for them: the populated
//! form of `running` (bare id strings, as on the beta) and the refusal of `start` for an unknown id.
//!
//! Separate test binary, because `WOOLLAMA_CONFIG_DIR` is process-global. `SETUP` serializes the
//! scenarios in this file around it.

use std::collections::HashSet;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

const EMBEDDER: &str = "Qwen/Qwen3-Embedding-0.6B";
const CHAT: &str = "Qwen/Qwen3-8B";
const UNKNOWN: &str = "no-such/model";

static SETUP: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// How the device refuses inference for a model that is not running.
#[derive(Clone, Copy, PartialEq)]
enum Firmware {
    /// tiiny_os 0.1.38
    Refuses404,
    /// The older beta
    Refuses503,
}

struct DeviceInner {
    firmware: Firmware,
    running: HashSet<String>,
    starts: Vec<String>,
    inference_calls: u32,
    /// Answer the next inference call with a 404 that does not mean "not loaded".
    plain_404_once: bool,
}

#[derive(Clone)]
struct Device(Arc<StdMutex<DeviceInner>>);

impl Device {
    fn starts_of(&self, id: &str) -> usize {
        self.0.lock().unwrap().starts.iter().filter(|s| *s == id).count()
    }
    fn evict(&self, id: &str) {
        self.0.lock().unwrap().running.remove(id);
    }
}

async fn handle_get(State(d): State<Device>, AxPath(rest): AxPath<String>) -> Response {
    if rest != "running" {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response();
    }
    let mut running: Vec<String> = d.0.lock().unwrap().running.iter().cloned().collect();
    running.sort();
    let instances: Vec<Value> = running.iter().map(|id| json!({"model_id": id, "capabilities": []})).collect();
    Json(json!({
        "object": "list",
        "running": running,
        "pending": [],
        "instances": {"running": instances, "pending": []},
    }))
    .into_response()
}

async fn handle_post(State(d): State<Device>, AxPath(rest): AxPath<String>) -> Response {
    let Some(id) = rest.strip_suffix("/start") else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response();
    };
    d.0.lock().unwrap().starts.push(id.to_string());
    if id == UNKNOWN {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": {"code": 404, "message": format!("{id:?} does not exist."), "type": "model_not_found"}})),
        )
            .into_response();
    }
    // The start is asynchronous: the model shows up in `running` only later.
    let (device, owned) = (d.clone(), id.to_string());
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        device.0.lock().unwrap().running.insert(owned);
    });
    Json(json!({"message": format!("start loading {id}"), "progress": 0})).into_response()
}

async fn handle_inference(State(d): State<Device>, Json(body): Json<Value>) -> Response {
    let id = body["model"].as_str().unwrap_or("").to_string();
    let mut inner = d.0.lock().unwrap();
    inner.inference_calls += 1;
    if inner.plain_404_once {
        inner.plain_404_once = false;
        return (StatusCode::NOT_FOUND, Json(json!({"error": {"code": 404, "message": "no such route", "type": "not_found"}})))
            .into_response();
    }
    if inner.running.contains(&id) {
        return Json(json!({"data": [{"embedding": [0.1, 0.2], "index": 0}]})).into_response();
    }
    match inner.firmware {
        Firmware::Refuses404 => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": {"code": 404, "message": format!("{id:?} is not loaded."), "type": "model_not_found"}})),
        )
            .into_response(),
        Firmware::Refuses503 => {
            (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": {"message": format!("{id} not loaded")}}))).into_response()
        }
    }
}

async fn spawn_router(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

struct Rig {
    device: Device,
    base: String,
    client: reqwest::Client,
    _cfg: tempfile::TempDir,
}

impl Rig {
    async fn up(firmware: Firmware) -> Rig {
        let device = Device(Arc::new(StdMutex::new(DeviceInner {
            firmware,
            running: HashSet::new(),
            starts: Vec::new(),
            inference_calls: 0,
            plain_404_once: false,
        })));
        let device_url = spawn_router(
            Router::new()
                .route("/api/v1/models/{*rest}", get(handle_get).post(handle_post))
                .route("/embeddings", post(handle_inference))
                .route("/chat/completions", post(handle_inference))
                .with_state(device.clone()),
        )
        .await;

        let cfg = tempfile::tempdir().unwrap();
        std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
        std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
        std::fs::write(
            cfg.path().join("inferencers.toml"),
            format!(
                "[inferencers.device]\nbase_url=\"{u}\"\nmanagement_url=\"{u}\"\nmanagement_protocol=\"device\"\n",
                u = device_url
            ),
        )
        .unwrap();

        let state = {
            let _guard = SETUP.lock().await;
            std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
            Arc::new(woollama_server::build_state().await)
        };
        let base = spawn_router(woollama_server::router(state)).await;
        Rig { device, base, client: reqwest::Client::new(), _cfg: cfg }
    }

    /// Bounded, so that a hang fails the test instead of stalling it.
    async fn post(&self, path: &str, body: Value) -> (u16, Value) {
        let send = self.client.post(format!("{}{path}", self.base)).json(&body).send();
        let resp = tokio::time::timeout(Duration::from_secs(20), send)
            .await
            .expect("woollamad must answer, not hang")
            .unwrap();
        let status = resp.status().as_u16();
        (status, resp.json().await.unwrap_or(Value::Null))
    }

    async fn embed(&self, id: &str) -> (u16, Value) {
        self.post("/v1/embeddings", json!({"model": format!("device/{id}"), "input": ["a"]})).await
    }

    async fn chat(&self, id: &str, stream: bool) -> u16 {
        let body = json!({
            "model": format!("device/{id}"),
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
        });
        self.post("/v1/chat/completions", body).await.0
    }
}

/// The measured 0.1.38 shapes drive a whole cold load: empty `running`, the `start` reply, the wait.
#[tokio::test]
async fn cold_embedding_loads_through_the_0_1_38_shapes() {
    let rig = Rig::up(Firmware::Refuses404).await;
    let (status, _) = rig.embed(EMBEDDER).await;
    assert_eq!(status, 200);
    assert_eq!(rig.device.starts_of(EMBEDDER), 1);
}

/// The reported failure, as it reaches a build that has the embeddings gate. One refusal is what
/// tells the pool, so the request that discovers the eviction still fails. The next one must load.
#[tokio::test]
async fn embedder_evicted_under_the_pool_reloads_after_a_404() {
    let rig = Rig::up(Firmware::Refuses404).await;
    assert_eq!(rig.embed(EMBEDDER).await.0, 200);
    rig.device.evict(EMBEDDER);

    let (status, body) = rig.embed(EMBEDDER).await;
    assert_eq!(status, 404);
    assert_eq!(body["error"]["type"], "model_not_found", "the device's own answer is relayed");

    assert_eq!(rig.embed(EMBEDDER).await.0, 200, "a stale belief must not fail every later request");
    assert_eq!(rig.device.starts_of(EMBEDDER), 2);
}

/// The beta firmware's refusal gets the same recovery on the embeddings route.
#[tokio::test]
async fn embedder_evicted_under_the_pool_reloads_after_a_503() {
    let rig = Rig::up(Firmware::Refuses503).await;
    assert_eq!(rig.embed(EMBEDDER).await.0, 200);
    rig.device.evict(EMBEDDER);

    assert_eq!(rig.embed(EMBEDDER).await.0, 503);
    assert_eq!(rig.embed(EMBEDDER).await.0, 200);
    assert_eq!(rig.device.starts_of(EMBEDDER), 2);
}

/// The chat route has the same belief and the same refusal, streaming or not.
#[tokio::test]
async fn chat_model_evicted_under_the_pool_reloads_after_a_404() {
    for stream in [false, true] {
        let rig = Rig::up(Firmware::Refuses404).await;
        assert_eq!(rig.chat(CHAT, stream).await, 200);
        rig.device.evict(CHAT);

        assert_eq!(rig.chat(CHAT, stream).await, 404, "stream={stream}");
        assert_eq!(rig.chat(CHAT, stream).await, 200, "stream={stream}: must reload, not keep failing");
        assert_eq!(rig.device.starts_of(CHAT), 2, "stream={stream}");
    }
}

/// Older behaviour, kept: a 503 on the chat route already led to a reload.
#[tokio::test]
async fn chat_model_evicted_under_the_pool_still_reloads_after_a_503() {
    let rig = Rig::up(Firmware::Refuses503).await;
    assert_eq!(rig.chat(CHAT, false).await, 200);
    rig.device.evict(CHAT);

    assert_eq!(rig.chat(CHAT, false).await, 503);
    assert_eq!(rig.chat(CHAT, false).await, 200);
    assert_eq!(rig.device.starts_of(CHAT), 2);
}

/// "Does not exist" is settled at `start`: one attempt per request, a clear error, no inference
/// call, and no state carried into the next request.
#[tokio::test]
async fn unknown_model_fails_clearly_without_a_load_loop() {
    let rig = Rig::up(Firmware::Refuses404).await;
    for attempt in 1..=2 {
        let (status, body) = rig.embed(UNKNOWN).await;
        assert_eq!(status, 502, "attempt {attempt}");
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains("start no-such/model failed"), "must name the failure: {body}");
        assert_eq!(rig.device.starts_of(UNKNOWN), attempt, "one start per request, never a retry loop");
    }
    assert_eq!(rig.device.0.lock().unwrap().inference_calls, 0);
}

/// A 404 that does not say the model is missing must not cold-load a healthy, resident model.
#[tokio::test]
async fn an_unrelated_404_does_not_force_a_reload() {
    let rig = Rig::up(Firmware::Refuses404).await;
    assert_eq!(rig.embed(EMBEDDER).await.0, 200);

    rig.device.0.lock().unwrap().plain_404_once = true;
    assert_eq!(rig.embed(EMBEDDER).await.0, 404);

    assert_eq!(rig.embed(EMBEDDER).await.0, 200);
    assert_eq!(rig.device.starts_of(EMBEDDER), 1, "the model never left, so nothing restarts it");
}
