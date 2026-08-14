//! `Gate`/`Slot`/`PoolRegistry` (Task 6) — TDD port of the Gate-level cases from
//! `tests/test_pool.py` (parallel serialization, queue_max/queue_timeout backpressure,
//! serving-model eviction protection) plus an end-to-end pooled-passthrough test through
//! the real HTTP surface (`POST /v1/chat/completions`).
//!
//! `FakeDevice` is a trimmed duplicate of `tests/pool_manager.rs`'s fixture: the same
//! management endpoints (GET .../running, POST .../{id}/start, POST .../{id}/stop), no
//! failure-injection/blocking knobs (not needed at the Gate level — those are Task 5's
//! `DeviceModelManager` concerns, already covered in `pool_manager.rs`), plus a
//! `/chat/completions` route so the end-to-end test can drive a real pooled dispatch.

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::sync::Notify;

use woollama_server::pool::{DeviceModelManager, Gate, PoolError, RestBackend};

#[derive(Default)]
struct DeviceInner {
    running: std::collections::HashSet<String>,
    calls: Vec<(String, String)>,
    last_chat_model: Option<String>,
}

#[derive(Clone)]
struct DeviceState {
    inner: Arc<StdMutex<DeviceInner>>,
}

struct FakeDevice {
    url: String,
    inner: Arc<StdMutex<DeviceInner>>,
}

impl FakeDevice {
    async fn spawn(running: &[&str]) -> Self {
        let inner = Arc::new(StdMutex::new(DeviceInner {
            running: running.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }));
        let state = DeviceState { inner: inner.clone() };
        let router = Router::new()
            .route("/api/v1/models/{*rest}", get(handle_get).post(handle_post))
            .route("/chat/completions", post(handle_chat))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        FakeDevice { url: format!("http://{addr}"), inner }
    }

    fn calls(&self) -> Vec<(String, String)> {
        self.inner.lock().unwrap().calls.clone()
    }

    fn last_chat_model(&self) -> Option<String> {
        self.inner.lock().unwrap().last_chat_model.clone()
    }
}

async fn handle_get(State(st): State<DeviceState>, AxPath(rest): AxPath<String>) -> Response {
    if rest != "running" {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response();
    }
    let inner = st.inner.lock().unwrap();
    let mut running: Vec<String> = inner.running.iter().cloned().collect();
    running.sort();
    (StatusCode::OK, Json(json!({"object": "list", "running": running, "pending": []}))).into_response()
}

async fn handle_post(State(st): State<DeviceState>, AxPath(rest): AxPath<String>) -> Response {
    if let Some(id) = rest.strip_suffix("/start") {
        let mut inner = st.inner.lock().unwrap();
        inner.calls.push(("start".to_string(), id.to_string()));
        inner.running.insert(id.to_string());
        return (StatusCode::OK, Json(json!({"ok": true}))).into_response();
    }
    if let Some(id) = rest.strip_suffix("/stop") {
        let mut inner = st.inner.lock().unwrap();
        inner.calls.push(("stop".to_string(), id.to_string()));
        inner.running.remove(id);
        return (StatusCode::OK, Json(json!({"ok": true}))).into_response();
    }
    (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response()
}

async fn handle_chat(State(st): State<DeviceState>, Json(body): Json<Value>) -> Response {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    st.inner.lock().unwrap().last_chat_model = Some(model);
    (StatusCode::OK, Json(json!({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}))).into_response()
}

fn mgr(device: &FakeDevice) -> Arc<DeviceModelManager> {
    Arc::new(DeviceModelManager::with_retry_after(
        Arc::new(RestBackend::tiiny(device.url.clone(), HashMap::new(), 0.01, 5.0)),
        5.0,
    ))
}

async fn spawn_router(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

// --- Gate-level tests (ported from test_pool.py's Gate cases) -----------------

/// `parallel=1` must serialize two concurrent `enter`/hold/exit — no interleaving.
#[tokio::test]
async fn gate_parallel_one_serializes() {
    let device = FakeDevice::spawn(&["A"]).await;
    let m = mgr(&device);
    let gate = Arc::new(Gate::new(m.clone(), 1, None, 5.0, None, 5.0));
    let order: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));

    async fn worker(gate: Arc<Gate>, order: Arc<StdMutex<Vec<String>>>, tag: &'static str, hold: Duration) {
        let slot = gate.enter("A").await.unwrap();
        order.lock().unwrap().push(format!("enter-{tag}"));
        tokio::time::sleep(hold).await;
        order.lock().unwrap().push(format!("exit-{tag}"));
        drop(slot);
    }

    let t1 = tokio::spawn(worker(gate.clone(), order.clone(), "1", Duration::from_millis(50)));
    let t2 = tokio::spawn(worker(gate.clone(), order.clone(), "2", Duration::from_millis(0)));
    t1.await.unwrap();
    t2.await.unwrap();

    let order = order.lock().unwrap().clone();
    assert!(
        order == vec!["enter-1", "exit-1", "enter-2", "exit-2"]
            || order == vec!["enter-2", "exit-2", "enter-1", "exit-1"],
        "expected non-interleaved order, got {order:?}"
    );
}

/// `queue_max` saturated (queue already holding `queue_max` waiters) → the next
/// `enter` is rejected immediately with `Backpressure`.
#[tokio::test]
async fn gate_queue_max_saturated_is_backpressure() {
    let device = FakeDevice::spawn(&["A"]).await;
    let m = mgr(&device);
    let gate = Arc::new(Gate::new(m.clone(), 1, Some(1), 5.0, None, 5.0));

    let holder_ready = Arc::new(Notify::new());
    let release_holder = Arc::new(Notify::new());

    let g1 = gate.clone();
    let hr = holder_ready.clone();
    let rh = release_holder.clone();
    let holder = tokio::spawn(async move {
        let slot = g1.enter("A").await.unwrap();
        hr.notify_one();
        rh.notified().await;
        drop(slot);
    });
    holder_ready.notified().await;

    // Waiter fills the single queue slot (blocked on the semaphore, since the
    // holder above holds the only parallel=1 permit).
    let g2 = gate.clone();
    let waiter = tokio::spawn(async move {
        let _slot = g2.enter("A").await.unwrap();
    });

    // Deterministically wait until the waiter has actually enqueued (queued == 1
    // == queue_max) rather than sleeping a guessed duration.
    while m.queued("A") < 1 {
        tokio::task::yield_now().await;
    }

    match gate.enter("A").await {
        Err(PoolError::Backpressure(_)) => {}
        Ok(_) => panic!("expected Backpressure, got Ok(Slot)"),
        Err(PoolError::Device(msg)) => panic!("expected Backpressure, got Device({msg})"),
    }

    release_holder.notify_one();
    holder.await.unwrap();
    waiter.await.unwrap();
}

/// A holder blocking the single `parallel=1` permit past `queue_timeout` makes the
/// next `enter` give up and return `Backpressure`.
#[tokio::test]
async fn gate_queue_timeout_is_backpressure() {
    let device = FakeDevice::spawn(&["A"]).await;
    let m = mgr(&device);
    let gate = Arc::new(Gate::new(m.clone(), 1, None, 0.05, None, 5.0));

    let holder_ready = Arc::new(Notify::new());
    let release_holder = Arc::new(Notify::new());

    let g1 = gate.clone();
    let hr = holder_ready.clone();
    let rh = release_holder.clone();
    let holder = tokio::spawn(async move {
        let slot = g1.enter("A").await.unwrap();
        hr.notify_one();
        rh.notified().await;
        drop(slot);
    });
    holder_ready.notified().await;

    match gate.enter("A").await {
        Err(PoolError::Backpressure(_)) => {}
        Ok(_) => panic!("expected Backpressure, got Ok(Slot)"),
        Err(PoolError::Device(msg)) => panic!("expected Backpressure, got Device({msg})"),
    }

    release_holder.notify_one();
    holder.await.unwrap();
}

/// A model with a held `Slot` (in-flight) is never chosen as an eviction victim —
/// loading a third model at capacity must Backpressure rather than stop a busy one.
#[tokio::test]
async fn gate_protects_serving_model_from_eviction() {
    let device = FakeDevice::spawn(&["A", "B"]).await;
    let m = mgr(&device);
    m.ensure_loaded("A", None).await.unwrap();
    m.ensure_loaded("B", None).await.unwrap();
    let gate = Arc::new(Gate::new(m.clone(), 1, None, 5.0, Some(2), 5.0));

    let ready = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());

    let ga = gate.clone();
    let ra = ready.clone();
    let rl_a = release.clone();
    let ta = tokio::spawn(async move {
        let slot = ga.enter("A").await.unwrap();
        ra.notify_one();
        rl_a.notified().await;
        drop(slot);
    });

    let gb = gate.clone();
    let rl_b = release.clone();
    let tb = tokio::spawn(async move {
        let slot = gb.enter("B").await.unwrap();
        rl_b.notified().await;
        drop(slot);
    });

    ready.notified().await;
    while m.in_flight("B") < 1 {
        tokio::task::yield_now().await;
    }

    match gate.enter("C").await {
        Err(PoolError::Backpressure(_)) => {}
        Ok(_) => panic!("expected Backpressure, got Ok(Slot)"),
        Err(PoolError::Device(msg)) => panic!("expected Backpressure, got Device({msg})"),
    }
    assert!(!device.calls().contains(&("stop".to_string(), "A".to_string())));
    assert!(!device.calls().contains(&("stop".to_string(), "B".to_string())));

    release.notify_waiters();
    ta.await.unwrap();
    tb.await.unwrap();
}

// --- end-to-end: pooled passthrough through the real HTTP surface -------------

/// `POST /v1/chat/completions {"model":"device/default"}` with a management-capable
/// "device" inferencer (config-only, `management_url` set): loads the configured
/// `virtual.default` on demand and resolves/forwards to it. A sibling inferencer
/// "device2" with `queue_max=0` forces immediate `Backpressure` on its very first
/// request (queued() starts at 0, which is already >= queue_max=0), proving the
/// 503 + Retry-After error path deterministically (no timing/concurrency needed).
///
/// Both scenarios share one env-var-scoped `build_state()` — `WOOLLAMA_CONFIG_DIR`
/// is process-global, so (mirroring `passthrough_config.rs`/`images_embeddings.rs`)
/// this file keeps its env-touching assertions in a single #[tokio::test].
#[tokio::test]
async fn pooled_passthrough_end_to_end() {
    let device = FakeDevice::spawn(&[]).await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        format!(
            "[inferencers.device]\nbase_url=\"{u}\"\nmanagement_url=\"{u}\"\n\
             [inferencers.device.virtual]\ndefault=\"Qwen/Coder\"\n\
             [inferencers.device2]\nbase_url=\"{u}\"\nmanagement_url=\"{u}\"\nqueue_max=0\n",
            u = device.url,
        ),
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn_router(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    // --- loads on demand + resolves 'default' to the configured virtual model ---
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "device/default", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "ok");
    assert!(
        device.calls().contains(&("start".to_string(), "Qwen/Coder".to_string())),
        "expected the device to have been asked to start Qwen/Coder; calls={:?}",
        device.calls()
    );
    assert_eq!(device.last_chat_model().as_deref(), Some("Qwen/Coder"));

    // --- forced Backpressure (queue_max=0) -> 503 + Retry-After ---
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "device2/SomeModel", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);
    assert_eq!(r.headers().get("retry-after").map(|v| v.to_str().unwrap()), Some("5"));

    // --- streaming: the Slot held inside the SSE body must release when the
    // stream completes, not when the handler returns. "device" has the default
    // parallel=1, so if the first stream's Slot leaked (never dropped), this
    // second streamed request to the SAME real model would starve waiting on the
    // semaphore permit (eventually 503 Backpressure after queue_timeout) instead
    // of succeeding promptly.
    for _ in 0..2 {
        let r = c
            .post(format!("{base}/v1/chat/completions"))
            .json(&json!({
                "model": "device/Qwen/Coder",
                "stream": true,
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        r.bytes().await.unwrap(); // fully drain -> the SSE body's generator finishes -> Slot drops
    }
}
