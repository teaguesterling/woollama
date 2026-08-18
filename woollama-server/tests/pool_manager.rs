//! `DeviceModelManager` (Task 5) — TDD port of `tests/test_pool.py`'s manager-level
//! cases, incl. the eviction-race regression.
//!
//! `FakeDevice` stands in for the device's :8800 model-management API (GET
//! .../running, POST .../{id}/start, POST .../{id}/stop) as a spawned `axum::Router`.
//! `block_stop` lets the eviction-race test deterministically hold a `/stop` call in
//! flight (via a `tokio::sync::Notify`) while a racer lands concurrent work on the
//! same model id — mirroring the Python fixture's `threading.Event`-gated handler.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;
use tokio::sync::Notify;

use woollama_server::pool::{DeviceModelManager, PoolError, RestBackend};

#[derive(Default)]
struct DeviceInner {
    running: HashSet<String>,
    calls: Vec<(String, String)>,
    fail_start: bool,
    fail_stop: bool,
    fail_running: bool,
    running_bad_json: bool,
    start_no_register: bool,
    block_stop: bool,
}

#[derive(Clone)]
struct DeviceState {
    inner: Arc<StdMutex<DeviceInner>>,
    stop_started: Arc<AtomicBool>,
    stop_release: Arc<Notify>,
}

struct FakeDevice {
    url: String,
    inner: Arc<StdMutex<DeviceInner>>,
    stop_started: Arc<AtomicBool>,
    stop_release: Arc<Notify>,
}

impl FakeDevice {
    async fn spawn(running: &[&str]) -> Self {
        let inner = Arc::new(StdMutex::new(DeviceInner {
            running: running.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }));
        let stop_started = Arc::new(AtomicBool::new(false));
        let stop_release = Arc::new(Notify::new());
        let state = DeviceState {
            inner: inner.clone(),
            stop_started: stop_started.clone(),
            stop_release: stop_release.clone(),
        };
        let router = Router::new()
            .route("/api/v1/models/{*rest}", get(handle_get).post(handle_post))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        FakeDevice { url: format!("http://{addr}"), inner, stop_started, stop_release }
    }

    fn calls(&self) -> Vec<(String, String)> {
        self.inner.lock().unwrap().calls.clone()
    }

    fn running(&self) -> HashSet<String> {
        self.inner.lock().unwrap().running.clone()
    }

    fn set_fail_start(&self, v: bool) {
        self.inner.lock().unwrap().fail_start = v;
    }

    fn set_fail_stop(&self, v: bool) {
        self.inner.lock().unwrap().fail_stop = v;
    }

    fn set_block_stop(&self, v: bool) {
        self.inner.lock().unwrap().block_stop = v;
    }

    /// Deterministic wait until a blocked `/stop` handler has actually landed on the
    /// device and is parked there — no sleep-based timing guess (mirrors the Python
    /// fixture's `while not device.stop_started.is_set(): await asyncio.sleep(0)`).
    async fn wait_stop_started(&self) {
        while !self.stop_started.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    }

    fn release_stop(&self) {
        self.stop_release.notify_one();
    }
}

async fn handle_get(State(st): State<DeviceState>, AxPath(rest): AxPath<String>) -> Response {
    if rest != "running" {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response();
    }
    let inner = st.inner.lock().unwrap();
    if inner.fail_running {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "running failed"}))).into_response();
    }
    if inner.running_bad_json {
        return (StatusCode::OK, "not json").into_response();
    }
    let mut running: Vec<String> = inner.running.iter().cloned().collect();
    running.sort();
    (StatusCode::OK, Json(json!({"object": "list", "running": running, "pending": []}))).into_response()
}

async fn handle_post(State(st): State<DeviceState>, AxPath(rest): AxPath<String>) -> Response {
    if let Some(id) = rest.strip_suffix("/start") {
        let mut inner = st.inner.lock().unwrap();
        inner.calls.push(("start".to_string(), id.to_string()));
        if inner.fail_start {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "start failed"}))).into_response();
        }
        if !inner.start_no_register {
            inner.running.insert(id.to_string());
        }
        return (StatusCode::OK, Json(json!({"ok": true}))).into_response();
    }
    if let Some(id) = rest.strip_suffix("/stop") {
        // Read (and drop) the block flag before any `.await` — never hold a
        // std::sync MutexGuard across an await point.
        let block = st.inner.lock().unwrap().block_stop;
        if block {
            st.stop_started.store(true, Ordering::SeqCst);
            st.stop_release.notified().await;
        }
        let mut inner = st.inner.lock().unwrap();
        inner.calls.push(("stop".to_string(), id.to_string()));
        if inner.fail_stop {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "stop failed"}))).into_response();
        }
        inner.running.remove(id);
        return (StatusCode::OK, Json(json!({"ok": true}))).into_response();
    }
    (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response()
}

/// Build a manager over a `RestBackend::device` for the given mock URL/config —
/// the same shape `DeviceModelManager::with_config` used to build directly.
fn manager_with_config(
    url: String,
    headers: HashMap<String, String>,
    poll_interval: f64,
    load_timeout: f64,
    retry_after: f64,
) -> DeviceModelManager {
    DeviceModelManager::with_retry_after(
        Arc::new(RestBackend::device(url, headers, poll_interval, load_timeout)),
        retry_after,
    )
}

fn mgr(device: &FakeDevice) -> DeviceModelManager {
    manager_with_config(device.url.clone(), HashMap::new(), 0.01, 5.0, 5.0)
}

#[tokio::test]
async fn ensure_loaded_starts_when_absent() {
    let device = FakeDevice::spawn(&[]).await;
    let m = mgr(&device);
    m.ensure_loaded("Qwen/Coder", None).await.unwrap();
    assert!(device.calls().contains(&("start".to_string(), "Qwen/Coder".to_string())));
    assert!(device.running().contains("Qwen/Coder"));
    assert_eq!(m.snapshot(), vec!["Qwen/Coder".to_string()]);
}

#[tokio::test]
async fn ensure_loaded_noop_when_already_loaded() {
    let device = FakeDevice::spawn(&["Qwen/Coder"]).await;
    let m = mgr(&device);
    m.ensure_loaded("Qwen/Coder", None).await.unwrap();
    m.ensure_loaded("Qwen/Coder", None).await.unwrap();
    assert!(device.calls().iter().all(|(verb, _)| verb != "start"));
}

#[tokio::test]
async fn concurrent_ensure_loaded_dedups_to_one_start() {
    let device = Arc::new(FakeDevice::spawn(&[]).await);
    let m = Arc::new(mgr(&device));
    let mut tasks = Vec::new();
    for _ in 0..5 {
        let m = m.clone();
        tasks.push(tokio::spawn(async move { m.ensure_loaded("Qwen/Coder", None).await }));
    }
    for t in tasks {
        t.await.unwrap().unwrap();
    }
    let starts: Vec<_> = device.calls().into_iter().filter(|(verb, _)| verb == "start").collect();
    assert_eq!(starts, vec![("start".to_string(), "Qwen/Coder".to_string())]);
}

#[tokio::test]
async fn start_failure_raises_device_error() {
    let device = FakeDevice::spawn(&[]).await;
    device.set_fail_start(true);
    let m = mgr(&device);
    match m.ensure_loaded("Qwen/Coder", None).await {
        Err(PoolError::Device(_)) => {}
        other => panic!("expected PoolError::Device, got {other:?}"),
    }
}

#[tokio::test]
async fn unreachable_device_raises_device_error() {
    let m = manager_with_config("http://127.0.0.1:1".to_string(), HashMap::new(), 0.01, 1.0, 5.0);
    match m.ensure_loaded("Qwen/Coder", None).await {
        Err(PoolError::Device(_)) => {}
        other => panic!("expected PoolError::Device, got {other:?}"),
    }
}

#[tokio::test]
async fn evicts_lru_idle_at_capacity() {
    let device = FakeDevice::spawn(&["A", "B"]).await;
    let m = mgr(&device);
    m.ensure_loaded("A", None).await.unwrap(); // last_used older
    m.ensure_loaded("B", None).await.unwrap(); // last_used newer
    m.ensure_loaded("C", Some(2)).await.unwrap(); // full -> evict LRU idle (A)
    assert!(device.calls().contains(&("stop".to_string(), "A".to_string())));
    assert!(!device.running().contains("A"));
    assert!(device.running().contains("C"));
}

/// Every loaded model busy at capacity reports `SwapBlocked`, NOT `Backpressure`.
///
/// This assertion changed with #39, deliberately. The manager's job is to report what it sees —
/// "capacity is full and nothing is evictable *yet*" — and that is a signal, not an answer to a
/// caller. Deciding how long that is worth waiting for belongs to `Gate`, which owns
/// `queue_timeout`; it converts this to `Backpressure` once the wait is spent. The refusal to
/// evict a busy model is unchanged and is what this test still guards.
#[tokio::test]
async fn no_evict_when_all_busy_reports_swap_blocked() {
    let device = FakeDevice::spawn(&["A", "B"]).await;
    let m = mgr(&device);
    m.ensure_loaded("A", None).await.unwrap();
    m.ensure_loaded("B", None).await.unwrap();
    m.acquire("A");
    m.acquire("B"); // both serving -> not evictable
    match m.ensure_loaded("C", Some(2)).await {
        Err(PoolError::SwapBlocked) => {}
        other => panic!("expected PoolError::SwapBlocked, got {other:?}"),
    }
    assert!(!device.calls().contains(&("stop".to_string(), "A".to_string())));
    assert!(!device.calls().contains(&("stop".to_string(), "B".to_string())));
}

#[tokio::test]
async fn running_query_non_2xx_raises_device_error() {
    let device = FakeDevice::spawn(&[]).await;
    device.inner.lock().unwrap().fail_running = true;
    let m = mgr(&device);
    match m.ensure_loaded("Qwen/Coder", None).await {
        Err(PoolError::Device(_)) => {}
        other => panic!("expected PoolError::Device, got {other:?}"),
    }
}

#[tokio::test]
async fn running_query_bad_json_raises_device_error() {
    let device = FakeDevice::spawn(&[]).await;
    device.inner.lock().unwrap().running_bad_json = true;
    let m = mgr(&device);
    match m.ensure_loaded("Qwen/Coder", None).await {
        Err(PoolError::Device(_)) => {}
        other => panic!("expected PoolError::Device, got {other:?}"),
    }
}

#[tokio::test]
async fn start_poll_timeout_raises_device_error() {
    let device = FakeDevice::spawn(&[]).await;
    device.inner.lock().unwrap().start_no_register = true;
    let m = manager_with_config(device.url.clone(), HashMap::new(), 0.01, 0.05, 5.0);
    match m.ensure_loaded("Qwen/Coder", None).await {
        Err(PoolError::Device(msg)) => assert!(msg.contains("not running"), "unexpected message: {msg}"),
        other => panic!("expected PoolError::Device, got {other:?}"),
    }
}

#[tokio::test]
async fn stop_failure_raises_device_error() {
    let device = FakeDevice::spawn(&["A", "B"]).await;
    device.set_fail_stop(true);
    let m = mgr(&device);
    m.ensure_loaded("A", None).await.unwrap();
    m.ensure_loaded("B", None).await.unwrap();
    match m.ensure_loaded("C", Some(2)).await {
        Err(PoolError::Device(_)) => {}
        other => panic!("expected PoolError::Device, got {other:?}"),
    }
    assert!(device.running().contains("A"), "stop failed; device state must be unchanged");
}

/// Round-2 regression (ported from `test_eviction_race_does_not_strand_or_lose_racer`):
/// `resolver::pick_eviction` snapshots "idle" at the instant the evictor decides on a
/// victim, but the evictor then `.await`s the device's `/stop` call — and during that
/// await a racer can legitimately land an `enqueue`/`ensure_loaded`/`acquire`
/// sequence (the Gate's real call pattern) against the very model being torn down.
/// Two invariants must hold across that window:
///   1. The racer's `ensure_loaded` must not take the pre-lock fast path on stale
///      "still loaded" truth while the device is mid-teardown — it must block until
///      the evictor is done, then re-check/reload for real.
///   2. The evictor's post-stop cleanup must never silently discard the racer's
///      in_flight/queued bookkeeping just because it happened to land on the
///      (about-to-be-popped) entry while the stop was in flight.
///
/// Runs on the `current_thread` flavor so the interleaving is driven purely by
/// `.await` points (no real thread-level preemption), the same cooperative-scheduling
/// assumption the Python `asyncio` original relies on for its deterministic ordering.
#[tokio::test(flavor = "current_thread")]
async fn eviction_race_does_not_strand_or_lose_racer() {
    let device = Arc::new(FakeDevice::spawn(&["A", "B"]).await);
    device.set_block_stop(true);
    let m = Arc::new(mgr(&device));

    m.ensure_loaded("A", None).await.unwrap(); // last_used older -> LRU victim
    m.ensure_loaded("B", None).await.unwrap(); // last_used newer

    let m_evict = m.clone();
    let evict_task = tokio::spawn(async move { m_evict.ensure_loaded("C", Some(2)).await });

    // Wait until the evictor's stop("A") request has actually landed on the device
    // and is being held there.
    device.wait_stop_started().await;

    // A racer arrives while A's stop() is in flight, mirroring the Gate's real call
    // pattern (enqueue before ensure_loaded, acquire after).
    let m_racer = m.clone();
    let racer_task = tokio::spawn(async move {
        m_racer.enqueue("A");
        m_racer.ensure_loaded("A", None).await.unwrap();
        m_racer.dequeue("A");
        m_racer.acquire("A");
    });
    // Let the racer run up to its blocking point (enqueue, then park on load_lock,
    // which the evictor still holds) before we release the stop.
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;

    device.release_stop(); // let the fake device finish stopping A
    evict_task.await.unwrap().unwrap();

    // Invariant 2: the evictor's cleanup must not have silently discarded A's
    // bookkeeping (the racer's enqueue already landed on it while stop() was in
    // flight) just because the stop was in flight when the pick was made. Checked
    // here, before awaiting the racer, on the `current_thread` runtime so nothing
    // else could have run between the evictor finishing and this assertion.
    assert_eq!(m.queued("A"), 1, "victim entry's queued count was lost mid-race (entry silently discarded)");

    racer_task.await.unwrap();

    // Invariant 1: the racer must end up bound to a genuinely (re)loaded model,
    // never a phantom / torn-down one.
    assert!(m.snapshot().contains(&"A".to_string()));
    assert!(device.running().contains("A"));
    assert_eq!(m.queued("A"), 0, "racer's dequeue should balance its enqueue");
    assert_eq!(m.in_flight("A"), 1, "racer's acquire should land cleanly");
}
