//! Queueing across a model swap (#39).
//!
//! `Gate` already queues requests *for* a model. It did not queue *across a swap*: when another
//! consumer held model A on a capacity-1 device, a request for B got `503` immediately, even
//! though B is not unservable — it is servable after work that is already draining.
//!
//! The refusal to evict a busy model is correct and is preserved. What changes is what happens
//! next: wait, bounded by `queue_timeout`, rather than answer immediately.
//!
//! `FakeDevice` is a trimmed duplicate of `pool_gate.rs`'s fixture (no chat route — nothing here
//! dispatches), with one addition that matters: `start` can be made SLOW, because a swap that
//! completes instantly cannot expose an ordering bug. A fixture kinder than the hardware is how
//! the first #38 fix passed while being wrong.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;
use tokio::sync::Notify;

use woollama_server::pool::{DeviceModelManager, Gate, PoolError, RestBackend};

#[derive(Default)]
struct DeviceInner {
    running: std::collections::HashSet<String>,
    calls: Vec<(String, String)>,
}

#[derive(Clone)]
struct DeviceState {
    inner: Arc<StdMutex<DeviceInner>>,
    start_delay_ms: Arc<AtomicU64>,
}

struct FakeDevice {
    url: String,
    inner: Arc<StdMutex<DeviceInner>>,
    start_delay_ms: Arc<AtomicU64>,
}

impl FakeDevice {
    async fn spawn(running: &[&str]) -> Self {
        let inner = Arc::new(StdMutex::new(DeviceInner {
            running: running.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }));
        let start_delay_ms = Arc::new(AtomicU64::new(0));
        let state = DeviceState { inner: inner.clone(), start_delay_ms: start_delay_ms.clone() };
        let router = Router::new()
            .route("/api/v1/models/{*rest}", get(handle_get).post(handle_post))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        FakeDevice { url: format!("http://{addr}"), inner, start_delay_ms }
    }

    /// Make loading take time, so a swap is a window rather than an instant.
    fn slow_start(&self, ms: u64) {
        self.start_delay_ms.store(ms, Ordering::SeqCst);
    }

    fn calls(&self) -> Vec<(String, String)> {
        self.inner.lock().unwrap().calls.clone()
    }

    fn stopped(&self, id: &str) -> bool {
        self.calls().contains(&("stop".to_string(), id.to_string()))
    }
}

async fn handle_get(State(st): State<DeviceState>, AxPath(rest): AxPath<String>) -> Response {
    if rest != "running" {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response();
    }
    let mut running: Vec<String> = st.inner.lock().unwrap().running.iter().cloned().collect();
    running.sort();
    (StatusCode::OK, Json(json!({"object": "list", "running": running, "pending": []}))).into_response()
}

async fn handle_post(State(st): State<DeviceState>, AxPath(rest): AxPath<String>) -> Response {
    if let Some(id) = rest.strip_suffix("/start") {
        let delay = st.start_delay_ms.load(Ordering::SeqCst);
        if delay > 0 {
            tokio::time::sleep(Duration::from_millis(delay)).await;
        }
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

fn mgr(device: &FakeDevice) -> Arc<DeviceModelManager> {
    Arc::new(DeviceModelManager::with_retry_after(
        Arc::new(RestBackend::device(device.url.clone(), HashMap::new(), 0.01, 5.0)),
        5.0,
    ))
}

/// Capacity-1 device with A resident, and a `Gate` whose `queue_timeout` is `timeout`.
async fn capacity_one(timeout: f64) -> (FakeDevice, Arc<DeviceModelManager>, Arc<Gate>) {
    let device = FakeDevice::spawn(&["A"]).await;
    let m = mgr(&device);
    m.ensure_loaded("A", Some(1)).await.unwrap();
    let gate = Arc::new(Gate::new(m.clone(), 1, None, timeout, Some(1), 5.0));
    (device, m, gate)
}

// ---------------------------------------------------------------------------------

/// The headline case: a request for a non-resident model waits for the swap instead of being
/// refused, and is served once the holder of the resident model lets go.
#[tokio::test]
async fn a_request_for_another_model_waits_for_the_swap_instead_of_503() {
    let (device, m, gate) = capacity_one(10.0).await;

    let held = gate.enter("A").await.expect("A is resident");

    let gb = gate.clone();
    let waiter = tokio::spawn(async move { gb.enter("B").await.map(drop) });

    // B must still be waiting while A is held — if it resolves here it either 503'd (the bug)
    // or evicted a busy model (much worse).
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!waiter.is_finished(), "B resolved while A was still in flight");
    assert!(!device.stopped("A"), "A was evicted while serving — eviction protection broke");

    drop(held);

    let outcome = tokio::time::timeout(Duration::from_secs(10), waiter)
        .await
        .expect("B never completed after A was released")
        .unwrap();
    assert!(outcome.is_ok(), "B should be served after the swap, got {outcome:?}");
    assert!(device.stopped("A"), "the swap should have evicted the now-idle A");
    assert_eq!(m.in_flight("B"), 0, "B's slot should have been dropped");
}

/// The starvation case, and the reason the fairness hold is load-bearing rather than a
/// refinement. `Gate::enter` enqueues BEFORE it loads, and `pick_eviction` skips anything with
/// `queued > 0` — so without a hold, a steady stream of new A requests keeps `A.queued` above
/// zero forever and the waiter times out anyway. Waiting alone converts an immediate failure
/// into a slow one.
#[tokio::test]
async fn a_steady_stream_for_the_resident_model_cannot_starve_the_waiter() {
    let (device, _m, gate) = capacity_one(10.0).await;

    // A flag, not a `Notify`: `notify_waiters` wakes only whoever is registered at that instant
    // and stores nothing, so a stop signal sent while the stream task is inside `enter` is lost
    // and the task loops forever. That hung this test rather than failing it — which is the
    // worse outcome, because a hang looks like infrastructure trouble rather than a bug.
    let stop = Arc::new(AtomicBool::new(false));
    let first_ready = Arc::new(Notify::new());

    // A pipeline of A requests, each briefly held, arriving continuously.
    let ga = gate.clone();
    let stop_a = stop.clone();
    let ready_a = first_ready.clone();
    let stream = tokio::spawn(async move {
        let mut served: u32 = 0;
        loop {
            if let Ok(slot) = ga.enter("A").await {
                served += 1;
                if served == 1 {
                    ready_a.notify_one();
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
                drop(slot);
            }
            if stop_a.load(Ordering::SeqCst) {
                return served;
            }
            tokio::task::yield_now().await;
        }
    });

    first_ready.notified().await;

    let gb = gate.clone();
    let waiter = tokio::spawn(async move { gb.enter("B").await.map(drop) });

    let outcome = tokio::time::timeout(Duration::from_secs(10), waiter)
        .await
        .expect("B was starved by the A stream — the fairness hold is missing or ineffective")
        .unwrap();
    assert!(outcome.is_ok(), "B should eventually be served, got {outcome:?}");

    stop.store(true, Ordering::SeqCst);
    let served = stream.await.unwrap();
    assert!(served > 0, "the A stream should have been served too, not blocked outright");
    assert!(device.stopped("A"), "the swap should have evicted A once it drained");
}

/// The wait is bounded. `503` is still the answer when the wait genuinely exceeds
/// `queue_timeout` — the complaint in #39 was that it came back immediately, not that it came
/// back at all.
#[tokio::test]
async fn the_wait_is_bounded_by_queue_timeout() {
    let (device, _m, gate) = capacity_one(0.5).await;

    let held = gate.enter("A").await.expect("A is resident");

    let started = std::time::Instant::now();
    match gate.enter("B").await {
        Err(PoolError::Backpressure(_)) => {}
        Ok(_) => panic!("expected Backpressure once the wait exceeded queue_timeout"),
        Err(PoolError::Device(msg)) => panic!("expected Backpressure, got Device({msg})"),
        // `Gate::enter` must convert this internally; leaking it would 503 with no
        // Retry-After budget and hide a missed conversion (#39).
        Err(PoolError::SwapBlocked) => panic!("SwapBlocked escaped Gate::enter"),
    }
    let waited = started.elapsed();

    assert!(
        waited >= Duration::from_millis(400),
        "it must WAIT before refusing — returned after {waited:?}, which is the #39 bug"
    );
    assert!(!device.stopped("A"), "a timed-out waiter must not evict a busy model");
    drop(held);
}

/// A swap must not preempt live work even under the new waiting behaviour: the eviction still
/// happens only once the victim is genuinely idle, never while it is serving.
#[tokio::test]
async fn the_swap_never_evicts_a_model_that_is_still_serving() {
    let (device, m, gate) = capacity_one(10.0).await;
    device.slow_start(150);

    let held = gate.enter("A").await.expect("A is resident");

    let gb = gate.clone();
    let waiter = tokio::spawn(async move { gb.enter("B").await.map(drop) });

    for _ in 0..20 {
        assert!(
            !device.stopped("A"),
            "A was stopped while still in flight (in_flight={})",
            m.in_flight("A")
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    drop(held);
    tokio::time::timeout(Duration::from_secs(10), waiter)
        .await
        .expect("B never completed")
        .unwrap()
        .expect("B should be served");
}

/// `queue_max` still bounds the queue after a fairness hold.
///
/// Found by reading the entry path rather than by a failure: the `queue_max` check happens
/// BEFORE the hold, so requests parked in `hold_for_swap` are not counted as queued. When the
/// reservation clears they all enqueue at once, and the queue can overshoot the limit the
/// operator configured — the change would have quietly widened a documented bound.
#[tokio::test]
async fn queue_max_is_rechecked_after_a_fairness_hold() {
    let device = FakeDevice::spawn(&["A"]).await;
    let m = mgr(&device);
    m.ensure_loaded("A", Some(1)).await.unwrap();
    // queue_max = 1: at most one request may be waiting in A's queue at a time.
    let gate = Arc::new(Gate::new(m.clone(), 1, Some(1), 10.0, Some(1), 5.0));

    let held = gate.enter("A").await.expect("A is resident");

    // Takes the swap reservation and waits, since A is in flight.
    let gb = gate.clone();
    let waiter = tokio::spawn(async move { gb.enter("B").await.map(drop) });
    while m.swap_reserved_by().is_none() {
        tokio::task::yield_now().await;
    }

    // Two more A requests, both parked in the fairness hold (they have NOT enqueued).
    let mut parked = Vec::new();
    for _ in 0..2 {
        let g = gate.clone();
        parked.push(tokio::spawn(async move { g.enter("A").await.map(drop) }));
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(m.queued("A"), 0, "parked requests must not be queued yet");

    drop(held);
    tokio::time::timeout(Duration::from_secs(10), waiter)
        .await
        .expect("B never completed")
        .unwrap()
        .expect("B should be served");

    let mut backpressured = 0;
    for t in parked {
        if let Err(PoolError::Backpressure(_)) = tokio::time::timeout(Duration::from_secs(10), t)
            .await
            .expect("a parked request never resolved")
            .unwrap()
        {
            backpressured += 1;
        }
    }
    assert!(
        backpressured >= 1,
        "with queue_max=1, releasing two parked requests must not admit both — \
         queue_max has to be re-checked after the hold, not only before it"
    );
}
