//! Protecting in-flight work from **device-forced** swaps (#47).
//!
//! Our eviction protection assumes woollama is the only party that evicts. On a capacity-bound
//! device with `pool_max` unset — the correct configuration, since `pool_max` counts models while
//! the hardware counts capacity — we never evict at all: we issue `start` and the device makes
//! room by killing something. Hardware measurement showed one in-flight request lost per swap,
//! deterministically, with **zero** `stop` calls in a logging proxy. The protection did not fail,
//! it never applied.
//!
//! We cannot decline a device-forced eviction. We can choose *when* to trigger it, which is a
//! stronger position: the proxy log showed nothing else on that device initiates one.
//!
//! The fixture backend is in-process rather than HTTP. These tests are about ordering — did the
//! load wait for the victim to drain — and an HTTP fixture would add a second source of timing
//! without testing anything the ordering doesn't already cover.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use woollama_server::pool::{DeviceBackend, DeviceModelManager, Gate, ModelCapabilities, PoolError, SwapForecast};

/// A device that can be told to demand an eviction, and records the order it was driven in.
struct ForcingDevice {
    running: StdMutex<HashSet<String>>,
    /// The victim `unload_candidate` names, if any.
    victim: StdMutex<Option<String>>,
    /// Set once `load` is called — the moment the device would kill the victim.
    loaded_at_tick: AtomicU64,
    tick: AtomicU64,
    /// How many times `unload_candidate` was consulted — the design claims once per LOAD, never
    /// per request, and ~192 ms per request would be unacceptable.
    asked: AtomicU64,
}

impl ForcingDevice {
    fn new(running: &[&str], victim: Option<&str>) -> Arc<Self> {
        Arc::new(ForcingDevice {
            running: StdMutex::new(running.iter().map(|s| s.to_string()).collect()),
            victim: StdMutex::new(victim.map(str::to_string)),
            loaded_at_tick: AtomicU64::new(0),
            tick: AtomicU64::new(0),
            asked: AtomicU64::new(0),
        })
    }
    fn bump(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::SeqCst) + 1
    }
    fn load_tick(&self) -> u64 {
        self.loaded_at_tick.load(Ordering::SeqCst)
    }
    fn ask_count(&self) -> u64 {
        self.asked.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl DeviceBackend for ForcingDevice {
    async fn list_loaded(&self) -> Result<HashSet<String>, PoolError> {
        Ok(self.running.lock().unwrap().clone())
    }
    async fn list_loaded_detailed(&self) -> Result<(HashSet<String>, ModelCapabilities), PoolError> {
        Ok((self.list_loaded().await?, ModelCapabilities::new()))
    }
    async fn load(&self, id: &str) -> Result<(), PoolError> {
        // The device makes room on load: whatever `unload_candidate` named is gone now.
        let victim = self.victim.lock().unwrap().clone();
        if let Some(v) = victim {
            self.running.lock().unwrap().remove(&v);
        }
        self.running.lock().unwrap().insert(id.to_string());
        self.loaded_at_tick.store(self.bump(), Ordering::SeqCst);
        Ok(())
    }
    async fn unload(&self, id: &str) -> Result<(), PoolError> {
        self.running.lock().unwrap().remove(id);
        Ok(())
    }
    async fn unload_candidate(&self, _id: &str) -> Result<SwapForecast, PoolError> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        Ok(match self.victim.lock().unwrap().clone() {
            Some(v) => SwapForecast::Evicts(vec![v]),
            None => SwapForecast::NoEviction,
        })
    }
}

/// A backend that cannot answer the question — the default trait impl.
struct SilentDevice {
    running: StdMutex<HashSet<String>>,
}

#[async_trait::async_trait]
impl DeviceBackend for SilentDevice {
    async fn list_loaded(&self) -> Result<HashSet<String>, PoolError> {
        Ok(self.running.lock().unwrap().clone())
    }
    async fn load(&self, id: &str) -> Result<(), PoolError> {
        self.running.lock().unwrap().insert(id.to_string());
        Ok(())
    }
    async fn unload(&self, id: &str) -> Result<(), PoolError> {
        self.running.lock().unwrap().remove(id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------

/// No eviction required ⇒ the load happens immediately, with no waiting.
#[tokio::test]
async fn no_forced_eviction_loads_immediately() {
    let dev = ForcingDevice::new(&["A"], None);
    let m = Arc::new(DeviceModelManager::new(dev.clone()));
    let gate = Arc::new(Gate::new(m.clone(), 1, None, 10.0, None, 5.0));

    let started = std::time::Instant::now();
    let slot = gate.enter("B").await.expect("B should load");
    assert!(started.elapsed() < Duration::from_secs(1), "must not wait when nothing is evicted");
    assert_eq!(dev.ask_count(), 1, "the device should have been consulted exactly once");
    drop(slot);
}

/// A forced eviction of an **idle** model ⇒ still immediate. The wait is for busy victims only.
#[tokio::test]
async fn a_forced_eviction_of_an_idle_model_does_not_wait() {
    let dev = ForcingDevice::new(&["A"], Some("A"));
    let m = Arc::new(DeviceModelManager::new(dev.clone()));
    m.ensure_loaded("A", None).await.unwrap();
    let gate = Arc::new(Gate::new(m.clone(), 1, None, 10.0, None, 5.0));

    let started = std::time::Instant::now();
    let slot = gate.enter("B").await.expect("B should load");
    assert!(started.elapsed() < Duration::from_secs(1), "an idle victim needs no wait");
    drop(slot);
}

/// **The defect.** A forced eviction of a model we are actively serving must wait for our own
/// in-flight work to drain before the load that kills it.
#[tokio::test]
async fn a_forced_eviction_waits_for_our_in_flight_work_to_drain() {
    let dev = ForcingDevice::new(&["A"], Some("A"));
    let m = Arc::new(DeviceModelManager::new(dev.clone()));
    m.ensure_loaded("A", None).await.unwrap();
    let gate = Arc::new(Gate::new(m.clone(), 1, None, 10.0, None, 5.0));

    // Hold A: this is the request the device would kill.
    let held = gate.enter("A").await.expect("A is resident");
    assert_eq!(m.in_flight("A"), 1);

    let g = gate.clone();
    let loader = tokio::spawn(async move { g.enter("B").await.map(drop) });

    // While A is in flight, the load must NOT have happened — issuing it is what kills A.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        dev.load_tick(),
        0,
        "the load fired while our own request was in flight — that request is now dead, \
         which is precisely #47"
    );

    drop(held);
    tokio::time::timeout(Duration::from_secs(10), loader)
        .await
        .expect("the load never completed after the victim drained")
        .unwrap()
        .expect("B should be served");
    assert!(dev.load_tick() > 0, "the load must happen once the victim is idle");
}

/// A victim that never drains must not pin the caller forever. After the deadline we load anyway:
/// the bound is on politeness, not a veto — refusing would trade a request we MIGHT have lost for
/// one we definitely lose.
#[tokio::test]
async fn a_victim_that_never_drains_still_gets_served_after_the_deadline() {
    let dev = ForcingDevice::new(&["A"], Some("A"));
    let m = Arc::new(DeviceModelManager::new(dev.clone()));
    m.ensure_loaded("A", None).await.unwrap();
    // Short queue_timeout so the deadline is reachable in a test.
    let gate = Arc::new(Gate::new(m.clone(), 1, None, 0.5, None, 5.0));

    let _held = gate.enter("A").await.expect("A is resident"); // never released
    let started = std::time::Instant::now();
    // BOUNDED deliberately. Without the deadline this waits forever, because the victim never
    // drains — so removing the deadline would HANG this test rather than fail it, and a test
    // whose failure mode is a stall is most of the way to no test. Confirmed by mutation.
    let slot = tokio::time::timeout(Duration::from_secs(5), gate.enter("B"))
        .await
        .expect("the load never happened — the deadline is missing, so a victim that never drains pins the caller forever")
        .expect("B must still be served, not refused");
    let waited = started.elapsed();

    assert!(waited >= Duration::from_millis(400), "it must WAIT first, got {waited:?}");
    assert!(dev.load_tick() > 0, "and then load anyway rather than refusing");
    drop(slot);
}

/// A backend that cannot answer behaves exactly as before — no new waiting, no new calls.
#[tokio::test]
async fn a_backend_that_cannot_say_is_unaffected() {
    let dev = Arc::new(SilentDevice { running: StdMutex::new(HashSet::new()) });
    let m = Arc::new(DeviceModelManager::new(dev.clone()));
    let gate = Arc::new(Gate::new(m.clone(), 1, None, 10.0, None, 5.0));

    let started = std::time::Instant::now();
    let slot = gate.enter("B").await.expect("B should load");
    assert!(started.elapsed() < Duration::from_secs(1), "default impl must not introduce a wait");
    drop(slot);
}

/// The pre-flight costs one call per LOAD, never one per request.
///
/// ~192 ms per request would be unacceptable on a 0.65 s warm call; per load it is free against a
/// 25-30 s cold load. The whole cost argument for this design rests on that distinction, so it is
/// pinned here rather than left to a comment — and mirrored by an explicit log line, since an
/// operator checking it on real hardware cannot run this test.
#[tokio::test]
async fn the_pre_flight_costs_one_call_per_load_not_per_request() {
    let dev = ForcingDevice::new(&[], None);
    let m = Arc::new(DeviceModelManager::new(dev.clone()));
    let gate = Arc::new(Gate::new(m.clone(), 4, None, 10.0, None, 5.0));

    // First request loads the model.
    drop(gate.enter("B").await.expect("B loads"));
    assert_eq!(dev.ask_count(), 1, "the load consults the device once");

    // Five more requests for the SAME, now-resident model must add nothing.
    for _ in 0..5 {
        drop(gate.enter("B").await.expect("B is resident"));
    }
    assert_eq!(
        dev.ask_count(),
        1,
        "requests for a resident model must not pay the pre-flight — it belongs on the load path, \
         and ~192ms per request would make this design cost more than it saves"
    );
}

// --- the device preset over HTTP -------------------------------------------------------
//
// Everything above drives an in-process fixture that answers `unload_candidate` by construction.
// That is why all of it was green while the `device` preset had NO implementation at all — the
// trait default returned "cannot say", woollamad never issued the GET, and a full hardware run
// was spent before anyone noticed. These tests exist so that cannot happen silently again: they
// assert the real backend issues the real call and parses the real response shape.

/// Serve one `unload_candidate` body and record what was asked for.
async fn spawn_forecast_device(body: serde_json::Value) -> (String, Arc<StdMutex<Vec<String>>>) {
    use axum::extract::{Path as AxPath, State};
    use axum::routing::get;
    use axum::{Json, Router};

    let seen: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let state = (seen.clone(), body);
    let app = Router::new()
        .route(
            "/api/v1/models/{*rest}",
            get(|State((seen, body)): State<(Arc<StdMutex<Vec<String>>>, serde_json::Value)>,
                 AxPath(rest): AxPath<String>| async move {
                seen.lock().unwrap().push(rest);
                Json(body)
            }),
        )
        .with_state(state);
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    (format!("http://{addr}"), seen)
}

/// The regression test for the actual bug: the `device` preset must ISSUE the call, and must key
/// on `model_id[]` — the full victim set — rather than the singular `candidate`.
///
/// The body is the shape measured on hardware: `candidate` names an IDLE embedder while
/// `model_id[]` also carries the BUSY 30B. Keying on `candidate` sees an idle victim, skips the
/// wait, loads, and the device kills the busy one.
#[tokio::test]
async fn the_device_preset_issues_the_call_and_reads_the_full_victim_list() {
    let (url, seen) = spawn_forecast_device(serde_json::json!({
        "unload_required": true,
        "candidate": { "model_id": "Qwen3-Embedding-0.6B", "active_request_count": 0 },
        "model_id": ["Qwen3-Embedding-0.6B", "Qwen3-30B-A3B-Instruct"],
        "npu_required": 55,
        "npu_available": 44
    }))
    .await;
    let b = woollama_server::pool::RestBackend::device(url, Default::default(), 0.01, 5.0);

    let got = b.unload_candidate("Qwen3.6-35B-A3B-turbo").await.expect("the call must succeed");

    assert_eq!(
        seen.lock().unwrap().as_slice(),
        &["Qwen3.6-35B-A3B-turbo/unload_candidate".to_string()],
        "the device preset must actually issue the GET — it previously never did, and the trait \
         default made that look like a device saying 'nothing to evict'"
    );
    match got {
        SwapForecast::Evicts(v) => assert_eq!(
            v,
            vec!["Qwen3-Embedding-0.6B".to_string(), "Qwen3-30B-A3B-Instruct".to_string()],
            "must read model_id[], the FULL victim set — `candidate` names only the idle embedder"
        ),
        other => panic!("expected Evicts, got {other:?}"),
    }
}

/// `unload_required: false` is a real "no", distinct from not knowing.
#[tokio::test]
async fn the_device_preset_reports_no_eviction_when_the_device_says_so() {
    let (url, _) = spawn_forecast_device(serde_json::json!({ "unload_required": false })).await;
    let b = woollama_server::pool::RestBackend::device(url, Default::default(), 0.01, 5.0);
    assert_eq!(b.unload_candidate("m").await.unwrap(), SwapForecast::NoEviction);
}

/// A body we cannot read is `Unknown`, never `NoEviction`.
///
/// This is the merge that hid the original bug, now impossible to express: reporting "nothing
/// will be evicted" because parsing failed is a different claim from the device saying so.
#[tokio::test]
async fn an_unreadable_forecast_is_unknown_not_no_eviction() {
    let (url, _) = spawn_forecast_device(serde_json::json!({ "something_else": 1 })).await;
    let b = woollama_server::pool::RestBackend::device(url, Default::default(), 0.01, 5.0);
    assert_eq!(b.unload_candidate("m").await.unwrap(), SwapForecast::Unknown);

    // "eviction required, but I won't say of what" is also unknown — waiting on an empty victim
    // set would silently skip the wait while looking like protection.
    let (url2, _) = spawn_forecast_device(serde_json::json!({ "unload_required": true, "model_id": [] })).await;
    let b2 = woollama_server::pool::RestBackend::device(url2, Default::default(), 0.01, 5.0);
    assert_eq!(b2.unload_candidate("m").await.unwrap(), SwapForecast::Unknown);
}
