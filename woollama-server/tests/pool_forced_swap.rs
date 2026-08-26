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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use woollama_server::pool::{DeviceBackend, DeviceModelManager, Gate, ModelCapabilities, PoolError};

/// A device that can be told to demand an eviction, and records the order it was driven in.
struct ForcingDevice {
    running: StdMutex<HashSet<String>>,
    /// The victim `unload_candidate` names, if any.
    victim: StdMutex<Option<String>>,
    /// Set once `load` is called — the moment the device would kill the victim.
    loaded_at_tick: AtomicU64,
    tick: AtomicU64,
    /// Records whether `unload_candidate` was consulted at all.
    asked: AtomicBool,
}

impl ForcingDevice {
    fn new(running: &[&str], victim: Option<&str>) -> Arc<Self> {
        Arc::new(ForcingDevice {
            running: StdMutex::new(running.iter().map(|s| s.to_string()).collect()),
            victim: StdMutex::new(victim.map(str::to_string)),
            loaded_at_tick: AtomicU64::new(0),
            tick: AtomicU64::new(0),
            asked: AtomicBool::new(false),
        })
    }
    fn bump(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::SeqCst) + 1
    }
    fn load_tick(&self) -> u64 {
        self.loaded_at_tick.load(Ordering::SeqCst)
    }
    fn was_asked(&self) -> bool {
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
    async fn unload_candidate(&self, _id: &str) -> Result<Option<String>, PoolError> {
        self.asked.store(true, Ordering::SeqCst);
        Ok(self.victim.lock().unwrap().clone())
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
    assert!(dev.was_asked(), "the device should still have been consulted");
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
