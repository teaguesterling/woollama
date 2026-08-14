//! `DeviceModelManager` — the load/evict actor for one management-capable inferencer.
//!
//! One long-lived object per inferencer that has a `management_url`. It owns which
//! models are loaded on the device and their per-model runtime counters (in-flight,
//! queued, last-used), loads/unloads on demand via the device's management API, and
//! evicts an idle LRU model to make room when `pool_max` is reached. Pure decision
//! logic (what to evict, how to resolve virtual models) lives in
//! `woollama_engine::resolver`; this module is the live I/O half — a direct port of
//! `woollama.pool.DeviceModelManager` (Python oracle, `src/woollama/pool.py`),
//! **including its eviction-race fix** (see `ensure_loaded` below). `Gate`/`Slot`
//! (Task 6) build the request-queueing layer on top of this.
//!
//! ## Lock model
//!
//! Per-model runtime state (`Entry { loaded, in_flight, queued, last_used }`) lives
//! behind a single `std::sync::Mutex<HashMap<String, Entry>>`. This is the direct
//! analogue of Python's `self._entries: dict[str, _Entry]`: in Python, a bare dict is
//! safe there because coroutines never preempt each other except at an `await`, so a
//! run of plain (non-async) statements is implicitly atomic. Rust has no such
//! guarantee under a real (potentially multi-threaded) executor, so the `std::sync`
//! `Mutex` is the explicit stand-in — locked for short, non-awaiting critical
//! sections only, exactly mirroring the granularity of Python's unguarded dict
//! accesses. Being a `std::sync::Mutex` (not `tokio::sync::Mutex`), it is safe to
//! lock from a synchronous, non-async context — required because Task 6's
//! `Slot::Drop` calls `release` synchronously and `Drop` cannot `.await`.
//!
//! The load/evict *critical section* — the check-running / evict / start sequence
//! that spans multiple `.await` points on device I/O — is serialized by a separate
//! `tokio::sync::Mutex<()>` (`load_lock`), the direct analogue of Python's
//! `self._load_lock: asyncio.Lock()`. It holds no data of its own (all data lives in
//! `entries`); its only job is to ensure at most one `ensure_loaded` runs the
//! load/evict sequence for this manager at a time, so concurrent loads never
//! double-`start` and evictions never overlap. The `std::sync::Mutex` is **never**
//! held across an `.await` — every read/mutation of `entries` is a short lock/unlock
//! that completes before or after (never during) a network call.
//!
//! `last_used` is an internally incrementing counter (via an `AtomicU64`), not a wall
//! clock — only relative ordering matters for LRU eviction, and a counter can never
//! be `NaN`, unlike a fallible wall-clock read. This also means tests get
//! deterministic LRU ordering for free (each `ensure_loaded`/`acquire`/`release`
//! ticks the counter) without needing an injected fake clock the way the Python
//! tests do.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;

use woollama_engine as engine;
use woollama_engine::resolver::{self, PoolEntry};

/// Errors from the device-management I/O path, shared with `Gate` (Task 6).
///
/// `Device` maps to an HTTP 502 (device unreachable, or a start/stop/running call
/// failed); `Backpressure(retry_after_secs)` maps to an HTTP 503 with a
/// `Retry-After` header (queue saturated, wait timed out, or the pool is full with
/// no idle model to evict).
#[derive(Debug, Clone, PartialEq)]
pub enum PoolError {
    Device(String),
    Backpressure(f64),
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::Device(msg) => write!(f, "{msg}"),
            PoolError::Backpressure(retry_after) => {
                write!(f, "backpressure; retry after {retry_after}s")
            }
        }
    }
}

impl std::error::Error for PoolError {}

/// Per-model runtime state, guarded by `DeviceModelManager::entries`.
#[derive(Debug, Clone, Copy)]
struct Entry {
    loaded: bool,
    in_flight: u32,
    queued: u32,
    last_used: f64,
}

impl Entry {
    fn new(last_used: f64) -> Self {
        Entry { loaded: false, in_flight: 0, queued: 0, last_used }
    }
}

/// True for any 2xx status — the single success predicate for all three device
/// endpoints (running/start/stop), kept consistent in one place.
fn ok(status: reqwest::StatusCode) -> bool {
    status.is_success()
}

/// First `n` chars of `s` (char-boundary safe), for bounding error-message bodies.
fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// A pluggable device-management transport: however a `DeviceModelManager` talks to
/// its inferencer to discover/load/unload models. `RestBackend` is the built-in
/// implementation for Tiiny's REST shape (`{url}/api/v1/models/...`); later tasks add
/// config-defined REST protocols and an Ollama adapter behind this same seam.
#[async_trait::async_trait]
pub trait DeviceBackend: Send + Sync {
    async fn list_loaded(&self) -> Result<HashSet<String>, PoolError>;
    async fn load(&self, id: &str) -> Result<(), PoolError>;
    async fn unload(&self, id: &str) -> Result<(), PoolError>;
}

/// The built-in `DeviceBackend` for Tiiny's device-management REST API
/// (`GET {url}/api/v1/models/running`, `POST .../{id}/start`, `POST .../{id}/stop`).
/// A direct, behavior-preserving extraction of what used to be
/// `DeviceModelManager`'s private `running`/`start`/`stop`/`apply_headers` methods.
pub struct RestBackend {
    client: reqwest::Client,
    base_url: String,
    headers: HashMap<String, String>,
    poll_interval: f64,
    load_timeout: f64,
}

impl RestBackend {
    /// The Tiiny device-management REST shape.
    pub fn tiiny(management_url: String, headers: HashMap<String, String>, poll_interval: f64, load_timeout: f64) -> RestBackend {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs_f64(30.0))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        RestBackend {
            client,
            base_url: management_url.trim_end_matches('/').to_string(),
            headers,
            poll_interval,
            load_timeout,
        }
    }

    fn apply_headers(&self, mut rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        for (k, v) in &self.headers {
            rb = rb.header(k.as_str(), v.as_str());
        }
        rb
    }
}

#[async_trait::async_trait]
impl DeviceBackend for RestBackend {
    async fn list_loaded(&self) -> Result<HashSet<String>, PoolError> {
        let rb = self.apply_headers(self.client.get(format!("{}/api/v1/models/running", self.base_url)));
        let r = rb
            .send()
            .await
            .map_err(|e| PoolError::Device(format!("device unreachable: {e}")))?;
        let status = r.status();
        if !ok(status) {
            let text = r.text().await.unwrap_or_default();
            return Err(PoolError::Device(format!("running query failed: {status} {}", truncate(&text, 200))));
        }
        let v: Value = r
            .json()
            .await
            .map_err(|e| PoolError::Device(format!("running query: bad JSON: {e}")))?;
        let running = v
            .get("running")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).map(String::from).collect())
            .unwrap_or_default();
        Ok(running)
    }

    async fn load(&self, real_id: &str) -> Result<(), PoolError> {
        let path = format!("{}/api/v1/models/{real_id}/start", self.base_url);
        let rb = self.apply_headers(self.client.post(path));
        let r = rb
            .send()
            .await
            .map_err(|e| PoolError::Device(format!("start {real_id}: unreachable: {e}")))?;
        let status = r.status();
        if !ok(status) {
            let text = r.text().await.unwrap_or_default();
            return Err(PoolError::Device(format!(
                "start {real_id} failed: {status} {}",
                truncate(&text, 200)
            )));
        }
        let deadline = Instant::now() + Duration::from_secs_f64(self.load_timeout);
        while Instant::now() < deadline {
            if self.list_loaded().await?.contains(real_id) {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_secs_f64(self.poll_interval)).await;
        }
        Err(PoolError::Device(format!("start {real_id}: not running after {}s", self.load_timeout)))
    }

    async fn unload(&self, real_id: &str) -> Result<(), PoolError> {
        let path = format!("{}/api/v1/models/{real_id}/stop", self.base_url);
        let rb = self.apply_headers(self.client.post(path));
        let r = rb
            .send()
            .await
            .map_err(|e| PoolError::Device(format!("stop {real_id}: unreachable: {e}")))?;
        let status = r.status();
        if !ok(status) {
            let text = r.text().await.unwrap_or_default();
            return Err(PoolError::Device(format!(
                "stop {real_id} failed: {status} {}",
                truncate(&text, 200)
            )));
        }
        Ok(())
    }
}

pub struct DeviceModelManager {
    backend: Arc<dyn DeviceBackend>,
    retry_after: f64,
    entries: StdMutex<HashMap<String, Entry>>,
    load_lock: AsyncMutex<()>,
    clock: AtomicU64,
}

impl DeviceModelManager {
    /// Production constructor: Python default (`retry_after=5.0`).
    pub fn new(backend: Arc<dyn DeviceBackend>) -> Self {
        Self::with_retry_after(backend, 5.0)
    }

    /// Test/tunable constructor.
    pub fn with_retry_after(backend: Arc<dyn DeviceBackend>, retry_after: f64) -> Self {
        DeviceModelManager {
            backend,
            retry_after,
            entries: StdMutex::new(HashMap::new()),
            load_lock: AsyncMutex::new(()),
            clock: AtomicU64::new(0),
        }
    }

    /// Monotonically-increasing tick, used only for `last_used` ordering.
    fn tick(&self) -> f64 {
        self.clock.fetch_add(1, Ordering::SeqCst) as f64
    }

    // --- per-model counters (sync; called by the Gate, incl. from `Slot::Drop`) ---

    pub fn acquire(&self, real_id: &str) {
        let tick = self.tick();
        let mut entries = self.entries.lock().unwrap();
        entries.entry(real_id.to_string()).or_insert_with(|| Entry::new(tick)).in_flight += 1;
    }

    pub fn release(&self, real_id: &str) {
        let tick = self.tick();
        let mut entries = self.entries.lock().unwrap();
        if let Some(e) = entries.get_mut(real_id) {
            if e.in_flight > 0 {
                e.in_flight -= 1;
            }
            e.last_used = tick;
        }
    }

    pub fn enqueue(&self, real_id: &str) {
        let tick = self.tick();
        let mut entries = self.entries.lock().unwrap();
        entries.entry(real_id.to_string()).or_insert_with(|| Entry::new(tick)).queued += 1;
    }

    pub fn dequeue(&self, real_id: &str) {
        let mut entries = self.entries.lock().unwrap();
        if let Some(e) = entries.get_mut(real_id) {
            if e.queued > 0 {
                e.queued -= 1;
            }
        }
    }

    /// Atomic queued→in-flight handoff: decrement `queued` (saturating, matching
    /// `dequeue`) and increment `in_flight` (matching `acquire`) under ONE `entries`
    /// lock, instead of `dequeue(...)` then `acquire(...)` as two separate critical
    /// sections.
    ///
    /// This closes a real race on the multi-threaded tokio runtime: with two
    /// separate locks, a concurrent evictor (another `ensure_loaded` running
    /// `pick_eviction`, which also reads `entries` under its own short lock) can
    /// observe the gap between them — `queued == 0 && in_flight == 0 && loaded ==
    /// true` — and pick this model as the idle LRU victim, stopping it on the
    /// device while this request is mid-handoff. Python's equivalent is safe only
    /// because asyncio coroutines never preempt each other outside an `await`; Rust
    /// under a real executor has no such guarantee, so the two updates must land as
    /// one indivisible step. Used by `Gate::enter`.
    ///
    /// Mirrors `acquire`: does NOT stamp `last_used` for an already-existing entry
    /// (only a freshly-inserted entry gets `last_used = tick` via `Entry::new`).
    /// Never holds the lock across an `.await` (there isn't one here).
    pub fn dequeue_acquire(&self, real_id: &str) {
        let tick = self.tick();
        let mut entries = self.entries.lock().unwrap();
        let e = entries.entry(real_id.to_string()).or_insert_with(|| Entry::new(tick));
        if e.queued > 0 {
            e.queued -= 1;
        }
        e.in_flight += 1;
    }

    pub fn queued(&self, real_id: &str) -> u32 {
        let entries = self.entries.lock().unwrap();
        entries.get(real_id).map(|e| e.queued).unwrap_or(0)
    }

    /// In-flight count for `real_id`. Not required by the Task 5 interface, but a
    /// natural counterpart to `queued()` — useful diagnostics for tests (used to
    /// verify the eviction-race fix doesn't lose a racer's in-flight bump) and for
    /// callers that want it.
    pub fn in_flight(&self, real_id: &str) -> u32 {
        let entries = self.entries.lock().unwrap();
        entries.get(real_id).map(|e| e.in_flight).unwrap_or(0)
    }

    /// Loaded model ids, most-recently-used first.
    pub fn snapshot(&self) -> Vec<String> {
        let entries = self.entries.lock().unwrap();
        let mut loaded: Vec<(&String, &Entry)> = entries.iter().filter(|(_, e)| e.loaded).collect();
        loaded.sort_by(|a, b| b.1.last_used.partial_cmp(&a.1.last_used).unwrap());
        loaded.into_iter().map(|(rid, _)| rid.clone()).collect()
    }

    // --- load / evict --------------------------------------------------------

    /// Ensure `real_id` is loaded on the device, evicting an idle LRU model first
    /// if `pool_max` is reached. Concurrent calls for the SAME id dedup to exactly
    /// one device `start` (serialized on `load_lock`, re-checked after acquiring
    /// it). Never evicts a model with `in_flight > 0` or `queued > 0`.
    pub async fn ensure_loaded(&self, real_id: &str, pool_max: Option<u32>) -> Result<(), PoolError> {
        if self.mark_used_if_loaded(real_id) {
            return Ok(());
        }
        let _guard = self.load_lock.lock().await;
        if self.mark_used_if_loaded(real_id) {
            return Ok(());
        }

        let running = self.backend.list_loaded().await?;
        self.reconcile(&running);
        if running.contains(real_id) {
            self.mark_loaded(real_id);
            return Ok(());
        }

        let (loaded_ids, pool_entries) = {
            let entries = self.entries.lock().unwrap();
            let loaded_ids: HashSet<String> =
                entries.iter().filter(|(_, e)| e.loaded).map(|(rid, _)| rid.clone()).collect();
            let pool_entries: Vec<PoolEntry> = entries
                .iter()
                .filter(|(_, e)| e.loaded)
                .map(|(rid, e)| PoolEntry {
                    model_id: rid.clone(),
                    in_flight: e.in_flight,
                    queued: e.queued,
                    last_used: e.last_used,
                })
                .collect();
            (loaded_ids, pool_entries)
        };

        if resolver::needs_eviction(&loaded_ids, real_id, pool_max) {
            let victim = resolver::pick_eviction(&pool_entries)
                .ok_or(PoolError::Backpressure(self.retry_after))?;

            // Eviction-race fix (ported from pool.py): flip the victim's `loaded`
            // flag off SYNCHRONOUSLY, before the `.await` on `_stop`, so a
            // concurrent `ensure_loaded(victim)` racer can no longer take the
            // pre-lock fast path on stale "still loaded" truth while we're
            // mid-teardown — it must block on `load_lock` (which we hold) and
            // re-check for real once we're done.
            {
                let mut entries = self.entries.lock().unwrap();
                if let Some(e) = entries.get_mut(&victim) {
                    e.loaded = false;
                }
            }
            self.backend.unload(&victim).await?;
            // Only discard the victim's bookkeeping if nothing referenced it
            // while the stop was in flight (a racer's enqueue()/acquire() land
            // directly on the entry, unguarded by load_lock). Never silently
            // drop a nonzero in_flight/queued count — leave the entry in place
            // (unloaded) so the next ensure_loaded reloads it and the racer's
            // counters survive.
            {
                let mut entries = self.entries.lock().unwrap();
                let idle = entries.get(&victim).map(|e| e.in_flight == 0 && e.queued == 0).unwrap_or(false);
                if idle {
                    entries.remove(&victim);
                }
            }
        }

        self.backend.load(real_id).await?;
        self.mark_loaded(real_id);
        Ok(())
    }

    /// Fast path: if `real_id` is already loaded, bump `last_used` and report done.
    fn mark_used_if_loaded(&self, real_id: &str) -> bool {
        let tick = self.tick();
        let mut entries = self.entries.lock().unwrap();
        if let Some(e) = entries.get_mut(real_id) {
            if e.loaded {
                e.last_used = tick;
                return true;
            }
        }
        false
    }

    fn mark_loaded(&self, real_id: &str) {
        let tick = self.tick();
        let mut entries = self.entries.lock().unwrap();
        let e = entries.entry(real_id.to_string()).or_insert_with(|| Entry::new(tick));
        e.loaded = true;
        e.last_used = tick;
    }

    /// Fold device truth into our state: mark loaded what the device runs; clear the
    /// loaded flag on anything the device dropped from under us (keep counters — a
    /// request may still be accounted against it).
    fn reconcile(&self, running: &HashSet<String>) {
        let tick = self.tick();
        let mut entries = self.entries.lock().unwrap();
        for rid in running {
            entries.entry(rid.clone()).or_insert_with(|| Entry::new(tick)).loaded = true;
        }
        for (rid, e) in entries.iter_mut() {
            if !running.contains(rid) {
                e.loaded = false;
            }
        }
    }

}

// --- Gate / Slot ----------------------------------------------------------------
//
// The request-queueing layer on top of `DeviceModelManager` — a direct port of
// `woollama.pool.Gate`/`Slot` (Python oracle). `Gate::enter` is the full gating
// protocol for one request: reject early if the per-model queue is already
// saturated; otherwise register a queue slot (which also protects the model from
// eviction — `ensure_loaded`'s eviction pass reads `queued`/`in_flight` via the
// manager's sync counters), ensure the model is loaded, then acquire a concurrency
// permit within `queue_timeout` and bump the in-flight ref-count. `queue_timeout`
// bounds ONLY the permit acquisition, not the preceding `ensure_loaded` await —
// time spent waiting on an in-progress load (serialized on the manager's
// `load_lock`, which can poll up to `load_timeout`, default 120s) is governed
// separately by `load_timeout` (mirrors the Python docstring on `Gate.enter`).

/// One per-model concurrency permit, held for the lifetime of a pooled request.
/// `Drop` releases both halves synchronously: the manager's in-flight counter
/// (`DeviceModelManager::release`, sync — see the Task 5 lock-model note at the
/// top of this module) and the semaphore permit (`OwnedSemaphorePermit`'s own
/// `Drop`). Because `Drop` runs at most once per value, release is idempotent by
/// construction — no Python-style `_released` guard flag is needed.
pub struct Slot {
    manager: Arc<DeviceModelManager>,
    real_id: String,
    // Always `Some` until `Drop` — held as an `Option` only so `Drop::drop` (which
    // takes `&mut self`, not `self`) can take it out; never actually `None` while
    // the `Slot` is alive.
    permit: Option<OwnedSemaphorePermit>,
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.manager.release(&self.real_id);
        // Dropping the permit (if still held) releases the semaphore synchronously.
        drop(self.permit.take());
    }
}

/// Serializes/queues requests for one management-capable inferencer's models around
/// a shared `DeviceModelManager`. One `Gate` per inferencer (see `PoolRegistry`).
pub struct Gate {
    manager: Arc<DeviceModelManager>,
    parallel: usize,
    queue_max: Option<u32>,
    queue_timeout: f64,
    pool_max: Option<u32>,
    retry_after: f64,
    sems: StdMutex<HashMap<String, Arc<Semaphore>>>,
}

impl Gate {
    pub fn new(
        manager: Arc<DeviceModelManager>,
        parallel: u32,
        queue_max: Option<u32>,
        queue_timeout: f64,
        pool_max: Option<u32>,
        retry_after: f64,
    ) -> Self {
        Gate {
            manager,
            parallel: parallel.max(1) as usize,
            queue_max,
            queue_timeout,
            pool_max,
            retry_after,
            sems: StdMutex::new(HashMap::new()),
        }
    }

    /// The per-`real_id` semaphore, created lazily with `parallel` permits (mirrors
    /// Python's `_sems` dict-of-`asyncio.Semaphore`, lazily populated the same way).
    fn sem(&self, real_id: &str) -> Arc<Semaphore> {
        let mut sems = self.sems.lock().unwrap();
        sems.entry(real_id.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(self.parallel)))
            .clone()
    }

    pub async fn enter(&self, real_id: &str) -> Result<Slot, PoolError> {
        if let Some(queue_max) = self.queue_max {
            if self.manager.queued(real_id) >= queue_max {
                return Err(PoolError::Backpressure(self.retry_after));
            }
        }
        self.manager.enqueue(real_id);

        // Python's `try/finally`: `dequeue` must run whether `ensure_loaded`/the
        // semaphore acquire succeeds or fails. The `async` block plays the role of
        // the `try` body; `dequeue` below plays `finally`.
        let outcome: Result<OwnedSemaphorePermit, PoolError> = async {
            self.manager.ensure_loaded(real_id, self.pool_max).await?;
            let sem = self.sem(real_id);
            match tokio::time::timeout(Duration::from_secs_f64(self.queue_timeout), sem.acquire_owned()).await {
                Ok(Ok(permit)) => Ok(permit),
                // The semaphore is never `close()`d, so this arm is unreachable in
                // practice; map it to a Device error rather than panicking/unwrapping.
                Ok(Err(_)) => Err(PoolError::Device("semaphore closed unexpectedly".to_string())),
                Err(_) => Err(PoolError::Backpressure(self.retry_after)),
            }
        }
        .await;

        // Python's `finally: dequeue()` runs unconditionally; but on the success
        // path the queued→in-flight handoff must be ATOMIC (see
        // `DeviceModelManager::dequeue_acquire`), not `dequeue` then `acquire` as
        // two separate critical sections — a concurrent evictor's `pick_eviction`
        // could otherwise observe this model as idle (queued == 0, in_flight == 0)
        // in the gap between them and stop it mid-handoff. No `.await` on either
        // branch (both sync).
        match outcome {
            Ok(permit) => {
                self.manager.dequeue_acquire(real_id);
                Ok(Slot { manager: self.manager.clone(), real_id: real_id.to_string(), permit: Some(permit) })
            }
            Err(e) => {
                self.manager.dequeue(real_id);
                Err(e)
            }
        }
    }
}

// --- PoolRegistry -----------------------------------------------------------------

/// One `(DeviceModelManager, Gate)` pair per management-capable inferencer, keyed by
/// inferencer name. Built once at startup from the config `Registry` (see
/// `from_registry`); consulted by the chat passthrough to take the pooled path.
pub struct PoolRegistry(HashMap<String, (Arc<DeviceModelManager>, Gate)>);

impl PoolRegistry {
    pub fn get(&self, provider: &str) -> Option<&(Arc<DeviceModelManager>, Gate)> {
        self.0.get(provider)
    }

    /// One manager+gate per inferencer that declares a `management_url` (mirrors the
    /// Python lifespan's `_pools` construction loop). Auth headers are best-effort —
    /// an inferencer with a required-but-unset API key env still gets a pool (with no
    /// auth headers against its management API), matching Python's `except
    /// InferencerError: _hdrs = {}`.
    pub fn from_registry(registry: &engine::Registry) -> PoolRegistry {
        let mut map = HashMap::new();
        for inf in registry.list() {
            let Some(management_url) = inf.management_url.clone() else { continue };
            let headers = inf.auth_headers().unwrap_or_default();
            let manager = Arc::new(DeviceModelManager::new(Arc::new(RestBackend::tiiny(management_url, headers, 0.5, 120.0))));
            let gate = Gate::new(manager.clone(), inf.parallel, inf.queue_max, inf.queue_timeout, inf.pool_max, 5.0);
            map.insert(inf.name.clone(), (manager, gate));
        }
        PoolRegistry(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `from_registry` should build a pool ONLY for inferencers that declare a
    /// `management_url`; a plain (non-device) inferencer is skipped entirely.
    #[test]
    fn from_registry_skips_inferencers_without_management_url() {
        let mut reg = engine::Registry::new();
        reg.insert(engine::Inferencer {
            name: "device".to_string(),
            base_url: "http://device.example/v1".to_string(),
            api_key_env: None,
            extra_body: serde_json::json!({}),
            models: Vec::new(),
            discover: false,
            model_patterns: Vec::new(),
            management_url: Some("http://device.example:8800".to_string()),
            management_protocol: None,
            parallel: 1,
            pool_max: None,
            queue_max: None,
            queue_timeout: 30.0,
            virtual_models: Default::default(),
        });
        reg.add("cloud".to_string(), "http://cloud.example/v1".to_string(), None, serde_json::json!({}));

        let pools = PoolRegistry::from_registry(&reg);
        assert!(pools.get("cloud").is_none(), "no management_url => no pool");
        assert!(pools.get("device").is_some(), "management_url present => pool built");
    }
}
