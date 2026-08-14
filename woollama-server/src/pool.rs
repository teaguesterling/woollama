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
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;
use tokio::time::Instant;

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

pub struct DeviceModelManager {
    url: String,
    headers: HashMap<String, String>,
    client: reqwest::Client,
    poll_interval: f64,
    load_timeout: f64,
    retry_after: f64,
    entries: StdMutex<HashMap<String, Entry>>,
    load_lock: AsyncMutex<()>,
    clock: AtomicU64,
}

impl DeviceModelManager {
    /// Production constructor: Python defaults (`poll_interval=0.5`,
    /// `load_timeout=120.0`, `retry_after=5.0`).
    pub fn new(management_url: String, headers: HashMap<String, String>) -> Self {
        Self::with_config(management_url, headers, 0.5, 120.0, 5.0)
    }

    /// Test/tunable constructor.
    pub fn with_config(
        management_url: String,
        headers: HashMap<String, String>,
        poll_interval: f64,
        load_timeout: f64,
        retry_after: f64,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs_f64(30.0))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        DeviceModelManager {
            url: management_url.trim_end_matches('/').to_string(),
            headers,
            client,
            poll_interval,
            load_timeout,
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

        let running = self.running().await?;
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
            self.stop(&victim).await?;
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

        self.start(real_id).await?;
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

    // --- device I/O ------------------------------------------------------------

    fn apply_headers(&self, mut rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        for (k, v) in &self.headers {
            rb = rb.header(k.as_str(), v.as_str());
        }
        rb
    }

    async fn running(&self) -> Result<HashSet<String>, PoolError> {
        let rb = self.apply_headers(self.client.get(format!("{}/api/v1/models/running", self.url)));
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

    async fn start(&self, real_id: &str) -> Result<(), PoolError> {
        let path = format!("{}/api/v1/models/{real_id}/start", self.url);
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
            if self.running().await?.contains(real_id) {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_secs_f64(self.poll_interval)).await;
        }
        Err(PoolError::Device(format!("start {real_id}: not running after {}s", self.load_timeout)))
    }

    async fn stop(&self, real_id: &str) -> Result<(), PoolError> {
        let path = format!("{}/api/v1/models/{real_id}/stop", self.url);
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
