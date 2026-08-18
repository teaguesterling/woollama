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
/// implementation for the built-in device-management REST shape (`{url}/api/v1/models/...`);
/// later tasks add config-defined REST protocols and an Ollama adapter behind this same seam.
#[async_trait::async_trait]
pub trait DeviceBackend: Send + Sync {
    async fn list_loaded(&self) -> Result<HashSet<String>, PoolError>;

    /// Loaded ids PLUS whatever the backend says each one can do, in ONE call.
    ///
    /// Default: ids only, no capabilities — a backend that publishes nothing is simply "unknown"
    /// everywhere, which callers must treat as eligible. Backends that do publish override this
    /// so discovery costs no extra round trip.
    async fn list_loaded_detailed(&self) -> Result<(HashSet<String>, ModelCapabilities), PoolError> {
        Ok((self.list_loaded().await?, ModelCapabilities::new()))
    }
    async fn load(&self, id: &str) -> Result<(), PoolError>;
    async fn unload(&self, id: &str) -> Result<(), PoolError>;
}

/// One HTTP call (`running`/`start`/`stop`) fully resolved against a base URL and
/// default headers, but still carrying an `{id}` placeholder in `url`/`body`/header
/// values — substituted per-call by `render` (there's no id yet at construction time
/// for `start`/`stop`, and never one for `running`).
struct CompiledEndpoint {
    method: reqwest::Method,
    url: String,
    body: Option<String>,
    headers: HashMap<String, String>,
}

impl CompiledEndpoint {
    /// Substitute `{id}` (when `id` is given) into `url`/`body`/every header value,
    /// and hand back everything a request needs. `running` calls this with `None`
    /// (no id in scope); `start`/`unload` with `Some(real_id)`.
    fn render(&self, id: Option<&str>) -> (reqwest::Method, String, Option<String>, HashMap<String, String>) {
        let sub = |s: &str| -> String {
            match id {
                Some(id) => s.replace("{id}", id),
                None => s.to_string(),
            }
        };
        let url = sub(&self.url);
        let body = self.body.as_deref().map(sub);
        let headers = self.headers.iter().map(|(k, v)| (k.clone(), sub(v))).collect();
        (self.method.clone(), url, body, headers)
    }
}

/// Method parsed case-insensitively from an optional config override, falling back to
/// `default` when unset OR unparseable (an invalid method string in config shouldn't
/// panic startup; it just loses the override — logged so the typo isn't silently
/// invisible).
fn resolve_method(configured: &Option<String>, default: reqwest::Method) -> reqwest::Method {
    match configured.as_deref() {
        None => default,
        Some(m) => reqwest::Method::from_bytes(m.to_ascii_uppercase().as_bytes()).unwrap_or_else(|_| {
            eprintln!("woollamad: management_protocols: invalid HTTP method '{m}', falling back to {default}");
            default
        }),
    }
}

/// Build a `CompiledEndpoint` from a config `EndpointSpec`: substitute `{base}` into
/// `url`/`body`/header values now (known at construction time), merge `spec.headers`
/// OVER `default_headers` (an endpoint header key overrides the shared Bearer auth —
/// keyed CASE-INSENSITIVELY, since HTTP header names are; both maps are folded to
/// lowercase keys so e.g. an endpoint header keyed `authorization` overrides a default
/// `Authorization`, never sending both), and default `content-type: application/json`
/// when a `body` is present and no `content-type` header was set either way.
/// Lowercase keys are also what actually go out on the wire (reqwest is fine with
/// that; header names are case-insensitive there too).
fn compile_endpoint(
    base: &str,
    default_headers: &HashMap<String, String>,
    spec: &engine::EndpointSpec,
    default_method: reqwest::Method,
) -> CompiledEndpoint {
    let sub_base = |s: &str| s.replace("{base}", base);
    let method = resolve_method(&spec.method, default_method);
    let url = sub_base(&spec.url);
    let body = spec.body.as_deref().map(sub_base);
    let mut headers: HashMap<String, String> = HashMap::new();
    for (k, v) in default_headers {
        headers.insert(k.to_ascii_lowercase(), v.clone());
    }
    for (k, v) in &spec.headers {
        headers.insert(k.to_ascii_lowercase(), sub_base(v));
    }
    if body.is_some() && !headers.contains_key("content-type") {
        headers.insert("content-type".to_string(), "application/json".to_string());
    }
    CompiledEndpoint { method, url, body, headers }
}

/// Dotted-path lookup into a JSON value (`""` => the value itself; `"a.b"` =>
/// `v["a"]["b"]`) — how `RestBackend::list_loaded` finds the running-models
/// array/object inside an arbitrary config-defined response shape.
fn get_dotted<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return Some(v);
    }
    let mut cur = v;
    for part in path.split('.') {
        cur = cur.get(part)?;
    }
    Some(cur)
}

/// The `DeviceBackend` for config-defined (and the built-in device-management REST
/// shape's) REST shapes: three HTTP calls (list-loaded/start/stop), each independently
/// templated (`RestBackend::from_spec`). `RestBackend::device` is the built-in `device`
/// preset, expressed as a `from_spec` call with its built-in endpoints.
pub struct RestBackend {
    client: reqwest::Client,
    running: CompiledEndpoint,
    start: CompiledEndpoint,
    stop: CompiledEndpoint,
    /// Dotted path (within the `running` response) to the array of loaded models;
    /// `None`/absent means the response body itself is that array.
    running_path: Option<String>,
    /// When the running array holds objects rather than bare id strings, the field
    /// to pluck from each element.
    running_id_field: Option<String>,
    /// Optional sibling array of per-model detail objects in the SAME running response, and the
    /// fields within it carrying the model id and its capability list. Vendor-specific by nature,
    /// so it is a per-backend detail rather than an assumption: absent ⇒ no discovery, which means
    /// "unknown" and therefore no behaviour change.
    detail: Option<DetailSpec>,
    poll_interval: f64,
    load_timeout: f64,
}

/// Where a backend publishes per-model capabilities inside its running response.
#[derive(Clone)]
pub struct DetailSpec {
    /// Dotted path to the array of detail objects.
    pub path: String,
    /// Field on each object holding the model id.
    pub id_field: String,
    /// Field holding the capability list.
    pub capabilities_field: String,
}

impl RestBackend {
    /// Build a `RestBackend` from a config `ProtocolSpec::Rest`'s three endpoints.
    /// `{base}` (→ `base_url` trimmed of a trailing `/`) is substituted into every
    /// `url`/`body`/header value now; `{id}` is substituted per-call (see
    /// `CompiledEndpoint::render`). Per-op method defaults: GET for `running`, POST
    /// for `start`/`stop` (an explicit `method` on the spec overrides).
    pub fn from_spec(
        base_url: &str,
        default_headers: &HashMap<String, String>,
        running: &engine::EndpointSpec,
        start: &engine::EndpointSpec,
        stop: &engine::EndpointSpec,
        poll_interval: f64,
        load_timeout: f64,
    ) -> RestBackend {
        let base = base_url.trim_end_matches('/');
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs_f64(30.0))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        RestBackend {
            client,
            running: compile_endpoint(base, default_headers, running, reqwest::Method::GET),
            start: compile_endpoint(base, default_headers, start, reqwest::Method::POST),
            stop: compile_endpoint(base, default_headers, stop, reqwest::Method::POST),
            running_path: running.path.clone(),
            running_id_field: running.id_field.clone(),
            // Config-declared REST protocols get no discovery: a `[management_protocols.*]` block
            // describes endpoints, not payload semantics, and guessing where a stranger's backend
            // puts capability data is exactly the inference this design refuses to make.
            detail: None,
            poll_interval,
            load_timeout,
        }
    }

    /// Declare where this backend publishes per-model capabilities within its running response.
    pub fn with_detail(mut self, detail: DetailSpec) -> RestBackend {
        self.detail = Some(detail);
        self
    }

    /// The built-in device-management REST shape (`GET {base}/api/v1/models/running`,
    /// `POST .../{id}/start`, `POST .../{id}/stop`), expressed as a `from_spec` call
    /// with its built-in endpoints — the preset every `management_protocol`
    /// resolution falls back to when an inferencer names none (or names `"device"`
    /// explicitly).
    pub fn device(management_url: String, headers: HashMap<String, String>, poll_interval: f64, load_timeout: f64) -> RestBackend {
        let no_headers = std::collections::BTreeMap::new();
        let running = engine::EndpointSpec {
            url: "{base}/api/v1/models/running".to_string(),
            method: None,
            body: None,
            headers: no_headers.clone(),
            path: Some("running".to_string()),
            id_field: None,
        };
        let start = engine::EndpointSpec {
            url: "{base}/api/v1/models/{id}/start".to_string(),
            method: None,
            body: None,
            headers: no_headers.clone(),
            path: None,
            id_field: None,
        };
        let stop = engine::EndpointSpec {
            url: "{base}/api/v1/models/{id}/stop".to_string(),
            method: None,
            body: None,
            headers: no_headers,
            path: None,
            id_field: None,
        };
        // The device publishes per-model capabilities in `instances.running[]`, a SIBLING of the
        // bare-string `running` array in the same response — so discovery costs no extra call.
        // Verified against a live device payload rather than inferred.
        RestBackend::from_spec(&management_url, &headers, &running, &start, &stop, poll_interval, load_timeout)
            .with_detail(DetailSpec {
                path: "instances.running".to_string(),
                id_field: "model_id".to_string(),
                capabilities_field: "capabilities".to_string(),
            })
    }

    /// Issue one templated call: apply `endpoint.render(id)`'s method/url/body/headers
    /// Fetch and parse the running response body once.
    async fn running_body(&self) -> Result<Value, PoolError> {
        let (status, r) = self
            .call(&self.running, None)
            .await
            .map_err(|e| PoolError::Device(format!("device unreachable: {e}")))?;
        if !ok(status) {
            let text = r.text().await.unwrap_or_default();
            return Err(PoolError::Device(format!("running query failed: {status} {}", truncate(&text, 200))));
        }
        r.json().await.map_err(|e| PoolError::Device(format!("running query: bad JSON: {e}")))
    }

    /// The loaded-model ids from a running response.
    ///
    /// `get_dotted` returning `None` (key/path absent) is normal and means "no running models" —
    /// the back-compat case where a device response has no `running` key at all must still resolve
    /// to an empty set, not an error. But a path that IS present and resolves to something other
    /// than an array is a config-typo signal, and gets a loud error naming the path and what it
    /// found, rather than silently becoming "no models" and surfacing later as a much more
    /// confusing load-timeout "not running".
    fn ids_from(&self, v: &Value) -> Result<HashSet<String>, PoolError> {
        let path = self.running_path.as_deref().unwrap_or("");
        let arr = match get_dotted(v, path) {
            None => Vec::new(),
            Some(Value::Array(items)) => items.clone(),
            Some(other) => {
                return Err(PoolError::Device(format!(
                    "running query: path '{path}' is present but not an array: {}",
                    truncate(&other.to_string(), 200)
                )));
            }
        };
        Ok(arr
            .iter()
            .filter_map(|item| match &self.running_id_field {
                Some(field) => item.get(field).and_then(Value::as_str).map(String::from),
                None => item.as_str().map(String::from),
            })
            .collect())
    }

    /// to `self.client`, send it, and hand back the parsed status + body text/bytes.
    async fn call(&self, endpoint: &CompiledEndpoint, id: Option<&str>) -> Result<(reqwest::StatusCode, reqwest::Response), reqwest::Error> {
        let (method, url, body, headers) = endpoint.render(id);
        let mut rb = self.client.request(method, url);
        for (k, v) in &headers {
            rb = rb.header(k.as_str(), v.as_str());
        }
        if let Some(b) = body {
            rb = rb.body(b);
        }
        let r = rb.send().await?;
        Ok((r.status(), r))
    }
}

#[async_trait::async_trait]
impl DeviceBackend for RestBackend {
    async fn list_loaded(&self) -> Result<HashSet<String>, PoolError> {
        let v = self.running_body().await?;
        self.ids_from(&v)
    }

    /// One fetch, both views: the bare-string id array the pool has always used, plus the sibling
    /// detail array when this backend declares one.
    async fn list_loaded_detailed(&self) -> Result<(HashSet<String>, ModelCapabilities), PoolError> {
        let v = self.running_body().await?;
        let running = self.ids_from(&v)?;
        let mut caps = ModelCapabilities::new();
        if let Some(d) = &self.detail {
            if let Some(Value::Array(items)) = get_dotted(&v, &d.path) {
                for item in items {
                    let Some(id) = item.get(&d.id_field).and_then(Value::as_str) else { continue };
                    let Some(Value::Array(list)) = item.get(&d.capabilities_field) else { continue };
                    let tokens: Vec<String> =
                        list.iter().filter_map(|c| c.as_str().map(str::to_lowercase)).collect();
                    if !tokens.is_empty() {
                        caps.insert(id.to_string(), tokens);
                    }
                }
            }
        }
        Ok((running, caps))
    }

    async fn load(&self, real_id: &str) -> Result<(), PoolError> {
        let (status, r) = self
            .call(&self.start, Some(real_id))
            .await
            .map_err(|e| PoolError::Device(format!("start {real_id}: unreachable: {e}")))?;
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
        let (status, r) = self
            .call(&self.stop, Some(real_id))
            .await
            .map_err(|e| PoolError::Device(format!("stop {real_id}: unreachable: {e}")))?;
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

/// The `DeviceBackend` for Ollama's native management API: `GET {base}/api/ps` lists
/// loaded models, `POST {base}/api/generate` both loads (empty-prompt warm-up) and
/// unloads (`keep_alive: 0`) a model — there is no separate start/stop endpoint pair
/// the way `RestBackend` has. `keep_alive` (when set) is forwarded on `load` so a
/// config author can control how long Ollama keeps a model resident after use; when
/// unset, the field is omitted entirely so Ollama's own default applies.
///
/// Unlike `RestBackend` (which sends `default_headers`, e.g. a Bearer token derived
/// from `api_key_env`), `OllamaBackend` sends NO auth headers at all — stock Ollama's
/// HTTP API is unauthenticated, so there is nothing to attach.
///
/// Latent accuracy note: `/api/ps` can return a name-NORMALIZED id (e.g. Ollama may
/// report `qwen3:latest` for a model this backend posted to `/api/generate` as plain
/// `qwen3`). `list_loaded`/`reconcile` compare ids by exact string match, so such a
/// normalization would make `DeviceModelManager` think the model it just loaded is
/// NOT running (and vice versa on eviction bookkeeping) — a reconcile/eviction-accuracy
/// gap, not currently handled here.
pub struct OllamaBackend {
    client: reqwest::Client,
    base_url: String,
    keep_alive: Option<String>,
}

impl OllamaBackend {
    /// Build an `OllamaBackend` against `base_url` (trailing `/` trimmed). `keep_alive`
    /// is `None` for the built-in `"ollama"` name (Ollama's default applies) or
    /// `Some(...)` when resolved from a config `ProtocolSpec::Ollama { keep_alive }`.
    pub fn new(base_url: String, keep_alive: Option<String>) -> OllamaBackend {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs_f64(30.0))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        OllamaBackend { client, base_url: base_url.trim_end_matches('/').to_string(), keep_alive }
    }

    /// `POST {base}/api/generate` with `body`, mapped to the shared `PoolError::Device`
    /// error shape (transport failure or non-2xx status), same message style as
    /// `RestBackend`.
    async fn generate(&self, real_id: &str, body: Value) -> Result<(), PoolError> {
        let url = format!("{}/api/generate", self.base_url);
        let r = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|e| PoolError::Device(format!("generate {real_id}: unreachable: {e}")))?;
        let status = r.status();
        if !ok(status) {
            let text = r.text().await.unwrap_or_default();
            return Err(PoolError::Device(format!(
                "generate {real_id} failed: {status} {}",
                truncate(&text, 200)
            )));
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl DeviceBackend for OllamaBackend {
    async fn list_loaded(&self) -> Result<HashSet<String>, PoolError> {
        let url = format!("{}/api/ps", self.base_url);
        let r = self
            .client
            .get(url)
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
            .get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| item.get("name").and_then(Value::as_str).map(String::from))
            .collect();
        Ok(running)
    }

    // No poll-for-readiness loop here (unlike `RestBackend::load`, which starts the
    // device then polls `list_loaded` until `load_timeout`): Ollama's `/api/generate`
    // itself blocks until the model is fully resident before it returns a 2xx, so by
    // the time `generate` below succeeds, the model is already loaded — polling would
    // be redundant.
    async fn load(&self, real_id: &str) -> Result<(), PoolError> {
        let mut body = serde_json::json!({ "model": real_id });
        if let Some(keep_alive) = &self.keep_alive {
            body["keep_alive"] = Value::String(keep_alive.clone());
        }
        self.generate(real_id, body).await
    }

    async fn unload(&self, real_id: &str) -> Result<(), PoolError> {
        let body = serde_json::json!({ "model": real_id, "keep_alive": 0 });
        self.generate(real_id, body).await
    }
}

/// `model id -> the capability tokens the backend reports for it`, e.g. `["embedding"]`.
/// Absent means the backend said nothing, NOT that the model can do nothing.
pub type ModelCapabilities = HashMap<String, Vec<String>>;

/// What the device is running, plus whether we could actually see it.
///
/// The distinction is load-bearing for anything reasoning about an EMPTY set. "The query succeeded
/// and nothing matched" and "the query failed" produce the same empty vec but justify opposite
/// advice — the first means the operator's expectations are wrong, the second means we are blind.
/// Conflating them yields a confident instruction derived from missing information, which
/// misdirects rather than confuses.
pub struct Residency {
    pub models: Vec<String>,
    /// What the backend says each resident can do, when it says anything at all (issue #20).
    pub capabilities: ModelCapabilities,
    /// False when the most recent device query failed. `models` is then a last-known view, and an
    /// empty one means "we could not see", NOT "nothing is loaded".
    pub current: bool,
}

pub struct DeviceModelManager {
    backend: Arc<dyn DeviceBackend>,
    retry_after: f64,
    entries: StdMutex<HashMap<String, Entry>>,
    load_lock: AsyncMutex<()>,
    clock: AtomicU64,
    /// When the device was last asked what it is running. Wall-clock, NOT `tick()` — `tick` is a
    /// logical counter for LRU ordering and has no time units; comparing it against a duration
    /// silently disables coalescing (it did, until a test caught 5 queries where 1 was expected).
    /// Purely a coalescing window for read-through (see `residency`) — never a claim that our view
    /// is authoritative between reads.
    residency_read_at: StdMutex<Option<std::time::Instant>>,
    /// Coalescing window for `residency`, in seconds; `0` reads every time. Captured at
    /// construction rather than read per call — an env read on the request path is both wasteful
    /// and, because the var is process-global, a race against any concurrently-running test.
    residency_ttl: f64,
    /// Last-seen per-model capabilities from the backend. Cached alongside residency because it
    /// arrives in the same payload and changes for the same reasons.
    discovered_caps: StdMutex<ModelCapabilities>,
    /// Models the backend has just failed a request for. The next `ensure_loaded` reloads them
    /// unconditionally instead of trusting either our view or the backend's (see
    /// `mark_needs_reload`).
    needs_reload: StdMutex<std::collections::HashSet<String>>,
    /// Single-flights concurrent read-throughs. Deliberately NOT `load_lock`: residency reads sit
    /// on the request path, and sharing that lock would make every `default` request queue behind
    /// an in-progress model load.
    residency_lock: AsyncMutex<()>,
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
            residency_read_at: StdMutex::new(None),
            residency_ttl: std::env::var("WOOLLAMA_POOL_RESIDENCY_TTL_MS")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .map(|ms| ms / 1000.0)
                .unwrap_or(1.0),
            discovered_caps: StdMutex::new(ModelCapabilities::new()),
            needs_reload: StdMutex::new(std::collections::HashSet::new()),
            residency_lock: AsyncMutex::new(()),
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
        // A model the backend just failed for is reloaded unconditionally: neither our view nor
        // the backend's can be trusted here, since the backend keeps listing a crashed instance
        // for seconds (see `mark_needs_reload`). Taken ONCE — a load that fails surfaces its error
        // rather than looping, so a model that genuinely cannot load fails cleanly.
        let forced = self.needs_reload.lock().unwrap().remove(real_id);
        if !forced && self.mark_used_if_loaded(real_id) {
            return Ok(());
        }
        let _guard = self.load_lock.lock().await;
        if !forced && self.mark_used_if_loaded(real_id) {
            return Ok(());
        }

        let running = self.backend.list_loaded().await?;
        self.reconcile(&running);
        if !forced && running.contains(real_id) {
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

    /// What the DEVICE is running, read through to the device itself.
    ///
    /// woollama is not the only consumer of a device's management API — the vendor's own UI loads
    /// models, and any caller needing endpoints woollama doesn't serve (images, embeddings, ASR)
    /// must drive the device directly. Exclusivity is therefore not achievable even in principle,
    /// so our `loaded` flags are a **cache of someone else's state**, never a ledger. Anything
    /// that would be expensive to get wrong reads through to the device and lets it arbitrate.
    ///
    /// The freshness window is a *coalescing* window, not a staleness budget: a burst of requests
    /// shares one query instead of issuing one each. `WOOLLAMA_POOL_RESIDENCY_TTL_MS`, default
    /// 1000; `0` queries every time.
    ///
    /// Best-effort: it runs on the request path, so a device that is down must not turn into an
    /// earlier, different error than the one the request itself will produce. A failed read leaves
    /// the previous view in place and is not latched.
    pub async fn residency(&self) -> Residency {
        if self.fresh_enough() {
            return self.residency_from_cache(true);
        }
        let _guard = self.residency_lock.lock().await;
        // Re-check: another request may have refreshed while we waited for the lock.
        if self.fresh_enough() {
            return self.residency_from_cache(true);
        }
        let mut current = true;
        match self.backend.list_loaded_detailed().await {
            Ok((running, caps)) => {
                self.reconcile(&running);
                *self.discovered_caps.lock().unwrap() = caps;
                *self.residency_read_at.lock().unwrap() = Some(std::time::Instant::now());
            }
            Err(e) => {
                eprintln!("woollamad: could not read device residency (using last known): {e}");
                current = false;
            }
        }
        self.residency_from_cache(current)
    }

    fn residency_from_cache(&self, current: bool) -> Residency {
        Residency {
            models: self.snapshot(),
            capabilities: self.discovered_caps.lock().unwrap().clone(),
            current,
        }
    }

    /// Mark `real_id` as needing a reload, because the backend just failed a request for it.
    ///
    /// Does NOT reconcile against the backend, and that is the whole point. Measured on real
    /// hardware: for **2–5 seconds** after an instance crash the device still reports the model as
    /// `running` (with a stale `active_request_count`) before reaping catches up. So a reconcile
    /// fired on the failure reads "still resident, all fine" and does nothing — the correct-looking
    /// fix, failing intermittently, because the stale belief is the *backend's* and reading through
    /// to the authority cannot help when the authority has not noticed yet.
    ///
    /// Nor does it wait out the reaping window: how long a backend takes to reap is not a constant
    /// either side should be encoding.
    ///
    /// Instead the next `ensure_loaded` for this model issues a load unconditionally — skipping
    /// both the "we believe it is loaded" short-circuit and the "the backend still lists it" one.
    /// Measured: `start` is accepted immediately after a crash even while the dead instance is
    /// still listed, so this recovers on the next request rather than after the window.
    pub fn mark_needs_reload(&self, real_id: &str) {
        let mut entries = self.entries.lock().unwrap();
        if let Some(e) = entries.get_mut(real_id) {
            e.loaded = false;
        }
        self.needs_reload.lock().unwrap().insert(real_id.to_string());
        eprintln!(
            "woollamad: the backend failed a request for '{real_id}'; it will be reloaded on the \
             next request rather than assumed resident"
        );
    }

    fn fresh_enough(&self) -> bool {
        if self.residency_ttl <= 0.0 {
            return false;
        }
        match *self.residency_read_at.lock().unwrap() {
            Some(at) => at.elapsed().as_secs_f64() < self.residency_ttl,
            None => false,
        }
    }

    /// Override the residency coalescing window (seconds; `0` reads every time). For tests and
    /// for callers that want an explicit value rather than the `WOOLLAMA_POOL_RESIDENCY_TTL_MS`
    /// default.
    pub fn with_residency_ttl(mut self, secs: f64) -> Self {
        self.residency_ttl = secs;
        self
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
/// Per-inferencer view onto the device pools. Two inferencers that name the SAME
/// `management_url` share one `(manager, gate)` pair — see `from_registry`.
pub struct PoolRegistry(HashMap<String, (Arc<DeviceModelManager>, Arc<Gate>)>);

impl PoolRegistry {
    pub fn get(&self, provider: &str) -> Option<&(Arc<DeviceModelManager>, Arc<Gate>)> {
        self.0.get(provider)
    }

    /// No pools — an empty registry. Used by tests, and available as a manual
    /// fallback; `from_registry` itself no longer needs one (see below).
    pub fn empty() -> PoolRegistry {
        PoolRegistry(HashMap::new())
    }

    /// One manager+gate per inferencer that declares a `management_url` (mirrors the
    /// Python lifespan's `_pools` construction loop). Auth headers are best-effort —
    /// an inferencer with a required-but-unset API key env still gets a pool (with no
    /// auth headers against its management API), matching Python's `except
    /// InferencerError: _hdrs = {}`.
    ///
    /// Each inferencer's `management_protocol` (default `"device"` when unset) is
    /// resolved to a `DeviceBackend`: the built-in `"device"` REST preset, the built-in
    /// `"ollama"` adapter (`OllamaBackend`, no configured `keep_alive`), a
    /// config-defined `[management_protocols.<name>]` REST shape (`from_spec`), or a
    /// config-defined `kind = "ollama"` block (`OllamaBackend` with its configured
    /// `keep_alive`). An unresolvable name (not `"device"`, not `"ollama"`, and absent
    /// from `protocols`) does NOT fail the whole registry — a typo in ONE
    /// inferencer's `management_protocol` must not silently disable pooling for
    /// every other device inferencer (that used to route them all to the plain,
    /// no-auto-load relay via `build_state`'s `Err` => `PoolRegistry::empty()`
    /// fallback). Instead: warn loudly (naming the inferencer and the unresolved
    /// protocol) and skip just that inferencer, leaving it absent from the
    /// resulting map (`get()` returns `None` for it) while every other inferencer
    /// still gets its pool built normally.
    ///
    /// Also warns (once, up front) if a config `[management_protocols.<name>]`
    /// block reuses a RESERVED built-in name (`"device"`/`"ollama"`) — that block is
    /// always shadowed by the built-in of the same name (see the `match` below),
    /// so silently ignoring it would hide a config mistake.
    pub fn from_registry(registry: &engine::Registry, protocols: &HashMap<String, engine::ProtocolSpec>) -> PoolRegistry {
        for reserved in ["device", "ollama"] {
            if protocols.contains_key(reserved) {
                eprintln!(
                    "woollamad: management_protocols: WARNING: config block '[management_protocols.{reserved}]' \
                     is shadowed by the built-in '{reserved}' protocol and will be ignored"
                );
            }
        }
        // Group by management_url FIRST. Two inferencers pointing at one device must share both
        // the residency view and the concurrency gate (issue #26):
        //   * separate managers mean neither sees the other's loads, so alternating between two
        //     routes onto one device swaps the model on every call;
        //   * separate gates mean `parallel` is enforced twice over, DOUBLING the effective
        //     in-flight concurrency against hardware where `parallel = 1` exists to stop a wedge.
        // The second is a safety property being silently halved, not a performance question.
        let mut groups: HashMap<String, Vec<engine::Inferencer>> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for inf in registry.list() {
            let Some(url) = inf.management_url.clone() else { continue };
            if !groups.contains_key(&url) {
                order.push(url.clone());
            }
            groups.entry(url).or_default().push(inf);
        }

        let mut map = HashMap::new();
        for url in order {
            let infs = &groups[&url];
            let lead = &infs[0];
            if infs.len() > 1 {
                let names: Vec<&str> = infs.iter().map(|i| i.name.as_str()).collect();
                eprintln!(
                    "woollamad: inferencers {names:?} share management_url '{url}' — they will \
                     share one device pool and one concurrency gate, using the most restrictive \
                     limits configured across them"
                );
                for other in &infs[1..] {
                    if other.management_protocol != lead.management_protocol {
                        eprintln!(
                            "woollamad: WARNING: inferencer '{}' declares a different \
                             management_protocol than '{}' for the same device; using '{}'s",
                            other.name, lead.name, lead.name
                        );
                    }
                }
            }
            let headers = lead.auth_headers().unwrap_or_default();
            let protocol_name = lead.management_protocol.as_deref().unwrap_or("device");
            let backend: Arc<dyn DeviceBackend> = match protocol_name {
                "device" => Arc::new(RestBackend::device(url.clone(), headers, 0.5, 120.0)),
                "ollama" => Arc::new(OllamaBackend::new(url.clone(), None)),
                other => match protocols.get(other) {
                    Some(engine::ProtocolSpec::Rest { running, start, stop }) => {
                        Arc::new(RestBackend::from_spec(&url, &headers, running, start, stop, 0.5, 120.0))
                    }
                    Some(engine::ProtocolSpec::Ollama { keep_alive }) => {
                        Arc::new(OllamaBackend::new(url.clone(), keep_alive.clone()))
                    }
                    None => {
                        eprintln!(
                            "woollamad: management_protocols: WARNING: inferencer '{}': unknown \
                             management_protocol '{other}' — skipping this inferencer (its device pool is \
                             disabled; other inferencers are unaffected)",
                            lead.name
                        );
                        continue;
                    }
                },
            };

            // Most restrictive wins for anything that bounds load on the device. `queue_timeout`
            // is the exception: a SHORTER wait causes spurious 503s on a slow cold load without
            // protecting the device, so the most forgiving value is the safe one there.
            let parallel = infs.iter().map(|i| i.parallel).min().unwrap_or(1);
            let queue_max = infs.iter().filter_map(|i| i.queue_max).min();
            let pool_max = infs.iter().filter_map(|i| i.pool_max).min();
            let queue_timeout = infs.iter().map(|i| i.queue_timeout).fold(0.0f64, f64::max);

            let manager = Arc::new(DeviceModelManager::new(backend));
            let gate = Arc::new(Gate::new(manager.clone(), parallel, queue_max, queue_timeout, pool_max, 5.0));
            for inf in infs {
                map.insert(inf.name.clone(), (manager.clone(), gate.clone()));
            }
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

        let pools = PoolRegistry::from_registry(&reg, &HashMap::new());
        assert!(pools.get("cloud").is_none(), "no management_url => no pool");
        assert!(pools.get("device").is_some(), "management_url present => pool built");
    }
}
