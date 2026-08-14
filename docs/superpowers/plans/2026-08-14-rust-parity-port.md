# Rust Router Parity Port (to Python v0.10.0) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `woollamad` (`woollama-server`) to feature parity with the Python reference at v0.10.0 — `/v1/images/generations`, `/v1/embeddings`, and the model-pooling stack (virtual models + on-demand load/evict + per-model queue/backpressure).

**Architecture:** Pure logic (inferencer fields, resolver/eviction) lands in `woollama-engine`; stateful runtime (`DeviceModelManager`, `Gate`) and the HTTP surface land in `woollama-server`. The manager is a `Mutex`-guarded struct + `reqwest` modeled on `ManagedAgents` (no channel-actor). The chat passthrough is first re-routed through the config `Registry` (which also fixes config inferencers being invisible today).

**Tech Stack:** Rust, tokio, axum, reqwest, serde_json, `tokio::sync::{Mutex, Semaphore}`. Tests are `#[tokio::test]` integration tests against a mock upstream `axum::Router` on an ephemeral TCP port.

**Spec:** `docs/superpowers/specs/2026-08-14-rust-parity-port-design.md` — read it alongside this plan. The **Python originals on this same branch are the behavior oracle**: `src/woollama/resolver.py`, `src/woollama/pool.py`, `src/woollama/router.py` (`_passthrough_images`/`_passthrough_embeddings`/`_passthrough_pooled`), `tests/test_resolver.py`, `tests/test_pool.py`, `tests/test_router.py`. Port their behavior; do not re-invent it.

## Global Constraints

- **Two crates, one workspace.** Pure logic → `woollama-engine` (rlib, **no pyo3**, must not gain a server dep). Stateful/HTTP → `woollama-server`. `woollama-core` (the cdylib wheel) is not touched.
- **Conformance untouched.** `engine::inferencer_to_json` (engine lib.rs:479-486) stays at its current 4 fields — do NOT surface the new inferencer fields in it, so the conformance suite (42) is unaffected.
- **Generic naming.** The example/test inferencer name is **`device`** (never "tiiny" — the repo was scrubbed of that brand). Field for the TOML `virtual` key is `virtual_models` (`virtual` is a Rust keyword; the TOML key stays `virtual`).
- **CI gate (Rust):** every task must leave `cargo test -p woollama-engine -p woollama-server --features test-fixtures` green and `cargo clippy -p woollama-engine -p woollama-server --all-targets --features test-fixtures -- -D warnings` clean. Clippy is a hard gate (`-D warnings`).
- **Additive / backward-compatible.** An inferencer without `management_url` behaves exactly as today. Existing tests must stay green.
- **Device API (verified live):** management base is `management_url`; `GET {management_url}/api/v1/models/running` → `{"running":[<real_id>,...], ...}` (parse the top-level `running` array), auth `Authorization: Bearer <key>`; load/unload `POST {management_url}/api/v1/models/{real_id}/start|stop`. Real ids contain slashes — send raw.
- **Error contract:** `Backpressure`→HTTP 503 + `Retry-After` (integer secs); `DeviceError`→HTTP 502; resolve failure→400.
- **Commits:** conventional subject; body ends with the two trailers:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MNbD4t3EqzQuXXoVZApnHu
  ```

---

### Task 1: Route the chat passthrough through the config `Registry`

The passthrough currently resolves via `engine::get_inferencer(provider)` (built-ins only), so config-defined inferencers are invisible and new config fields can't reach the handler. Route it through `state.inferencers`. Prerequisite for pooling; also fixes the config-inferencer gap.

**Files:**
- Modify: `woollama-engine/src/lib.rs` (make `Registry::resolve` public, :531)
- Modify: `woollama-server/src/lib.rs` (`chat_completions` passthrough, :840; also the ollama-native branch :862 uses the inferencer)
- Test: `woollama-server/tests/passthrough_config.rs` (new)

**Interfaces:**
- Produces: `engine::Registry::resolve(&self, provider: &str) -> Option<Inferencer>` becomes `pub`.
- The passthrough resolves `let inf = state.inferencers.resolve(provider)` instead of `engine::get_inferencer(provider)`.

- [ ] **Step 1: Write the failing test**

Create `woollama-server/tests/passthrough_config.rs`. Mirror the harness in `woollama-server/tests/discovery.rs` (spawn a mock upstream `axum::Router` on an ephemeral port; point `WOOLLAMA_CONFIG_DIR` at a temp dir holding an `inferencers.toml` that defines `[inferencers.device]` with `base_url` = the mock URL; build via `build_state().await`; mount `router(state)`; drive with an HTTP client). The mock upstream serves `POST /chat/completions` returning `{"choices":[{"message":{"content":"ok"}}]}`. Assert a `POST /v1/chat/completions` with `{"model":"device/somemodel","messages":[...]}` returns 200 with that body (proving a **config-only** inferencer is reachable on the passthrough).

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p woollama-server --test passthrough_config`
Expected: FAIL — the config inferencer `device` is unknown to `get_inferencer`, so the handler returns 400 "unknown model namespace".

- [ ] **Step 3: Make `Registry::resolve` public**

In `woollama-engine/src/lib.rs:531`, change `fn resolve` to `pub fn resolve`.

- [ ] **Step 4: Route the passthrough through `state.inferencers`**

In `woollama-server/src/lib.rs`, in `chat_completions`'s passthrough path (around :840), replace `engine::get_inferencer(provider)` with `state.inferencers.resolve(provider)`. Keep the same `None` → 400 "unknown model namespace" behavior (the error message that lists known providers should now use `state.inferencers.names()`).

- [ ] **Step 5: Run tests**

Run: `cargo test -p woollama-server` then `cargo clippy -p woollama-server --all-targets --features test-fixtures -- -D warnings`
Expected: new test PASS; existing passthrough/orchestrate/discovery tests still PASS; clippy clean.

- [ ] **Step 6: Commit**

```bash
git add woollama-engine/src/lib.rs woollama-server/src/lib.rs woollama-server/tests/passthrough_config.rs
git commit   # "feat(server): resolve chat passthrough via the config Registry" + trailers
```

---

### Task 2: Engine `Inferencer` — pooling fields + `images_url`/`embeddings_url`

Add the six config fields (mirroring the Python dataclass) and the two URL builders. Compiler-enforced across every struct literal. `inferencer_to_json` stays unchanged.

**Files:**
- Modify: `woollama-engine/src/lib.rs` (struct :79-94; `impl` :165-176; `get_inferencer` literals :99-107/:114-122/:124-132; `Registry::add` :503-507; `build_config_registry` insert :474 + key reads near :461-473)
- Test: `woollama-engine/tests/inferencer_pooling.rs` (new)

**Interfaces:**
- Produces on `engine::Inferencer` (all `pub`): `management_url: Option<String>`, `parallel: u32`, `pool_max: Option<u32>`, `queue_max: Option<u32>`, `queue_timeout: f64`, `virtual_models: std::collections::BTreeMap<String, String>`.
- Produces methods: `Inferencer::images_url(&self) -> String` (`{base_url}/images/generations`), `Inferencer::embeddings_url(&self) -> String` (`{base_url}/embeddings`).
- Defaults used in every non-config literal: `management_url: None, parallel: 1, pool_max: None, queue_max: None, queue_timeout: 30.0, virtual_models: BTreeMap::new()`.

- [ ] **Step 1: Write the failing test**

Create `woollama-engine/tests/inferencer_pooling.rs`: set `WOOLLAMA_CONFIG_DIR` to a temp dir with
```toml
[inferencers.device]
base_url = "http://dev/v1"
management_url = "http://dev:8800"
parallel = 2
pool_max = 3
queue_max = 8
queue_timeout = 45
virtual = { default = "Qwen/Coder", coder = "Qwen/Coder" }
```
Call `engine::Registry::from_config().unwrap()`, `resolve("device")`, and assert: `management_url == Some("http://dev:8800")`, `parallel == 2`, `pool_max == Some(3)`, `queue_max == Some(8)`, `queue_timeout == 45.0`, `virtual_models["default"] == "Qwen/Coder"`, and `images_url() == "http://dev/v1/images/generations"`, `embeddings_url() == "http://dev/v1/embeddings"`. Also assert a built-in (`resolve("anthropic")`) has `management_url == None`, `parallel == 1`, `queue_timeout == 30.0`, empty `virtual_models`.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p woollama-engine --test inferencer_pooling`
Expected: FAIL to compile — the fields/methods don't exist.

- [ ] **Step 3: Add the fields + methods + defaults**

Add the six fields to `pub struct Inferencer` (lib.rs:79-94). Add `images_url`/`embeddings_url` to `impl Inferencer` (:165-176) beside `chat_url`. Add the default field values to every `Inferencer { … }` literal the compiler flags: the `cloud` closure and each arm in `get_inferencer` (:99-137), and `Registry::add` (:503-507).

- [ ] **Step 4: Read the new keys in `build_config_registry`**

In `build_config_registry` (:430-474), after the existing `discover`/`str_list` reads and before the final `reg.insert`, extract (following the existing idioms):
- `management_url`: `spec.get("management_url").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string).or_else(|| base.as_ref().and_then(|b| b.management_url.clone()))`
- `parallel`: `spec.get("parallel").and_then(Value::as_u64).map(|v| v as u32).unwrap_or_else(|| base.as_ref().map_or(1, |b| b.parallel))`
- `pool_max` / `queue_max`: `spec.get("pool_max").and_then(Value::as_u64).map(|v| v as u32).or_else(|| base.as_ref().and_then(|b| b.pool_max))` (same for queue_max)
- `queue_timeout`: `spec.get("queue_timeout").and_then(Value::as_f64).unwrap_or_else(|| base.as_ref().map_or(30.0, |b| b.queue_timeout))`
- `virtual_models`: read `spec.get("virtual")` as an object → `BTreeMap<String,String>` (string values), else inherit `base.virtual_models` or empty.
Add all six to the `Inferencer { … }` constructor at :474. Leave `inferencer_to_json` unchanged.

- [ ] **Step 5: Run tests + clippy**

Run: `cargo test -p woollama-engine` then `cargo clippy -p woollama-engine --all-targets --features test-fixtures -- -D warnings`
Expected: PASS; conformance tests unaffected; clippy clean.

- [ ] **Step 6: Commit**

```bash
git add woollama-engine/src/lib.rs woollama-engine/tests/inferencer_pooling.rs
git commit   # "feat(engine): pooling fields + images/embeddings URLs on Inferencer" + trailers
```

---

### Task 3: `/v1/images/generations` + `/v1/embeddings` handlers

Two non-streaming passthrough handlers + routes, siblings of the chat passthrough non-stream path. Port `router.py::_passthrough_images`/`_passthrough_embeddings`.

**Files:**
- Modify: `woollama-server/src/lib.rs` (add handlers; register routes near :214)
- Test: `woollama-server/tests/images_embeddings.rs` (new)

**Interfaces:**
- Consumes: `state.inferencers.resolve` (Task 1), `Inferencer::images_url`/`embeddings_url`/`auth_headers` (Task 2), `forward_post`/`relay_json` (lib.rs:274-297).
- Produces: `async fn images_generations(State<Arc<AppState>>, Json<Value>) -> Response` and `async fn embeddings(...)`; routes `POST /v1/images/generations`, `POST /v1/embeddings`.

- [ ] **Step 1: Write the failing test**

Create `woollama-server/tests/images_embeddings.rs`. Mock upstream serves `POST /images/generations` → `{"data":[{"b64_json":"x"}]}` and `POST /embeddings` → `{"data":[{"embedding":[0.1,0.2]}]}`, recording the received `model` field. Config `[inferencers.device]` base_url = mock. Assert: `POST /v1/images/generations {"model":"device/img","prompt":"a cat"}` → 200 with the image body AND the upstream saw `model=="img"` (prefix stripped); same for embeddings. Assert an unknown provider (`"nope/x"`) → 400.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p woollama-server --test images_embeddings`
Expected: FAIL — routes 404 (not registered).

- [ ] **Step 3: Implement the handlers**

Add `images_generations` and `embeddings`, each: parse `model`, `let (provider, _) = split at first '/'`, `state.inferencers.resolve(provider)` → 400 if None, `inf.auth_headers()` (map `EngineError` via `engine_err_response`), rewrite `body["model"]` to the bare id, `forward_post(inf.images_url()/embeddings_url(), &fwd, &headers, timeout)` (images timeout 300, embeddings 180) then `relay_json`. Register both routes near lib.rs:214.

- [ ] **Step 4: Run tests + clippy**

Run: `cargo test -p woollama-server --test images_embeddings` then clippy.
Expected: PASS; clippy clean.

- [ ] **Step 5: Commit**

```bash
git add woollama-server/src/lib.rs woollama-server/tests/images_embeddings.rs
git commit   # "feat(server): /v1/images/generations + /v1/embeddings passthrough" + trailers
```

---

### Task 4: Engine `resolver` module (pure)

Port `src/woollama/resolver.py` to a pure engine module. No I/O, no async.

**Files:**
- Create: `woollama-engine/src/resolver.rs`; add `pub mod resolver;` to `woollama-engine/src/lib.rs`
- Test: unit tests in `resolver.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Produces `engine::resolver::`:
  - `pub struct PoolEntry { pub model_id: String, pub in_flight: u32, pub queued: u32, pub last_used: f64 }`
  - `pub struct ResolveError(pub String);`
  - `pub fn resolve(bare: &str, virtual_models: &BTreeMap<String,String>, loaded: &[String], default: Option<&str>) -> Result<String, ResolveError>` — `bare=="default"` → `loaded[0]` if any, else `default` param, else `Err`; alias hit → target; else `bare.to_string()`.
  - `pub fn needs_eviction(loaded: &HashSet<String>, target: &str, pool_max: Option<u32>) -> bool` — true iff `pool_max` is `Some(n)` with `n>0`, `!loaded.contains(target)`, and `loaded.len() as u32 >= n`.
  - `pub fn pick_eviction(entries: &[PoolEntry]) -> Option<String>` — the `model_id` of the min-`last_used` entry among those with `in_flight==0 && queued==0`; `None` if none idle.

- [ ] **Step 1: Write the failing tests**

In `resolver.rs`, add `#[cfg(test)] mod tests` porting every case from `tests/test_resolver.py` verbatim in intent: real-id passthrough; `default` prefers `loaded[0]`; `default` falls back to configured default; `default` with nothing loaded + no fallback → `Err`; alias→target; unknown alias→itself; `needs_eviction` truth table (capped-full-absent true; already-loaded false; room false; `None` false; `0` false); `pick_eviction` LRU idle; never-busy; skip-busy-return-idle.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p woollama-engine resolver`
Expected: FAIL to compile — module doesn't exist.

- [ ] **Step 3: Implement `resolver.rs`**

Port `resolver.py` (same three functions + `PoolEntry`), exact semantics above. Pure; only `std::collections::{BTreeMap, HashSet}`. Add `pub mod resolver;` to lib.rs.

- [ ] **Step 4: Run tests + clippy**

Run: `cargo test -p woollama-engine resolver` then clippy.
Expected: PASS; clippy clean.

- [ ] **Step 5: Commit**

```bash
git add woollama-engine/src/resolver.rs woollama-engine/src/lib.rs
git commit   # "feat(engine): pure resolver (virtual-model + eviction logic)" + trailers
```

---

### Task 5: `pool::DeviceModelManager` (server)

The load/evict actor. `Mutex`-guarded state + `reqwest`, modeled on `ManagedAgents` (managed_agents.rs:83-96 struct, :116-148 HTTP, :180 `ensure_agent`). Port `pool.py`'s `DeviceModelManager` **including the eviction-race fix** (flip victim not-loaded before stop; conditional drop only if still idle).

**Files:**
- Create: `woollama-server/src/pool.rs`; add `pub mod pool;` to `woollama-server/src/lib.rs`
- Test: `woollama-server/tests/pool_manager.rs` (new)

**Interfaces:**
- Consumes: `engine::resolver::{PoolEntry, needs_eviction, pick_eviction}` (Task 4).
- Produces `pool::`:
  - `pub enum PoolError { Device(String), Backpressure(f64) }` (shared with the Gate; `Backpressure(retry_after_secs)`).
  - `pub struct DeviceModelManager` with `new(management_url: String, headers: HashMap<String,String>) -> Self` and a test ctor `with_config(url, headers, poll_interval: f64, load_timeout: f64, retry_after: f64)`.
  - `pub async fn ensure_loaded(&self, real_id: &str, pool_max: Option<u32>) -> Result<(), PoolError>`
  - sync `pub fn acquire/release/enqueue/dequeue(&self, real_id: &str)` and `pub fn queued(&self, real_id: &str) -> u32`, `pub fn snapshot(&self) -> Vec<String>` (loaded ids, MRU first). (Counters live behind the same `Mutex` as loaded state; these lock internally — keep them non-async by using `std::sync::Mutex` for the counter map OR make them async; **decision:** use one `tokio::sync::Mutex<PoolState>` and make the counter ops `async` — simpler than splitting locks. Update the Gate's calls accordingly.)
  - Internal: `Entry { loaded: bool, in_flight: u32, queued: u32, last_used: f64 }`; `_running()`/`_start()`/`_stop()` via `reqwest` (parse top-level `running` array; non-2xx or transport error → `PoolError::Device`).

**Note on `last_used`/clock:** use `tokio::time::Instant`/a monotonic counter for `last_used` (no wall-clock needed — only ordering matters). For tests, an injected monotonic counter keeps eviction deterministic (mirror the Python `clock` injection).

- [ ] **Step 1: Write the failing tests (with a FakeDevice)**

Create `woollama-server/tests/pool_manager.rs`. Build a `FakeDevice` = an `axum::Router` (spawned via the ephemeral-port helper) with shared state (`Arc<Mutex<{running: HashSet<String>, calls: Vec<(String,String)>, fail_start, fail_stop, running_status, block_stop: Option<Notify>}>>`) serving: `GET /api/v1/models/running` → `{"running":[...]}`; `POST /api/v1/models/{id}/start` (record; add to running unless `fail_start`); `POST .../{id}/stop`. Port these `test_pool.py` cases: load-on-demand issues one `start`; already-loaded → no `start`; concurrent `ensure_loaded` same id → exactly one `start` (`tokio::join!` several); `fail_start`→`PoolError::Device`; unreachable url→`Device`; evict LRU idle at `pool_max`; no-evict-when-all-busy→`Backpressure`; **the eviction-race test** (block `/stop`; while stopping the victim, a concurrent `ensure_loaded`/`acquire` on it must not strand it or lose counters — port `test_eviction_race_does_not_strand_or_lose_racer`).

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p woollama-server --test pool_manager`
Expected: FAIL to compile — `pool` module doesn't exist.

- [ ] **Step 3: Implement `pool.rs` (manager only)**

Port `pool.py`'s manager: `state: Mutex<PoolState>` (a single `tokio::sync::Mutex`), `ensure_loaded` (fast-path if loaded; else under the lock: `_running()` + reconcile, evict-to-fit via `resolver::needs_eviction`/`pick_eviction`, flip victim `loaded=false` before `_stop`, drop only if still idle after, `_start` + poll `running` until ready or `load_timeout`). Counter ops + `snapshot`. `PoolError`. Add `pub mod pool;` to lib.rs.

- [ ] **Step 4: Run tests + clippy**

Run: `cargo test -p woollama-server --test pool_manager` then clippy.
Expected: PASS; clippy clean.

- [ ] **Step 5: Commit**

```bash
git add woollama-server/src/pool.rs woollama-server/src/lib.rs woollama-server/tests/pool_manager.rs
git commit   # "feat(server): DeviceModelManager load/evict actor" + trailers
```

---

### Task 6: `pool::Gate` + wire pooled passthrough + `PoolRegistry`

Add the queue/backpressure gate, the `PoolRegistry` on `AppState`, and wire the pooled path into the chat passthrough. Port `pool.py::Gate`/`Slot` and `router.py::_passthrough_pooled`.

**Files:**
- Modify: `woollama-server/src/pool.rs` (add `Gate`, `Slot`, `PoolRegistry`)
- Modify: `woollama-server/src/lib.rs` (`AppState` field :56-75; `build_state` :150-159; `chat_completions` passthrough :839-917; add a 503+Retry-After response builder)
- Test: `woollama-server/tests/pool_gate.rs` (new) + extend `passthrough_config.rs` if useful

**Interfaces:**
- Consumes: `DeviceModelManager` (Task 5).
- Produces `pool::`:
  - `pub struct Gate` with `new(manager: Arc<DeviceModelManager>, parallel: u32, queue_max: Option<u32>, queue_timeout: f64, pool_max: Option<u32>, retry_after: f64)`.
  - `pub async fn enter(&self, real_id: &str) -> Result<Slot, PoolError>` — early-reject if `queued(real_id) >= queue_max`; `enqueue`; `ensure_loaded(real_id, pool_max)`; acquire a per-`real_id` `Arc<Semaphore>` permit within `queue_timeout` (`tokio::time::timeout` → `PoolError::Backpressure`); `dequeue`; `acquire`. Return `Slot` holding the permit + `Arc<DeviceModelManager>` + `real_id`.
  - `pub struct Slot` whose `Drop` releases the permit (held `OwnedSemaphorePermit`) and calls `manager.release`. (Because `Drop` can't be async, hold the permit as an `OwnedSemaphorePermit` — dropping it releases synchronously — and do the in-flight decrement via a `std::sync::Mutex` counter path OR spawn a detached release; **decision:** keep the in-flight counter behind a `std::sync::Mutex` inside the manager so `release` is sync and callable from `Drop`. Reconcile with Task 5's lock choice: the load-critical section uses the `tokio::Mutex`; the fast counters use a separate `std::sync::Mutex<Counters>`. Document this split.)
  - `pub struct PoolRegistry(HashMap<String, (Arc<DeviceModelManager>, Gate)>)` with `get(&self, provider) -> Option<&(Arc<DeviceModelManager>, Gate)>` and a `from_registry(&engine::Registry) -> PoolRegistry` builder (iterate `registry.list()`, for each `Inferencer` with `management_url`, build a manager from `management_url` + `auth_headers()` and a `Gate` from its knobs).
- Produces on `AppState`: `pub pools: Arc<pool::PoolRegistry>` (added to the struct and the `build_state` literal).

**⚠️ Lock-model note for the implementer:** Task 5 tentatively used a single `tokio::sync::Mutex`. `Slot::Drop` needs a **sync** `release`. Resolve by holding per-model runtime counters (`in_flight`, `queued`, `last_used`) behind a `std::sync::Mutex<Counters>` (sync, `Drop`-safe) and the load/evict critical section behind a `tokio::sync::Mutex<LoadState>` (async, held across `.await`). `ensure_loaded` reads counters (to protect busy models from eviction) via the sync mutex. Pick this split in Task 5 if you reach it first; otherwise refactor here. Do not hold the sync counter mutex across an `.await`.

- [ ] **Step 1: Write the failing tests**

Create `woollama-server/tests/pool_gate.rs` (reuse the `FakeDevice` from `pool_manager.rs` — extract it into a shared `mod common` or duplicate minimally). Port from `test_pool.py`: `parallel=1` serializes two concurrent `enter`/hold/exit (assert non-interleaved); `queue_max` saturated → third `enter` is `Backpressure`; `queue_timeout` exceeded while a holder holds → `Backpressure`; a serving model (in-flight via a held `Slot`) is protected from eviction (`enter` a third model at capacity → `Backpressure`, no `stop` on the busy ones). Add an end-to-end pooled-passthrough test (mock device `/v1/chat/completions` + management endpoints, config `[inferencers.device] management_url=…`): `POST /v1/chat/completions {"model":"device/default"}` loads-on-demand + resolves to the loaded model; and a forced `Backpressure` path returns **503 with a `Retry-After` header**.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p woollama-server --test pool_gate`
Expected: FAIL to compile — `Gate`/`PoolRegistry`/`AppState.pools` don't exist.

- [ ] **Step 3: Implement `Gate`/`Slot`/`PoolRegistry`**

Port `pool.py::Gate`/`Slot` with the ordering guarantee (enqueue before ensure_loaded; no `.await` between dequeue and the in-flight bump — trivial with the sync counter mutex). Implement `PoolRegistry::from_registry`.

- [ ] **Step 4: Wire `AppState` + `build_state`**

Add `pub pools: Arc<pool::PoolRegistry>` to `AppState` (lib.rs:56-75). In `build_state` (:150-159), after `inferencers` is built (:107), `let pools = Arc::new(pool::PoolRegistry::from_registry(&inferencers));` and add it to the struct literal.

- [ ] **Step 5: Wire the pooled path into `chat_completions`**

In the passthrough path (after Task 1's `resolve`): if `state.pools.get(provider)` is `Some((manager, gate))` **and** `inf.management_url.is_some()`, take the pooled path: `let real = engine::resolver::resolve(bare, &inf.virtual_models, &manager.snapshot(), inf.virtual_models.get("default").map(String::as_str))` (map `ResolveError`→400); rewrite `body["model"]=real`; `let slot = gate.enter(&real).await` mapping `PoolError::Backpressure(secs)`→ a **503 response with `Retry-After: <secs as int>`** (a small builder: `Response::builder().status(503).header("retry-after", secs.to_string())…`, pattern from the SSE header at lib.rs:913-915) and `PoolError::Device(msg)`→ `engine_err_response(EngineError::new(msg,"server_error",502))`; then dispatch via the existing `forward_post`/`passthrough_stream`; hold `slot` across the dispatch (for streaming, move it into the stream body so it drops when the stream ends). Non-`management_url` inferencers keep today's path unchanged.

- [ ] **Step 6: Run tests + clippy + full suite**

Run: `cargo test -p woollama-engine -p woollama-server --features test-fixtures` then `cargo clippy -p woollama-engine -p woollama-server --all-targets --features test-fixtures -- -D warnings`
Expected: all PASS; clippy clean.

- [ ] **Step 7: Commit**

```bash
git add woollama-server/src/pool.rs woollama-server/src/lib.rs woollama-server/tests/pool_gate.rs
git commit   # "feat(server): Gate + pooled passthrough (503/502, virtual models)" + trailers
```

---

## Self-Review

**Spec coverage:** images/embeddings (Task 3) ✓; engine inferencer fields + URL builders (Task 2) ✓; pure resolver in engine (Task 4) ✓; DeviceModelManager (Task 5) + Gate/backpressure (Task 6) ✓; passthrough routed through config Registry (Task 1) ✓; `PoolRegistry` on `AppState` + wiring (Task 6) ✓; error contract 503+Retry-After / 502 / 400 (Task 6) ✓; conformance untouched (`inferencer_to_json` unchanged, stated in Task 2) ✓; eviction-race fix ported (Task 5) ✓. `/v1/responses` pooling and packaging are out of scope per the spec.

**Placeholder scan:** logic bodies are delegated to the named Python originals on this branch (a port, not greenfield) with exact Rust signatures + test cases given; no vague "add error handling" — the error mapping is specified per case in Task 6.

**Type consistency:** `Inferencer.virtual_models`/`management_url`/`parallel`/`pool_max`/`queue_max`/`queue_timeout` (Task 2) are consumed identically in Tasks 4/6; `engine::resolver::{PoolEntry, resolve, needs_eviction, pick_eviction}` (Task 4) are consumed by Tasks 5/6; `pool::{PoolError, DeviceModelManager, Gate, Slot, PoolRegistry}` names are consistent across Tasks 5/6; `Registry::resolve` made `pub` in Task 1 and used in Tasks 1/3/6.

**Open implementation decision flagged, not hidden:** the lock model (`tokio::Mutex` for the load critical section vs a sync `Mutex` for `Drop`-safe counters) is called out explicitly in Tasks 5 and 6 with a recommended split — the implementer resolves it in whichever task lands first.
