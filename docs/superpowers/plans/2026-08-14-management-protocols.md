# Pluggable Device Management Protocols — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make woollamad's pooling backend pluggable — a `DeviceBackend` trait with a built-in `tiiny` preset + `ollama` adapter, plus config-defined `rest` protocols (`[management_protocols.<name>]`) — selected per-inferencer via `management_protocol`, defaulting to `tiiny` for back-compat.

**Architecture:** Extract the manager's hardcoded Tiiny HTTP into a `DeviceBackend` behind an `Arc<dyn DeviceBackend>` (via `#[async_trait::async_trait]`, as the crate's existing `StoreProvider`/`PatternBackend` do). Two kinds: `RestBackend` (config-parameterized; `tiiny` is a preset of it) and `OllamaBackend`. The resolver, gate, queue, eviction-race fix, and pooled passthrough are untouched — only what talks HTTP moves.

**Tech Stack:** Rust, tokio, reqwest, serde_json, `async-trait`. Tests are `#[tokio::test]` integration tests using the mock-backend fixture already on this branch.

**Spec:** `docs/superpowers/specs/2026-08-14-management-protocols-design.md` — read it alongside this plan.

## Global Constraints

- **woollamad (Rust) only.** No Python changes. `woollama-engine` stays pyo3-free.
- **Back-compat is mandatory.** `management_url` with no `management_protocol` ⇒ the `tiiny` preset, byte-identical to today's behavior. Every existing `woollama-server` test must stay green — the `tiiny` `RestBackend` reproduces the current `running`/`start`/`stop` exactly.
- **Conformance untouched:** do NOT surface `management_protocol` (or anything new) in `engine::inferencer_to_json`.
- **Test fixture (already on this branch):** `woollama-server/tests/common/mod.rs` — `spawn_rest(RestMockConfig)` / `spawn_ollama()` returning a `MockHandle` (`base_url`, `requests()`/`requests_to()`, `loaded()`, `set_loaded()`, `set_fail_start/stop()`). Use `mod common;` in new test files. `RunningShape::{Strings{field}, Objects{field,id_key}}`, `IdLoc::{Path, Body{field}}`.
- **`dyn DeviceBackend`** requires `#[async_trait::async_trait]` on the trait and impls (native async-fn-in-trait isn't dyn-safe). Match `conversations.rs::StoreProvider`.
- **CI gate per task:** `~/.cargo/bin/cargo test -p woollama-engine -p woollama-server --features test-fixtures` green AND `~/.cargo/bin/cargo clippy -p woollama-engine -p woollama-server --all-targets --features test-fixtures -- -D warnings` clean (`-D warnings` is hard). cargo is at `~/.cargo/bin/cargo` (1.97.1); a bare `cargo` fails to find rustc.
- **Commits:** conventional subject; body ends with:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MNbD4t3EqzQuXXoVZApnHu
  ```

---

### Task 1: `DeviceBackend` trait + extract the `tiiny` `RestBackend`; make the manager backend-driven

Pure refactor — no new behavior. Move the manager's Tiiny HTTP (`running`/`start`/`stop`/`apply_headers`, `woollama-server/src/pool.rs`) into a `RestBackend` behind a `DeviceBackend` trait; the manager holds `Arc<dyn DeviceBackend>`.

**Files:**
- Modify: `woollama-server/src/pool.rs` (add trait + `RestBackend`; refactor `DeviceModelManager` fields/ctors/`ensure_loaded`; update `from_registry`)
- Modify: `woollama-server/tests/pool_manager.rs`, `woollama-server/tests/pool_gate.rs` (construct managers via the new ctor)

**Interfaces:**
- Produces: `#[async_trait::async_trait] pub trait DeviceBackend: Send + Sync { async fn list_loaded(&self) -> Result<HashSet<String>, PoolError>; async fn load(&self, id: &str) -> Result<(), PoolError>; async fn unload(&self, id: &str) -> Result<(), PoolError>; }`
- Produces: `pub struct RestBackend` + `RestBackend::tiiny(management_url: String, headers: HashMap<String,String>, poll_interval: f64, load_timeout: f64) -> RestBackend` (pre-fills the Tiiny endpoints). Its `DeviceBackend` impl: `list_loaded` = today's `running()`; `load` = today's `start()` (POST start + poll `list_loaded` until present or `load_timeout`); `unload` = today's `stop()`.
- Changes: `DeviceModelManager` drops `url/headers/client/poll_interval/load_timeout`, gains `backend: Arc<dyn DeviceBackend>`; keeps `retry_after/entries/load_lock/clock`. New ctors: `pub fn new(backend: Arc<dyn DeviceBackend>) -> Self` (retry_after=5.0) and `pub fn with_retry_after(backend: Arc<dyn DeviceBackend>, retry_after: f64) -> Self`. `ensure_loaded` calls `self.backend.{list_loaded,load,unload}` in place of `self.{running,start,stop}` (reconcile/eviction/counters/lock logic unchanged, incl. the eviction-race fix).

- [ ] **Step 1: Add the trait + `RestBackend` (move the HTTP verbatim)**

In `pool.rs`, add `#[async_trait::async_trait] pub trait DeviceBackend …` and `pub struct RestBackend { client: reqwest::Client, base_url: String, headers: HashMap<String,String>, poll_interval: f64, load_timeout: f64 }` with `RestBackend::tiiny(...)`. Move `apply_headers`, `running` (→ `list_loaded`), `start` (→ `load`, keeping the poll loop but polling `self.list_loaded()`), and `stop` (→ `unload`) into `impl RestBackend` / the `DeviceBackend` impl, verbatim in body (same URLs, same `PoolError` messages, same `ok()` check).

- [ ] **Step 2: Refactor the manager to hold the backend**

Replace the manager's `url/headers/client/poll_interval/load_timeout` fields with `backend: Arc<dyn DeviceBackend>`. Rewrite `new`/`with_config` → `new(backend)` / `with_retry_after(backend, retry_after)`. In `ensure_loaded`, swap `self.running()`→`self.backend.list_loaded()`, `self.start(x)`→`self.backend.load(x)`, `self.stop(x)`→`self.backend.unload(x)`. Delete the now-moved `running`/`start`/`stop`/`apply_headers` from the manager.

- [ ] **Step 3: Update `from_registry`**

`DeviceModelManager::new(management_url, headers)` → `DeviceModelManager::new(Arc::new(RestBackend::tiiny(management_url, headers, 0.5, 120.0)))`. Behavior identical (tiiny defaults).

- [ ] **Step 4: Update the existing pool tests to the new ctor**

In `pool_manager.rs`/`pool_gate.rs`, wherever a manager is built from a mock URL (`DeviceModelManager::with_config(url, headers, poll, timeout, retry)`), build `DeviceModelManager::with_retry_after(Arc::new(RestBackend::tiiny(url, headers, poll, timeout)), retry)`. Add a tiny local helper if it reduces churn. The FakeDevice in `pool_manager.rs` already serves the Tiiny shape, so assertions are unchanged.

- [ ] **Step 5: Run the gate**

Run: `~/.cargo/bin/cargo test -p woollama-server --features test-fixtures` then the clippy line.
Expected: ALL pool tests (manager, gate, race, backpressure, streaming, from_registry) green — proves the refactor is behavior-preserving; clippy clean.

- [ ] **Step 6: Commit**

```bash
git add woollama-server/src/pool.rs woollama-server/tests/pool_manager.rs woollama-server/tests/pool_gate.rs
git commit   # "refactor(server): DeviceBackend trait + tiiny RestBackend behind DeviceModelManager" + trailers
```

---

### Task 2: Engine — `management_protocol` field + `[management_protocols]` parsing

Add the per-inferencer selector and parse config-defined protocol blocks into typed specs the server will consume.

**Files:**
- Modify: `woollama-engine/src/lib.rs` (`Inferencer` field + literals + config-merge; new protocol-spec types + parser)
- Test: `woollama-engine/tests/management_protocols.rs` (new)

**Interfaces:**
- Produces on `engine::Inferencer`: `pub management_protocol: Option<String>` (default `None`). Added to every `Inferencer { … }` literal (compiler-enforced: `get_inferencer` arms, `Registry::add`, `Registry::insert`, `build_config_registry`) and read in `build_config_registry` from the TOML key `management_protocol` (inherit-on-unset like the other keys).
- Produces (pub, in engine): 
  - `pub struct EndpointSpec { pub url: String, pub method: Option<String>, pub body: Option<String>, pub headers: std::collections::BTreeMap<String,String>, pub path: Option<String>, pub id_field: Option<String> }` (`path`/`id_field` used only for the `running` endpoint).
  - `pub enum ProtocolSpec { Rest { running: EndpointSpec, start: EndpointSpec, stop: EndpointSpec }, Ollama { keep_alive: Option<String> } }`
  - `pub fn load_management_protocols() -> Result<std::collections::HashMap<String, ProtocolSpec>, EngineError>` — reads `[management_protocols.<name>]` from `$config/inferencers.toml` (via the existing `config_dir()` + `expand_env`), validates `kind` ∈ {`rest`,`ollama`}; `rest` requires `endpoints.running` (with `path`), `endpoints.start`, `endpoints.stop`; unknown/malformed → `EngineError` naming the offending protocol/key. Missing file / absent `[management_protocols]` ⇒ empty map.

- [ ] **Step 1: Write failing parser + field tests**

Create `woollama-engine/tests/management_protocols.rs`: point `WOOLLAMA_CONFIG_DIR` at a temp `inferencers.toml` containing a `[inferencers.dev] management_url=… management_protocol="mybox"` block plus a `[management_protocols.mybox] kind="rest"` with `endpoints.running/start/stop` (one path-based, and assert `${VAR}` expansion in a header), and a `[management_protocols.oll] kind="ollama"`. Assert: `Registry::from_config().resolve("dev").management_protocol == Some("mybox")`; `load_management_protocols()` returns a `Rest` spec with the right urls/method-defaults/headers and an `Ollama` spec. Add error cases: unknown `kind`, and a `rest` block missing `endpoints.stop` → `Err`.

- [ ] **Step 2: Run to verify it fails**

Run: `~/.cargo/bin/cargo test -p woollama-engine --test management_protocols`
Expected: FAIL to compile — the field/types/fn don't exist.

- [ ] **Step 3: Add the `Inferencer` field + merge**

Add `pub management_protocol: Option<String>` to `Inferencer` with all literals defaulting `None`; read `spec.get("management_protocol")` (str, inherit-on-unset) in `build_config_registry`. Leave `inferencer_to_json` unchanged.

- [ ] **Step 4: Add the protocol-spec types + parser**

Implement `EndpointSpec`, `ProtocolSpec`, and `load_management_protocols()` per the interface. Parse the nested `[management_protocols.<name>.endpoints.<op>]` tables; apply the `method`/`path`/`id_field` handling; validate. `expand_env` the whole file first (matching `load_inferencers_toml`).

- [ ] **Step 5: Run the gate**

Run: `~/.cargo/bin/cargo test -p woollama-engine` then clippy.
Expected: PASS; conformance suite unaffected.

- [ ] **Step 6: Commit**

```bash
git add woollama-engine/src/lib.rs woollama-engine/tests/management_protocols.rs
git commit   # "feat(engine): management_protocol field + [management_protocols] parsing" + trailers
```

---

### Task 3: Config-driven `RestBackend` + protocol resolution in `from_registry`

Generalize `RestBackend` to build from a `ProtocolSpec::Rest` (templated URLs/method/body/headers, running `path`/`id_field`), keep `tiiny` as a preset, and resolve `management_protocol` → backend at startup.

**Files:**
- Modify: `woollama-server/src/pool.rs` (`RestBackend` from-spec constructor + templating; `from_registry` resolution)
- Modify: `woollama-server/src/lib.rs` (`build_state`: load protocols, pass to `from_registry`)
- Test: `woollama-server/tests/pool_protocols.rs` (new) — uses the fixture

**Interfaces:**
- Consumes: `engine::{ProtocolSpec, EndpointSpec, load_management_protocols}` (Task 2), the `DeviceBackend`/`RestBackend` (Task 1), the fixture.
- Produces: `RestBackend::from_spec(base_url: &str, default_headers: &HashMap<String,String>, running: &EndpointSpec, start: &EndpointSpec, stop: &EndpointSpec, poll_interval: f64, load_timeout: f64) -> RestBackend`. Templating: replace `{base}` (→ base_url, trimmed) and `{id}` in every `url`, `body`, and header value; per-op method default (GET running, POST start/stop); headers merged over `default_headers` (Bearer); if `body` present and no `content-type` header, add `application/json`; `list_loaded` extracts via `running.path` (dotted) + optional `id_field` (elements are strings when absent).
- `RestBackend::tiiny(...)` (Task 1) is re-expressed as `from_spec` with the built-in Tiiny endpoints.
- Changes: `PoolRegistry::from_registry(registry: &engine::Registry, protocols: &HashMap<String, ProtocolSpec>) -> Result<PoolRegistry, EngineError>` — for each `management_url` inferencer, resolve `inf.management_protocol.as_deref().unwrap_or("tiiny")`: `"tiiny"`→tiiny preset; `"ollama"`→Task 4 (stub/return error until Task 4 lands, or land Task 4 first — see ordering note); else look up in `protocols`; **unknown name → `EngineError`** (fail fast at startup). `build_state` calls `engine::load_management_protocols()` and passes the map.

- [ ] **Step 1: Write failing fixture-driven tests**

Create `woollama-server/tests/pool_protocols.rs` (`mod common;`). Test A (**custom rest**): `spawn_rest` with a body-based config (`IdLoc::Body{field:"model"}`, `RunningShape::Objects{field:"data", id_key:"id"}`, a custom route + a custom header). Build an `engine::Registry` (via `insert`) with an inferencer whose `management_protocol="custom"`, and a `protocols` map holding a matching `ProtocolSpec::Rest` pointed at the mock's `base_url` (`{base}` = base_url). Build a manager via `from_registry`, call `ensure_loaded("m1")`, then assert through `MockHandle::requests()` that the backend issued the configured start (right method/path/body/header) and that `snapshot()`/`list_loaded` reflect the mock's loaded set. Test B (**back-compat**): an inferencer with `management_url` and no `management_protocol`, `spawn_rest` in tiiny shape → `from_registry` builds the tiiny preset and `ensure_loaded` works. Test C (**unknown protocol**): `management_protocol="nope"` not in the map → `from_registry` returns `Err`.

- [ ] **Step 2: Run to verify failure**

Run: `~/.cargo/bin/cargo test -p woollama-server --test pool_protocols`
Expected: FAIL — `from_spec`/the new `from_registry` signature don't exist.

- [ ] **Step 3: Implement `from_spec` + templating**

Implement `RestBackend::from_spec` (templating, method defaults, header merge, content-type default, `running.path`/`id_field` extraction — a small dotted-path getter + optional element field). Re-express `RestBackend::tiiny` via `from_spec`.

- [ ] **Step 4: Resolution in `from_registry` + `build_state`**

Change `from_registry` to take `&protocols` and resolve the protocol name → backend (tiiny preset / config rest / unknown→Err). Update `build_state` (`lib.rs`) to `engine::load_management_protocols()?` and pass it; surface a load error the same way other startup config errors are handled.

- [ ] **Step 5: Gate**

Run: `~/.cargo/bin/cargo test -p woollama-engine -p woollama-server --features test-fixtures` then clippy.
Expected: new tests + all existing green; clippy clean.

- [ ] **Step 6: Commit**

```bash
git add woollama-server/src/pool.rs woollama-server/src/lib.rs woollama-server/tests/pool_protocols.rs
git commit   # "feat(server): config-defined rest protocols + management_protocol resolution" + trailers
```

---

### Task 4: `OllamaBackend` + wire `kind = "ollama"`

**Files:**
- Modify: `woollama-server/src/pool.rs` (add `OllamaBackend`; resolve `"ollama"` + `ProtocolSpec::Ollama`)
- Test: `woollama-server/tests/pool_ollama.rs` (new) — uses `spawn_ollama`

**Interfaces:**
- Produces: `pub struct OllamaBackend { client: reqwest::Client, base_url: String, keep_alive: Option<String> }` + `DeviceBackend` impl: `list_loaded` = `GET {base}/api/ps` → collect `.models[].name` (String); `load(id)` = `POST {base}/api/generate` `{"model": id, "keep_alive": <keep_alive or omit>}`; `unload(id)` = `POST {base}/api/generate` `{"model": id, "keep_alive": 0}`. Non-2xx / transport → `PoolError::Device`.
- Resolution: in `from_registry`, `"ollama"` (built-in, no config) and `ProtocolSpec::Ollama{keep_alive}` → `OllamaBackend`.

- [ ] **Step 1: Write failing fixture-driven tests**

Create `woollama-server/tests/pool_ollama.rs` (`mod common;`, `spawn_ollama`). An inferencer with `management_protocol="ollama"` (built-in) → `from_registry` → manager; `ensure_loaded("qwen3:14b")` issues the warm-up `/api/generate` (assert via `requests_to("/api/generate")` the body has the model + keep_alive) and `list_loaded` reflects `/api/ps`; force an eviction (`pool_max=1`, load a second) → the victim gets `/api/generate` with `keep_alive:0`. Also a config `[management_protocols.x] kind="ollama"` path.

- [ ] **Step 2: Run to verify failure**

Run: `~/.cargo/bin/cargo test -p woollama-server --test pool_ollama`
Expected: FAIL — `OllamaBackend` doesn't exist.

- [ ] **Step 3: Implement `OllamaBackend` + resolution**

Implement the struct + `DeviceBackend` impl + the resolution arms.

- [ ] **Step 4: Gate**

Run: `~/.cargo/bin/cargo test -p woollama-engine -p woollama-server --features test-fixtures` then clippy.
Expected: all PASS; clippy clean.

- [ ] **Step 5: Commit**

```bash
git add woollama-server/src/pool.rs woollama-server/tests/pool_ollama.rs
git commit   # "feat(server): ollama management backend (auto-load + keep_alive)" + trailers
```

**Ordering note for the executor:** Task 3's `from_registry` resolves `"ollama"`. Either land Task 4's `OllamaBackend` first and have Task 3 call it, or in Task 3 make the `"ollama"` arm a temporary `EngineError("ollama backend not yet implemented")` and replace it in Task 4. Prefer the temporary-error approach so each task ships green independently; the Task 4 review confirms the arm is real.

---

## Self-Review

**Spec coverage:** `DeviceBackend` seam (Task 1) ✓; `rest` config-defined (Task 3) ✓; `tiiny` preset + back-compat default (Tasks 1+3) ✓; `ollama` adapter (Task 4) ✓; `management_protocol` field + `[management_protocols]` parsing (Task 2) ✓; unknown-protocol startup error (Task 3) ✓; resolver/gate/eviction untouched (all tasks preserve them) ✓; conformance untouched (Task 2 leaves `inferencer_to_json`) ✓. Non-goals (multi-step endpoints, response-field success, retries, config-driven ollama bodies, Python parity) are respected — none introduced.

**Placeholder scan:** none — each step names concrete types/functions, the fixture API, and the exact gate commands. Task 1 bodies are "move verbatim"; the readiness poll + eviction-race fix are explicitly preserved.

**Type consistency:** `DeviceBackend`/`RestBackend`/`OllamaBackend`/`PoolError` (Tasks 1/3/4); `EndpointSpec`/`ProtocolSpec`/`load_management_protocols` (Task 2 → consumed 3/4); `RestBackend::{tiiny, from_spec}` (Task 1 → generalized Task 3); `from_registry(registry, protocols) -> Result<…>` signature is introduced in Task 3 and its caller (`build_state`) updated in the same task; `Inferencer.management_protocol` (Task 2 → read in Task 3's resolution).
