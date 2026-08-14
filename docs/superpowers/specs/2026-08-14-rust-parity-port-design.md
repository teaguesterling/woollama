# Rust Router Parity Port (to Python v0.10.0) — Design

**Date:** 2026-08-14 · **Branch:** `rust-port` · **Status:** implemented

## Goal

Bring `woollamad` (the `woollama-server` crate — the canonical Rust router) to
feature parity with the Python reference at **v0.10.0**. The router port itself
shipped (slices 0–9 in `docs/rust-router-port.md`); this closes the three
features that landed in Python *after* the cutover and are absent in Rust:

1. `POST /v1/images/generations` passthrough (Python v0.9.0)
2. `POST /v1/embeddings` passthrough (Python v0.9.0)
3. The **model-pooling stack** (Python v0.10.0): virtual-model resolution +
   on-demand load/evict + per-model request queuing/backpressure, for inferencers
   that declare a `management_url`.

End state: `woollamad == python -m woollama` at v0.10.0 for these paths, verified
by the differential oracle.

## Motivation

The Rust server is the product; the Python server is the reference oracle. Every
release where a feature lives only in Python widens the gap the oracle is meant
to close. The pooling design (`docs/superpowers/specs/2026-08-13-model-pooling-design.md`)
was written anticipating this port — "the pure Resolver/eviction logic is a
migration candidate… mirrorable into the Rust oracle later" — so the port is the
planned completion of that design, not new scope.

## Discovered constraint (drives slice 1)

The Rust chat passthrough resolves the provider with the free function
`engine::get_inferencer(provider)` (`woollama-server/src/lib.rs:840`), which
returns a **built-in** `Inferencer` and **ignores `state.inferencers`** (the
config-merged `Registry`). Consequences:

- Config-defined inferencers (e.g. a `device` block in `inferencers.toml`) are
  **not visible to the chat passthrough today** — a latent parity gap independent
  of pooling.
- Any new config field (`management_url`, `virtual`, …) cannot reach the handler
  until resolution routes through `state.inferencers`.

So the first structural change is to route passthrough resolution through the
config `Registry` (making `Registry::resolve` public — currently
`woollama-engine/src/lib.rs:531`). This is a prerequisite for pooling **and**
fixes the config-inferencer passthrough gap.

## Architecture

Placement mirrors the Python split, adapted to the Rust workspace:

- **Pure decision logic → `woollama-engine`** (the reusable heart): the inferencer
  fields and the `resolver` (resolve + eviction pick). Pure, unit-testable, and
  pinnable by the Rust conformance suite against the Python `resolver.py`.
- **Stateful runtime → `woollama-server`**: the `DeviceModelManager` (loaded-model
  state + device I/O) and `Gate` (per-model semaphore + backpressure). Live I/O,
  never in the engine.
- **HTTP surface → `woollama-server`**: images/embeddings handlers, the
  passthrough refactor, error→HTTP mapping.

No channel-actor is used (the Python asyncio queue-marshaling pattern is
unnecessary in Rust and has no template here); the manager is modeled on the
existing `ManagedAgents` (`Mutex`-guarded state + `reqwest`, on-demand
`ensure_*` under the lock) plus `tokio::sync::Semaphore` for the gate.

### 1. Engine — inferencer fields + URL builders (`woollama-engine/src/lib.rs`)

Add to `pub struct Inferencer` (lib.rs:79–94), mirroring the Python dataclass:

- `management_url: Option<String>`, `parallel: u32` (default 1),
  `pool_max: Option<u32>`, `queue_max: Option<u32>`, `queue_timeout: f64`
  (default 30.0), `virtual_models: BTreeMap<String, String>` (`virtual` is a Rust
  keyword — field named `virtual_models`, TOML key stays `virtual`).
- `impl Inferencer` (165–176): add `images_url()` → `{base_url}/images/generations`
  and `embeddings_url()` → `{base_url}/embeddings`, alongside `chat_url()`.
- Every struct literal must gain the new fields (compiler-enforced): the built-in
  constructors in `get_inferencer` (99–107, 114–122, 124–132), `Registry::add`
  (503–507), and `build_config_registry` (474).
- **Config merge** in `build_config_registry` (423–477): read the new keys with
  the existing `spec.get(...)` idioms (follow `discover`/`str_list` at 462–473),
  field-by-field inherit-on-unset like the other keys. `load_inferencers_toml`
  (382–418) needs **no change** (it passes keys through verbatim).
- `inferencer_to_json` (479–486) is left at its current 4 fields → the JSON
  surface is unchanged → **conformance suite untouched**.

### 2. Engine — pure `resolver` module (`woollama-engine`, new)

Direct port of `resolver.py`:

- `struct PoolEntry { model_id: String, in_flight: u32, queued: u32, last_used: f64 }`
- `resolve(bare: &str, virtual_models: &BTreeMap<String,String>, loaded: &[String], default: Option<&str>) -> Result<String, ResolveError>` — `default` → `loaded[0]` if any, else configured `default`, else `ResolveError`; alias hit → target; else `bare` unchanged.
- `needs_eviction(loaded: &HashSet<String>, target: &str, pool_max: Option<u32>) -> bool`
- `pick_eviction(entries: &[PoolEntry]) -> Option<String>` — LRU among idle
  (`in_flight==0 && queued==0`); `None` if none idle.

Unit-tested against the same cases as `tests/test_resolver.py`.

### 3. Server — images/embeddings handlers (`woollama-server/src/lib.rs`)

Two handlers siblings to the passthrough non-stream path (839–877):

- Split provider, resolve via `state.inferencers` (see slice 5), `inf.auth_headers()`,
  rewrite `model` to bare, `forward_post(inf.images_url()/embeddings_url(), &fwd, &headers, timeout)`,
  `relay_json`. Always non-streaming (images uses a generous timeout like the
  Python 300s; embeddings the default). Unknown provider → 400, matching chat.
- Register `POST /v1/images/generations` and `POST /v1/embeddings` (near lib.rs:214).

### 4. Server — `pool` module: `DeviceModelManager` + `Gate` (new)

**`DeviceModelManager`** — one per management-capable inferencer:

- `struct DeviceModelManager { url: String, headers: HashMap<String,String>, state: Mutex<PoolState>, client cfg, poll_interval, load_timeout, retry_after, clock }` where `PoolState` holds `entries: HashMap<String, Entry>` (`Entry { loaded, in_flight, queued, last_used }`) and load serialization.
- `ensure_loaded(&self, real_id, pool_max)` under the lock (like `ManagedAgents::ensure_agent`, managed_agents.rs:180): fast-path if loaded; else reconcile against `GET {url}/api/v1/models/running` (parse top-level `running` list), `POST .../{id}/start`, poll running until ready or `load_timeout`. Evict-to-fit via `resolver::needs_eviction`/`pick_eviction`; **flip victim `loaded=false` before the stop** and only drop the entry if still idle after (the eviction-race fix from the Python port, `pool.py` — port it, don't re-derive).
- Sync counter ops `acquire`/`release`/`enqueue`/`dequeue`/`queued`, `snapshot()` (loaded ids MRU-first). Errors → `DeviceError` (→502).

**`Gate`** — per-model concurrency + backpressure:

- Per-`real_id` `tokio::sync::Semaphore` (size `parallel`); `queue_max`/`queue_timeout`.
- `enter(real_id) -> Result<Slot, Backpressure>`: early-reject if `queued >= queue_max`; `enqueue`; `ensure_loaded`; acquire a permit within `queue_timeout` (`tokio::time::timeout`) else `Backpressure`; `dequeue`→`acquire` handoff with no `.await` between (the non-idle invariant). `Slot` releases the permit + decrements in-flight on drop (idempotent). Streaming holds the `Slot` for the stream's lifetime.

`PoolRegistry` = `HashMap<inferencer_name, (Arc<DeviceModelManager>, Gate)>`, built in `build_state` (lib.rs:85–160) by iterating `inferencers.list()` for `management_url` entries; added as `pools: Arc<PoolRegistry>` on `AppState` (56–75).

### 5. Server — passthrough refactor + wiring

- Make `Registry::resolve` (engine lib.rs:531) `pub`; route the chat passthrough
  (lib.rs:839) through `state.inferencers` instead of `engine::get_inferencer`.
  For an inferencer **with** `management_url`: resolve virtual model
  (`resolver::resolve` with `manager.snapshot()`), `gate.enter(real)`, dispatch
  (existing `forward_post`/`passthrough_stream`), release. **Without**: today's
  stateless passthrough, unchanged.
- `DeviceError` → 502 via `EngineError::new(msg, "server_error", 502)` +
  `engine_err_response`. `Backpressure` → 503 + `Retry-After`: a small dedicated
  builder (`Response::builder().status(503).header("retry-after", secs)…`, the
  pattern already used for the SSE content-type at lib.rs:913–915) — the generic
  error helpers can't set headers.

## Data flow (`POST /v1/chat/completions`, pooled)

1. `chat_completions` → passthrough branch → resolve provider via `state.inferencers`.
2. If `inf.management_url` set → `real = resolver::resolve(bare, inf.virtual_models, manager.snapshot(), inf.virtual_models.get("default"))`.
3. `slot = gate.enter(real).await?` — enqueue → `ensure_loaded` (evict-to-fit) → semaphore permit (queue/timeout) → in-flight.
4. Rewrite body model to `real`; `forward_post(inf.chat_url())` / `passthrough_stream` (slot held across the stream).
5. Drop `slot` → release permit + decrement in-flight.

## Error handling

| Situation | Behavior |
|---|---|
| Requested model not loaded | `ensure_loaded` loads it; never a bare 503 |
| Queue saturated / `queue_timeout` | `503` + `Retry-After` |
| `start` fails / device unreachable | `502` with the device message |
| Capacity full, no idle to evict | `503` + `Retry-After` |
| Eviction candidate becomes in-use mid-decision | re-check; never evict (port the Python fix) |
| `ResolveError` (default, none loaded, no fallback) | `400` |

## Config surface

No new user-facing config — the keys already exist on `[inferencers.<name>]`
(Python v0.10.0): `management_url`, `parallel`, `pool_max`, `queue_max`,
`queue_timeout`, `virtual`. This port makes `woollamad` honor them. An inferencer
without `management_url` behaves exactly as today.

## Testing strategy

- **Engine (pure):** unit tests for `resolver` (resolve/eviction) mirroring
  `tests/test_resolver.py`; the config-merge picks up the new fields. Add the
  new fields to any conformance fixture only if it compares field sets (it does
  not today — `inferencer_to_json` unchanged).
- **Server (integration, mock device):** a mock HTTP upstream exposing
  `/api/v1/models/{running,start,stop}` + `/v1/chat/completions` (mirroring the
  Python `FakeDevice`): load-on-demand, concurrent-load de-dup, evict-LRU-idle,
  no-evict-while-serving, semaphore serialization, `queue_max`/`queue_timeout`
  backpressure (503 + Retry-After), streaming slot lifetime. Images/embeddings:
  mock-upstream relay + unknown-provider 400.
- **Differential oracle:** the HTTP/SDK live tests already repoint at `woollamad`;
  images/embeddings/passthrough parity ride the same oracle. Pooling's live path
  is device-specific (env-gated), consistent with the Python plan.

## Slice ordering (risk-front-loaded; each ships green)

1. **Passthrough → config registry** (`Registry::resolve` public; route chat
   passthrough through `state.inferencers`). The load-bearing refactor; also fixes
   the config-inferencer passthrough gap. Gate: existing passthrough tests still
   green + a config-inferencer passthrough test.
2. **Engine inferencer fields + URL builders + config merge.** Mechanical but
   compiler-wide (every struct literal). Gate: config-merge unit test; conformance
   unchanged.
3. **Images + embeddings handlers + routes.** Small, non-streaming. Gate:
   mock-upstream relay + 400.
4. **Engine `resolver`.** Pure; unit tests mirror `test_resolver.py`.
5. **`pool` module: `DeviceModelManager`** (load/evict + counters, incl. the
   eviction-race fix). Gate: mock-device tests.
6. **`Gate`** (semaphore + backpressure) + **wire into passthrough** (pooled path,
   503+Retry-After, 502) + `PoolRegistry` on `AppState`. Gate: mock-device
   serialization/backpressure/streaming-slot tests.

## Scope / non-goals

- **In:** parity to Python v0.10.0 for images, embeddings, and pooling on
  `/v1/chat/completions`.
- **Out (matches Python):** pooling on the `/v1/responses` path (deferred there
  too); `device/auto` by request shape; multi-backend load balancing; the
  managed-agents live wire reconciliation (separate, pre-existing).
- **Out:** publishing `woollamad` to crates.io / lackpy re-pin (packaging, not
  this feature).

## Key attachment points (from the codebase map)

- Passthrough handler: `woollama-server/src/lib.rs:809` (`chat_completions`),
  passthrough 839–877, stream 901–917; `forward_post` 274–291, `relay_json`
  293–297.
- Engine `Inferencer`/`Registry`: `woollama-engine/src/lib.rs:79–94` (struct),
  165–176 (urls), 98–139 (built-ins), 423–477 (`build_config_registry`), 491–534
  (`Registry`, `resolve` at 531), 479–486 (`inferencer_to_json`).
- Config parse: `load_inferencers_toml` engine lib.rs:382–418 (no change).
- Manager template: `woollama-server/src/managed_agents.rs:83–96` (state),
  180 (`ensure_agent`), 116–148 (HTTP).
- `AppState`: `woollama-server/src/lib.rs:56–75`; `build_state` 85–160.
- Error→HTTP: `EngineError` engine lib.rs:29–45; `engine_err_response` lib.rs:264–272,
  `err_response` 260–262; header-setting pattern 913–915.
