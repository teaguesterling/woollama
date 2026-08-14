# Model Pooling, Request Queuing & On-Demand Loading — Design

**Date:** 2026-08-13 · **Branch:** `model-pooling` · **Status:** design, pre-implementation

## Goal

Make woollama **device-aware**: instead of a stateless passthrough that fails when
the requested model isn't loaded, woollama should load models on demand, serialize
and queue requests around the backend's real concurrency limits, and expose stable
**virtual model names** that resolve to whatever's appropriate — without the caller
ever seeing a 503-for-not-loaded or a hang.

## Motivation (the pains this removes)

Observed against a management-capable device:

- **Not-loaded → hard failure.** The device serves a model only while it's loaded
  (`:8800/api/v1/models/{id}/start|stop`); other ids return 503/hang and it does
  **not** auto-load. A client that names a model that isn't loaded (e.g. a Hermes
  config pinned to `Qwen/Qwen3-Coder-30B-A3B-Instruct`) just fails.
- **`--parallel 1`.** The device's chat model serializes requests; concurrent callers
  (the desktop app, an OpenCode agent, Hermes) contend, and naive clients **retry-loop
  into a wedge** rather than back off.
- **State churn.** Which model is loaded changes out from under callers; there's no
  stable name that "just works," and no way to pool requests for a model before it's
  swapped away.

woollama is the right place to fix this: it already sits between every client and the
device as the OpenAI/MCP router.

## Architecture: three units, hybrid placement

Per the mid-migration Python→Rust split (`inferencers.py` is the Python oracle for the
Rust `woollama-engine`), the **stateful runtime lives in the Python server layer**
(it is live I/O, not "server-free" logic), while the **pure decision logic** sits at
the model-resolution seam and is mirrorable into the Rust core later.

### 1. `DeviceModelManager` — the loaded-model actor (stateful, Python server)

One long-lived async actor per management-capable inferencer, modeled on the existing
`manager.ServerManager` (`src/woollama/manager.py:49-132`: a dedicated asyncio task +
`asyncio.Queue` + `_ready` Event + `start`/`stop`).

**Owns:** the set of models currently loaded on the device and their per-model state
(loaded/loading, in-flight count, queue depth, last-used).

**Interface:**
- `ensure_loaded(real_id) -> awaitable`: if loaded, return immediately; else acquire
  the load-lock, **evict-to-fit** per policy (see Eviction), `POST management_url/api/v1/models/{real_id}/start`,
  poll `/api/v1/models/running` until ready, then return. Concurrent `ensure_loaded`
  calls for the same id share one load; loads never overlap.
- `acquire(real_id) / release(real_id)`: ref-count in-flight requests (protects a
  model from eviction while serving).
- `snapshot() -> pool state`: read-only view for the Resolver.

**Depends on:** an httpx client to `management_url` (the device `:8800`), and the
device's `/api/v1/models/{running,start,stop}` endpoints. **This actor never enters the
Rust core** — it is live device I/O.

### 2. `Gate` — the request queue (stateful, Python server)

A single gate both upstream entry points call, so it is **not passthrough-only**:
- Python `_passthrough` (`src/woollama/router.py:655-677`, wrap the httpx POST).
- The core-bound `/v1/responses` path (`router.py:538` → `woollama-engine`), wired
  the same way so both paths share the limiter.

**Per request:** `ensure_loaded(real_id)` → acquire a **per-model concurrency slot**
(an `asyncio.Semaphore`; size from `parallel`, default **1** to match the device) →
dispatch the existing upstream call → release. FIFO within a model.

**Backpressure over failure:** if the per-model queue exceeds `queue_max` or a request
waits past `queue_timeout`, return **`503` with `Retry-After`** — a clean signal so
clients back off instead of retry-looping into a wedge. In-flight and queued requests
are never dropped by eviction.

### 3. Resolver — virtual-model resolution (pure logic, resolution seam)

A pure function `resolve(provider, requested_id, pool_snapshot) -> real_id`, inserted at
the existing `split("/")→bare` step — Python `_passthrough` (`router.py:661-663`) and
Rust `build_request` (`woollama-engine/src/lib.rs:210,247`). Because it's pure it is
unit-testable server-free and mirrorable into the Rust oracle later.

**Resolves:**
- `device/<real-id>` → `<real-id>` (today's behavior, unchanged).
- `device/default` → whichever model is currently loaded (from `pool_snapshot`); if none
  loaded, the inferencer's configured default.
- `device/<alias>` → the real id from the inferencer's `virtual` map (e.g.
  `coder → Qwen/Qwen3-Coder-30B-A3B-Instruct`).

## Data flow (`POST /v1/chat/completions`)

1. **Resolve** `(provider, requested_id)` (`router.py:239-240`), then `real_id =
   Resolver.resolve(...)` using `DeviceModelManager.snapshot()`.
2. **Ensure loaded:** `await manager.ensure_loaded(real_id)` — queues *during a load*,
   evicts an idle model to fit if needed.
3. **Acquire slot:** `Gate` acquires the per-`real_id` semaphore — queues if busy
   (`--parallel 1`); `503 + Retry-After` if the queue is saturated.
4. **Dispatch:** the existing upstream call (httpx passthrough / engine reqwest), with
   the body's `model` set to `real_id`.
5. **Release** slot + decrement in-flight ref-count.

The `/v1/responses` path runs the same 1-4 around the core call.

## Eviction policy (queue-aware, conservative)

- **Never evict a model that is in-flight or has a non-empty queue.** Eviction targets
  only **idle** models (no in-flight, empty queue), LRU among them.
- To load a new model at capacity (`pool_max` reached): evict the LRU *idle* model. If
  none is idle, the new request **waits** (queued behind the load) up to `queue_timeout`,
  else `503 + Retry-After`. No surprise unloads of another session's active model.
- **Anti-thrash / "pool before evict" (target behavior):** because the Gate holds
  per-model queues, a model with pending requests is protected until its queue drains —
  so requests for one model batch and complete before it's swapped away, instead of
  load→serve-1→evict→reload. MVP achieves this passively (queued ⇒ not idle ⇒ not
  evictable); a later phase can add explicit drain-before-evict scheduling and fairness.

## Config surface (additive, backward-compatible)

New optional keys on `[inferencers.<name>]`, whitelisted in `config.load_inferencers`
(`src/woollama/config.py:241-254`), threaded through the `Inferencer` dataclass
(`src/woollama/inferencers.py:34-51`) and its merge in `_registry` (`inferencers.py:116-132`),
and mirrored on the Rust `Inferencer`/`Registry` (`woollama-engine/src/lib.rs:461,496-526`)
to keep the conformance oracle honest:

- `management_url` (str) — the device management base (`:8800`); presence enables the
  `DeviceModelManager` for this inferencer. Absent ⇒ today's stateless passthrough,
  unchanged.
- `parallel` (int, default 1) — per-model concurrency slot size.
- `pool_max` (int, optional) — max concurrently-loaded models before eviction kicks
  in. **Count-based**, since the device's memory capacity isn't queryable. Unset ⇒ no
  cap and **no auto-eviction** (models accumulate; a device out-of-memory on `start`
  surfaces as `502`). Set it to enable evict-to-fit.
- `queue_max` (int) and `queue_timeout` (seconds) — backpressure limits.
- `virtual` (table) — alias → real-id map; supports the reserved `default` key.

Existing configs with none of these behave exactly as today.

## Error handling

| Situation | Behavior |
|---|---|
| Requested model not loaded | `ensure_loaded` loads it (queue during load); never a bare 503 |
| Queue saturated / `queue_timeout` exceeded | `503` + `Retry-After` (backpressure, not a wedge) |
| Load in progress | requests await the load; not errored |
| `start` fails on the device | clear `502`/error surfaced with the device's message; no hang |
| Capacity full, no idle model to evict | request queues, then `503 + Retry-After` if it can't be served |
| Device management endpoint unreachable | `502`; do not evict/thrash |
| Eviction candidate becomes in-use mid-decision | re-check ref-count; never evict in-use |

## Testing strategy

- **Resolver (pure):** unit tests for `device/<id>`, `device/default` (loaded vs none),
  aliases, unknown alias. Server-free; add to the Rust conformance oracle later.
- **Eviction policy (pure):** unit tests — evict LRU idle; refuse to evict in-use/queued;
  capacity-full-no-idle path.
- **`DeviceModelManager` + `Gate` (async):** tests against a **fake device** (a small
  aiohttp/ASGI stub for `/api/v1/models/{running,start,stop}` + `/v1/chat/completions`)
  — load-on-demand, concurrent `ensure_loaded` de-dup, semaphore serialization, queue
  backpressure (`503`+`Retry-After`), no-evict-while-serving.
- **One guarded live integration test** against a real device, opt-in (env-gated),
  mindful of `--parallel 1` and single-call-only (no retry loops).
- Follows existing patterns: `tests/test_routing.py`, `tests/test_managed_agents.py`,
  and the server-free guard `tests/test_core_is_server_free.py`.

## Scope / phasing (YAGNI)

**MVP (this branch):**
- One management-capable inferencer (`device`) via `management_url`.
- `DeviceModelManager` with `ensure_loaded` + ref-counting + evict-LRU-idle.
- `Gate` on the **passthrough path first**, per-model semaphore + backpressure, written
  so the `/v1/responses` path calls the same gate.
- Resolver: `device/default` + config aliases.
- Config keys above; fake-device tests + pure-logic tests.

**Deferred (not now):**
- Explicit drain-before-evict scheduling and cross-client fairness/priority.
- Smart `device/auto` that picks a model from request shape (images → vision, etc.).
- Load balancing across *multiple* backends / cloud fallback.
- Porting the runtime into Rust (only the pure Resolver/eviction logic is a migration
  candidate; the actor/queue stay server-layer).

## Non-goals

- Changing the behavior of inferencers **without** `management_url` (stays a stateless
  passthrough).
- Managing non-chat model types (embeddings/ASR/image) through the load/queue path —
  they have separate endpoints and are out of scope here.

## Key attachment points (from the architecture map)

- Model resolution seam: `router.py:661-663` (passthrough), `woollama-engine/src/lib.rs:210,247` (core).
- Upstream entry points to gate: `router.py:655-677` (+ `_passthrough_stream:778`), core `router.py:538/562`.
- Actor pattern to copy: `manager.ServerManager` (`manager.py:49-132`, queue at `:63`).
- Config parse/whitelist: `config.py:200-258` (keys at `:241-254`); dataclass `inferencers.py:34-51`; merge `_registry:116-132`; Rust twin `lib.rs:461,496-526`.
- Registry lookup wrapped by the pool: `inferencers.get` (`router.py:240`) / `get_inferencer` (`lib.rs:98`).
