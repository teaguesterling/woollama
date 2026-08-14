# Pluggable Device Management Protocols — Design

**Date:** 2026-08-14 · **Branch:** `mgmt-protocols` · **Status:** design, pre-implementation

## Goal

Make woollamad's model-pooling backend **pluggable**: instead of the hardcoded
Tiiny/Houmo `:8800` API, an inferencer's `management_url` is driven by a named
**management protocol** — either a built-in adapter (`tiiny`, `ollama`) or one
**defined entirely in config** (a generic `rest` protocol: URLs, methods,
bodies, headers, and the JSON path to the loaded-model list). Adding a new
list/start/stop-style device becomes a config edit, no recompile.

## Motivation

The pooling stack (virtual models + gate + queue + eviction) is already
backend-agnostic; the *only* device-specific code is three request builders +
one JSON parse in `DeviceModelManager` (`woollama-server/src/pool.rs`). Today
those hardcode the Tiiny protocol, so `management_url` silently *means* "speaks
the Tiiny `:8800` API." There is no industry-standard model-management protocol
(Ollama auto-loads with `keep_alive`; LM Studio has load/unload over REST with a
body; vLLM is one-model-per-instance), so pluggability must cover both a
config-parameterized REST family and a couple of semantically-distinct built-ins.

## Scope

- **woollamad (Rust) first.** The Python reference keeps its current behavior
  (Tiiny, now the built-in default). The differential oracle stays valid because
  both default to `tiiny`. Python parity is a follow-on.
- **v1 includes:** the `DeviceBackend` seam, the `rest` kind (config-defined
  protocols), the `tiiny` built-in preset (a `rest` preset), the `ollama`
  built-in adapter, and back-compat (`management_url` + no `management_protocol`
  ⇒ `tiiny`).

## Architecture

### The seam: `DeviceBackend`

A trait in `woollama-server` — the only code that knows a backend's wire
protocol:

```rust
#[async_trait]
pub trait DeviceBackend: Send + Sync {
    /// Ids of models currently loaded on the backend.
    async fn list_loaded(&self) -> Result<Vec<String>, DeviceError>;
    /// Load `id`; idempotent; returns only once the model is ready to serve.
    async fn load(&self, id: &str) -> Result<(), DeviceError>;
    /// Unload/evict `id`.
    async fn unload(&self, id: &str) -> Result<(), DeviceError>;
}
```

`DeviceModelManager` holds an `Arc<dyn DeviceBackend>` instead of the hardcoded
`url`/`headers`. Its internals change minimally:
- `ensure_loaded`: reconcile via `backend.list_loaded()`, then `backend.load(real_id)`
  (the backend owns readiness — for `rest` that's POST-then-poll; for `ollama`
  a warm-up call). Eviction calls `backend.unload(victim)`.
- The load-serialization lock, the `std::sync::Mutex<PoolState>` counters, the
  eviction-race fix, `snapshot`, and the whole `Gate`/`Slot`/resolver/queue layer
  are **unchanged** — they never touched HTTP.

`DeviceError` is unchanged (→ HTTP 502).

### Two backend kinds

**`RestBackend`** — the config-parameterized list/start/stop family. Holds the
inferencer's `management_url` as `{base}`, the default auth headers (Bearer from
`api_key_env`), the manager's `poll_interval`/`load_timeout`, and three resolved
endpoint specs (running/start/stop). Behavior:
- `list_loaded`: issue the `running` request; extract ids from the response JSON
  via `path` (+ optional `id_field`).
- `load(id)`: issue the `start` request (with `{id}` substituted); then re-poll
  `list_loaded` until `id` appears or `load_timeout` (generic readiness).
- `unload(id)`: issue the `stop` request.
- Success = any `2xx`; non-2xx or transport error → `DeviceError`.

**`OllamaBackend`** — auto-load semantics; not expressible as `rest`. Holds the
Ollama base URL + a `keep_alive` TTL. Behavior:
- `list_loaded`: `GET {base}/api/ps` → `.models[].name`.
- `load(id)`: `POST {base}/api/generate` `{"model": id, "keep_alive": <ttl>}`
  (empty prompt — Ollama loads the model without generating).
- `unload(id)`: `POST {base}/api/generate` `{"model": id, "keep_alive": 0}`.

### Protocol resolution & presets

At startup, `PoolRegistry::from_registry` resolves each management-capable
inferencer's `management_protocol` name:
1. Built-in presets first: `tiiny` (a `RestBackend` preset with the Tiiny URLs
   baked in) and `ollama` (an `OllamaBackend`).
2. Then config-defined `[management_protocols.<name>]` blocks.
3. Unknown name → a clear startup config error.
4. **Back-compat:** `management_url` present with no `management_protocol` ⇒
   `tiiny`. Existing configs are unchanged.

`{base}` in every URL/body/header value is the inferencer's `management_url`;
`{id}` is the model id. `${VAR}` env-expansion applies to config-defined protocol
values (same as the rest of `inferencers.toml`).

## Config surface (`inferencers.toml`)

Per-inferencer selector (additive; default `tiiny`):

```toml
[inferencers.tiiny]
management_url      = "${TIINY_URL}:8800"
management_protocol = "tiiny"     # optional; omitted + management_url present ⇒ "tiiny"
```

Config-defined protocols — a new top-level `[management_protocols.<name>]`
section. For `kind = "rest"`, one nested table per operation:

```toml
[management_protocols.mybox]
kind = "rest"
  [management_protocols.mybox.endpoints.running]
  url  = "{base}/api/v1/models/running"
  path = "running"                  # dotted path to the loaded-id array
  # id_field = "id"                  # optional: pluck this field from each element (omit ⇒ elements are strings)
  [management_protocols.mybox.endpoints.start]
  url  = "{base}/api/v1/models/{id}/start"
  [management_protocols.mybox.endpoints.stop]
  url  = "{base}/api/v1/models/{id}/stop"

# body-based backend (LM Studio-shaped) with its own content-type:
[management_protocols.lmstudio]
kind = "rest"
  [management_protocols.lmstudio.endpoints.running]
  url = "{base}/api/v0/models"
  path = "data"
  id_field = "id"
  [management_protocols.lmstudio.endpoints.start]
  url     = "{base}/api/v0/models/load"
  body    = '{"model": "{id}"}'
  headers = { "Content-Type" = "application/json" }
  [management_protocols.lmstudio.endpoints.stop]
  url  = "{base}/api/v0/models/unload"
  body = '{"model": "{id}"}'

# custom (non-Bearer) auth:
[management_protocols.vendorbox]
kind = "rest"
  [management_protocols.vendorbox.endpoints.running]
  url = "{base}/models"
  path = "loaded"
  headers = { "X-API-Key" = "${VENDOR_KEY}" }
  [management_protocols.vendorbox.endpoints.start]
  url = "{base}/models/{id}/load"
  headers = { "X-API-Key" = "${VENDOR_KEY}" }
  [management_protocols.vendorbox.endpoints.stop]
  url = "{base}/models/{id}/unload"
  headers = { "X-API-Key" = "${VENDOR_KEY}" }

# ollama built-in kind:
[management_protocols.local-ollama]
kind = "ollama"
# keep_alive = "30m"   # optional; passed to load; unload uses 0
```

### Per-endpoint table fields (`kind = "rest"`)

Each `[...endpoints.<op>]` (`op` ∈ `running` | `start` | `stop`):
- `url` — **required**; `{base}`/`{id}` substituted.
- `method` — optional; default `GET` for `running`, `POST` for `start`/`stop`.
- `body` — optional; a **raw string** sent verbatim after `{base}`/`{id}`
  substitution. Omit ⇒ no body.
- `headers` — optional; a map of header → value (env-expanded, `{base}`/`{id}`
  substituted), **merged over** the default `Authorization: Bearer <api_key_env>`
  (a header here overrides the default).
- `running` only: `path` (**required** — dotted path to the array) + optional
  `id_field`.

Rules: default `Authorization: Bearer <api_key_env>` on every op unless
`headers` overrides it (absent `api_key_env` ⇒ no Bearer); if a `body` is present
and no `Content-Type` header is set, default `application/json`; a `rest`
protocol MUST define all three endpoints (`running`, `start`, `stop`).

## Data flow

Unchanged request path. The only new work is at **startup** in
`PoolRegistry::from_registry`: resolve the protocol name → construct the
`DeviceBackend` → build the `DeviceModelManager` around it. At request time the
pooled path is identical to today; `ensure_loaded`/evict just call the trait.

## Where it lives

- `DeviceBackend` trait + `RestBackend` + `OllamaBackend` + the `tiiny` preset:
  `woollama-server/src/pool.rs` (or a new `woollama-server/src/backend.rs`
  it re-exports — decide during planning to keep files focused).
- `[management_protocols.<name>]` parsing: the config layer that already reads
  `inferencers.toml` (the engine's `load_inferencers_toml` seam), surfaced as
  typed `ProtocolSpec` data the server consumes. Kept as inert data in the
  engine (no I/O); the server turns specs into `DeviceBackend`s. (This keeps the
  Python follow-on symmetric.)

## Error handling

| Situation | Behavior |
|---|---|
| Unknown `management_protocol` name | startup config error naming the inferencer + the missing protocol |
| `[management_protocols.x]` malformed / missing an endpoint / bad `kind` | startup config error with the offending key |
| `rest` op returns non-2xx / transport error | `DeviceError` → 502 (as today) |
| `load` readiness times out (id never appears in `running`) | `DeviceError` (as today) |
| `running` response JSON lacks `path` / wrong shape | `DeviceError` (surfaced clearly) |

## Testing strategy

Rust integration tests against mock `axum` upstreams (mirroring the existing
`pool_manager.rs`/`pool_gate.rs` harness):
- **Custom-`rest` protocol:** a mock device with *different* URLs, a `POST` body,
  a nested `path`+`id_field`, and a custom auth header — driven via a
  config-defined `[management_protocols.x]` — proves the config parameterization
  end-to-end (list/load/poll-until-ready/unload).
- **`ollama` adapter:** a mock serving `GET /api/ps` and `POST /api/generate`,
  asserting `load` warms up (keep_alive) and `unload` sends keep_alive=0.
- **Back-compat:** the existing Tiiny-shaped pool tests pass unchanged via the
  default `tiiny` preset (no config change).
- **Config errors:** unknown protocol name and a malformed protocol block each
  produce a clear startup error.

Gate per the repo convention: `cargo test -p woollama-engine -p woollama-server
--features test-fixtures` + `cargo clippy … -D warnings`.

## Scope / non-goals

- **In:** the seam; `rest` (config-defined, method/body/headers/path); `tiiny`
  preset; `ollama` adapter; back-compat default.
- **Out (v1):** Python parity (follow-on); **multi-step per-op sequences**
  (`endpoints.start` stays a single table — an array-of-tables is a
  forward-compatible later extension only if a device needs it); response-body
  *success* checks (success is `2xx`); retries / pagination / auth-flow
  negotiation; making `ollama`'s methods/bodies config-driven (its adapter is
  baked). Surfacing the new inferencer field in `inferencer_to_json` (leave it —
  conformance untouched).

## Key attachment points

- Backend seam target: `woollama-server/src/pool.rs` — `DeviceModelManager`
  (the `list_loaded`/`start`/`stop` HTTP methods become the `tiiny` `RestBackend`),
  `PoolError`/`DeviceError`, `PoolRegistry::from_registry` (protocol resolution +
  backend construction).
- Config seam: the engine's `inferencers.toml` loader (`load_inferencers_toml` /
  the config module) — add `[management_protocols]` parsing beside inferencers;
  `Inferencer` already carries `management_url` (add `management_protocol:
  Option<String>`).
- Unchanged: `woollama-engine/src/resolver.rs`, the `Gate`/`Slot`, the eviction
  policy, and the pooled passthrough wiring in `woollama-server/src/lib.rs`.
