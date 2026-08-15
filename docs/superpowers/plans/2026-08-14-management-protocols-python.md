# Pluggable Management Protocols — Python Parity Port

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the Python reference server (`src/woollama/`) to parity with the merged Rust `woollamad` for pluggable device management protocols — a `DeviceBackend` seam with built-in `tiiny`/`ollama` + config-defined `rest` protocols — so the differential oracle is honest again.

**Architecture:** Mirror the Rust design (already shipped): extract `DeviceModelManager`'s hardcoded device HTTP (`pool.py::_running/_start/_stop`) into a `DeviceBackend` protocol; `RestBackend` (config-parameterized; `tiiny` is a preset) + `OllamaBackend`; a `management_protocol` selector defaulting to `tiiny`; `[management_protocols.<name>]` config parsing. The resolver/Gate/Slot/eviction-race logic is untouched.

**Tech Stack:** Python 3.12, asyncio, httpx, pytest (`asyncio_mode="auto"`).

**Spec:** `docs/superpowers/specs/2026-08-14-management-protocols-design.md`.
**Behavior authority (the reference / oracle-in-reverse):** the merged Rust implementation — `woollama-server/src/pool.rs` (`DeviceBackend`, `RestBackend::{tiiny,from_spec}`, `OllamaBackend`, `PoolRegistry::from_registry`) and `woollama-engine/src/lib.rs` (`EndpointSpec`/`ProtocolSpec`/`load_management_protocols`, `management_protocol` field). The Python MUST match its behavior: same protocol shapes, `{base}`/`{id}` templating, case-insensitive header merge over the default Bearer, `path`+`id_field` extraction, per-op method/body/headers defaults, skip-bad-protocol-with-warning, reserved-name shadow warning, and the `ollama` `/api/ps` + `/api/generate`(keep_alive) semantics.

## Global Constraints

- **Interpreter:** `/home/teague/Projects/Tiiny/.venv/bin/python -m pytest` (a bare `python`/`pytest` lacks woollama). Lint gate: `/home/teague/Projects/Tiiny/.venv/bin/python -m ruff check .` clean (CI runs `ruff check .` hard). The whole existing suite must stay green.
- **Back-compat:** `management_url` with no `management_protocol` ⇒ `tiiny`; the existing `tests/test_pool.py` (which uses a Tiiny-shaped `FakeDevice`) stays green — the `tiiny` `RestBackend` reproduces `_running`/`_start`/`_stop` byte-for-byte.
- **Match the Rust reference** exactly for every behavior the differential oracle could observe. When a choice arises, read the Rust and mirror it.
- Generic example names only (`device`, `mybox`, `oll`) — never a product brand.
- **Commits:** conventional subject; body ends with:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MNbD4t3EqzQuXXoVZApnHu
  ```

---

### Task 1: `DeviceBackend` protocol + extract the `tiiny` `RestBackend`; manager backend-driven

Behavior-preserving refactor (`src/woollama/pool.py`). Existing `tests/test_pool.py` is the safety net.

**Files:**
- Modify: `src/woollama/pool.py`
- Modify: `tests/test_pool.py` (construct managers via the new ctor)
- Modify: `src/woollama/router.py` (lifespan `_pools` build — construct a `tiiny` `RestBackend`)

**Interfaces:**
- Produces: a `DeviceBackend` protocol (use `typing.Protocol` with three async methods, or an `abc.ABC` — match the codebase's seam style, e.g. how `conversations`/`tooling` define their provider seams):
  - `async def list_loaded(self) -> set[str]`
  - `async def load(self, real_id: str) -> None`  (idempotent; returns only when ready)
  - `async def unload(self, real_id: str) -> None`
- Produces: `class RestBackend` + `RestBackend.tiiny(management_url, *, headers, client=None, poll_interval=0.5, load_timeout=120.0, clock=time.monotonic) -> RestBackend`. Its methods = today's `_running` (→`list_loaded`), `_start` (→`load`: POST start then poll `list_loaded` until present or `load_timeout`), `_stop` (→`unload`), moved verbatim (same URLs, `DeviceError` messages, `_ok` check, headers).
- Changes: `DeviceModelManager.__init__(self, backend: DeviceBackend, *, retry_after=5.0, clock=time.monotonic)` — drops `management_url/headers/client/poll_interval/load_timeout` (they move to `RestBackend`); keeps `retry_after`, `_entries`, `_load_lock`, `_clock`. `ensure_loaded` calls `self._backend.list_loaded/load/unload` in place of `self._running/_start/_stop`; the reconcile / eviction-race fix / counters / lock are UNCHANGED. `aclose` delegates to the backend (add `async def aclose(self)` on `RestBackend` that closes its owned client).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pool.py` (or a new `tests/test_backend.py`): a test that builds `DeviceModelManager(RestBackend.tiiny(url, headers={}, ...))` against the existing `FakeDevice` and drives `ensure_loaded` — asserting the same load-on-demand behavior. (Most existing tests will need the ctor change in Step 4; this one pins the new shape.)

- [ ] **Step 2: Run to verify it fails**

Run: `/home/teague/Projects/Tiiny/.venv/bin/python -m pytest tests/test_pool.py -k backend -x -q`
Expected: FAIL — `RestBackend` doesn't exist.

- [ ] **Step 3: Implement the protocol + `RestBackend`; make the manager backend-driven**

Add `DeviceBackend`, `RestBackend` (+ `.tiiny`), move `_running`/`_start`/`_stop`/`aclose` into `RestBackend` verbatim, and rewrite `DeviceModelManager.__init__`/`ensure_loaded`/`aclose` to use `self._backend`.

- [ ] **Step 4: Update existing pool tests + the router lifespan to the new ctor**

In `tests/test_pool.py`, replace every `DeviceModelManager(url, headers=…, poll_interval=…, …)` with `DeviceModelManager(RestBackend.tiiny(url, headers=…, poll_interval=…, …), retry_after=…)`. In `src/woollama/router.py` lifespan (~line 145), `pool.DeviceModelManager(_inf.management_url, headers=_hdrs)` → `pool.DeviceModelManager(pool.RestBackend.tiiny(_inf.management_url, headers=_hdrs))`.

- [ ] **Step 5: Run the gate**

Run: `/home/teague/Projects/Tiiny/.venv/bin/python -m pytest -q` then `… -m ruff check .`
Expected: the ENTIRE suite green (proves the refactor is behavior-preserving); ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/woollama/pool.py tests/test_pool.py src/woollama/router.py
git commit   # "refactor: DeviceBackend protocol + tiiny RestBackend behind DeviceModelManager (py)" + trailers
```

---

### Task 2: `management_protocol` field + `[management_protocols]` parsing

Mirror the Rust `Inferencer.management_protocol` + `load_management_protocols`.

**Files:**
- Modify: `src/woollama/inferencers.py` (dataclass field + `_registry` merge)
- Modify: `src/woollama/config.py` (new `load_management_protocols`)
- Test: `tests/test_management_protocols.py` (new)

**Interfaces:**
- Produces: `Inferencer.management_protocol: str | None = None` (dataclass field, defaulted; threaded through the `_registry` merge next to `management_url`, i.e. `management_protocol=spec.get("management_protocol", base.management_protocol if base else None)`).
- Produces (in `config.py`, plain dataclasses/typed dicts — mirror the Rust `EndpointSpec`/`ProtocolSpec`):
  - `EndpointSpec` = `{url: str, method: str | None, body: str | None, headers: dict[str,str], path: str | None, id_field: str | None}`
  - a `ProtocolSpec` union: rest = `{"kind":"rest","running":EndpointSpec,"start":EndpointSpec,"stop":EndpointSpec}`; ollama = `{"kind":"ollama","keep_alive": str | None}`.
  - `def load_management_protocols() -> dict[str, ProtocolSpec]` — reads `[management_protocols.<name>]` from `config_dir()/inferencers.toml` (reuse the existing file-read + `_expand_env`, mirroring `load_inferencers`); `kind` ∈ {rest,ollama}; rest requires `endpoints.{running(with path),start,stop}` each with a `url`; unknown/missing → `ValueError` naming the protocol + key; missing file/absent section → `{}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_management_protocols.py`: set `WOOLLAMA_CONFIG_DIR` to a temp `inferencers.toml` with `[inferencers.dev] management_url=… management_protocol="mybox"`, a `[management_protocols.mybox] kind="rest"` (nested `[management_protocols.mybox.endpoints.running/start/stop]`, one with `${VAR}` in a header), and `[management_protocols.oll] kind="ollama"`. Assert `inferencers.get("dev").management_protocol == "mybox"`; `config.load_management_protocols()` returns the rest spec (urls/method-defaults/headers, `${VAR}` expanded) + the ollama spec. Error cases: unknown `kind`, and a rest block missing `endpoints.stop` → `ValueError`.

- [ ] **Step 2: Run to verify failure**

Run: `/home/teague/Projects/Tiiny/.venv/bin/python -m pytest tests/test_management_protocols.py -q`
Expected: FAIL — the field/function don't exist.

- [ ] **Step 3: Implement the field + the parser**

Add the dataclass field + merge; implement `load_management_protocols` per the interface (validate; name offenders; expand env).

- [ ] **Step 4: Run the gate**

Run: `/home/teague/Projects/Tiiny/.venv/bin/python -m pytest -q` then ruff.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/woollama/inferencers.py src/woollama/config.py tests/test_management_protocols.py
git commit   # "feat: management_protocol field + [management_protocols] parsing (py)" + trailers
```

---

### Task 3: config-driven `RestBackend.from_spec` + protocol resolution in the lifespan

**Files:**
- Modify: `src/woollama/pool.py` (`RestBackend.from_spec`)
- Modify: `src/woollama/router.py` (lifespan `_pools` build: resolve `management_protocol` → backend)
- Test: `tests/test_pool_protocols.py` (new)

**Interfaces:**
- Produces: `RestBackend.from_spec(base_url, *, default_headers, running, start, stop, poll_interval=0.5, load_timeout=120.0, client=None, clock=time.monotonic) -> RestBackend` where `running/start/stop` are `EndpointSpec` dicts. Templating: substitute `{base}` (base_url stripped of trailing `/`) and `{id}` in every url/body/header value; per-op method default GET(running)/POST(start,stop) honoring explicit `method`; `body` raw string sent verbatim (as `content=`); endpoint `headers` merged OVER `default_headers` **case-insensitively** (lowercase all keys, endpoint wins — match the Rust fix); if a `body` is present and no `content-type` header set → default `application/json`; `list_loaded` extracts via a dotted `path` (present-but-not-a-list → `DeviceError`; absent key → empty set, matching Rust); `id_field` plucks from object elements (else string elements); `load` = start-then-poll; `unload` = stop.
- `RestBackend.tiiny(...)` re-expressed via `from_spec` with the built-in Tiiny endpoints (existing tests stay green).
- Changes: the lifespan `_pools` loop resolves `_inf.management_protocol or "tiiny"`: `"tiiny"` → `RestBackend.tiiny(...)`; `"ollama"` → a temporary `raise`/skip until Task 4 (per the Rust ordering ruling — use a clear "not yet implemented" that Task 4 replaces, OR land Task 4 first); a config `rest` spec → `RestBackend.from_spec(...)`; **unknown name → skip that inferencer with a `log.warning` naming it (do NOT drop the other pools)**; and a one-time `log.warning` if a config `[management_protocols]` block shadows a reserved name (`tiiny`/`ollama`). Load the protocols via `config.load_management_protocols()` in the lifespan.

- [ ] **Step 1: Write failing tests (with a configurable fake backend)**

`tests/test_pool_protocols.py`: add a small configurable fake device (mirror the Rust fixture / extend `test_pool.py`'s `FakeDevice`) that can serve a **body-based** shape (running `data[].id`, load/unload by JSON body field, a custom header). Test A: build the lifespan `_pools` (or call the extracted resolver helper) with an inferencer `management_protocol="custom"` + a matching `rest` `ProtocolSpec` at the fake's URL; drive `ensure_loaded`; assert (via the fake's recorded requests) the right method/path/body/header went out and `list_loaded` reflects it. Test B (back-compat): no `management_protocol` + tiiny fake → works. Test C (unknown): `management_protocol="nope"` → that inferencer is skipped (not in `_pools`) while a sibling good inferencer IS pooled. Test D: an endpoint header keyed `authorization` (lowercase) overrides a real default Bearer → exactly one auth header, endpoint value.

- [ ] **Step 2: Run to verify failure**

Run: `/home/teague/Projects/Tiiny/.venv/bin/python -m pytest tests/test_pool_protocols.py -q`
Expected: FAIL — `from_spec`/the resolution don't exist.

- [ ] **Step 3: Implement `from_spec` + templating; re-express `tiiny` via it**

- [ ] **Step 4: Wire protocol resolution into the lifespan**

Extract the per-inferencer backend selection into a small helper (e.g. `pool.build_backend(inf, protocols) -> DeviceBackend | None`) so it's unit-testable, and call it from the lifespan; skip+warn on unknown; reserved-name warning; load `protocols` once.

- [ ] **Step 5: Gate**

Run: `/home/teague/Projects/Tiiny/.venv/bin/python -m pytest -q` then ruff.
Expected: new + existing green.

- [ ] **Step 6: Commit**

```bash
git add src/woollama/pool.py src/woollama/router.py tests/test_pool_protocols.py
git commit   # "feat: config-defined rest protocols + management_protocol resolution (py)" + trailers
```

---

### Task 4: `OllamaBackend` + wire `kind = "ollama"`

**Files:**
- Modify: `src/woollama/pool.py` (`OllamaBackend`)
- Modify: `src/woollama/router.py` (resolve `"ollama"` + `ProtocolSpec` ollama → `OllamaBackend`; remove the Task-3 temporary)
- Test: `tests/test_pool_ollama.py` (new)

**Interfaces:**
- Produces: `class OllamaBackend` with `__init__(self, base_url, *, keep_alive: str | None = None, client=None)` + the `DeviceBackend` methods: `list_loaded` = `GET {base}/api/ps` → `{m["name"] for m in json["models"]}`; `load(id)` = `POST {base}/api/generate` json `{"model": id}` plus `"keep_alive": <v>` only when `keep_alive` is set (omit when None); `unload(id)` = `POST {base}/api/generate` json `{"model": id, "keep_alive": 0}` (numeric 0). Non-2xx/transport → `DeviceError`. No readiness poll (Ollama's generate blocks until resident — add a comment noting this + that it sends no auth headers, mirroring the Rust comments).
- Resolution: `"ollama"` (built-in) and a config ollama spec → `OllamaBackend`.

- [ ] **Step 1: Write failing tests (fake ollama)**

`tests/test_pool_ollama.py`: a fake serving `/api/ps` + `/api/generate` (record model + keep_alive). An inferencer `management_protocol="ollama"` → build backend → `ensure_loaded` issues the warm-up generate (assert body model, no keep_alive for the keyless built-in) and `list_loaded` reflects `/api/ps`; force eviction (`pool_max=1`) → victim gets `/api/generate` with `keep_alive:0`; a config `kind="ollama"` with `keep_alive="5m"` forwards it in the load body.

- [ ] **Step 2: Run to verify failure**

Run: `/home/teague/Projects/Tiiny/.venv/bin/python -m pytest tests/test_pool_ollama.py -q`
Expected: FAIL — `OllamaBackend` doesn't exist.

- [ ] **Step 3: Implement + wire**

Implement `OllamaBackend`; replace the Task-3 temporary arms with real construction.

- [ ] **Step 4: Gate**

Run: `/home/teague/Projects/Tiiny/.venv/bin/python -m pytest -q` then ruff.
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/woollama/pool.py src/woollama/router.py tests/test_pool_ollama.py
git commit   # "feat: ollama management backend (py)" + trailers
```

**Ordering note:** like the Rust plan — Task 3 leaves `"ollama"` as a temporary skip/raise; Task 4 makes it real, so each task ships green independently.

## Self-Review

**Spec/reference coverage:** `DeviceBackend` seam (T1) ✓; config `rest` (T3) ✓; `tiiny` preset + back-compat default (T1+T3) ✓; `ollama` (T4) ✓; `management_protocol` field + parsing (T2) ✓; skip-bad-protocol + reserved-name warning (T3) ✓; case-insensitive header merge (T3, mirrors the Rust fix) ✓; resolver/Gate/eviction untouched (all tasks) ✓. Matches the merged Rust so the differential oracle is honest.

**Placeholder scan:** none — each step names concrete Python signatures, the interpreter/ruff commands, and the Rust reference to mirror. T1 bodies are "move verbatim."

**Type consistency:** `DeviceBackend`/`RestBackend`/`OllamaBackend`/`DeviceError`/`Backpressure` (T1/T3/T4); `EndpointSpec`/`ProtocolSpec`/`load_management_protocols` (T2 → consumed T3/T4); `RestBackend.{tiiny, from_spec}` (T1 → generalized T3); `Inferencer.management_protocol` (T2 → read in T3's resolution); `build_backend(inf, protocols)` helper (T3 → extended T4).
