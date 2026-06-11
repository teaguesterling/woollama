# Dist-split: server onto the Rust `woollama.core`

Status: **done** (see the DONE section at the end). The server runs its
inference/orchestration on the Rust core (slices 1–3, `woollama-core/`); the live
integration gate is green. This document is kept as the migration record.

## The constraint (why this isn't a swap)

The Rust `woollama-core` wheel provides `woollama.core` as a single compiled
extension exporting only the **engine**: `complete` / `complete_sync` /
`complete_stream` / `orchestrate` / `orchestrate_events` / `ModelRegistry` /
`EventIter` / `InferenceError` / `provider_names`.

But today `woollama.core` is a **Python package** that also ships, and the server
imports, much more:

- `core.config` — `inferencers.toml` / `recipes.toml` / `mcp.json` loading, dirs.
- `core.recipes` — `Recipe` TypedDict + `make_recipe`.
- `core.tooling` — `ToolSpec` / `ToolResult` / `ToolProvider` / `render_tool_result`
  / `Capabilities` (the seam; the server's `RegistryToolProvider` builds these).
- `core.ollama_native` — native `/api/chat` helper.
- `core.inferencers` — `Inferencer` dataclass, `InferencerError`, module-level
  `get`/`all`/`names` + **discovery** (`models`/`discover`/`model_patterns`) for
  `GET /v1/models`, **and** `ModelRegistry`.

Three hard facts, all empirically confirmed:

1. **Subset** — the Rust core covers the engine, not the support modules above.
2. **Namespace collision** — a `.so` named `core` cannot coexist with a `core/`
   package; the package shadows the `.so`. (Seen live: with both installed in one
   venv, `from woollama import core` resolves to the *Python* package.)
3. **Packaging-style clash** — the server's `woollama` is a regular package
   (`__init__.py`); the Rust wheel ships `woollama` as a PEP 420 namespace.

So `woollama.core` must become engine-only (Rust), and everything else must move
out from under it.

## Target layout

```
woollama/                      ← PEP 420 namespace (no __init__.py)
  core   (woollama-core wheel) ← Rust engine: complete*, orchestrate*,
                                 ModelRegistry, InferenceError, EventIter
  config        (server)       ← was core.config
  recipes       (server)       ← was core.recipes   (Recipe, make_recipe)
  tooling       (server)       ← was core.tooling    (ToolSpec/ToolResult/…)
  ollama_native (server)       ← was core.ollama_native
  inferencers   (server)       ← was core.inferencers MINUS ModelRegistry
                                 (keeps Inferencer dataclass + all()/get() +
                                  discovery for /v1/models)
  manager / router / mcp_server / … (server, unchanged location)
```

`core.inference` (the Python `complete`) and `core.orchestrate` are **deleted** —
their callers move to `woollama.core` (Rust).

## The `inferencers` split + two registries (the key decision)

`core.inferencers` does two jobs; they separate:

- **Orchestration registry** → Rust `woollama.core.ModelRegistry` (slice 3 already
  loads `inferencers.toml`, merge faithfully ported). The server passes
  `ModelRegistry.from_config()` into `orchestrate_events(..., registry=…)`.
  (Without it the Rust loop silently regresses to built-ins.)
- **`/v1/models` discovery** → stays Python `woollama.inferencers` (`Inferencer`
  dataclass, `all()`, `discover`/`model_patterns`, the live `/v1/models` query) —
  none of which the Rust core provides.

Consequence: **two registries read the same `inferencers.toml`** — the Rust one for
the loop, the Python one for discovery. Their merge semantics must agree; the
slice-3 conformance suite pins the orchestration-relevant fields against the Python
oracle. Discovery-only fields (`models`/`discover`/`model_patterns`) live only in
the Python side. This duplication is accepted for now (a later slice could port
discovery + `/v1/models` to Rust and collapse it).

## Staged migration (server suite green at every stage)

0. **Baseline** — `uv run --extra dev pytest tests/` green. ✅ (244 passed)
1. **Relocate support modules** `config`/`recipes`/`tooling`/`ollama_native`
   `core.X → woollama.X`; update their import sites. Pure Python, reversible.
2. **Split `inferencers`** — move the Python discovery half to `woollama.inferencers`
   (drop `ModelRegistry`; the dataclass/`all`/`get`/discovery stay). Update
   `/v1/models` + router import sites.
3. **Engine swap** — add `woollama-core` as a dependency; point the router's
   `orchestrate_events` delegation and `complete` at Rust `woollama.core`, passing
   `registry=woollama.core.ModelRegistry.from_config()`. Delete `core.inference` /
   `core.orchestrate`.
4. **Namespace + packaging** — make `woollama` a PEP 420 namespace (drop the trivial
   top `__init__.py`; `__version__` via `importlib.metadata`); the server dist stops
   shipping `woollama/core/`; `pyproject` depends on `woollama-core`;
   `test_core_is_server_free` updated (the boundary is now a wheel boundary).
5. **Cleanup** — remove any transition shims; docs.

Each stage is its own commit; if a stage can't stay green it stops there for review.

## Status / discovery (stage 3b cutover)

Stages 1, 2, 3a, 3b-1 landed green. The structural cutover (3b-2 / stage 4) is
**done and proven**: `woollama` is a PEP 420 namespace, `woollama.core` resolves to
the **Rust** `.so` (namespace merge verified — `woollama.__path__` spans the server
src + the woollama-core wheel), the router is rewired to the flat Rust surface with
`registry=ModelRegistry.from_config()`, and the non-engine suite is green
(config/recipes/tooling/smoke/server-free/claude_code).

**But the cutover surfaced a test-harness incompatibility** (the genuine remaining
work): the server's orchestration tests fake the inferencer by
`monkeypatch.setattr(httpx, "AsyncClient", …)` — patching httpx **in-process**. The
Python engine used httpx so the patch worked; the **Rust engine uses reqwest**, which
the patch can't intercept, so those tests issue real HTTP and hit the live ollama
(slow inference → effective hang).

The fix is the same pattern the Rust conformance suite already uses: a threaded **mock
HTTP server** + `$WOOLLAMA_OLLAMA_URL` (or per-recipe `base_url`) pointed at it, so the
Rust core's reqwest hits the mock. Scope: rework the shared `mock_inferencer` fixture
(test_routing) and the equivalent httpx-patching in the orchestration-driving files
(test_responses, test_responses_stream, test_router, test_mcp_server), plus a few
non-hang failures in test_inferencers / test_ollama_native (Python-path tests, likely
the ollama-URL normalization). The file-store / discovery httpx patches
(test_store_backend, test_http_store_provider) are Python-path and stay as-is.

This is bounded but is its own focused pass — pending.

### Second discovery: the router needs structured `InferenceError` fields

Reworking the `mock_inferencer` fixture to a mock HTTP server **removed the hangs**
(test_routing: 11 happy-path orchestration tests green). The remaining failures are a
*separate* gap: the router maps orchestration errors to HTTP via
`OrchestrationError.{message,kind,status,payload}` (and raises
`OrchestrationError(msg, kind, status[, payload=…])`). `OrchestrationError` is now the
Rust `core.InferenceError`, which carries only a message — the **structured
`InferenceError` fields (kind/status/payload)** the port deferred. The router depends
on them; the Python `InferenceError` had them.

So greening the cutover needed, in addition to the harness rework:

- **Rust: make `InferenceError` a structured exception** — `InferenceError(message,
  kind=None, status=None, payload=None)` with `.message`/`.kind`/`.status`/`.payload`
  attributes, raises populated to match `orchestrate.py`. (A
  `#[pyclass(extends=PyException)]`; construct POSITIONALLY — PyO3 wires `#[new]` as
  __new__ only, so a `payload=` kwarg reaches the inherited `BaseException.__init__`
  which rejects it.)
- Propagate the mock-server fixture to every orchestration-driving test (test_routing,
  test_responses, test_responses_stream, test_router, test_mcp_server,
  test_inferencers, test_ollama_native, test_store_backend); leave the Python-path
  httpx patches (passthrough, /v1/models discovery, file-store) as-is.

## DONE ✅

Both pieces landed. The server's **inference and orchestration run on the Rust
`woollama.core`** (`complete`/`complete_stream`/`orchestrate_events` +
`ModelRegistry` + structured `InferenceError`); the support surface stays Python —
passthrough (`ollama_native`), `/v1/models` discovery (`woollama.inferencers`),
recipes, config, tooling. ("Entirely on Rust" would overstate it.)

**Verification:** 226 hermetic server tests + 42 Rust conformance green, ruff clean.
The obsolete Python-engine tests (test_core_{inference,orchestrate,models}) were
removed — their coverage is the Rust conformance suite + the (now wire-mocked) server
integration tests; `make_recipe` coverage preserved in test_recipes.py. Packaging
proven from a clean rebuild (`rm -rf .venv && uv sync --extra dev && pytest`).

**Live integration gate: 25/25 green** (run against real Ollama + the user's Claude
auth + the real Anthropic API): 17 free-tier (ollama orchestration incl. streaming
tool-loop reassembly, store-backed journeys, MCP stdio+HTTP) + 5 claude-code (incl.
the three delegation/lockdown security gates) + 3 anthropic (managed-agents journey,
requires_action, anthropic-compat inferencer). The gate surfaced **one latent bug**
(NOT a cutover regression — the deleted Python engine had it too): a tool-less recipe
sent `tools: []`, which Anthropic's stricter endpoint rejects. Fixed by omitting the
key when empty (commit `e221780`); conformance + the live test confirm it.

Remaining deferred (named, NOT blocking the migration): the registry's discovery
fields (`models`/`discover`/`model_patterns`) + a Rust `/v1/models` (still Python in
`woollama.inferencers`); publishing `woollama-core` to PyPI so the server dep isn't a
path source; and re-pinning lackpy from the git rev to the published wheel.
