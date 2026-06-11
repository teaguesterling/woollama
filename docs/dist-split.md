# Dist-split: server onto the Rust `woollama.core`

Status: **in progress** (migration). This is the last gate before the server runs
its inference/orchestration on the Rust core (slices 1–3, `woollama-core/`).

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
