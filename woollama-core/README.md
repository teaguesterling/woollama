# woollama-core (Rust)

The embeddable woollama model-management core, in Rust with Python bindings
(PyO3 / maturin) — **the first slice of the woollama v1.0 Rust port**.

The Python extraction (`docs/core-extraction.md`) froze the core's API and left a
hermetic test suite behind; that suite is the **conformance oracle** for this Rust
implementation. We port against a spec, not blind. A Rust core also serves the
"embeddable, light" goal better than Python: near-zero Python dependencies, so a
consumer (e.g. lackpy) gets a single compiled wheel instead of the whole server
stack.

## Slice 1 — what's here

Callback-free, and enough to fully serve lackpy (which only needs `complete`):

- the **built-in inferencer registry** (ollama / anthropic / openai / groq /
  together / openrouter; ollama honors `$WOOLLAMA_OLLAMA_URL`);
- **`complete(model, messages, *, options, params, api_key, base_url)`** — an
  **awaitable** (`await complete(...)`, the drop-in for async embedders like
  lackpy), backed by async `reqwest` on a tokio runtime (pyo3-async-runtimes);
- **`complete_sync(...)`** — the blocking variant (HTTP off the GIL);
- **`complete_stream(model, messages, …)`** — an **async iterator** over assistant
  text deltas (`async for d in complete_stream(...)`): the `/v1` SSE, parsed in
  Rust, yielded incrementally (num_ctx-native routing is non-stream only, matching
  Python; a setup error raises on the first pull);
- all do ollama-native `num_ctx` routing (→ `/api/chat`, non-stream), top-level
  `params` (temperature, …), and per-call `api_key`/`base_url` overrides;
- **`InferenceError`** + `provider_names()`.

Behavior mirrors `woollama.core.complete` (Python) — verified by
`tests/test_complete_conformance.py` (request shape, routing, params, auth,
fail-fast on missing key, unknown provider; sync + the async awaitable) and live
against ollama. One minor difference: the async awaitable binds to the running
event loop at creation, so it must be created inside the loop (moot for the embed
case — lackpy always `await`s inside an async function).

## Slice 2 — the recipe↔tool loop (`orchestrate`)

The first **callback** slice: the core now drives the inferencer↔tool loop and
calls *back* into Python for the tools.

- **`orchestrate(recipe, user_msgs, tools, *, api_key, base_url)`** — an awaitable
  returning the final OpenAI response dict. It prepends `recipe["system"]`, offers
  the recipe's allow-listed tools, dispatches the ones the model calls through the
  Python `tools` **`ToolProvider`** (sync `tools_for(allow) -> [ToolSpec]`, **async**
  `dispatch(name, args) -> ToolResult`), feeds results back, and repeats (≤8 turns).
  This is the Python port's `core.orchestrate` (the drainer over
  `orchestrate_events`).
- The **allow-list is a boundary, not a hint**: a tool_call for anything off it is
  *refused without dispatching* and the refusal is fed back as the tool result (so
  the loop recovers) — the adversarial property, enforced in Rust.
- A `dispatch` that **raises** becomes an `ERROR: {Type}: {msg}` tool result and the
  loop continues — it never propagates (matches `orchestrate.py`).
- Built-in inferencers carry `extra_body` merged into each orchestration request
  (ollama → `{"options":{"temperature":0}}`, anthropic → `max_tokens`, clouds →
  `temperature:0`); the recipe's `params` override it. `ToolResult` rendering
  (text-join, JSON fallback, `[tool error]` prefix) is reimplemented in Rust over
  duck-typed `.blocks`/`.is_error`, so the core still imports no Python woollama.

The genuinely novel mechanic — awaiting the Python `dispatch` coroutine from inside
the Rust async loop (`pyo3_async_runtimes`' `into_future`, task-locals propagated
through `future_into_py`) — was spiked in isolation first. Verified by
`tests/test_orchestrate_conformance.py` (dispatch→final, the out-of-list refusal,
`is_error`/exception rendering, `extra_body`/`params` merge, unsupported inferencer)
with a mock `ToolProvider` + scripted inferencer, and **live against ollama**: a
`math.add` recipe → qwen3 calls the tool → Rust dispatches it → `"…is 42."`.

## Proven with lackpy (the thesis)

lackpy's `WoollamaProvider` runs on this Rust core **unchanged** — its provider
does `from woollama.core import complete` and `await`s it, which is exactly the
Rust surface. Verified live against ollama (the provider's prompt-building → Rust
`complete` → ollama → a generated lackpy program):

```python
import asyncio
from lackpy.infer.providers.woollama import WoollamaProvider  # does `from woollama.core import complete`

p = WoollamaProvider(model="ollama/qwen3:14b-iq4xs", temperature=0.2)
out = asyncio.run(p.generate("count the rows",
                  namespace_desc="kernel.select(expr)\nkernel.count()"))
# -> 'count = kernel.count()\nprint(count)'   (generated via the Rust core)
```

The maturin build ships the extension at the `woollama.core` import path directly
(`module-name = "woollama.core"`, a submodule of the PEP 420 `woollama`
namespace), so `from woollama.core import complete` resolves to the Rust `complete`
with no `sys.modules` swap. Productionizing is the remaining packaging step
(below): make the server dist stop shipping its own `woollama.core` and depend on
this, so the server and lackpy both consume the Rust core directly.

## Deferred (later slices)

The per-event generator (`orchestrate_events` — `delta`/`tool_call`/`tool_result`
progress events) and streamed orchestration; config-file (`inferencers.toml`)
loading + an explicit `ModelRegistry` (`orchestrate` uses built-in inferencers
only); structured `InferenceError` fields (kind/status/payload); and packaging so
this provides `woollama.core` for the server dist (the dist-split).

## Build & test

```sh
uv venv && uv pip install maturin pytest
maturin develop                      # builds + installs `woollama.core` into the venv
python -m pytest tests/ -q
```
