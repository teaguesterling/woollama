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

## Slice 2 — the recipe↔tool loop (`orchestrate_events` + `orchestrate`)

The first **callback** slice: the core now drives the inferencer↔tool loop and
calls *back* into Python for the tools. There is **one loop** —
`orchestrate_events`, authored as a Rust `stream!` of events — and `orchestrate`
drains it (matching how Python's `orchestrate` is a thin drainer over the
generator).

- **`orchestrate_events(recipe, user_msgs, tools, *, api_key, base_url, stream=False)`**
  — an **async iterator** (`async for ev in …`) of progress dicts: `tool_call`
  (`turn`/`name`/`args`), `tool_result` (`turn`/`name`/`ok`), and a terminal `final`
  (`{"type":"final","response": <openai dict>}`). It prepends `recipe["system"]`,
  offers the recipe's allow-listed tools, dispatches the ones the model calls through
  the Python `tools` **`ToolProvider`** (sync `tools_for(allow) -> [ToolSpec]`,
  **async** `dispatch(name, args) -> ToolResult`), feeds results back, repeats (≤8
  turns). This is the server's surface.
- **`stream=True`** runs each turn over **SSE**: the iterator also yields `delta`
  (assistant content) events as they arrive, fragmented `tool_calls` are reassembled
  by index across chunks, and the `final` `response` is *synthesized*
  (`{object:"chat.completion", …, finish_reason}`) since no single upstream object
  exists. `stream=False` does one non-stream POST and passes the raw response through.
- **`orchestrate(recipe, user_msgs, tools, *, api_key, base_url)`** — the awaitable
  drainer returning just the final OpenAI dict (Python's `core.orchestrate`).
- The **allow-list is a boundary, not a hint**: a tool_call for anything off it is
  *refused without dispatching* (the `tool_result` carries `ok: false`) and the
  refusal is fed back so the loop recovers — the adversarial property, in Rust.
- A `dispatch` that **raises** becomes an `ERROR: {Type}: {msg}` tool result
  (`ok: false`) and the loop continues — it never propagates (matches `orchestrate.py`).
- Built-in inferencers carry `extra_body` merged into each orchestration request
  (ollama → `{"options":{"temperature":0}}`, anthropic → `max_tokens`, clouds →
  `temperature:0`); the recipe's `params` override it. `ToolResult` rendering
  (text-join, JSON fallback, `[tool error]` prefix) is reimplemented in Rust over
  duck-typed `.blocks`/`.is_error`, so the core still imports no Python woollama.
- **Eager setup** (a deliberate divergence from Python's lazy generator): an
  unsupported inferencer / missing key raises on the *call*, not on first `__anext__`.

Three novel mechanics, each spiked in isolation first: awaiting the Python `dispatch`
coroutine from inside the Rust async loop (`pyo3_async_runtimes`' `into_future`,
task-locals propagated through `future_into_py`); a `stream!`-backed async-iterator
pyclass that yields across `__anext__` calls with an `await` between yields; and the
fragmented-tool_call reassembly (id/name in one SSE chunk, `arguments` dribbling
across the next). Verified by the conformance suite (dispatch→final, the out-of-list
refusal, `is_error`/exception rendering, parallel tool_calls, max-turns,
`extra_body`/`params` merge, the event sequence + `ok` flags, eager-raise, and
streaming — `delta` events, the synthesized final, a tool call whose `arguments` are
split across chunks, the SSE error) with mock `ToolProvider` + scripted (JSON & SSE)
inferencers, and **live against ollama**: a `math.add` recipe → qwen3 calls the tool
→ `tool_call`/`tool_result`/(`delta`×N)/`final` → `"…is 42."`, both non-stream and
streamed.

The SSE reader buffers **raw bytes** and decodes whole lines, so a multibyte UTF-8
char split across network chunks (`resp.chunk()` boundaries don't align to chars) is
never corrupted — regression-tested with a chunked mock that splits an em-dash, and
the fix is shared with slice-1's `complete_stream`.

## Slice 3 — config-driven inferencers (`ModelRegistry`)

Until now the loop resolved built-in providers only. **`ModelRegistry`** adds the
config path so the server's `inferencers.toml` providers reach the Rust loop.

- **`ModelRegistry.from_config()`** — built-ins overlaid by
  `$WOOLLAMA_CONFIG_DIR/inferencers.toml` (env precedence:
  `$WOOLLAMA_CONFIG_DIR` → `$XDG_CONFIG_HOME/woollama` → `~/.config/woollama`;
  `${VAR}` expanded in values — braced form only, a braceless `$VAR` is left
  literal). The built-in `ollama` honors `$WOOLLAMA_OLLAMA_URL` as the *root* and
  appends `/v1` (normalizing a trailing `/` or `/v1`), matching Python. Also
  `ModelRegistry()` + `add(name, base_url, *, api_key_env, extra_body)` for
  in-memory building, and `get`/`names`/`all`.
- The merge is **field-by-field** (mirrors `inferencers._registry`), with the
  oracle's two inheritance idioms preserved faithfully: `base_url`/`extra_body`
  inherit on **falsy** (a config `extra_body = {}` keeps the built-in's), while
  `api_key_env` inherits on **absence** (extend `anthropic` with only an
  `extra_body` and it still resolves `ANTHROPIC_API_KEY`). A *new* provider must
  supply `base_url`, else an error. Parsing into a JSON value (not a defaulted
  struct) is what preserves present-vs-absent.
- Pass it to **`orchestrate`/`orchestrate_events` via `registry=`**; omitting it
  uses the built-ins (the hermetic default — so a server migration must pass
  `ModelRegistry.from_config()` or it silently regresses to built-ins).

Verified by `tests/test_registry_conformance.py` (both merge idioms, new-provider
`base_url` error, `${VAR}` expansion, missing-file → built-ins, and the loop
resolving a **config-only** provider through `registry=` with *no* `base_url`
override — so registry resolution is genuinely exercised), and **live**:
`ModelRegistry.from_config()` resolves `ollama` and the request reaches it. The
discovery fields (`models`/`discover`/`model_patterns`) and `/v1/models` are
deferred (they don't feed the loop).

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

The registry's discovery fields (`models`/`discover`/`model_patterns`) + a
`/v1/models` surface (they don't feed the recipe loop); structured `InferenceError`
fields (kind/status/payload); and packaging so this provides `woollama.core` for the
server dist (the **dist-split**). With the loop streaming *and* config-driven, the
last gate before the server can sit on the core is the dist-split — the server just
has to pass `ModelRegistry.from_config()` (opt-in, else it regresses to built-ins).

## Build & test

```sh
uv venv && uv pip install maturin pytest
maturin develop                      # builds + installs `woollama.core` into the venv
python -m pytest tests/ -q
```
