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

## Deferred (later slices)

Config-file (`inferencers.toml`) loading + an explicit `ModelRegistry`; structured
`InferenceError` fields (kind/status/payload); packaging so this provides
`woollama.core` for consumers; and (server-port territory) the recipe loop + the
Python `ToolProvider` callback.

## Build & test

```sh
uv venv && uv pip install maturin pytest
maturin develop                      # builds + installs `woollama_core` into the venv
python -m pytest tests/ -q
```
