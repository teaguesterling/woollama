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
- **`complete(model, messages, *, options, params, api_key, base_url)`** — one
  stateless turn against `<provider>/<model>`, with ollama-native `num_ctx`
  routing (→ `/api/chat`), top-level `params` (temperature, …), and per-call
  `api_key`/`base_url` overrides; HTTP runs off the GIL;
- **`InferenceError`** + `provider_names()`.

Behavior mirrors `woollama.core.complete` (Python) exactly — verified by
`tests/test_complete_conformance.py` (request shape, routing, params, auth,
fail-fast on missing key, unknown provider) and live against ollama.

## Deferred (later slices)

`complete_stream` (SSE), async bindings (pyo3-async-runtimes, so `await complete`
works unchanged for embedders), config-file (`inferencers.toml`) loading, an
explicit `ModelRegistry`, structured `InferenceError` fields (kind/status/payload),
and the recipe loop + the Python `ToolProvider` callback.

## Build & test

```sh
uv venv && uv pip install maturin pytest
maturin develop                      # builds + installs `woollama_core` into the venv
python -m pytest tests/ -q
```
