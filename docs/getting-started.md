# Getting started

## Install (development)

woollama uses [`uv`](https://docs.astral.sh/uv/) for environment management.

```sh
git clone https://github.com/teaguesterling/woollama
cd woollama
uv sync                           # creates .venv and installs deps
uv run woollama                   # starts the router; prints its address
```

## Discover the address

woollama serves on **two transports at once** and never binds to `0.0.0.0`
without an explicit opt-in:

- a **Unix socket** at `$XDG_RUNTIME_DIR/woollama.sock` (mode `0600` — the
  default for local MCP clients, since a connectable socket can spend the
  router's API keys);
- an **ephemeral loopback TCP port**, written to
  `$XDG_RUNTIME_DIR/woollama.addr` for clients to discover.

In a second shell:

```sh
cat "${XDG_RUNTIME_DIR:-/tmp}/woollama.addr"
```

That printed `host:port` is the `<port>` used in the examples below.

## Drive it from an OpenAI client

The router is OpenAI-compatible, so any OpenAI client can drive it:

```python
import openai
c = openai.OpenAI(base_url="http://127.0.0.1:<port>/v1", api_key="x")

# Pass-through to Ollama
r = c.chat.completions.create(
    model="ollama/qwen3:14b-iq4xs",
    messages=[{"role": "user", "content": "Hi"}],
)

# Orchestrated: a recipe (system prompt + tools + model), transparent to the
# client. The chat-loop happens inside woollama; the client sees only the final
# answer.
r = c.chat.completions.create(
    model="woollama/streamer",
    messages=[{"role": "user", "content": "Please count to 4."}],
)
```

`stream=True` works on both paths: on `<provider>/<model>` it relays the
upstream SSE verbatim; on `woollama/<recipe>` it streams the answer as OpenAI
SSE with the tool loop hidden.

## Configuration

woollama is file-driven. Three files:

- `mcp.json` — MCP servers to discover (`command` / `args` / `env`).
- `recipes.toml` — named recipes (system prompt + tools + inferencer).
- `inferencers.toml` — OpenAI-compatible backends (merges over the built-ins;
  supports `${VAR}` expansion), e.g. a self-hosted vLLM endpoint.

See [Architecture](architecture.md) for how these compose.

## Tests & lint

```sh
uv run --extra dev pytest        # hermetic suite (live tests are opt-in: -m integration)
uv run ruff check .              # lint — the CI gate
```

CI (`.github/workflows/ci.yml`) runs both on every push to `main` and every PR.
For the same lint gate locally on commit, opt into the pre-commit hook:

```sh
uv tool install pre-commit && pre-commit install
```

The project does not use `ruff format` (lines are hand-wrapped, `E501` is
ignored), so there is no formatter step in either gate.

## Build the docs locally

This site is built with [MkDocs](https://www.mkdocs.org/) + the
[Material](https://squidfunk.github.io/mkdocs-material/) theme:

```sh
uv run --with-requirements docs/requirements.txt mkdocs serve   # live-reload at http://127.0.0.1:8000
uv run --with-requirements docs/requirements.txt mkdocs build --strict   # the gate CI/RTD use
```
