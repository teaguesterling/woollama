# Getting started

## Install

The router is **`woollamad`** — a small Rust daemon. Install it from crates.io:

```sh
cargo install woollama-server     # installs the `woollamad` binary
woollamad                         # starts the router; prints its address
```

`cargo install` ships only the binary, so bring your own `mcp.json` (or point
`WOOLLAMA_EXAMPLES_DIR` at this repo's `examples/` for the bundled demo servers).

### From a checkout (with the bundled example servers)

```sh
git clone https://github.com/teaguesterling/woollama
cd woollama
cargo build --release             # builds target/release/woollamad
./target/release/woollamad        # starts the router; prints its address
```

### The Python reference server

The original Python implementation still runs and is the differential-test
**oracle** that keeps `woollamad` honest. It's on PyPI (`pip install woollama`,
which also pulls the native `woollama-core` engine), or run it from the checkout
with [`uv`](https://docs.astral.sh/uv/):

```sh
uv sync                           # creates .venv and installs deps
uv run woollama                   # the Python reference server
```

!!! note "Prerequisite for the examples"
    The examples below use `ollama/qwen3:14b-iq4xs`. Install
    [Ollama](https://ollama.ai), run `ollama serve`, and
    `ollama pull qwen3:14b-iq4xs`. **No Ollama?** Swap in the keyless
    `claude-code/haiku` (needs the `claude` CLI logged in), or any cloud model
    with its key set (see [Configuration](configuration.md) and
    [Environment variables](environment.md)).

## The address

On startup the router prints its `OpenAI base_url` — copy that into your client:

```
OpenAI base_url:      http://127.0.0.1:<port>/v1
```

It serves on **two transports at once** and never binds off-loopback without an
explicit opt-in ([`WOOLLAMA_ADDRESS`](environment.md)):

- a **Unix socket** at `$XDG_RUNTIME_DIR/woollama.sock` (mode `0600` — the
  default for local MCP clients, since a connectable socket can spend the
  router's API keys);
- an **ephemeral loopback TCP port**, also written to
  `$XDG_RUNTIME_DIR/woollama.addr` for programmatic discovery.

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

### Stateful conversations (`/v1/responses`)

The OpenAI **Responses** surface adds multi-turn state. woollama routes the
conversation *handle*; a state-owning backend keeps the transcript (here
`claude-code/<model>` → the Claude session). `stream=True` works too (Responses
SSE):

```python
r = c.responses.create(
    model="claude-code/haiku",
    input="Remember the codeword: banana.",
    store=True,                       # create a backing conversation
)
# Continue it — woollama resumes the same session by its conversation id:
r2 = c.responses.create(
    model="claude-code/haiku",
    input="What was the codeword?",
    conversation=r.conversation.id,
)
print(r2.output_text)                 # → "banana"
```

Models with no state-owning backend (ollama/cloud/recipe) are stateless — use
`store=False` (the caller owns history). See the
[Conversations API](conversations-api-design.md) for the full surface.

## Configuration

woollama is file-driven (in `$WOOLLAMA_CONFIG_DIR`, default
`~/.config/woollama`). Three files:

- `mcp.json` — MCP servers to discover (`command` / `args` / `env`).
- `recipes.toml` — named recipes (system prompt + tools + inferencer).
- `inferencers.toml` — OpenAI-compatible backends (field-merge over the
  built-ins; `${VAR}` expansion), e.g. a self-hosted vLLM endpoint, or surfacing
  cloud models in `/v1/models`.

Full field-by-field schemas: **[Configuration reference](configuration.md)**.
Configurable env vars: **[Environment variables](environment.md)**.

## Tests & lint

```sh
# Rust (woollamad) — the daemon's own suites:
cargo test --tests -p woollama-server -p woollama-engine --features test-fixtures

# Python (reference server) + lint:
uv run --extra dev pytest        # hermetic suite (live tests are opt-in: -m integration)
uv run ruff check .              # lint — the CI gate
```

The live **differential oracle** runs the *same* integration suite against
whichever implementation you select — `woollamad` by default (build it first so
the suite can spawn it), or the Python reference via `WOOLLAMA_TEST_CMD`:

```sh
cargo build --release
uv run --extra dev pytest -m integration                      # targets woollamad
WOOLLAMA_TEST_CMD="python -m woollama" \
  uv run --extra dev pytest -m integration                    # targets the Python server
```

CI runs the Rust + Python gates on every push to `main` and every PR
(`.github/workflows/ci.yml`); `.github/workflows/wheels.yml` builds the
cross-platform `woollama-core` wheels. For the same lint gate locally on commit,
opt into the pre-commit hook:

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
