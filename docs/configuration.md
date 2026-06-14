# Configuration reference

woollama is file-driven. Config lives in `$WOOLLAMA_CONFIG_DIR` (default
`$XDG_CONFIG_HOME/woollama`, i.e. `~/.config/woollama`). All three files are
optional — woollama falls back to bundled defaults for `mcp.json` and
`recipes.toml`, and to its built-in provider list when there's no
`inferencers.toml`.

| File | Purpose | Fallback |
|---|---|---|
| `mcp.json` | MCP servers to discover (tools/prompts/resources) | bundled default (hello + textops examples) |
| `recipes.toml` | Named recipes (system prompt + tools + inferencer) | bundled default |
| `inferencers.toml` | OpenAI-compatible inference backends | built-in providers only |

`${VAR}` references are expanded from the environment in `mcp.json` and
`inferencers.toml` (e.g. `base_url = "${VLLM_URL}/v1"`).

## `mcp.json`

Shape matches Claude Code's `mcpServers` block:

```json
{
  "mcpServers": {
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git"],
      "env": { "GIT_AUTHOR_NAME": "woollama" }
    }
  }
}
```

| Field | Required | Description |
|---|---|---|
| `command` | ✅ | Executable to launch the server (stdio MCP). |
| `args` | — | Argument list (default `[]`). |
| `env` | — | Extra environment for the server process (default `{}`). |

woollama starts one long-lived connection per server and aggregates their tools
(namespaced `<server>.<tool>`).

> **Interpreter & `PATH`.** `command` is resolved against woollama's *own*
> environment when the server is spawned. A bare name (`python`, `uvx`, `node` —
> including in the `conversationStore` examples below) picks whatever is first on
> `PATH` at spawn time, which need not be the interpreter that has the server's
> dependencies when woollama runs **outside its virtualenv** (launched by an
> absolute path, or as a `systemd` unit with a minimal `PATH`). Because a
> downstream server that fails to start **aborts woollama startup**, pin `command`
> to an absolute interpreter (e.g. your venv's `python`) if startup is sensitive
> to which environment launched it.

### Selecting a conversation store

An **external** conversation store makes non-claude models stateful (issue #2) —
it owns the transcript bytes while woollama stays a client. Select one with the
top-level `conversationStore` key (a sibling of `mcpServers`). woollama ships two
reference stores and the seam is transport-agnostic, so the key takes two typed
forms:

**MCP store** — a server in `mcpServers` exposing `create_thread` / `get_thread` /
`append_turn` / `delete_thread` (reference: `examples/mcp-convstore`):

```json
{
  "conversationStore": { "type": "mcp", "server": "convstore" },
  "mcpServers": {
    "convstore": {
      "command": "python",
      "args": ["${WOOLLAMA_EXAMPLES_DIR}/mcp-convstore/server.py"]
    }
  }
}
```

A bare string is shorthand for the MCP form: `"conversationStore": "convstore"`
≡ `{ "type": "mcp", "server": "convstore" }`.

**HTTP store** — a REST endpoint with `PUT`/`GET`/`PATCH`/`DELETE /threads/{id}`
(reference: `examples/rest-convstore`, file-backed so transcripts persist):

```json
{
  "conversationStore": { "type": "http", "url": "http://127.0.0.1:9000" }
}
```

| Field | Required | Description |
|---|---|---|
| `conversationStore` | — | The store to use. A string (= MCP server name), `{type:"mcp", server}`, or `{type:"http", url}`. Omitted (the default) ⇒ non-claude models are stateless. An `mcp` server not present in `mcpServers` is warned and ignored. |

Once set, **every** non-claude model (`ollama/*`, cloud providers, and
`woollama/<recipe>`) becomes stateful on `/v1/responses` + `/v1/conversations`.
See the [Conversations design](conversations-api-design.md) §10 for the contract.

## `recipes.toml`

A recipe binds a system prompt + an allow-listed tool set + an inferencer into a
single `woollama/<name>` model.

```toml
[recipes.streamer]
inferencer = "ollama/qwen3:14b-iq4xs"   # <provider>/<model> — who runs inference
system = "You are concise."             # system prompt
tools = ["hello.count_to"]              # allow-list of <server>.<tool> (may be [])

[recipes.cc-counter]
inferencer = "claude-code/haiku"        # a claude-code recipe WITH tools delegates
system = "Use the count_to tool."
tools = ["hello.count_to"]
```

| Field | Required | Description |
|---|---|---|
| `inferencer` | ✅ | `<provider>/<model>` that runs the recipe's inference. |
| `system` | ✅ | System prompt (whitespace-trimmed). |
| `tools` | ✅ | Allow-list of `<server>.<tool>` names; `[]` for a tool-less recipe. **Enforced as a security boundary** — a recipe can't call a tool outside this list (in the in-loop path *and* in claude-code delegation). |

## `inferencers.toml`

OpenAI-compatible backends. **Merged field-by-field over the built-ins** (`ollama`,
`anthropic`, `openai`, `groq`, `together`, `openrouter`) — a same-named entry
overlays only the keys it sets, so you can extend a built-in (e.g. add `models`
to `anthropic`) without restating its `base_url`. A *new* provider must supply
`base_url`.

```toml
# New self-hosted provider (no auth)
[inferencers.vllm]
base_url = "${VLLM_URL}/v1"
extra_body = { temperature = 0.5 }

# Surface specific cloud models in GET /v1/models (issue #3) — no base_url needed
# (extends the built-in anthropic)
[inferencers.anthropic]
models = ["claude-opus-4-8", "claude-haiku-4-5"]

# Live-discover a provider's catalog, filtered so it doesn't flood the picker
[inferencers.openrouter]
discover = true
model_patterns = ["anthropic/*", "openai/gpt-4*"]
```

| Field | Required | Description |
|---|---|---|
| `base_url` | ✅ for a new provider | OpenAI-compatible base, **without** `/chat/completions`. |
| `api_key_env` | — | **Name** of the env var holding the bearer key (not the key itself). Omit for no-auth (local). |
| `extra_body` | — | Fields merged into each orchestration request (e.g. `temperature`, ollama's `options`). |
| `models` | — | Static model ids to list in `GET /v1/models` as `<provider>/<id>` (no key needed to *list*). |
| `discover` | — | If `true`, live-query the provider's own `/v1/models` and list those too (needs the key). `ollama` defaults to `true`. |
| `model_patterns` | — | fnmatch globs that filter `discover` results (e.g. `["claude-*"]`); empty = list all discovered. |

Models are still **routable by raw id** (`anthropic/claude-opus-4-8`) whether or
not they're listed — `models`/`discover` only control *discoverability* in
`GET /v1/models` (what a list-backed picker can offer).

See also: [Environment variables](environment.md) · [Security model](security.md).
