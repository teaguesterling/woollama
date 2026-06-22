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

### Fabric backend

The top-level `fabric` key (a sibling of `mcpServers`) puts a
[fabric](https://github.com/danielmiessler/fabric) deployment **behind woollama**:
its pattern library appears on `/w1/patterns`, and fabric's REST API is
transparently proxied at `/fabric/*`. woollama either spawns + supervises
`fabric --serve` (managed) or routes to an externally-run one (`url`).

```jsonc
{
  "fabric": {
    "managed": true,                            // spawn + supervise `fabric --serve`
    "default_model": "ollama/qwen3:14b-iq4xs"   // fabric patterns have no bound model;
                                                //   this is the fallback (and what makes
                                                //   them woollama/<name> in /v1/models)
  }
}
// or route to an externally-run fabric:
{ "fabric": { "url": "http://127.0.0.1:8999" } }
```

| Field | Required | Description |
|---|---|---|
| `managed` | — | `true` ⇒ woollama spawns + supervises `fabric --serve` (loopback). Reuse + graceful-kill: the address is persisted, so a restart reuses the live fabric; killed only on clean shutdown. |
| `url` | — | Route to an externally-run fabric at this base URL instead of spawning. Takes precedence over `managed`. |
| `command` | — | The fabric binary for managed mode (default `"fabric"`, resolved on `PATH`). |
| `address` | — | Fixed `host:port` to bind in managed mode (default: a persisted free loopback port). |
| `default_model` | — | Fallback `<provider>/<model>` for fabric patterns when a run omits `model`. Required for a fabric pattern to be addressable as `woollama/<name>` via `/v1/chat/completions` (which has no per-call model slot). |

> **Why here and not `inferencers.toml`?** fabric is not OpenAI-compatible, and the
> engine's `inferencers.toml` loader requires every entry to have a `base_url` —
> a fabric entry there would break config load. The fabric backend is a
> server-layer plugin, not an engine inferencer.

**Resilience.** The fabric pattern list is cached and kept fresh two ways: it is
**re-sourced on a TTL** as requests arrive (fabric hot-reloads its pattern dir, so
patterns added/removed at runtime show up — eventually; the triggering call still
sees the prior list), and it is re-sourced after any respawn. In **managed** mode a
dead or hung fabric is **respawned on the same address and the request retried
once** (single-flight, so concurrent requests don't race spawns); in `url` mode
woollama re-probes but never respawns (the process isn't woollama's to own). The
TTL defaults to 60s; override with the env var `WOOLLAMA_FABRIC_REFRESH_SECS`
(`0` = refresh on every read).

Omitted (the default) ⇒ no fabric backend. See [Pattern templating](patterns.md)
for the `/w1/` + `/fabric/` surfaces, and [Extending woollama](extending.md) to
add your own backend.

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

> **Recipes are also `/w1/` patterns.** A `system` prompt may contain `{{var}}`
> tokens that the [pattern surface](patterns.md) substitutes per call (and that
> MCP clients see as prompt arguments). Plain recipes simply have no `{{var}}`.

### `[patterns]` — a fabric-style pattern directory (optional)

Discover patterns from a directory of `<name>/system.md` files (e.g. fabric's
pattern library on disk), with no fabric process — woollama reads the files and
renders/runs them natively. Opt-in via a `[patterns]` block in `recipes.toml`:

```toml
[patterns]
dir = "~/.config/fabric/patterns"          # each <name>/system.md becomes a pattern
default_inferencer = "ollama/qwen3:14b-iq4xs"  # model for these patterns
```

| Field | Required | Description |
|---|---|---|
| `dir` | ✅ | Directory scanned for `<name>/system.md` files. `~` is expanded. A missing dir is ignored (no patterns). |
| `default_inferencer` | — | `<provider>/<model>` the discovered patterns run on. |

A `recipes.toml` recipe **wins** over a scanned pattern of the same name. For a
*live* fabric instance (the full library + fabric's own assembly) instead of a
file scan, use the [fabric backend](#fabric-backend) below.

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

See also: [Pattern templating](patterns.md) · [Extending woollama](extending.md) ·
[Environment variables](environment.md) · [Security model](security.md).
