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

`${VAR:-default}` supplies a fallback when the variable is unset **or empty** (POSIX `:-`
semantics). Use it to say a missing variable is intended:

```json
"args": ["${WOOLLAMA_EXAMPLES_DIR:-/nonexistent}/mcp-hello/server.py"]
```

The default runs to the first `}`, so it cannot itself contain one.

**A bare `${VAR}` that is not set is a hard load error.** woollama refuses the file, naming every
offending variable. This is safe to do only because `:-` exists: without a way to say "absent is
intended", refusing would break configurations that legitimately depend on an optional value —
including woollama's own bundled default.

A variable that is **set but empty** is not missing. `FOO=` is an explicit operator choice, and
treating it as an error would make being deliberate indistinguishable from a typo'd name.

The check runs on the **parsed** structure, so a `${VAR}` that appears only in documentation is
prose rather than configuration and never fails a load: `_`-prefixed keys (in JSON *and* TOML) and
TOML comments are skipped. MCP **server names** are not — `_disabled` is a server that loads, not
a comment.

It applies to `mcp.json` and `inferencers.toml`, in both implementations. `woollamad check-config`
reports it and exits non-zero, so a reload can be gated on it.

Anything security-relevant that survives the check is validated by its **consumer**, where the
consequence is known: an `Authorization` header expanding to a bare `Bearer ` invalidates that
server (see below), because there an empty value is unambiguously a missing credential.

Two things are **warned** about rather than refused, because refusing them would be wrong:

- a value inside a server's **`env` block** that resolves to empty — including via `${VAR:-}`.
  That is the one position woollama cannot check on the consumer's behalf: the consumer is a child
  process, and an empty `API_KEY=` may mean "auth disabled", "anonymous mode", or "this child will
  talk to a remote service unauthenticated". Many servers treat an empty string as absent and
  proceed.
- a **malformed** reference. `${VAR:-default}` does not nest, so `${DIR:-${HOME}/fb}` emits a
  literal `${HOME` rather than expanding it.

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

A server is **either** a stdio subprocess (`command`) **or** a Streamable-HTTP
endpoint (`url`). Setting both is an error, as is setting neither.

| Field | Required | Description |
|---|---|---|
| `command` | one of | Executable to launch the server (stdio MCP). |
| `args` | — | Argument list (default `[]`). Stdio only. |
| `env` | — | Extra environment for the server process (default `{}`). Stdio only. |
| `url` | one of | Streamable-HTTP MCP endpoint, e.g. another woollamad's `/mcp`. |
| `headers` | — | Headers sent with every request to `url` (default `{}`). HTTP only. |

woollama starts one long-lived connection per server and aggregates their tools
(namespaced `<server>.<tool>`).

#### The `url` form — consuming a remote MCP server

Across a network, use `https` — the credential is on the wire on every request:

```json
{
  "mcpServers": {
    "shelf": {
      "url": "https://mcp.example.lan/mcp",
      "headers": { "Authorization": "Bearer ${SHELF_TOKEN}" }
    }
  }
}
```

Same host, over loopback, plain `http` is fine — nothing leaves the machine. This
is the common shape when a tools-only instance publishes to `127.0.0.1` only and
still requires its token, so the bind address isn't load-bearing for auth:

```json
{
  "mcpServers": {
    "suite": {
      "url": "http://127.0.0.1:9200/mcp",
      "headers": { "Authorization": "Bearer ${WOOLLAMA_TOKEN}" }
    }
  }
}
```

> If the *consuming* router runs in a container, `127.0.0.1` is that container's
> own network namespace, not the host's loopback. Use host networking, or address
> the host explicitly — a loopback URL that works from a shell will fail from
> inside a container, and the connection-refused error looks identical to the
> downstream being down.

Because woollama also *serves* Streamable HTTP at `/mcp`, the `url` form is how
one woollamad consumes another — an inference-holding instance reaching a
tools-only instance's namespace without either spawning the other's servers.

> **What this is verified against.**
>
> *In CI:* `woollama-server/tests/federation_auth.rs` runs two real `woollamad`
> processes — a tools-only leaf requiring a bearer on every request (loopback
> included), and a consumer reaching it over the `url` form with the credential
> from `${WOOLLAMA_TOKEN}`. It asserts the leaf's tools federate through the
> enforced credential, and that a consumer holding the *wrong* credential
> federates nothing, stays running, and says so.
>
> *End to end, with real tools:* two `woollamad` 0.13.0 processes, the downstream
> fronting DuckDB-backed MCP servers as stdio children and the consumer reaching
> it over `url`. Eleven tools federated, and a real `tools/call` through
> consumer → HTTP → downstream → stdio → server → DuckDB → a ZIM archive → back.
> Longest federated wire name was 42 characters against the 64-char limit, so the
> hash fallback does not fire at one level of federation with realistic names.
>
> **Still not covered:** a LAN hop, a reverse proxy, TLS, or a containerised
> consumer. Those remain a materially different failure surface — in particular,
> a consumer *inside* a container resolves `127.0.0.1` to its own namespace.

**Credentials go in `headers` as `${VAR}`, never inline.** There is no separate
secrets mechanism: `${VAR}` expansion already applies to `mcp.json`, the same
shape as `api_key_env` for inferencers.

Header values are validated at load. An empty value, or a bare `Authorization`
scheme with nothing after it, makes **that server** invalid: it is logged and
skipped, and the rest of `mcp.json` loads normally. This is deliberate: `${VAR}`
expansion resolves an **unset** variable to the empty string, so
`"Bearer ${SHELF_TOKEN}"` with `SHELF_TOKEN` unset would otherwise produce the
literal header `Bearer ` — a well-formed request carrying no credential, which a
permissive downstream accepts while everything reports healthy.

The same per-server scoping applies to every other invalid entry (setting both
`command` and `url`, setting neither, a non-string `env`/`headers` value): the
bad server is skipped with a warning naming it, never taking its siblings with
it. Only `mcp.json` being unparseable JSON — where nothing is recoverable —
leaves woollama with no servers at all.

### Downstream reconnect and introspection

A downstream that is unreachable at startup is retried on a per-server exponential backoff
(1s doubling to a ceiling), so a peer that comes up later is picked up without restarting
woollama. A server whose *transport* could not be built — an unusable header, say — is reported
`failed` and **not** retried: that is a config fault, and retrying it would only produce noise
until someone edits the file. With retry disabled (`WOOLLAMA_MCP_RETRY_MAX_SECS=0`) an
unreachable server is likewise reported `failed`, not `retrying` — nothing will try again, and
saying otherwise would defeat the point of distinguishing the two.

> **Reconnect covers "down at startup, comes up later" — not the reverse.** A downstream that
> dies *after* connecting is not currently re-detected: its health stays `connected` and
> `GET /v1/tools` keeps reporting its last known tool count until woollama restarts. Dispatches
> through it fail in the meantime.

| Env var | Default | Meaning |
|---|---|---|
| `WOOLLAMA_MCP_RETRY_MAX_SECS` | `60` | Backoff ceiling for downstream reconnect. `0` disables retry entirely. |
| `WOOLLAMA_MCP_MAX_NESTING` | `2` | Federation levels a re-exported tool may carry. `0` disables the cap. |

Refresh is **background-only** — a request never triggers a downstream fetch. That is
deliberate: if it did, then in a federated topology one router's `tools/list` would fetch from
the next, whose `tools/list` would fetch back, recursing at request time. Serving from a cached
snapshot is what keeps that impossible.

`WOOLLAMA_MCP_MAX_NESTING` exists because reconnect makes unbounded growth reachable. Tool names
gain one namespace level per federation hop, so in a mutual topology (A consumes B, B consumes A)
each refresh ingests a roster that already carries the previous round's nesting — one level per
tick, forever. The cap bounds it; capped tools are logged with a count rather than silently
dropped.

**`GET /v1/tools`** reports what the router actually has:

```json
{
  "tools": ["shelf.search"],
  "data":  [{"name": "mcp__shelf__search", "server": "shelf", "tool": "search"}],
  "servers": [
    {"name": "shelf", "transport": "http", "health": "connected", "tools": 1},
    {"name": "git", "transport": "stdio", "health": "retrying", "tools": 0,
     "attempts": 4, "last_error": "No such file or directory (os error 2)"}
  ]
}
```

A downstream that is **down appears here with its reason**, rather than being omitted — absence
and not-yet-connected are indistinguishable from outside, and a router showing neither would look
healthy with its tools quietly gone. Each tool names its originating server, which is how to read
a federated namespace without driving an MCP handshake by hand.

> **Check your config before a reload.** Because a bad entry is skipped rather
> than fatal, the only trace at runtime is a line in the boot log. Run
> `woollamad check-config` to make that actionable — it validates `mcp.json`,
> `recipes.toml` and `inferencers.toml`, reports which servers are usable and
> which were skipped and why, and **exits non-zero if anything is wrong**. It
> connects to nothing and binds nothing, so it is safe to run against a live
> deployment's config dir, and it is the thing to gate a `systemctl reload` (or
> a CI job) on.
>
> ```console
> $ woollamad check-config
> error: mcp.json: server 'bad' header 'Authorization' is the bare auth scheme
>        'Bearer' with no credential — an unset ${VAR} expands to nothing
> mcp.json: 1 server(s) usable (good), 1 skipped
> recipes.toml: 4 recipe(s)
> inferencers.toml: OK
> 1 problem(s) found
> ```

woollama also warns when a server sends `headers` to a plain `http://` URL that
isn't loopback, since that puts the credential on the network in cleartext on
every request.

`env` has no meaning for a `url` server (there is no child process), and the
child-env scrub below likewise applies only to the stdio form. A `url` server
also cannot be handed to **claude-code delegation**: a recipe whose tools
reference one is rejected with a 400 naming the server, rather than having the
child `claude` process connect to the downstream directly, outside woollama's
allow-list boundary.

> **The `url` form is `woollamad`-only.** The Python reference server (the
> differential-test oracle) requires `command` and will fail to load a config
> containing a `url` server. If you share one config dir between the two, keep
> `url` entries out of it.

> **Interpreter & `PATH`.** `command` is resolved against woollama's *own*
> environment when the server is spawned. A bare name (`python`, `uvx`, `node` —
> including in the `conversationStore` examples below) picks whatever is first on
> `PATH` at spawn time, which need not be the interpreter that has the server's
> dependencies when woollama runs **outside its virtualenv** (launched by an
> absolute path, or as a `systemd` unit with a minimal `PATH`). Pin `command` to an
> absolute interpreter (e.g. your venv's `python`) if startup is sensitive to
> which environment launched it.

> **On a downstream server that fails to start, the two implementations
> differ.** `woollamad` (the canonical router) logs a warning and skips it, so
> the router comes up in a known-degraded state with that server's tools absent.
> The Python reference server aborts startup instead. Don't design around either
> behaviour without checking which one you're running.

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
| `command` | — | The fabric binary (default `"fabric"`, resolved on `PATH`). Used for managed `--serve` **and** for the one-shot `fabric -a` CLI on the [vision path](patterns.md#vision-image-input-for-fabric-patterns). |
| `address` | — | Fixed `host:port` to bind in managed mode (default: a persisted free loopback port). |
| `default_model` | — | Fallback `<provider>/<model>` for fabric patterns when a run omits `model`. Required for a fabric pattern to be addressable as `woollama/<name>` via `/v1/chat/completions` (which has no per-call model slot). |

> **Why here and not `inferencers.toml`?** fabric is not OpenAI-compatible, and the
> engine's `inferencers.toml` loader requires every entry to have a `base_url` —
> a fabric entry there would break config load. The fabric backend is a
> server-layer plugin, not an engine inferencer.

> **Vision needs a vision-capable model.** Image input (`image_url`) on a fabric
> pattern is dispatched via `fabric -a` (see [patterns.md](patterns.md#vision-image-input-for-fabric-patterns)).
> Pass a vision `model` (e.g. `ollama/llama3.2-vision`) on the request; a text-only
> `default_model` won't see the image.

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

#### `[recipes.<name>.variables.<var>]` — variable metadata (optional)

Annotate a recipe's `{{var}}` tokens with a default, an allowed-value list, and a
description. Each field is optional, and the whole overlay is optional — a recipe
with no `[variables]` table behaves exactly as before (bare names).

```toml
[recipes.streamer]
inferencer = "ollama/qwen3:14b-iq4xs"
system = "Write a {{tone}} summary in {{language}}."

[recipes.streamer.variables.tone]
default = "neutral"                      # used when the caller omits {{tone}}
choices = ["neutral", "terse", "wry"]    # surfaced for UIs; NOT enforced
description = "Writing tone"
# {{language}} needs no annotation — it stays a name-only variable.
```

| Field | Required | Description |
|---|---|---|
| `default` | — | Value substituted when the caller doesn't supply this variable on `render`/`run`. A caller-supplied value always wins; a variable with no default is left verbatim. |
| `choices` | — | The allowed values, surfaced in `/w1/patterns` discovery (e.g. for a UI picker). **Advisory only** — not server-enforced, so a caller may still pass a value outside the list. |
| `description` | — | Human-readable docs; shown in `/w1/patterns` and carried across to the MCP prompt argument. |

The overlay is keyed by variable name; the `{{var}}` tokens in `system` stay
authoritative for *which* variables exist and their order. An entry whose name
isn't actually in `system` is simply unused. Metadata applies to **native recipes
only** — fabric-library patterns carry none. See the
[`GET /w1/patterns`](patterns.md#get-w1patterns--discovery) shape for how it surfaces.

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
`base_url`. Pass-through covers chat (`/v1/chat/completions`), images
(`/v1/images/generations`), and embeddings (`/v1/embeddings`) — all forwarded to
the inferencer's own OpenAI-compatible endpoints under `base_url`.

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
| `extra_body` | — | Fields merged into each **orchestration** request (e.g. `temperature`, ollama's `options`). **Not applied to pass-through** — a `<provider>/<model>` request is relayed as the client sent it. Set the field client-side for pass-through; it forwards faithfully. |
| `models` | — | Static model ids to list in `GET /v1/models` as `<provider>/<id>` (no key needed to *list*). |
| `discover` | — | If `true`, live-query the provider's own `/v1/models` and list those too (needs the key). `ollama` defaults to `true`. |
| `model_patterns` | — | fnmatch globs that filter `discover` results (e.g. `["claude-*"]`); empty = list all discovered. |

Models are still **routable by raw id** (`anthropic/claude-opus-4-8`) whether or
not they're listed — `models`/`discover` only control *discoverability* in
`GET /v1/models` (what a list-backed picker can offer).

### Model pooling / device-aware inferencers (optional)

An inferencer that declares `management_url` becomes **device-aware**: woollama
loads models on it on demand instead of failing with a not-loaded error, resolves
stable **virtual model names**, and serializes/queues requests around the
backend's real concurrency limit — returning `503` + `Retry-After` under
backpressure instead of hanging. Fully additive: an inferencer with no
`management_url` behaves exactly as the stateless pass-through above.

```toml
[inferencers.device]
base_url = "http://127.0.0.1:8800/v1"
management_url = "http://127.0.0.1:8800"   # enables pooling for this inferencer
parallel = 1                                # concurrent requests the backend serves per model
pool_max = 2                                # max concurrently-loaded models before eviction
queue_max = 8                               # requests queued per model before 503
queue_timeout = 30                          # seconds a queued request waits before 503

[inferencers.device.virtual]
default = "big-model-7b"     # device/default -> whatever's currently loaded, else this
coder = "code-model-14b"     # device/coder -> code-model-14b
```

| Field | Required | Description |
|---|---|---|
| `management_url` | — | Base URL of the backend's model-management API (`GET/POST /api/v1/models/{running,start,stop}`). Its presence is what turns on pooling for this inferencer. |
| `parallel` | — | How many requests **woollama** sends the backend concurrently per loaded model (default `1`). Sizes the per-model queue semaphore. |
| `pool_max` | — | Max models kept loaded at once. When a new model is needed at capacity, the LRU **idle** model is evicted to fit (never a model that's in-flight or has a queued request). If every loaded model is busy, the request **waits** for one to free up rather than being refused — see *Queueing across a model swap* below. Unset ⇒ no cap and no auto-eviction. |
| `queue_max` | — | Max requests queued per model before woollama returns `503` + `Retry-After` instead of enqueuing more. Unset ⇒ no queue-depth limit (only `queue_timeout` bounds the wait). |
| `queue_timeout` | — | Seconds a request waits for its turn before woollama gives up and returns `503` + `Retry-After` (default `30`). Bounds three waits: a place in a model's queue, a cold load, and — since v0.15 — a **model swap** on a full device. |
| `virtual` | — | Table of alias → real model id. The reserved key `default` resolves `<provider>/default` against the backend's **current residency, read from the backend itself** at request time, falling back to this table entry if nothing is loaded. Other keys are ordinary aliases (`<provider>/<alias>` → the real id). |

#### Model capabilities — what `default` is allowed to pick

`<provider>/default` is never asked in the abstract; it is asked **at an endpoint**. A device can
hold an embedding model, a reranker and a chat model at once, and only one of them can serve
`/v1/chat/completions`. woollama drops residents it *positively knows* cannot serve the endpoint.

**Discovered, where the backend says so.** The `device` protocol's running response carries a
sibling `instances.running[]` array with a `capabilities` list per model, in the same payload
woollama already fetches — so this costs no extra call and needs no configuration. A resident the
backend labels `embedding` or `rerank` is not a candidate for a chat request.

**Declared, for backends that publish nothing:**

```toml
[inferencers.device.capabilities]
embedding = ["*Embedding*"]
rerank    = ["*Reranker*"]
```

Each key is a capability; each value is a list of glob patterns over model ids. **Declare what a
model *is*, not what may chat.** A positive allow-list ("these models can chat") looks safer but
requires naming every model that might ever legitimately serve the endpoint — on a shared device
that means predicting what other consumers will load, and a model nobody predicted gets refused.
That failure has been observed in production. Declaring exclusions survives models you did not
foresee, because a new chat model matches none of them and stays eligible.

Declared capability is also enforced **before dispatch** and surfaced in `GET /v1/models`:

- A request naming a model declared for a different capability is refused with `400`, naming what
  the model *is*, rather than being sent to the backend. Some backends answer an unsupported
  request by failing in a way that takes the whole model service down until it is reloaded, so a
  refusal woollama can make cheaply is worth more than the backend's own answer.
- `GET /v1/models` carries a `capabilities` array on entries that have declarations. Entries
  without one are simply undeclared — **absent never means "cannot"**.

Only *declared* capability is used for those two, not discovered: discovery describes what is
currently **resident**, while both of these concern model ids that may not be loaded at all.

Precedence: **config beats discovery beats unknown**, and *unknown always means eligible*. So a
backend that publishes nothing and has no declarations behaves exactly as it did before, and an
operator correcting a backend has the last word.

#### Knowing which model will actually answer

`GET /v1/models` is a catalogue: it lists what an inferencer declares plus whatever discovery
found. For a **management-capable** inferencer it also reports what is *resident right now*, so a
caller can tell "this model exists" from "this model will answer" without sending a request that
may `503` after a thirty-second load:

```json
{"id": "device/Qwen3-30B-Instruct", "object": "model", "owned_by": "device", "loaded": true}
{"id": "device/Qwen3-Coder-30B",    "object": "model", "owned_by": "device", "loaded": false}
{"id": "device/GLM-4.7-Flash",      "object": "model", "owned_by": "device",
 "loaded": true, "undeclared": true}
```

- **`loaded`** answers *"is this model in memory"* — which is **necessary but not sufficient** for
  *"will this call succeed"*. Do not build a pre-flight check on it alone. On real hardware the two
  differ in at least three ways: a resident model may not be **servable on the endpoint you called**
  (an embedding model on the chat path — see capabilities above), may **crash on certain inputs**,
  and may be **evicted by another consumer** between your check and your call. Treat it as "worth
  trying" rather than "will work".

  There is also a window where `loaded` is **affirmatively wrong** rather than merely incomplete:
  after an instance crash a backend may keep reporting the model as running — measured at 2–5
  seconds on one device, with a stale in-flight count alongside it — so `loaded: true` can describe
  a model that is already gone. That window is exactly when someone is most likely to be reading
  this field and drawing conclusions from it.
- `loaded` appears only for inferencers that declare a `management_url`. An inferencer with no pool
  cannot know, and **absent never means "no"**.
- A model that is resident but **not** in the inferencer's `models` list is still routable as
  `<provider>/<id>`, and it is the one that will answer — so it is listed, flagged `undeclared`
  because the operator did not promise it.
- If the residency read itself fails, `loaded` is omitted rather than reported as `false`: not
  seeing is not the same as not loaded.

The read shares the same coalescing window `<provider>/default` uses, so listing models does not
add a backend round trip per request.

#### Recovering from a model that disappears

A backend can drop a model woollama believes it is running — an instance crash, an eviction by
another consumer. On an upstream `5xx`, woollama re-checks what the backend is actually running and
stops believing in anything that has gone; the next request loads it again, and the drop is logged
naming the model.

This asks the backend rather than interpreting its error. Whether a particular message means
"unloaded" is vendor wording; whether a model is running is a question the backend can answer.

Recovery happens on the **next** request — the one that hit the failure still fails.

#### woollama is not the device's only consumer

Pool state is a **cache of the backend's state, not a ledger of it**. The vendor's own UI may load
models; any caller needing endpoints woollama does not serve (images, embeddings, ASR) has to
drive the backend directly, and that traffic changes residency. Exclusivity is therefore not
achievable even in principle, and two things follow:

- **`<provider>/default` reads through to the backend.** It asks what is actually running rather
  than trusting woollama's own bookkeeping, which would be empty after a restart and stale after
  anyone else's load. Concrete: without this, `default` returns *"no model is loaded"* while the
  backend is running one, or falls through to the `virtual.default` entry and evicts a perfectly
  good resident model to load it. Concrete model ids and ordinary aliases need no such round trip.
- **`parallel` bounds what woollama sends, not what the backend receives.** Another consumer can
  drive it concurrently regardless. Treat it as self-restraint, not a guarantee.

| Env var | Default | Meaning |
|---|---|---|
| `WOOLLAMA_POOL_RESIDENCY_TTL_MS` | `1000` | Coalescing window for residency reads — a burst of `default` requests shares one query. `0` reads every time. This is a *coalescing* window, not a staleness budget: it bounds request amplification, and does not license trusting the cached view for longer. |

**Inferencers sharing a `management_url` share one pool and one gate.** Two routes onto the same
device are two views of one truth, so keeping them separate meant neither saw the other's loads —
alternating between them swapped the model on *every* call — and it enforced `parallel` once per
route, **doubling** the concurrency the device actually saw. The shared gate takes the most
restrictive limit configured across them, except `queue_timeout`, where the most forgiving value
wins (a shorter wait causes spurious `503`s on a slow cold load without protecting the device).

#### Queueing across a model swap

On a capacity-bound device, serving model B can mean evicting model A. woollama used to answer
such a request with an immediate `503`: A was busy, so nothing was evictable, so B was treated as
unservable. But B is not unservable — it is servable after work that is already draining, and
making the caller run a retry loop over a resource woollama is already sequencing pushes a
cold-load-length decision onto every client.

A request for a non-resident model now **waits for the swap**, bounded by `queue_timeout`:

- A busy model is still never evicted. That protection is unchanged.
- While a swap is pending, arriving requests for the *resident* model queue behind it, so it can
  drain and become evictable. Without this, steady traffic for the resident model would keep it
  permanently busy and the waiter would time out anyway — a slow failure instead of a fast one.
  Work already in flight is never interrupted; only new work is held back.
- `503` + `Retry-After` is still the answer when the wait genuinely exceeds `queue_timeout`.
- A request held this way is **not** counted against `queue_max` while it waits — it has not
  joined the queue yet. `queue_max` is re-checked at the moment it does, so a burst released
  together cannot overshoot the limit; the surplus gets `503` as it would have on arrival.

Each swap serves at least the request that caused it, so two consumers competing for one slot
alternate rather than thrash. The cost of alternation is a cold load per switch, so
`queue_timeout` needs the same headroom described below — more, if you expect contention.

> **`queue_timeout` must exceed your backend's COLD-LOAD time, and the default may not.**
> The first request for a model that isn't resident waits for the backend to load it, and that
> is backend- and model-dependent — measured at **33s** for a 30B model on one NPU device, where
> the 30s default would have returned `503` on first use. The failure looks exactly like
> pooling being broken: the request that should have triggered the load is the one that times
> out, so the model never becomes warm and every retry repeats the wait. Measure a cold load on
> your own hardware and set `queue_timeout` comfortably above it.

Pooling applies to `/v1/chat/completions` (in both `woollama` and `woollamad`);
the `/v1/responses` path is not pooled yet. Raw real model ids remain routable
alongside virtual names — `virtual` only adds aliases, it doesn't restrict.

See also: [Pattern templating](patterns.md) · [Extending woollama](extending.md) ·
[Environment variables](environment.md) · [Security model](security.md).
