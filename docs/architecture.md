# A model, tool, executor router — architecture

Status: **the target design.** Co-designed 2026-05-31 (validated then by a
throwaway probe, since obsolete); now the implemented project lives in this repo
as **woollama** (naming settled — see `naming.md`). This document describes the
full intended shape; much of it is built and some is still aspirational.

> **Implementation status:** for what's actually built vs. still planned, see
> [`roadmap.md`](roadmap.md); for the slice-by-slice history, see
> [`build-log.md`](build-log.md). In particular, several specifics below are now
> realized differently or more concretely than first sketched — e.g. the MCP
> HTTP surface is **Streamable HTTP mounted at `/mcp`** (not `/mcp/sse`), and
> the bundled inferencer set + `inferencers.toml` is the current provider
> mechanism. Treat code + `roadmap.md` as authoritative where they differ.

## What it is

A small daemon that **routes inference requests, tool calls, and executor
choice** between AI clients and AI backends, using two standard wire formats
and inventing none of its own.

Three axes of routing, one daemon:

| | what it routes | namespace | discovery |
|---|---|---|---|
| **models** | inference requests (raw / pattern / variant / recipe) | `<provider>/<name>` | `GET /v1/models` + MCP resources |
| **tools** | tool calls during a chat-loop | `<server>.<tool>` | MCP `tools/list` |
| **executors** | which backend handles a given model | implicit in the model name's `<provider>` prefix | per-provider config |

## What it is not

- Not an inference engine. It uses other engines.
- Not a UI. Cosmic clients (the panel, CLI, Claude Desktop) connect to it.
- Not a tool host (except for a small built-in set). Tools live in MCP servers.
- Not a fabric clone or extension. fabric is one possible pattern source.
- Not opinionated about agent loops, scratchpads, or chain-of-thought — those
  belong in patterns or in the model.

## Binding — local-only and ephemeral by default

Same pattern as the existing fabric subprocess wrapping: **bind to a random
free loopback port; persist the chosen address to `$XDG_RUNTIME_DIR/<name>.addr`
so clients can discover it; never bind to `0.0.0.0` without explicit
opt-in.** The router holds API keys and routes to local resources; it must
not be LAN-reachable by default.

Surfaces:
- **Unix socket** (default for local MCP clients — the panel, the CLI):
  `$XDG_RUNTIME_DIR/<name>.sock`. No network at all.
- **HTTP loopback on a random free port** (for OpenAI-compatible clients
  that need HTTP): `127.0.0.1:<random>`, address persisted to
  `$XDG_RUNTIME_DIR/<name>.addr`.
- **LAN bind** (`0.0.0.0:<port>`): only when explicitly configured AND with
  required `api_key`, mirroring how fabric upstream forces auth on LAN.

Override hierarchy:
1. `$ROUTER_ADDRESS=host:port` env (explicit, highest precedence)
2. `[server] bind = "0.0.0.0:8889"` in config (explicit)
3. Random free loopback port (default)

The persisted address file is the discovery mechanism — clients read it the
same way `cosmic-fabric fabric-url` works today.

## Two inbound surfaces (same port, path-routed)

```
GET  /v1/models                — OpenAI-compat list of all addressable models
POST /v1/chat/completions      — OpenAI-compat chat (the inference primitive)
POST /v1/embeddings            — OpenAI-compat embeddings (when needed)

POST /mcp/sse, /mcp/messages   — MCP over HTTP+SSE for network clients
stdio (subprocess)             — MCP over stdio for local clients
```

OpenAI surface: any tool that speaks OpenAI is a client without code changes.
Cursor, Aider, Continue, the `openai` Python/JS SDKs, anything with
`OPENAI_API_BASE` — they all just work.

MCP surface: rich clients that want tool discovery, prompts, resources, and
bidirectional callbacks. The cosmic panel is one such client.

## Two outbound protocols

```
MCP        ←→ tool servers (lackpy, fabric-mcp, filesystem, git, sqlite, …)
              prompts, tools, resources, callbacks
OpenAI     ←→ inference backends (Ollama native, Anthropic compat shim,
              vLLM, llama.cpp, Together, Groq, OpenRouter, …)
              chat completions with tools + streaming
```

These cover different concerns:
- **MCP** is the discovery + control + tool primitive
- **OpenAI** is the inference primitive

Composing them in the router gives orchestrated chat without inventing
anything.

## Four kinds of model — one namespace

```
model: "ollama/qwen3:14b-iq4xs"     raw inferencer — pass-through
model: "anthropic/claude-opus-4-7"  raw inferencer — pass-through
model: "fabric/scribe-summarize"    pattern — fetch system prompt, route to
                                    pattern's configured inferencer
model: "cosmic/qwen3-spicy"         variant — model + sampling config bundle
model: "cosmic/deep-research"       recipe — pattern + tools + inferencer +
                                    sampling, fully orchestrated
```

Resolution table:

| model field | client experience | router behavior |
|---|---|---|
| `provider/raw-model`, no tools | standard OpenAI | pass-through to backend |
| `provider/raw-model`, tools supplied | standard OpenAI | pass-through; tool_calls returned to client (client handles) |
| `fabric/pattern`, pattern is `tool_use = false` | one final answer | fetch pattern → prepend system → route to pattern's inferencer |
| `fabric/pattern`, pattern is `tool_use = true` | one final answer | fetch pattern → resolve tool allow-list → chat-loop with internal tool dispatch |
| `cosmic/variant` | standard OpenAI | resolve to underlying provider/model + apply sampling |
| `cosmic/recipe` | one final answer | resolve full composition → chat-loop |

Mechanical dispatch on parse of the `model` field plus a policy lookup. No
extension to OpenAI's wire format. The pattern-as-model and recipe-as-model
concepts are how the router exposes "stored prompts" (which OpenAI doesn't
have natively) through the OpenAI client surface.

## Recipes — pre-packaged compositions

```toml
[recipes."deep-research"]
pattern = "fabric/scribe-look-it-up"           # system prompt source
prompt_arguments = { depth = "really" }        # template args
inferencer = "anthropic/claude-opus-4-7"       # override pattern's default
tools = ["http_get_wikipedia", "http_get_arxiv"]
sampling.temperature = 0.7
description = "Deep research with Opus + Wikipedia + arXiv. Long answers."
```

Recipes are first-class models. `GET /v1/models` lists them. Any OpenAI client
can use `cosmic/deep-research` without knowing fabric, MCP, or tools exist.

Recipes also surface as MCP **prompts** for MCP clients that want the same
composition through that surface.

## Built-in tools (TL.0 — router-native)

The router ships a small set of in-process tools, distinct from MCP-discovered
tools. The current set:

| tool | mode | notes |
|---|---|---|
| `http_get` | daemon | URL fetch via Jina; per-instance `allow_domains` |
| `read_file` | daemon | cwd-jailed UTF-8 read; per-instance `roots` |
| `run_shell_confirmed` | panel-confirm | argv array, no shell expansion, requires panel approval |

(Plus the hygiene rules for any tool result: empty-result sentinel, schema
validation, exception wrapping, truncation cap.)

These exist alongside MCP-discovered tools in the registry, namespaced the
same way (`cosmic.http_get_wikipedia` if we want to be fully consistent —
TBD).

## Bidirectional MCP (TL.3)

When a handler we're calling needs to invoke our tools (e.g., lackpy's
program calls our `read_file`), the model is: **two MCP connections,
symmetric roles**. The router is server-to-the-handler for callbacks, while
also being client-to-the-handler for delegation. Per-handler `roles =
["server", "client"]` config declares the directionality. Validated end-to-end
in `examples/mcp-hello/probe_client.py`'s elicitation test.

## Tool visibility (the per-client filter inversion — deferred)

Tools have descriptive properties (`internal`, `conversational`, `research`,
etc.); each named client (panel, CLI, ollama-mcp, lackpy-mcp) declares what
properties or explicit tools it consumes. The router applies the filter at
exposure time.

This is *premature* until the named clients exist in config. Until then, the
per-pattern/per-recipe `tools = [...]` allow-list is the working
approximation. Real client filters land when the panel and at least one
sub-inferencer are named entries.

## Configuration

Configuration mirrors Claude Code's `.mcp.json` shape for familiarity, with
extensions for our specific needs:

```jsonc
// ~/.config/<name>/mcp.json
{
  "mcpServers": {
    "fabric": {
      "command": "fabric-mcp",
      "args": ["--transport", "stdio"],
      "env": { "FABRIC_BASE_URL": "http://localhost:11434" },
      "roles": ["server"],
      "features": { "streaming": "progress-typed-events" }
    },
    "lackpy": {
      "command": "lackpy",
      "args": ["mcp"],
      "roles": ["server", "client"]            // bidirectional
    }
  },
  "inferencers": {
    "ollama":   { "url": "http://localhost:11434" },
    "anthropic":{ "url": "https://api.anthropic.com",
                  "api_key": "$ANTHROPIC_API_KEY",
                  "compat_path": "/v1" }
  }
}
```

Plus a separate `policy.toml` for recipes, variants, tool instances, and
per-pattern metadata.

## Vendored MCP servers (the bundle)

The router ships with a curated bundle of MCP servers:

| package | source | shape |
|---|---|---|
| `fabric-mcp` | fork of `ksylvan/fabric-mcp` | fixed streaming bug + variable substitution |
| `lackpy-mcp` | `teaguesterling/lackpy` | use as-is |

Inference backends speak OpenAI-compat natively, so we don't ship wrappers for
Ollama, Anthropic, etc. — the router talks to their OpenAI endpoints directly.

Forks live as their own repos (`<namespace>/cosmic-mcp-fabric` etc.) with
clear README diffs from upstream. Sync periodically. Users can override any
bundled server by setting `command` in mcp.json.

## What this collapses from the prior cosmic-fabric prototype

About 60% of yesterday's tool-calling prototype:

| in prototype | replaced by |
|---|---|
| `core.FabricClient` (fabric REST adapter) | the (forked) fabric-mcp server |
| `core.run_with_tools` against Ollama `/api/chat` | OpenAI chat-completions client + MCP tool dispatch |
| Bespoke socket protocol panel↔daemon | MCP over stdio |
| `core.assemble_prompt` | fabric-mcp's `prompts/get` (with vars in the fork) |
| meta.toml sidecars | `[patterns.X]` and `[recipes.X]` in cos-fab's policy |
| Pattern frontmatter parser | gone with meta.toml |
| `core.inst_to_options` (sampling-knob translation) | direct openai-compat fields in recipes |

What survives:
- Hygiene rules (sanitize, validate, exception-wrap)
- Chat-loop *shape* (re-implemented against OpenAI + MCP)
- Built-in tool callbacks (http_get, read_file, run_shell_confirmed)
- Per-pattern tool allow-list semantics (moved to recipes)

## What stays open

1. ~~**Naming.**~~ Settled: **woollama** (see `naming.md`).
2. **Tools as named tables instead of bare names.** Will let tool entries
   carry per-tool metadata (description, version, deprecation) alongside the
   name. Deferred.
3. **Recipe inheritance.** One recipe extending another. Deferred until two
   recipes share enough config to motivate it.
4. **Per-client visibility filters.** Real client filters land when the panel
   and at least one sub-inferencer are named entries in mcp.json.

## What the probe demonstrated (historical)

> The validation probe (once at `/tmp/router_probe/`) is **gone** — its job was
> to prove the shape, which it did; the real implementation now lives in
> `src/woollama/` and is far past this. Kept as a record of the original
> end-to-end validation.

The architecture compiled into ~200 LoC of Python:

- FastAPI HTTP server with `/v1/models` and `/v1/chat/completions`
- Per-request stdio connection to the hello MCP server
- Two model namespaces: `ollama/*` (pass-through) and `cosmic/*` (recipe)
- One hardcoded recipe (`cosmic/streamer`) that bundles a system prompt + the
  hello server's `count_to` tool + qwen3 as the inferencer
- The full chat-loop: OpenAI client → router → Ollama with tools → MCP for
  tool execution → result back to Ollama → final answer back to OpenAI client

Tested end-to-end with the `openai` Python SDK as the client. Validates the
entire architecture in code, not just in conversation.
