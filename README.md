# woollama

**Web Over Ollama (and Llamas).** An MCP + OpenAI router for AI desktops.

woollama sits between AI clients (Cursor, the OpenAI SDK, Claude Desktop,
cosmic-fabric, anything that speaks OpenAI or MCP) and AI backends (Ollama,
Anthropic, fabric, lackpy, filesystem MCPs, anything that speaks OpenAI or
MCP). It composes them into orchestrated calls without inventing a new
protocol.

```
                          ┌─────────────────────┐
                          │   AI clients        │
                          │   (any OpenAI or    │
                          │    MCP client)      │
                          └──────────┬──────────┘
                                     │
                  ┌──────────────────┴───────────────────┐
                  │            woollama                  │
                  │  OpenAI server  +  MCP server        │
                  │  ───────────────────────────────     │
                  │  routes models, tools, executors     │
                  │  composes patterns + tools + models  │
                  │  into named recipes                  │
                  └──────────────────┬───────────────────┘
                                     │
                  ┌──────────────────┴───────────────────┐
                  │                                      │
              ┌───┴────┐                            ┌────┴────┐
              │ MCP    │  tools, prompts, resources │ OpenAI  │  inference
              │ tool   │                            │ compat  │
              │ servers│                            │ backends│
              └────────┘                            └─────────┘
              fabric-mcp, lackpy,                   Ollama, Anthropic,
              filesystem, git, …                    vLLM, llama.cpp, …
```

## Status

**v0.1 — Python prototype.** Works end-to-end as an OpenAI-compatible router
with hidden chat-loop orchestration. Spawns one MCP tool server (the bundled
hello example). One hardcoded example recipe. Local-only ephemeral binding
(random loopback port). Not production-ready; the architecture is validated
and the shape of v1 is settled.

> **Implementation note: woollama will be a Rust program at v1.0.**
> The Python in `src/woollama/` is a prototype used to iterate the
> architecture quickly. The Rust port lands when the design surface is
> stable. See [`docs/rust-transition.md`](docs/rust-transition.md) for the
> explicit transition criteria.

See `docs/architecture.md` for the full design.

## Quick taste

The router is OpenAI-compatible, so any OpenAI client can drive it:

```python
import openai
c = openai.OpenAI(base_url="http://127.0.0.1:<port>/v1", api_key="x")

# Pass-through to Ollama
r = c.chat.completions.create(
    model="ollama/qwen3:14b-iq4xs",
    messages=[{"role": "user", "content": "Hi"}],
)

# Orchestrated: pattern + tools + model, transparent to the client.
# The chat-loop happens inside woollama; client sees only the final answer.
r = c.chat.completions.create(
    model="woollama/streamer",
    messages=[{"role": "user", "content": "Please count to 4."}],
)
```

The `<port>` is ephemeral by default — woollama binds to a free loopback
port at startup and writes it to `$XDG_RUNTIME_DIR/woollama.addr` for clients
to discover. Same pattern as a local `fabric --serve` instance.

## Install (development)

```sh
git clone https://github.com/<you>/woollama
cd woollama
uv sync                           # creates .venv and installs deps
uv run woollama                   # starts the router; prints its address
```

In a second shell:

```sh
# Discover the address
cat "${XDG_RUNTIME_DIR:-/tmp}/woollama.addr"
# Then point an OpenAI client at it (see Quick taste above).
```

## Design principles

1. **Two standards, neither extended.** MCP for tool/prompt/resource
   discovery and execution; OpenAI chat-completions for the inference
   primitive. woollama is a router between them.
2. **Local-only, ephemeral by default.** Random loopback port, persisted
   address file for discovery, never `0.0.0.0` without explicit opt-in. The
   router holds API keys and routes to local resources — it should not be
   LAN-reachable.
3. **The model namespace is the universal addressing scheme.** Raw inferencers
   (`ollama/X`), patterns (`fabric/X`), variants (`woollama/X`), and full
   recipes (`woollama/X`) are all addressable through OpenAI's standard
   `model` field. No new wire format.
4. **woollama owns routing, not inference or tools.** It uses other people's
   inference engines (Ollama, Anthropic) and other people's tool servers
   (lackpy, filesystem, git). It composes them.
5. **she talks to llamas.**

## What v0.1 includes

- HTTP server at `/v1/models` and `/v1/chat/completions` (OpenAI-compat)
- `ollama/X` pass-through to local Ollama at `localhost:11434`
- One hardcoded example recipe (`woollama/streamer`) that exercises a
  bundled MCP tool (`count_to` from `examples/mcp-hello/`)
- Per-request MCP subprocess for tool dispatch (correct but not optimal —
  long-lived connection pooling is a follow-on)
- Ephemeral loopback binding + address discovery file

## What v0.1 does not include (yet)

- Streaming on either the OpenAI side or MCP side (call the existing
  endpoints non-streaming for now)
- Real configuration file (recipes + MCP servers are hardcoded in v0.1)
- Multiple MCP server discovery + the unified tool registry
- Cloud inference backends (Anthropic, OpenAI, etc.) — Ollama only
- Unix socket transport alongside HTTP loopback
- The MCP server side (woollama as an MCP server to its clients)

These are all sized and planned; see `docs/architecture.md` for the full
target shape.

## Origin

woollama is the production-grade rewrite of an architecture co-designed
in [cosmic-fabric](https://github.com/teaguesterling/cosmic-fabric), which
remains a frontend (and will use woollama as its router engine). The design
docs that brought woollama here:

- `docs/architecture.md` — the model/tool/executor router design
- `docs/naming.md` — how we landed on this name

## License

MIT — see `LICENSE`.
