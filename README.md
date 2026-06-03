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

**Python prototype — multi-backend router, both surfaces live.** woollama works
end-to-end as:

- an **OpenAI-compatible server** (`/v1/chat/completions`, `/v1/models`) with
  pass-through *and* hidden chat-loop orchestration of recipes;
- an **MCP server** to its own clients — over **stdio** (`woollama mcp`) and
  over **Streamable HTTP** at `/mcp`, mounted on the *same port* as `/v1/*`. It
  re-exports every discovered downstream tool (namespaced) plus a `chat` verb,
  i.e. it's an MCP aggregator.

It routes inference across **multiple backends** by `<provider>/<model>` —
`ollama` (local), `anthropic`, `openai`, `groq`, `together`, `openrouter`, and
**any OpenAI-compatible endpoint** you add in `inferencers.toml` (e.g.
self-hosted vLLM) — plus `claude-code/<model>`, a keyless path to Claude via the
local CLI. Config is file-driven (`mcp.json`, `recipes.toml`, `inferencers.toml`).
Long-lived MCP connections; local-only ephemeral binding by default.

Not production-ready. **Current status and what's next live in
[`docs/roadmap.md`](docs/roadmap.md).**

> **Implementation note: woollama will be a Rust program at v1.0.**
> The Python in `src/woollama/` is a prototype used to iterate the
> architecture quickly. The Rust port lands when the design surface is
> stable. See [`docs/rust-transition.md`](docs/rust-transition.md) for the
> explicit transition criteria.

See `docs/architecture.md` for the full target design and
`docs/build-log.md` for the slice-by-slice history.

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

## What works today

- OpenAI surface: `/v1/models`, `/v1/chat/completions` (pass-through +
  recipe orchestration), `/v1/tools` introspection
- Multi-backend routing by `<provider>/<model>`: ollama, anthropic, openai,
  groq, together, openrouter, `claude-code`, + any OpenAI-compatible endpoint
  via `inferencers.toml`
- MCP server side: stdio (`woollama mcp`) **and** Streamable HTTP at `/mcp` on
  the same port — recipes as prompts, a `chat` verb, and every downstream tool
  re-exported (aggregator)
- File-driven config (`mcp.json`, `recipes.toml`, `inferencers.toml`), multi-
  MCP-server discovery + unified tool registry, long-lived MCP connections
- Recipe allow-list enforced as a security boundary; ephemeral loopback binding
  + address discovery file

## Not yet (next on the roadmap)

- Streaming (OpenAI SSE out + MCP progress events) — biggest open item
- Unix socket transport alongside HTTP loopback
- Stateful Conversations/Responses surface (design in
  `docs/conversations-api-design.md`)
- The Rust v1.0 port

Full scorecard, ordering, and pending verifications:
**[`docs/roadmap.md`](docs/roadmap.md)**.

## Origin

woollama is the production-grade rewrite of an architecture co-designed
in [cosmic-fabric](https://github.com/teaguesterling/cosmic-fabric), which
remains a frontend (and will use woollama as its router engine). The design
docs that brought woollama here:

- `docs/architecture.md` — the model/tool/executor router design
- `docs/naming.md` — how we landed on this name

## License

MIT — see `LICENSE`.
