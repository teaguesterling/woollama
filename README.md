# woollama

**Web Over Ollama (and Llamas).** An MCP + OpenAI router for AI desktops.

📖 **Documentation: [woollama.readthedocs.io](https://woollama.readthedocs.io/)**

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

- an **OpenAI-compatible server**: `/v1/chat/completions` (pass-through *and*
  hidden chat-loop orchestration of recipes, both with `stream:true` → OpenAI
  SSE), `/v1/models`, `/v1/tools`, and a **stateful surface** —
  `/v1/responses` + `/v1/conversations` (OpenAI Responses/Conversations shape;
  see below);
- an **MCP server** to its own clients — over **stdio** (`woollama mcp`) and
  over **Streamable HTTP** at `/mcp`, mounted on the *same port* as `/v1/*`. It
  re-exports every discovered downstream tool (namespaced, with `output_schema`)
  plus a `chat` verb that emits live tool-progress notifications — i.e. it's an
  MCP aggregator.

It routes inference across **multiple backends** by `<provider>/<model>` —
`ollama` (local), `anthropic`, `openai`, `groq`, `together`, `openrouter`, and
**any OpenAI-compatible endpoint** you add in `inferencers.toml` (e.g.
self-hosted vLLM) — plus `claude-code/<model>`, a keyless path to Claude via the
local CLI (tool-less, or as an **executor** that runs a recipe's allow-listed
MCP tools itself — tool delegation). Config is file-driven (`mcp.json`,
`recipes.toml`, `inferencers.toml`).

**Stateful conversations** route *handles*; backends own the *state* — woollama
never stores transcripts in its own system. Two state-owning backends:
`claude-resume` (`claude --resume`, for `claude-code` models; keyless, the Claude
session owns the bytes) and `managed-agents` (Anthropic's Managed Agents, for
`claude-agent` models; `ANTHROPIC_API_KEY`, Anthropic hosts the session — and
exposes the transcript, so `/v1/conversations/{id}/items` works). Models with no
state-owning backend (ollama/cloud/recipe) are stateless — the caller owns
history (`store:false`). Long-lived MCP
connections. Served on **both a Unix socket** (`$XDG_RUNTIME_DIR/woollama.sock`,
mode 0600 — the default for local MCP clients) and an ephemeral loopback TCP
port; never `0.0.0.0` without explicit opt-in.

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

# Orchestrated: a recipe (system prompt + tools + model), transparent to the
# client. The chat-loop happens inside woollama; client sees only the final answer.
r = c.chat.completions.create(
    model="woollama/streamer",
    messages=[{"role": "user", "content": "Please count to 4."}],
)
```

woollama serves on **two transports at once**: a Unix socket at
`$XDG_RUNTIME_DIR/woollama.sock` (mode 0600 — the default for local MCP clients,
since a connectable socket can spend the router's API keys) and an ephemeral
loopback TCP port written to `$XDG_RUNTIME_DIR/woollama.addr` for clients to
discover. The `<port>` above is that ephemeral port. Same pattern as a local
`fabric --serve` instance.

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

### Tests & lint

```sh
uv run --extra dev pytest        # hermetic suite (live tests are opt-in: -m integration)
uv run ruff check .              # lint — the CI gate
```

CI (`.github/workflows/ci.yml`) runs both on every push to `main` and every PR.
For the same lint gate locally on commit, opt into the pre-commit hook:

```sh
uv tool install pre-commit && pre-commit install
```

Lint only — the project does not use `ruff format` (lines are hand-wrapped,
`E501` is ignored), so there is no formatter step in either gate.

## Design principles

1. **Two standards, neither extended.** MCP for tool/prompt/resource
   discovery and execution; OpenAI chat-completions for the inference
   primitive. woollama is a router between them.
2. **Local-only, ephemeral by default.** Random loopback port, persisted
   address file for discovery, never `0.0.0.0` without explicit opt-in. The
   router holds API keys and routes to local resources — it should not be
   LAN-reachable.
3. **The model namespace is the universal addressing scheme.** Raw inferencers
   (`<provider>/<model>`, e.g. `ollama/X`, `anthropic/X`, `claude-code/X`) and
   full recipes (`woollama/<recipe>`) are all addressable through OpenAI's
   standard `model` field. No new wire format.
4. **woollama owns routing, not inference or tools.** It uses other people's
   inference engines (Ollama, Anthropic, …) and other people's tool servers
   (any MCP server — filesystem, git, lackpy, …). It composes them.
5. **she talks to llamas.**

## What works today

- OpenAI surface: `/v1/models`, `/v1/chat/completions` (pass-through +
  recipe orchestration, both with `stream:true` → OpenAI SSE), `/v1/tools`
  introspection
- **Stateful surface**: `/v1/responses` (stateless subset + stateful) and
  `/v1/conversations` (create/list/get/delete, plus `items` where the backend
  exposes its transcript). woollama routes conversation *handles*; backends own
  state (woollama never stores transcripts itself) — `claude-resume` for
  `claude-code` models, `managed-agents` (Anthropic Managed Agents) for
  `claude-agent` models; models with no state-owning backend are stateless
  (`store:false`)
- Multi-backend routing by `<provider>/<model>`: ollama, anthropic, openai,
  groq, together, openrouter, `claude-code`, + any OpenAI-compatible endpoint
  via `inferencers.toml`
- **Tool delegation**: a `claude-code` recipe with tools runs as an *executor* —
  Claude owns the agentic loop and calls the recipe's allow-listed MCP tools
  itself (per-recipe `--mcp-config` + `--allowedTools` containment)
- MCP server side: stdio (`woollama mcp`) **and** Streamable HTTP at `/mcp` on
  the same port — recipes as prompts, a `chat` verb (with live tool-progress
  notifications), and every downstream tool re-exported with its `output_schema`
  (aggregator)
- File-driven config (`mcp.json`, `recipes.toml`, `inferencers.toml`), multi-
  MCP-server discovery + unified tool registry, long-lived MCP connections
- Recipe allow-list enforced as a security boundary (in-loop AND in delegation);
  served on a **Unix socket + loopback TCP**, address discovery file; CI
  (ruff + hermetic suite, 3.11/3.12)

## Not yet (next on the roadmap)

- The live, interactive Claude-in-tmux session backend (a separate Rust session
  driver) and the interactive `requires_action` path — gated on spikes that need
  a real terminal
- cosmic-fabric actually consuming the conversations surface (the last v1.0 gate)
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
