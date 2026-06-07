# woollama

**Web Over Ollama (and Llamas).** An MCP + OpenAI router for AI desktops.

woollama sits between AI clients (Cursor, the OpenAI SDK, Claude Desktop,
cosmic-fabric — anything that speaks OpenAI or MCP) and AI backends (Ollama,
Anthropic, fabric, lackpy, filesystem MCPs — anything that speaks OpenAI or
MCP). It composes them into orchestrated calls **without inventing a new
protocol**.

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

## What it is

A small daemon that **routes inference requests, tool calls, and executor
choice** between AI clients and AI backends, using two standard wire formats
and inventing none of its own. Three axes of routing, one daemon:

- **models** — `<provider>/<model>` (e.g. `ollama/qwen3`, `anthropic/…`,
  `claude-code/…`) and full recipes (`woollama/<recipe>`), all addressed through
  OpenAI's standard `model` field.
- **tools** — `<server>.<tool>`, discovered from any MCP server and re-exported.
- **executors** — which backend handles a model, including **tool delegation**
  to Claude Code (Claude owns the agentic loop and calls a recipe's allow-listed
  MCP tools itself).

See [Architecture](architecture.md) for the full design.

## Status

!!! warning "Python prototype — not production-ready"
    woollama works end-to-end as both an **OpenAI-compatible server** and an
    **MCP server**, across multiple inference backends and both surfaces. It is
    a prototype; **v1.0 will be a Rust rewrite** once the design stabilizes (see
    [Rust transition](rust-transition.md)). Authoritative live status lives in
    the [Roadmap](roadmap.md).

What works today:

- **OpenAI surface** — `/v1/chat/completions` (pass-through *and* hidden
  recipe orchestration, both streaming → OpenAI SSE), `/v1/models`, `/v1/tools`,
  plus a **stateful surface**: `/v1/responses` + `/v1/conversations`.
- **MCP surface** — stdio (`woollama mcp`) and Streamable HTTP at `/mcp` on the
  same port; an aggregator that re-exports every downstream tool (namespaced,
  with `output_schema`) plus a `chat` verb with live tool-progress events.
- **Multi-backend routing** — ollama, anthropic, openai, groq, together,
  openrouter, `claude-code`, and any OpenAI-compatible endpoint via
  `inferencers.toml`.
- **Stateful conversations** route *handles*; backends own the *state* —
  woollama never stores transcripts in its own system (see the
  [Conversations API](conversations-api-design.md)).

## Where to go next

- **[Getting started](getting-started.md)** — install, run, and drive it from an
  OpenAI client.
- **[Architecture](architecture.md)** — the model/tool/executor router design.
- **[Conversations API](conversations-api-design.md)** — the stateful surface and
  the handles-not-state principle.
- **[Roadmap](roadmap.md)** — the authoritative scorecard of what's built and
  what's next.

## Origin

woollama is the production-grade rewrite of an architecture co-designed in
[cosmic-fabric](https://github.com/teaguesterling/cosmic-fabric), which remains a
frontend (and will use woollama as its router engine). See [Naming](naming.md)
for how the project got its name.
