# Changelog

## v0.1.0 — 2026-05-31

First public version. **Working router; Python prototype, not production.**
Architecture validated end-to-end; v0.2 will harden, configure, and expand
the prototype. **v1.0 is a Rust rewrite** once the architecture stabilizes —
see `docs/rust-transition.md` for the explicit criteria.

### What v0.1 does

- **OpenAI-compatible HTTP surface** at `/v1/models` and `/v1/chat/completions`.
- **Model namespace routing**:
  - `ollama/<name>` — pure pass-through to local Ollama at `localhost:11434`
  - `woollama/<recipe>` — orchestrated chat-loop using the named recipe
- **One bundled example recipe** (`woollama/streamer`) demonstrating
  pattern + tools + inferencer composition.
- **MCP tool dispatch** via per-request stdio connection to the bundled
  hello server (`examples/mcp-hello/server.py`).
- **Ephemeral local-only binding** — random free loopback port at startup,
  persisted to `$XDG_RUNTIME_DIR/woollama.addr` for client discovery. Never
  binds to `0.0.0.0` without explicit `WOOLLAMA_ADDRESS` override.
- **Smoke tests** that don't require Ollama or network.

### Design ideas validated

- MCP + OpenAI compose as complementary standards without extension
- The model namespace (`<provider>/<name>`) is a sufficient addressing
  scheme for raw / pattern / variant / recipe model kinds
- Recipe orchestration is invisible to OpenAI clients — they get one final
  answer; the chat-loop happens inside the router
- Ephemeral local binding works for the OpenAI SDK out of the box (clients
  read the addr-file)

### Known limitations / queued for v0.2

- **No streaming** on either side (non-streaming round-trips only)
- **One hardcoded recipe** — real `~/.config/woollama/recipes.toml` to follow
- **One MCP server** — multi-server discovery + unified tool registry to follow
- **Ollama only** — Anthropic / OpenAI / vLLM via OpenAI-compat to follow
- **No Unix socket** transport — HTTP loopback only
- **woollama as MCP server** to its own clients is not yet implemented
- **No CI** — manual smoke tests; pytest config added but no GitHub Actions yet
- **Per-request MCP subprocess** is correct but slow; connection pooling to follow

### Origin

woollama is the rewrite of an architecture co-designed in [cosmic-fabric](
https://github.com/teaguesterling/cosmic-fabric), which remains as a frontend
client. The full design context lives in `docs/architecture.md` and
`docs/naming.md`.
