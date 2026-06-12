# woollama-server (`woollamad`)

The **woollama router daemon** — a small Rust service that fronts your local and cloud
models behind one OpenAI-compatible + MCP endpoint, and orchestrates recipes (system prompt
+ tools + model) so clients see only the final answer.

```sh
cargo install woollama-server   # installs the `woollamad` binary
woollamad                       # starts the router; prints its address
```

`woollamad` serves the same surface on two local transports at once: a unix socket at
`$XDG_RUNTIME_DIR/woollama.sock` (mode 0600 — the default for local MCP clients) and an
ephemeral loopback TCP port, persisted to `$XDG_RUNTIME_DIR/woollama.addr` for discovery.
`woollamad mcp` instead serves woollama's MCP surface over stdio.

What it does:

- **OpenAI-compatible HTTP** — `/v1/models`, `/v1/chat/completions` (passthrough +
  orchestration, streaming), `/v1/responses`, `/v1/conversations`.
- **MCP aggregator** — re-exports downstream MCP servers' tools (namespaced, schema-mirrored)
  and exposes woollama recipes as an MCP `chat` tool + prompts, at `/mcp` and over stdio.
- **Routing** — `ollama/<model>`, cloud providers, `claude-code/<model>` (the Claude Code
  executor), and `woollama/<recipe>` orchestration bundles.

Configuration lives in `$XDG_CONFIG_HOME/woollama/` (`mcp.json`, `recipes.toml`,
`inferencers.toml`); bundled defaults ship in the repo. `cargo install` ships only the
binary, so supply your own `mcp.json` (or point `WOOLLAMA_EXAMPLES_DIR` at the repo's
`examples/` for the bundled demo servers).

Part of [woollama](https://github.com/teaguesterling/woollama). The Python package is the
auxiliary embedding surface; this crate is the canonical router.
