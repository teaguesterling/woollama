# Environment variables

Every environment variable woollama reads, grouped by purpose. All are optional
unless you use the feature that needs them.

## Provider API keys

Read **at call time** (never stored); woollama only holds the *name* of the var.
Required only to *use* (or live-`discover`) the matching cloud provider.

| Variable | For |
|---|---|
| `ANTHROPIC_API_KEY` | `anthropic/<model>` and the `managed-agents` backend (`claude-agent/<model>`) |
| `OPENAI_API_KEY` | `openai/<model>` |
| `GROQ_API_KEY` | `groq/<model>` |
| `TOGETHER_API_KEY` | `together/<model>` |
| `OPENROUTER_API_KEY` | `openrouter/<model>` |
| *(custom)* | whatever `api_key_env` a provider in [`inferencers.toml`](configuration.md) names |

`ollama` and `claude-code` (CLI/subscription) need **no** key.

## Binding & discovery

| Variable | Default | Effect |
|---|---|---|
| `WOOLLAMA_ADDRESS` | *(unset)* | Override the bind address (`host[:port]`). **The only way to bind off-loopback** (e.g. `0.0.0.0:8000`) — opt-in, since the router holds API keys. |
| `XDG_RUNTIME_DIR` | `/tmp` fallback | Where the Unix socket (`woollama.sock`, mode 0600) and the address file (`woollama.addr`) are written. |

## Backends & config

| Variable | Default | Effect |
|---|---|---|
| `WOOLLAMA_CONFIG_DIR` | `$XDG_CONFIG_HOME/woollama` (`~/.config/woollama`) | Where `mcp.json` / `recipes.toml` / `inferencers.toml` are read from. |
| `XDG_CONFIG_HOME` | `~/.config` | Base for the default config dir (above). |
| `WOOLLAMA_OLLAMA_URL` | `http://localhost:11434` | The local Ollama endpoint (the `ollama/` provider's base + its `/v1/models` discovery). |
| `WOOLLAMA_CLAUDE_BIN` | `claude` (on `PATH`) | Path to the Claude Code CLI for `claude-code/<model>` (inference, delegation, and the `claude-resume` conversation backend). |

> The **conversation store** (issue #2) that makes non-claude models stateful is
> selected in config, not by an env var — the top-level `conversationStore` key
> in [`mcp.json`](configuration.md#using-an-mcp-server-as-the-conversation-store).

## Claude Code child process

When woollama runs the `claude` CLI (as an inferencer, an executor, or the
`claude-resume` backend), the child gets an **allow-listed** environment only —
operational vars, **no provider keys or secrets**. The allow-list is:

`HOME`, `PATH`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `TZ`, `TMPDIR`, `LANG`,
`LANGUAGE`, `LC_*`, and the proxy vars (`HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`,
lower-case too).

See [Security model](security.md) for why, and [Configuration](configuration.md)
for the files these complement.
