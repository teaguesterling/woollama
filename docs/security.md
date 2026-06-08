# Security model

woollama is a local router that holds API keys and can run a Claude Code
executor. This page is the threat model and the controls that are already in
place. It's a **prototype** — local-first by design, not hardened for hostile
multi-tenant exposure.

## Network exposure — local-first, opt-in only

- woollama binds an **ephemeral loopback TCP port** and a **Unix socket** at
  `$XDG_RUNTIME_DIR/woollama.sock` (mode **0600** — owner-only; a connectable
  socket can spend the router's API keys). The socket is the default for local
  MCP clients.
- It **never binds off-loopback** (e.g. `0.0.0.0`) unless you explicitly set
  [`WOOLLAMA_ADDRESS`](environment.md). There is no LAN exposure by default.
- The loopback port is random and written to `$XDG_RUNTIME_DIR/woollama.addr`
  for clients to discover.

**Implication:** anything that can reach the socket or the loopback port can use
your configured providers (and your keys). Keep it local; don't reverse-proxy it
to the internet without your own auth in front.

## API key custody

- Provider keys are read from **environment variables at call time** and never
  written to disk or logged. woollama stores only the *name* of the env var (from
  [`inferencers.toml`](configuration.md)'s `api_key_env`), not the value.
- `GET /v1/models` lists a provider's `models`/`discover` entries **without**
  needing the key (listing ≠ calling); the key is only used when a request
  actually routes to that provider.

## Recipe allow-list (a hard boundary)

A recipe declares an explicit `tools = [...]` allow-list. woollama enforces it as
a **security boundary**, not a hint: a recipe can only dispatch the tools it
lists — both in the in-process orchestration loop **and** in claude-code tool
delegation. A recipe cannot reach a tool (or an MCP server) it didn't allow-list.

## Claude Code executor containment

When a `claude-code/<model>` recipe runs as an **executor** (Claude owns the
agentic loop and calls the recipe's allow-listed MCP tools itself), the child
`claude` process is locked down:

- **`--tools ""`** — the built-in tool set is an allow-list of **nothing**. This
  is robust against whatever a deployment's global Claude config / plugins enable
  (a deny-list can't enumerate those; an allow-list of none can't be widened by
  them). Bash, file tools, LSP, harness meta-tools — all absent, not merely
  denied.
- **Only the recipe's MCP tools are reachable** — woollama writes a per-recipe
  `--mcp-config` containing *only* the servers the allow-list references, plus
  `--allowedTools` listing *only* those tools, with `--permission-mode dontAsk`
  (a hard deny for anything unlisted) and `--strict-mcp-config`.
- **`ENABLE_TOOL_SEARCH=false`** so the recipe's MCP tools load up front once the
  built-in tool search is gone.
- **`--setting-sources project`** so the child doesn't inherit host `~/.claude`
  settings.
- **Allow-listed child environment** — the child gets operational vars only
  (`HOME`, `PATH`, `LANG`, proxies, …; see [Environment](environment.md)). **No
  provider keys, no secrets, no parent-harness vars** are passed through.
- Runs in a neutral temp working directory (no host `CLAUDE.md`/settings).

This was hardened in an adversarial review: a provider-key env leak (the child
env is now an allow-list), a host-settings undercut (`--setting-sources
project`), and a tool-name injection vector were all closed; an out-of-list tool
is hard-denied and a shell-exec attempt fails because the tool is absent. The
authoritative detail lives in `src/woollama/claude_code.py`'s module docstring
and `docs/build-log.md`.

## What is NOT in scope (prototype)

- No authentication/authorization on the local surface (the socket's 0600 mode
  and loopback binding *are* the access control).
- No multi-tenant isolation, rate limiting, or audit logging.
- The `managed-agents` backend runs on Anthropic's infrastructure under your
  `ANTHROPIC_API_KEY` — its sessions are **billed**, and a woollama restart
  orphans live sessions (see the conversations design doc).
