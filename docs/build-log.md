# woollama build log (slice-by-slice)

The running, chronological record of how woollama was built — one section per
slice, each with the decisions, the load-bearing findings, and what was
verified. For *current status and what's next*, see
[`roadmap.md`](roadmap.md); this file is the detailed history behind it.

(Started life as the slice (e) handoff note; it now spans slices e → k. The
original slice (e) plan is preserved verbatim below, followed by each follow-on.)

## Index

| Slice | What | Commit |
|---|---|---|
| (e) | woollama as an MCP server (stdio) — prompts + chat verb | `9a19be0` |
| (f) | MCP aggregator (tool re-export) + routing demo + allow-list boundary | `dd53122` |
| (h) | MCP over Streamable HTTP, mounted on one port | `58addb2` |
| (i) | Claude Code as a (tool-less) inference backend | `1fc0a24` |
| (j) | OpenAI-compat inferencer seam — Anthropic | `35c2dc0` |
| (k) | the rest of the providers — cloud built-ins + config-file inferencers | `5b2996e` |

(Slices a–d — the OpenAI router, config files, multi-MCP discovery, test
backfill — predate this log; see `git log`. There is no slice (g): the
lettering jumped f→h during the session and we kept it faithful to the commits.)

---

# Slice (e) — woollama as an MCP server (handoff / pickup note)

Status: **DONE (2026-06-01).** Implemented + tested. (Counts below are as-of
that slice; the suite has grown since — see the per-slice "Verified" lines and
`roadmap.md` for the current totals.)

## Where this came from (session recap, 2026-05-31 → 06-01)

- woollama is the graduation of a throwaway router probe (originally
  `/tmp/router_probe/`, now obsolete — do not look for it). The architecture
  was co-designed in the sibling `cosmic-fabric` repo; woollama is the actual
  router implementation. cosmic-fabric remains as the COSMIC **frontend
  client** and its design docs are now stale (still cite `/tmp`, still read
  the naming question as open — it's settled: `woollama`).
- **Mount move gotcha**: the project moved `/mnt/fast/...` →
  `/srv/physical/fast/...` (also reachable as `/home/teague/Projects/woollama`
  via bind mount). This broke the `.venv` editable install (the
  `_editable_impl_woollama.pth` pointed at the dead `/mnt/fast` path). Fix
  already applied: `uv sync --extra dev` repointed it. **All 43 tests pass.**
  If imports break again as "unknown location", re-run `uv sync --extra dev`.
- Dev deps live in the `dev` **optional-dependency extra** (not a
  `dependency-groups` table). Run tests with `uv run --extra dev pytest`.

## Slices so far

```
78580a3 test suite: backfill manager + router + live integration; 19 → 43 tests
fa4d766 slice (c): multi-MCP-server discovery, long-lived connections, namespacing
fc05a04 slice (b): real config files — mcp.json + recipes.toml
1ef2c50 woollama v0.1.0 — Python prototype of the MCP + OpenAI router
```

Slice (d) was folded in; (e) is next.

## What slice (e) is

Project woollama's **inbound OpenAI surface onto an outbound MCP surface** so
MCP clients (Claude Desktop, the cosmic-fabric panel) can drive it natively.
This unlocks the panel-as-MCP-client direction.

woollama-as-MCP-server exposes:
- **recipes → MCP `prompts`** (`prompts/list`, `prompts/get` returns the
  rendered system message)
- **an orchestration verb → MCP `tool`** (`tools/list`, `tools/call`)
- **capability negotiation** on `initialize`

## Decisions locked (user, 2026-06-01)

1. **Transport: stdio first.** Subprocess + stdio — matches what the panel and
   Claude Desktop need locally; easiest to test via subprocess. HTTP/SSE can
   follow as a later slice.
2. **Orchestration verb is named `chat`.** Mirrors the `/v1/chat/completions`
   surface. (Considered `run_recipe` / `woollama_chat`; chose `chat`.)
3. **`tools/list` scope: start with just the `chat` verb, but build the
   projection to be EXTENSIBLE** so re-exporting discovered downstream tools
   (`textops.*`, `hello.*`) can be added later without a redesign. i.e. the
   tools/list builder should be a function that today returns `[chat]` but is
   structured to concatenate `registry`-derived tools when we flip that on.

## Implementation plan — TDD, tests are the spec

New file `tests/test_mcp_server.py`, five happy-path tests (all default suite),
each fails today; implement minimally to pass each in order:

1. `test_initialize_advertises_expected_capabilities`
   client connects → server announces `{ tools, prompts, ... }` caps.
2. `test_prompts_list_returns_loaded_recipes`
   `prompts/list` → sees the loaded recipe names (e.g. `streamer`,
   `textcounter`). Each recipe IS a prompt.
3. `test_prompts_get_returns_rendered_system_message`
   `prompts/get("streamer")` → returns the recipe's system text (the
   `assemble_prompt` equivalent on the MCP side).
4. `test_tools_list_includes_chat_orchestration_verb`
   `tools/list` → includes `chat`; its input schema has
   `(recipe?, model?, messages, ...)`.
5. `test_chat_tool_orchestrates_end_to_end`
   `tools/call chat {recipe: "streamer", messages: [...]}` → final assistant
   message (loop hidden, same as the OpenAI surface). **Mock the underlying
   inferencer call — this is a unit test, not integration.**

Plus **one opt-in integration test**: spawn woollama and drive it as an MCP
server over stdio with a real MCP client (mark it like the existing live/
integration tests so it skips when Ollama isn't up).

## How (e) maps onto existing code (read these first next session)

The MCP server is a thin projection over machinery that already exists:

- `src/woollama/recipes.py` — `recipes.names()`, `recipes.get(name)` →
  `Recipe` dict with keys `system`, `inferencer`, `tools`. **prompts/list and
  prompts/get project directly off this.** (NOTE: I had not finished re-reading
  recipes.py/manager.py when this note was written — read both before coding.)
- `src/woollama/manager.py` — `Registry`: `openai_tools_for(tool_names)`,
  `dispatch(namespaced_name, args)` (returns result with `.content` list of
  items carrying `.text`), `all_tool_names()`, `start_all()`/`stop_all()`,
  `ServerManager(name, command, args)`.
- `src/woollama/router.py` — `_orchestrate_recipe(recipe, body)` is the
  **core chat-loop the `chat` tool must reuse.** Do NOT duplicate it. Refactor
  the loop body out of the FastAPI handler into a transport-agnostic
  coroutine (e.g. `orchestrate(recipe, messages) -> final_message`) that BOTH
  `/v1/chat/completions` and the MCP `chat` tool call. The loop currently:
  prepends `recipe["system"]`, requires an `ollama/` inferencer, builds tools
  via `registry.openai_tools_for(recipe["tools"])`, loops ≤8 turns hitting
  Ollama, dispatches tool_calls via `registry.dispatch`, returns the final
  message. Extract this so the MCP path doesn't reimplement it.
- `src/woollama/config.py` — `load_mcp_servers()` (mcp.json), recipe loading.
- `src/woollama/__main__.py` — CLI wiring. **Add an `mcp` subcommand**
  (`woollama mcp`) that starts the stdio MCP server, alongside the existing
  HTTP serve command. This is what clients put in their mcp.json:
  `{ "command": "woollama", "args": ["mcp"] }`.
- The MCP server itself: use **FastMCP** (already a dev dep; the
  `examples/mcp-hello/server.py` is the reference shape). Mount prompts +
  the `chat` tool on a `FastMCP("woollama")` instance.

## Open sub-questions raised by the test names (decide while writing tests)

- `chat` tool input schema: `recipe` (name) vs `model` (`woollama/<recipe>`)
  vs both? Lean: accept `recipe` (bare name) primarily; optionally accept
  `model` for symmetry with the OpenAI surface.
- Registry lifecycle under stdio: the stdio server needs the same
  long-lived `registry.start_all()` / `stop_all()` that the FastAPI lifespan
  does. Wire it into the MCP server's startup/shutdown.
- tools/list extensibility hook: make the builder
  `def mcp_tools_list() -> list[Tool]: return [_CHAT_TOOL]  # + future registry tools`
  so flipping on re-export is a one-line concat, per decision #3.

## First concrete action next session

1. `uv sync --extra dev` (guard against mount-move staleness), confirm
   `uv run --extra dev pytest -q` is green (43 passing).
2. Read `recipes.py` and `manager.py` fully.
3. Refactor the orchestration loop out of `router.py` into a shared coroutine.
4. Write `tests/test_mcp_server.py` (5 tests) — RED.
5. Implement the FastMCP server + `woollama mcp` CLI subcommand — GREEN.
6. Add the opt-in stdio integration test.
7. Commit as `slice (e): woollama as MCP server (stdio) — prompts + chat verb`.

## What shipped (2026-06-01)

- **`src/woollama/router.py`** — extracted the chat-loop into a
  transport-agnostic `orchestrate(recipe, user_msgs, reg) -> resp_dict`
  coroutine + an `OrchestrationError(message, kind, status, payload)` that each
  transport maps to its own surface. `_orchestrate_recipe` (HTTP) is now a thin
  adapter. **No loop duplication** — the MCP `chat` tool reuses `orchestrate`.
- **`src/woollama/mcp_server.py`** (new) — `build_server(registry) -> FastMCP`:
  one MCP prompt per recipe (rendering returns the system message; per-iter
  closure binding avoids the late-binding bug), the `chat` tool (`messages`,
  `recipe`, optional `model="woollama/<name>"`), and a `lifespan` that
  `start_all()/stop_all()` the registry **inside the serving loop** (the
  load-bearing detail — `ServerManager` binds futures/tasks to that loop).
  `_chat_tool()` builds the orchestration verb. `serve()` runs stdio with
  `show_banner=False` (stdout is the JSON-RPC channel). (Decision #3 / tool
  re-export landed as a follow-on — see below.)
- **`src/woollama/__main__.py`** — `woollama mcp` subcommand → `mcp_server.serve()`,
  logging forced to stderr so stdout stays clean.
- **`tests/test_mcp_server.py`** (new, 5 unit tests, in-memory `fastmcp.Client`
  over a bare `Registry()`): initialize caps, prompts/list, prompts/get,
  tools/list has `chat`, chat orchestrates end-to-end (inferencer mocked).
- **`tests/test_integration.py`** — +2 opt-in tests:
  - `test_mcp_stdio_surface_with_started_registry` (not Ollama-gated): spawns
    `woollama mcp` over REAL stdio with a STARTED registry (hello + textops
    example servers). Proves `registry.start_all()` runs and the server comes
    up clean over stdio (where the documented anyio cancel-scope bug would
    bite) — the unit tests use an empty registry and can't catch that. Stops
    short of orchestration (the unknown-recipe call short-circuits before
    dispatch).
  - `test_mcp_stdio_chat_orchestrates_end_to_end` (`@needs_ollama`): drives a
    full `chat` orchestration over stdio — the MCP counterpart of the HTTP
    `test_orchestrated_recipe...` parity test. Gives the MCP transport the same
    end-to-end coverage HTTP has.
- **Verified**: default suite green / 5 deselected; the non-Ollama stdio
  integration test passes (~5s, really spawns the example servers). The
  Ollama-gated end-to-end test runs when qwen3:14b-iq4xs is available.
  (Counts grew with the tool-re-export follow-on below.)

### Client mcp.json entry
```json
{ "command": "woollama", "args": ["mcp"] }
```

### Natural next slices
- ~~HTTP/SSE transport~~ — DONE as Streamable HTTP, mounted (see follow-on below).
- Re-export `output_schema` pass-through on proxy tools (currently content +
  structuredContent only).
- Legacy SSE transport, if a client ever needs it (FastMCP `transport="sse"`).

## Follow-on slice — tool re-export (decision #3, 2026-06-01)

woollama is now an **MCP aggregator**: a client connecting over MCP sees the
union of every configured downstream server's tools (namespaced
`<server>.<tool>`, e.g. `hello.count_to`, `textops.word_count`) **plus** the
`chat` verb.

Implementation note that corrects the original plan: decision #3 described
re-export as a "one-line concat in the tools/list builder." That turned out to
be wrong — a server's tools are only known **after** `registry.start_all()`,
which (per the loop constraint) runs inside the lifespan, not at
`build_server()` time. So re-export is a **lifespan-time dynamic registration**,
not a build-time concat:

- **`_ProxyTool(Tool)`** — a FastMCP `Tool` subclass carrying the downstream
  tool's own name + input schema; its `run()` dispatches through the unified
  `Registry` (the SAME long-lived connection layer the chat path uses — NOT a
  second FastMCP client stack, which `as_proxy`/mount would have introduced).
  It's raw passthrough (no `orchestrate`), so it owns its own error handling:
  dispatch exceptions and `CallToolResult.isError` both become `ToolError`.
  (`Tool.from_function` can't be used here — it rejects `**kwargs` handlers and
  derives the schema from the signature; we need an arbitrary downstream schema.)
  `run()` passes the downstream `structuredContent` through (so dict-returning
  tools like `hello.count_to` reach the client as structured data, not just
  JSON-as-text). It deliberately does NOT mirror the downstream `output_schema`
  onto the proxy tool — that would couple every call to schema validation and
  break content-only results; output_schema pass-through is a later refinement.
- **`_register_reexported_tools(mcp, reg)`** — called in the lifespan right
  after `start_all()`; iterates `reg.servers` and `add_tool`s a `_ProxyTool`
  per discovered tool. Schemas are defensively copied (HTTP's
  `openai_tools_for` reads the same spec objects).
- **Tests**: +1 unit test (stubbed `ServerManager`: re-exported tool appears in
  tools/list with its schema and dispatches) and the non-Ollama stdio
  integration test extended to assert real `hello.count_to` / `textops.word_count`
  are re-exported and that `hello.count_to {n:3}` dispatches end-to-end over
  real stdio (content-block fidelity — the unit test's canned content can't
  prove that).
- **Verified**: default suite 46 passed / 5 deselected; full integration suite
  (incl. the Ollama-gated chat-over-stdio test) green.

## Follow-on slice — routing demonstration + allow-list boundary (2026-06-01)

A cohesive demonstration of the whole routing topology, plus a real fix the
demonstration surfaced.

**The allow-list is now a BOUNDARY, not a hint (behavior change in `orchestrate`).**
Previously the recipe's `tools` allow-list only governed which tools were
*offered* to the model (`openai_tools_for`); `dispatch` would execute whatever
namespaced name the model emitted — so a recipe scoped to `hello.*` could reach
`textops.*` if the model emitted it. Now `orchestrate` computes
`allowed = set(recipe["tools"])` once and refuses any out-of-list tool_call:
it does NOT dispatch, `log.warning`s the denial, and feeds an `ERROR: ... not
permitted` result back as the tool message (every tool_call still needs a
matching tool result, so the loop continues and the model can recover —
feed-back-and-continue, not hard-fail). Scope: `orchestrate` only — the
re-exported `_ProxyTool` aggregator surface is deliberately left open (different
concept: what a *recipe* may reach vs. what the aggregator exposes).

**`tests/test_routing.py`** (new, 10 hermetic tests) — executable documentation
of the routing map. Inferencer mocked with scripted turns; each downstream
session is a stub `ServerManager` that records what IT received, so tests assert
the RIGHT call reached the RIGHT session (per-session recording, not a global
counter). Headline: `woollama/textcounter` (allow-lists `textops.word_count` +
`hello.count_to`) fans one chat out to two sessions — proven over BOTH the HTTP
and MCP `chat` transports. Plus proxy-routing, passthrough, and a rejection
matrix: unknown namespace (400) / recipe (404) / non-ollama inferencer (501) /
MCP unknown recipe (ToolError) / dispatch to unknown provider+tool (KeyError) /
**out-of-allow-list tool refused without dispatch** (the boundary test asserts
the forbidden session's `call_tool` was never invoked).

**`tests/test_integration.py`** — +1 `@needs_ollama` test
(`test_two_provider_recipe_uses_tools_from_two_sessions`): real `textcounter`
chat end-to-end; loose assert (non-empty content, no leaked tool_calls) since
the hermetic matrix carries the deterministic routing proof.

**`examples/routing_demo.py`** (new, runnable) — `python examples/routing_demo.py`
spins up both surfaces over the bundled defaults and prints every routing
activity live (discovery, passthrough, two-provider orchestration, MCP
aggregator + direct proxy calls, rejections). Parts needing Ollama skip cleanly
when it's down. Confirmed: the live run shows the model calling word_count then
count_to across two sessions, returning a single hidden-loop answer.

- **Verified**: default suite 56 passed / 6 deselected; full integration suite
  6 passed (incl. the live two-provider chat); the demo runs end-to-end.

## Follow-on slice — MCP over Streamable HTTP, mounted on one port (2026-06-01)

woollama now serves BOTH surfaces on a single port: `/v1/*` (OpenAI) and `/mcp`
(MCP over Streamable HTTP), over ONE shared registry — the router thesis. Both
`/mcp` and `/mcp/` work; the mount does not shadow `/v1/*`. (User decisions:
mount-into-FastAPI over standalone; Streamable HTTP over legacy SSE.)

The load-bearing design (Design Y — registry owned in one place):
- **Single shared registry** (`router.registry`) started once by router's
  FastAPI lifespan and used by BOTH the OpenAI orchestration path and the MCP
  chat/proxy tools. Under uvicorn there's one event loop, so `start_all()` and
  every request (both surfaces) run on it — the `ServerManager` loop-binding
  invariant holds for free.
- **`build_server(registry, *, manage_registry=False)`** for the mounted server:
  it must NOT start/stop the registry (FastAPI's lifespan owns it) — avoids a
  double-start. The stdio path keeps `manage_registry=True`.
- **Lifespan composition** in `router.lifespan`: populate + `start_all` →
  `register_reexported_tools(_mcp, registry)` (re-export is dynamic, post-start,
  and — verified — visible over the HTTP transport even though tools are added
  after `http_app()` is constructed) → `async with _mcp_app.lifespan(app)` (the
  Streamable HTTP session manager) → yield → `stop_all`.
- **Import cycle broken**: `mcp_server` lazy-imports `orchestrate`/
  `OrchestrationError` inside the `chat` tool, so `router` can import
  `build_server`/`register_reexported_tools` at module top and mount at import
  (`_mcp_app = _mcp.http_app(path="/")`; `app.mount("/mcp", _mcp_app)`).
- **`register_reexported_tools`** is now public (was `_register_…`) since both
  `build_server`'s stdio lifespan and router's HTTP lifespan call it.

Client mcp.json entry (HTTP): `{ "url": "http://<host>:<port>/mcp" }`.

**Tests** (`tests/test_integration.py`, opt-in): `test_mcp_over_http_shares_
one_port_and_registry` (not Ollama-gated) — `/v1/models` still 200 (mount didn't
shadow it), `/mcp` lists chat + re-exported tools, and `hello.count_to{n:3}`
dispatches over HTTP returning the structured dict; `test_mcp_over_http_chat_
orchestrates_end_to_end` (`@needs_ollama`) — full `chat` orchestration over the
mounted HTTP surface.

**`examples/routing_demo.py`** updated: its MCP-aggregator section now connects
to `/mcp` on the SAME running HTTP server (Streamable HTTP) instead of spawning
a separate `woollama mcp` stdio process — so the demo actually showcases the
one-port-both-surfaces capability. Confirmed end-to-end against real Ollama.

**Coupling note (intentional):** `router.lifespan` now starts the shared
registry entangled with the mounted MCP app's lifespan, and `_mcp`/`_mcp_app`
are built at import. So the OpenAI surface (`/v1/*`) now has a hard dependency
on the MCP server constructing + its lifespan entering cleanly — if MCP mounting
throws, `/v1/*` goes down with it. Acceptable for "one unified router," but it
widens the OpenAI surface's blast radius.

- **Verified**: default suite 56 passed / 6 deselected (the import-time mount
  construction is exercised by every `import woollama.router`); full integration
  suite 8 passed (incl. both new HTTP-mount tests + the live chat-over-HTTP);
  the one-port demo runs end-to-end.

## Follow-on slice — Claude Code as a (tool-less) inference backend (2026-06-01)

A recipe whose inferencer is `claude-code/<model>` routes to the local `claude`
CLI in headless print mode (`src/woollama/claude_code.py`), using the user's
EXISTING Claude auth — no `ANTHROPIC_API_KEY`. A keyless path to Claude. (Chosen
over the OpenAI-compat Anthropic-API shim because there's no API key in the env;
the user asked for "a claude code adapter".)

**Scope: TOOL-LESS completions only.** `orchestrate` dispatches by inferencer
provider: `claude-code/` (recipe with `tools=[]`) → `claude_code.run_completion`;
`claude-code/` WITH tools → 501 ("tool delegation not yet supported"); `ollama/`
→ the existing woollama-owned loop; else → 501. Serves both the HTTP and MCP
`chat` transports (both call `orchestrate`). Tool DELEGATION (Claude Code owns
the loop and runs the recipe's MCP tools) is a deliberately separate, larger
concept — an *executor*, not an inferencer — and a later slice with its own
adversarial safety pass.

**Subprocess, not the Agent SDK** — the SDK needs the `claude` CLI on PATH
anyway, so shelling out is fewer deps and trivially mockable (tests patch
`claude_code._invoke`).

**Findings pinned empirically against `claude` v2.1.160 (docs were wrong):**
- `--output-format json` emits a JSON **array** of events (system → assistant →
  `result`); the final text is the `result` event's `result` field. (Docs say a
  single object.)
- 🔴 **`--permission-mode dontAsk` does NOT make it tool-less** — it auto-RUNS
  read-only Bash (verified: `echo` executed, `permission_denials: []`). The docs'
  "dontAsk denies everything unallowed" is false for read-only commands. So a
  naive tool-less backend would let crafted input run shell on the host.
- There is **no clean "disable all tools" flag**. The lockdown used:
  `--permission-mode dontAsk` (non-interactive, no hang) + `--strict-mcp-config`
  (zero MCP servers) + `--disallowedTools "Bash,Read,Write,Edit,NotebookEdit,
  WebFetch,WebSearch,Glob,Grep,Task"` (closes the read-only-Bash gap + the other
  exec/file/network/subagent vectors; other tools are auto-denied by dontAsk).
  Plus a neutral temp cwd (don't inherit host CLAUDE.md/settings/plugins) and
  `ANTHROPIC_API_KEY` stripped from the child env (force subscription auth).

**Verification split (honest):** the runtime safety boundary could NOT be
verified from inside this Claude Code session — nested `claude` invocations crash
the harness (the `H.replace` glitch). So:
- Hermetic `tests/test_claude_code.py` asserts the command is built locked-down
  (strict-mcp-config, dontAsk, Bash in disallowedTools), the API key is stripped,
  and the JSON-array output is parsed / errors surfaced. It CANNOT prove the
  lockdown actually holds at runtime.
- Opt-in `tests/test_integration.py::test_claude_code_backend_completes_and_
  refuses_shell` (`@needs_claude_code`: requires `claude` on PATH AND
  `WOOLLAMA_TEST_CLAUDE_CODE=1`, default-skipped — REAL subscription cost) is the
  runtime check: a real completion works AND a shell-exec attempt does NOT create
  an absolute-path canary file. **This is the test to run (outside a nested
  Claude Code session) to confirm the safety boundary before trusting it.**

**Bundled recipe:** `cc-assistant` (`inferencer = "claude-code/haiku"`, `tools =
[]`) — a tool-less example to demo/test against. `haiku` keeps live cost low.

**Roadmap honesty:** this does NOT close "cloud inferencers." The OpenAI-compat
HTTP seam architecture.md calls for (vLLM/Together/Groq/OpenRouter/anthropic-api)
is still unbuilt; claude-code is an alternative, keyless path to Claude via a
totally different mechanism (subprocess delegation), not that seam.

- **Verified**: default suite 68 passed / 9 deselected; integration suite 8
  passed + 1 skipped (the claude-code live test — awaiting opt-in). Runtime
  tool-lockdown verification PENDING the opt-in live run.

## Follow-on slice — OpenAI-compat inferencer seam (Anthropic) (2026-06-02)

The cloud-inferencer track architecture.md calls for: woollama routes
`<provider>/<model>` to any OpenAI-compatible chat-completions backend. New
`src/woollama/inferencers.py` is the registry; `orchestrate` and the
pass-through path are now provider-generic instead of ollama-hardcoded.

- **`Inferencer(name, base_url, api_key_env, extra_body)`** + a built-in
  registry rebuilt per call (so env overrides are live): `ollama` (local, no
  auth, native `options` body) and `anthropic` (Claude API's OpenAI-compat
  endpoint, `Authorization: Bearer $ANTHROPIC_API_KEY`, default `max_tokens` +
  clamped `temperature`). Adding vLLM/Together/Groq/OpenRouter is just more
  entries; **config-file-driven inferencers** (architecture.md's `inferencers`
  block) is the natural follow-on.
- **`orchestrate` dispatch**: `claude-code/` → subprocess backend (prior slice);
  otherwise resolve the provider in the registry. Unknown provider → 501; known
  provider with a missing API key → **400** (distinct, fail-fast via
  `inf.headers()` before any network call). The loop posts to `inf.chat_url()`
  with `inf.headers()` and merges `inf.extra_body` into the request.
- **Pass-through** (`<provider>/<model>` with no recipe) generalized the same
  way (was ollama-only). `claude-code/` is NOT a pass-through provider (recipe-
  only) → falls through to the 400 unknown-namespace error.
- **Anthropic compat-endpoint facts (from docs, not memory):** base
  `https://api.anthropic.com/v1/` + `chat/completions`; Bearer auth; `max_tokens`
  supported (not required); `temperature` capped to [0,1]; system messages
  hoisted+concatenated (we send one — fine). **Tools/function-calling ARE fully
  supported** (`tools[].function`, response `tool_calls`, assistant tool_calls,
  tool-role messages) — so this is FULL orchestration over Anthropic, not chat-
  only. Only `strict` schema enforcement is dropped (use the native API for
  that). `OLLAMA_URL` module const removed; `/v1/models` now derives the ollama
  URL from the registry.
- **Tests**: `tests/test_inferencers.py` — registry units + capturing-fake
  routing (orchestrate→anthropic asserts URL/Bearer/bare-model/max_tokens;
  ollama-unchanged regression; pass-through→anthropic). The old "non-ollama →
  501" tests became "unknown-provider → 501" (anthropic is supported now) +
  "anthropic-without-key → 400". Opt-in `@needs_anthropic` live test
  (`test_anthropic_inferencer_completes_live`, skips without a key) does a real
  tool-less round-trip.
- **Roadmap:** this BUILDS the OpenAI-compat seam → **anthropic-API now done**;
  vLLM/Together/Groq/OpenRouter are config entries away (pending config-file
  inferencers). Still open: streaming, Unix socket, the conversations surface,
  the Rust port. (claude-code remains a separate keyless mechanism, not this seam.)
- **Verified**: default suite 76 passed / 9 deselected; integration 8 passed +
  2 skipped (claude-code & anthropic live, both opt-in). Ollama path confirmed
  unchanged end-to-end (the generalization's regression canary). Caveats: the
  Anthropic tool-loop is doc-confirmed + unit-tested on the EMIT side (woollama
  sends the right request); the live round-trip is tool-LESS, and a live
  tool-using round-trip is unverified (no key). PENDING a run with
  ANTHROPIC_API_KEY.

## Follow-on slice — the rest of the providers (config-file + cloud built-ins) (2026-06-02)

Completes the multi-provider story two ways: more verified built-in clouds, and
a config file so ANY OpenAI-compat backend (self-hosted vLLM/llama.cpp, niche
clouds, overrides) is a one-entry addition — the durable answer architecture.md
calls for, since you can't hardcode someone's vLLM host.

- **Built-in clouds** (base URLs verified from each vendor's docs, not memory —
  verification caught that Together is `api.together.ai`, NOT the `.xyz` I'd have
  guessed): `openai` (`api.openai.com/v1`), `groq` (`api.groq.com/openai/v1`),
  `together` (`api.together.ai/v1`), `openrouter` (`openrouter.ai/api/v1`) — each
  Bearer + `<PROVIDER>_API_KEY`, `temperature=0`. Plus the existing ollama +
  anthropic.
- **`config.load_inferencers()`** reads an optional `$config/inferencers.toml`
  (`[inferencers.<name>]` with `base_url`, optional `api_key_env`, optional
  `extra_body`; `${VAR}` expanded). **MERGED OVER the built-ins** (same name
  overrides) — a deliberate departure from recipes/mcp.json *replace* semantics:
  an inferencer registry is infrastructure you extend, not content you own
  wholesale, so adding one provider must not wipe the defaults.
  `inferencers._registry()` builds built-ins then overlays config, rebuilt per
  call (live env/edits).
- **Tests** (`tests/test_inferencers.py`, +autouse clean-config fixture for
  hermeticity): the cloud built-ins resolve with the right URL/Bearer; config
  adds a custom `vllm` (no auth) and overrides a built-in's base_url while the
  other built-ins survive; `${VAR}` expansion; missing `base_url` errors.
- **Roadmap:** the OpenAI-compat cloud-inferencer seam is now **functionally
  complete** — 6 providers built in, and anything else is a config entry. (Live
  round-trips for the cloud providers remain unverified without keys, same
  posture as anthropic.) Lint note: ruff is configured (E/F/W/I/B) but not run
  by pytest and has drifted (~28 issues tree-wide, mostly the `;` pattern in
  earlier test files) — a quick `ruff --fix` cleanup is a worthwhile small slice.
- **Verified**: default suite 81 passed / 10 deselected; changed files lint-clean.

## Follow-on slice — tool delegation to Claude Code (executor) (2026-06-06)

The "executor" concept the slice-i note deferred: a `claude-code/<model>` recipe
WITH a non-empty `tools` list (previously a 501) now **delegates** — Claude Code
owns the agentic loop and calls the recipe's allow-listed MCP tools itself;
woollama returns only the final answer. Distinct from the tool-less inferencer
(`run_completion`) and from the ollama recipes where woollama runs the tool loop.

- **Design fork settled by a spike, not by argument.** The open question was how
  Claude reaches the tools and whether the allow-list stays a hard boundary. A
  headless `claude -p` spike (haiku, vs. the bundled `hello` server) established:
  `--allowedTools X` + `--permission-mode dontAsk` is a **hard deny-all-else**
  (an out-of-list MCP tool was denied and recorded in `permission_denials`; the
  allowed one ran and the loop terminated); a direct-server `--mcp-config` exposes
  tools as the clean `mcp__<server>__<tool>` (no dot). → **Option B (config
  containment)**: hand Claude a per-recipe `--mcp-config` with ONLY the servers
  the allow-list references + `--allowedTools` with ONLY those tools. Defense-in-
  depth (config containment AND allow-list AND the slice-i built-in lockdown),
  clean naming, smallest blast radius, no loopback re-entrancy. Option A (point
  Claude at woollama's own `/mcp` aggregator) + per-session filtering is deferred.
- **`claude_code.run_delegated`**: writes a temp `--mcp-config`, maps
  `<server>.<tool>` → `mcp__<server>__<tool>` for `--allowedTools`, keeps
  `--strict-mcp-config` + `_DENY_TOOLS` + neutral cwd, caps `--max-turns`.
- **Nested-contamination fix.** The spike's child inherited THIS session's
  harness (its meta-tools) via `CLAUDECODE`/`CLAUDE_CODE_*`. New `_child_env()`
  strips that family (plus `ANTHROPIC_API_KEY`), used by the tool-less path too.
  A residual leak from global `~/.claude` is not env-fixable, so the **live gate
  is plain-terminal-only** (like slice-i).
- **Routing**: `router.orchestrate_events` claude-code branch delegates when
  `recipe["tools"]` is non-empty; `_delegate_mcp_servers` resolves the referenced
  servers from `config.load_mcp_servers()` (400 if one is missing — never a
  partial toolset). Works through every surface (HTTP chat, `/v1/responses`, MCP
  `chat`) for free. Bundled `cc-counter` recipe added.
- **Tests**: hermetic unit (argv/config containment, the exact allow-list
  boundary, env strip) + routing (delegates not 501; missing-server 400). Live
  gate (`@needs_claude_code`, plain terminal): delegated count + a shell-exec
  attempt refused in delegation mode. Positive path verified at the EVENT level
  through woollama's GENERATED config (resolving `${WOOLLAMA_EXAMPLES_DIR}`): the
  hello server launched and Claude invoked `mcp__hello__count_to`
  (`result.is_error=False`, no permission denials) — not merely a final-text
  claim. (The adversarial Bash-in-delegation refusal stays plain-terminal-only;
  it rests on construction + the shared `_DENY_TOOLS`, not yet observed.)
- **Verified**: default suite 132 passed / 16 deselected; ruff clean.

## Follow-on slice — claude-code tool lockdown: deny-list → allow-list-of-none (2026-06-06)

Running the delegation live gate surfaced a real robustness gap. The lockdown was
a DENY-LIST (`_DENY_TOOLS`: Bash/Read/Write/…). On a machine whose global Claude
config / plugins enable extra tools NOT in that list (this box has Skill,
Workflow, Cron, Monitor, ToolSearch, LSP, …), those tools stayed reachable: a
deny-list can't enumerate every deployment's tools, and `--permission-mode
dontAsk` does NOT deny them (it hard-denies only tools that would *prompt*;
auto-approved extras pass through — confirmed: a spike showed `Skill` launching,
not being denied). What looked like "nested-session contamination" was actually
this machine's global config: a fully-scrubbed `env -i` child still showed the
extras, so they're not a session leak.

Fix (authoritative `claude` flags, confirmed via the claude-code guide + probes):
- **`--tools ""`** — the primary lockdown: an ALLOW-LIST of built-in tools set to
  NONE. Removes the entire built-in/plugin set at the source, robust against any
  deployment's extras. (Added to BOTH the tool-less and delegation arg builders.)
- **`ENABLE_TOOL_SEARCH=false`** (in `_child_env`) — `--tools ""` also disables the
  built-in ToolSearch that surfaces *deferred* MCP tools (Claude Code's default),
  so the recipe's MCP tools would vanish; this loads them UPFRONT instead.
- `_DENY_TOOLS` kept as defense-in-depth, now carrying **`LSP`** — the one tool
  `--tools ""` leaves behind (only `--bare` drops it, and `--bare` breaks
  subscription auth — rejected: a probe returned "Not logged in").
- Rejected `--bare` (kills keychain/credential reads → no auth).

Verified at the event level through woollama's real `_build_delegate_args` +
`_child_env`: the delegated Claude now exposes ONLY the recipe's MCP server tools
(no built-ins, no LSP, no Skill/Workflow), the out-of-list MCP tool is hard-denied,
the delegated tool runs, and a shell-exec attempt is refused (Bash absent). Because
the harness tools are now stripped, the delegation + tool-less live gates pass
trustworthily even nested. 154 hermetic pass; ruff clean.

## Follow-on slice — executor adversarial safety pass (2026-06-06)

The "adversarial safety pass" the executor was always flagged as needing. A
focused security review of the delegation path + lockdown + stored store (with
each finding verified against the code, not speculated) found three real issues,
now fixed:

- **HIGH — provider keys leaked to the child.** `_child_env` was a DENY-LIST of 3
  vars, so `OPENAI_API_KEY`/`GROQ_API_KEY`/`TOGETHER_API_KEY`/`OPENROUTER_API_KEY`
  (and any `inferencers.toml` `api_key_env`) flowed into the `claude` child AND
  the MCP servers it spawns. A key-custody router must not do that. Fixed:
  `_child_env` is now an ALLOW-LIST (`_CHILD_ENV_ALLOW`: HOME/PATH/locale/proxy/
  TERM/… only) — no provider keys/secrets reach the child. Verified live that
  subscription auth still works with just that set (HOME carries `~/.claude`).
- **MEDIUM — host settings could undercut the sibling boundary.** The child read
  the host's `~/.claude/settings.json`; a `permissions.allow: mcp__*` rule there
  would auto-approve a tool and slip past `dontAsk`. Fixed: `--setting-sources
  project` on both arg builders (neutral temp cwd → loads no settings). Verified
  auth survives it.
- **MEDIUM — comma in a recipe tool name injected an extra `--allowedTools`
  entry** (a same-server sibling grant, e.g. `count_to,mcp__hello__hello`). Fixed:
  reject commas/whitespace in tool names — at the router (`_delegate_mcp_servers`
  → 400) and defensively in `_mcp_tool_name` (ValueError).

Verified SAFE (checked, hold up): no SQL injection (all duckdb queries
parameterized), no `--mcp-config` JSON injection (`json.dump` + server-name
validated against config), no argv/shell injection (`create_subprocess_exec` list
form), no tempdir/path traversal (random `TemporaryDirectory`/`mkdtemp`), turn +
timeout bounds present. Accepted LOW / out-of-scope: `--tools ""` empty-string
semantics are CLI-version-coupled (deny-list `_DENY_TOOLS` is the backstop);
conversations have no per-caller ownership (mitigated by 122-bit unguessable
`uuid4` ids; woollama is local single-user).

New: a live **sibling-denial** test (`test_claude_code_delegation_denies_same_
server_sibling`) — the review's top previously-untested risk — asserts at the
event level that an un-allow-listed same-server tool never executes. 156 hermetic
pass; ruff clean; all three `@needs_claude_code` live gates green.
