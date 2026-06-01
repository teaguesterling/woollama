# Slice (e) — woollama as an MCP server (handoff / pickup note)

Status: **DONE (2026-06-01).** Implemented + tested. Default suite 45 passed
(+5 MCP unit tests), plus 1 opt-in stdio integration test (4 deselected). See
"What shipped" at the bottom. The rest of this note is the original plan, kept
for context.

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
  `_chat_tools()` is the extensibility hook from decision #3: returns `[chat]`
  today, one-line concat to re-export registry tools later. `serve()` runs
  stdio with `show_banner=False` (stdout is the JSON-RPC channel).
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
- **Verified**: default suite 45 passed / 5 deselected; the non-Ollama stdio
  integration test passes (~5s, really spawns the example servers). The
  Ollama-gated end-to-end test runs when qwen3:14b-iq4xs is available.

### Client mcp.json entry
```json
{ "command": "woollama", "args": ["mcp"] }
```

### Natural next slices
- Re-export discovered downstream tools onto tools/list (flip the decision-#3
  concat in `_chat_tools`).
- HTTP/SSE transport (FastMCP supports it; `serve()` picks the transport).
