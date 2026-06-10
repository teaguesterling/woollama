# Core library extraction — `woollama.core`

> **Status: design of record, not yet implemented.** This describes the target
> split: a server-free `woollama.core` subpackage that other Python projects
> (first consumer: **lackpy**) embed for model management, routing, and recipe
> orchestration — with the FastAPI/MCP **router** layered on top of it. Signatures
> below are the *proposed* surface; current code is referenced where it differs.

## Why

woollama today is a **server**: importing `woollama.router` pulls in FastAPI,
uvicorn, the MCP server build, and a config-file read at import time, and the
inference entry points reach for module-global singletons. That's correct for a
router you run — but it makes woollama unusable as an *embedded library*.

lackpy wants exactly the embeddable half: it should stop doing model/provider
management itself and delegate to woollama, while staying (a) the scoped
**interpreters** and (b) the **MCP/tool layer**. A running woollama sidecar would
be heavy-handed for that; a library is the right coupling.

The valuable parts are *already* nearly server-free — `inferencers` and `config`
import no FastAPI, and `orchestrate_events` already takes the tool registry as a
parameter. So this is mostly **relocation + de-globalisation**, not a rewrite.

The guiding idea: a **recipe** (`{model, system, tools, params}`) and a lackpy
**scoped interpreter** are the same kind of object — a bound configuration of
model + program + tools. If `woollama.core` owns that type and the loop that runs
it, the mapping between them is *identity*, and lackpy's interpreter configs
become recipes it builds in memory.

## The split

```
src/woollama/
  core/                    # server-free — NO fastapi / uvicorn / mcp imports
    __init__.py            # public API re-exports
    config.py              # MOVED — inferencer/recipe loading + ${VAR} (pure file parsing)
    inferencers.py         # MOVED verbatim (already import-clean)
    recipes.py             # MOVED + in-memory Recipe constructor
    ollama_native.py       # MOVED — num_ctx native routing
    inference.py           # NEW — complete() / complete_stream()
    orchestrate.py         # NEW — orchestrate() / orchestrate_events()
    tooling.py             # NEW — ToolProvider, ToolSpec, ToolResult, Capabilities, renderer
  router.py                # SERVER — imports woollama.core, adds FastAPI
  manager.py               # SERVER — Registry implements core.ToolProvider
  mcp_server.py            # SERVER
  conversations.py         # SERVER — conversation state stays OUT of core
  binding.py, __main__.py  # SERVER
```

**The server-free guarantee is a test.** `tests/test_core_is_server_free.py`
imports `woollama.core` and asserts `fastapi`, `uvicorn`, `mcp`, and
`woollama.router` are absent from `sys.modules` afterwards. That test *is* the
contract that keeps the subpackage embeddable — break it and CI fails.

Packaging: a **subpackage** (`woollama.core`) first, not a separate distribution.
The subpackage proves the seam with the least friction; a later `woollama-core`
dist split (so lackpy doesn't pull fastapi/uvicorn/mcp transitively) is a clean
follow-on once the boundary holds.

## What stays OUT of core (scope discipline)

These are **router** concerns and do **not** move into the embeddable core:

- the FastAPI app, the mounted MCP server, the Unix-socket binding;
- conversation **state**: the handle table, persistence, store providers
  (MCP/HTTP), attach-by-key (see [Conversations API](conversations-api-design.md)).

The core is **config + provider/model routing + the recipe loop**, full stop. If
an embedder later wants stateful conversations, that is a separate, deliberate
decision — it is not pulled in by default.

## Public API — `woollama.core`

### Model registry & inference

```python
@dataclass(frozen=True)
class Inferencer:                       # moved as-is
    name: str
    base_url: str
    api_key_env: str | None = None
    extra_body: dict = field(default_factory=dict)
    models: tuple[str, ...] = ()
    discover: bool = False
    model_patterns: tuple[str, ...] = ()
    capabilities: "Capabilities | None" = None   # NEW — per-model render targeting (below)
    # headers(), chat_url() unchanged

class ModelRegistry:
    """Provider/model resolution. Built from config OR constructed in memory."""
    @classmethod
    def from_config(cls) -> "ModelRegistry": ...        # built-ins + inferencers.toml
    def __init__(self, inferencers: dict[str, Inferencer] | None = None): ...
    def add(self, inf: Inferencer) -> None: ...          # in-memory registration
    def resolve(self, model: str) -> tuple[Inferencer, str]: ...   # "ollama/x" -> (inf, "x")

class InferenceError(Exception):        # was OrchestrationError
    def __init__(self, message: str, kind: str, status: int, payload: dict | None = None): ...

async def complete(model: str, messages: list[dict], *, registry: ModelRegistry,
                   options: dict | None = None,
                   api_key: str | None = None,        # NEW — per-call override
                   base_url: str | None = None) -> str: ...

async def complete_stream(model, messages, *, registry, ...) -> "AsyncIterator[str]": ...

def complete_sync(*args, **kw) -> str: ...   # asyncio.run wrapper; errors inside a running loop
```

Two deliberate additions the extraction is the right moment for, both real
library limitations today:

- **per-call `api_key` / `base_url` overrides** on `complete` (the env-var-only
  model breaks multi-key / multi-tenant embedders);
- **`ModelRegistry` as an object** so an embedder builds its provider set in
  memory and never reads `$WOOLLAMA_CONFIG_DIR`. The server uses `from_config()`.

### Recipes & the system prompt

```python
@dataclass(frozen=True)
class Recipe:
    model: str                          # "provider/model", matches /v1/models
    system: str | None = None           # DEFAULT system prompt (caller-owned)
    tools: tuple[str, ...] = ()          # allow-list, resolved by the ToolProvider
    params: dict = field(default_factory=dict)   # temperature, max_tokens, num_ctx, ...

async def orchestrate(recipe: Recipe, messages: list[dict], *, registry: ModelRegistry,
                      tools: "ToolProvider", system: str | None = None,
                      max_turns: int = 8) -> dict: ...

async def orchestrate_events(recipe, messages, *, registry, tools,
                             stream: bool = False) -> "AsyncIterator[dict]": ...
```

**System prompt — where it lives (the question that prompted this section).** It
is not ignored; it has a precedence, chosen so a lackpy interpreter whose
*program is generated per run* is first-class:

- `orchestrate` builds the final message list as `[system] + messages`, where
  `system` = the per-call `system=` argument **if given**, else `Recipe.system`
  **if set**, else none. `messages` stay **user/assistant only**.
- `complete` (raw, no recipe) does **no** system handling — the caller's
  `messages` are authoritative and carry their own system if they want one.

So a static prompt lives in `Recipe.system`; a dynamic prompt is passed per call
via `system=`; the recipe is the *vehicle* and the caller owns it. (Current code:
`recipes.Recipe` is a TypedDict `{inferencer, tools, system}` loaded from
`recipes.toml`; the move makes it an in-memory-constructible dataclass with the
TOML loader as just one producer.)

`orchestrate_events` stays the **single source of truth** for the loop
(system-prepend, allow-list boundary, tool dispatch, max-turns guard); the
non-streaming `orchestrate` is a thin drainer, exactly as today.

## Tools — OpenAI vs MCP, without pretending they're the same

The model on the far end of the loop speaks **OpenAI function-calling**:
`{name, description, parameters}` in, one **string** result out. No output schema,
no typed/multimodal results, no mid-call interaction.

An **MCP tool** is richer: `inputSchema` **and** `outputSchema`, tool
`annotations` (`readOnlyHint`/`destructiveHint`/`idempotentHint`/`openWorldHint`),
and a `CallToolResult` that is a **list of typed content blocks**
(text/image/audio/resource/resource-link) plus `structuredContent`, `isError`,
`_meta` — and the protocol supports progress, elicitation, and sampling.

Reconciling them naïvely (advertise an OpenAI schema, return `Any`, keep only text
blocks) is **lossy in three ways**, one of which is a bug:

1. `isError` dropped → a *failed* tool is indistinguishable from a successful
   empty result (the model can't tell it failed). **A defect, not a simplification.**
2. `structuredContent` and non-text blocks dropped silently.
3. Name mapping hidden: OpenAI function names must match `^[a-zA-Z0-9_-]{1,64}$` —
   woollama's namespaced `server.tool` contains a **dot**, which strict providers
   (OpenAI/Azure) reject (ollama is lenient). The `server.tool ↔ server__tool`
   round-trip has to live *somewhere*.

### Principle: lossless at the boundary, lossy only at render

The `ToolProvider` adapter **mirrors** the MCP result faithfully — it never drops
anything. A separate, **per-model renderer** decides what *this* target can
actually receive. The loss becomes a property of the target model (a text-only
model can't see an image), explicit and pluggable — not baked into the adapter.

```python
@dataclass(frozen=True)
class ToolSpec:                 # the advertisement
    name: str                  # OpenAI-legal name the model sees (mapped from source)
    schema: dict               # OpenAI function schema → the ONLY thing the model reads
    # MCP metadata carried through LOSSLESSLY for the policy / permission / UX layer:
    source_name: str | None = None     # original namespaced/MCP name (dispatch maps back)
    output_schema: dict | None = None
    annotations: dict | None = None    # readOnly/destructive/idempotent/openWorld hints
    meta: dict | None = None

@dataclass
class ToolResult:              # a faithful mirror of MCP CallToolResult
    blocks: list               # ALL content blocks, typed (text/image/audio/resource/...)
    structured: dict | None = None     # structuredContent
    is_error: bool = False             # isError — carried, never dropped
    meta: dict | None = None

@dataclass(frozen=True)
class Capabilities:            # render target profile (per Inferencer/model)
    accepts_image_parts: bool = False
    accepts_audio: bool = False
    accepts_structured: bool = False
    # default = text-only

class ToolProvider(Protocol):  # THE seam — the MCP Registry and lackpy each implement it
    def tools_for(self, allow: "Sequence[str]") -> list[ToolSpec]: ...
    async def dispatch(self, name: str, args: dict) -> ToolResult: ...

def render_tool_result(result: ToolResult, *, caps: Capabilities) -> "str | list[dict]":
    """Build the `tool` message content for THIS target — string for a text model,
    an array of parts for a multimodal one."""
```

The loop sends `[s.schema for s in specs]` to the model and, on a tool call, does
`content = render_tool_result(await tools.dispatch(...), caps=caps_for(inferencer))`
then appends `{"role": "tool", "tool_call_id": ..., "content": content}`. The
tool's `annotations` flow to the embedder's capability/permission layer (lackpy's
greyed-hints policy) even though the model never sees them.

### "Support per model, start by dropping"

Phase 1 ships **one** renderer, `TextRenderer`:

- text blocks → joined; `structured` → JSON-dumped and appended;
- `is_error` → surfaced in the content (e.g. an `[error]` prefix) — **fixes the
  silent-failure bug**;
- non-text blocks → a **placeholder** line (`[image omitted: <mime>]` / a resource
  link), *not* a silent drop.

Because `ToolResult` is already lossless, adding an image-capable renderer later
is **non-breaking** and touches **zero adapters** — you register a renderer for
models whose `Capabilities` accept image parts. The support is structural from day
one; only the rendering *coverage* grows.

### Deliberate non-goals of this seam

The stateless loop does **not** carry MCP **elicitation**, **sampling**, or
**progress** — those need an interactive path (woollama's `requires_action` /
managed-agents shape), not a request/response loop. This is named so it's a
conscious boundary, not an accidental omission.

## What the router becomes

Almost mechanical, because the logic already lives in registry-parameterised
functions:

- `router.complete_stateless` → thin wrapper over `core.complete(..., registry=…)`
  for `<provider>/<model>` and `core.orchestrate(...)` for `woollama/<recipe>`,
  passing the MCP `Registry` as `tools`.
- `manager.Registry` is adapted to satisfy `core.ToolProvider` — it already has
  `openai_tools_for` (→ `tools_for`, now returning `ToolSpec`s) and `dispatch`
  (now returning a `ToolResult` built from the `CallToolResult`).
- `config.load_recipes` now *produces* `core.Recipe` instances — one of several
  producers; lackpy's interpreter configs are another.
- Conversation state, store providers, attach-by-key, persistence — untouched.

## The lackpy integration shape

```python
class LackpyTools:                      # wraps lackpy's assorted tool definitions behind ONE interface
    def tools_for(self, allow):
        return [ToolSpec(name=..., schema=..., annotations=...) for t in ...]
    async def dispatch(self, name, args):
        return ToolResult(blocks=..., structured=..., is_error=...)

models = core.ModelRegistry(inferencers={...lackpy's providers...})   # no config files

# an interpreter scope ≡ a Recipe
recipe = core.Recipe(model="ollama/qwen3", system=interpreter.program,
                     tools=interpreter.toolbox_names, params={"temperature": 0.2})

result = await core.orchestrate(recipe, messages, registry=models, tools=LackpyTools())
```

lackpy sheds model/provider management entirely; keeps (a) the scoped interpreters
(now *producers* of `Recipe`) and (b) the tool layer (which *is* the
`ToolProvider`). Its "messy assortment of tool definitions" is normalised exactly
once, in its `tools_for`/`dispatch` adapter — woollama-core never sees the mess.

## Phased plan

Each step ships independently; the hermetic suite stays green throughout.

1. **Create `woollama/core/`; move the import-clean modules** (`config`,
   `inferencers`, `recipes`, `ollama_native`) in, leaving re-export shims at the
   old paths (zero behaviour change). Add `test_core_is_server_free.py`.
2. **Extract `core/inference.py`** (`complete` / `complete_stream`) from the
   router's non-recipe path; the router calls it. Add per-call `api_key`/`base_url`.
3. **Move `orchestrate` / `orchestrate_events` → `core/orchestrate.py`**,
   parameterised on `ToolProvider`; define `ToolProvider`/`ToolSpec`/`ToolResult`/
   `Capabilities` + `TextRenderer` in `core/tooling.py`; adapt `manager.Registry`
   to satisfy the protocol (incl. the `server.tool ↔ server__tool` name round-trip
   and carrying `isError`/`annotations`).
4. **Add `ModelRegistry`, the in-memory `Recipe` dataclass, sync wrappers.**
5. **Delete the shims; update server imports to `woollama.core`.** Optionally split
   a `woollama-core` distribution once the boundary has held.

## Decisions

**Settled:** subpackage (not a separate dist) first; core is async with a
`complete_sync` wrapper; per-call key/base_url overrides land with the extraction;
`Recipe.model` stays a single `"provider/model"` string; the core speaks
OpenAI-shaped messages/responses (the lingua franca — no bespoke native type);
conversation state stays server-only.

**Deferred (named, not dropped):** non-text renderers (image/audio) — structural
support now, coverage later; a `woollama-core` dist split; MCP
elicitation/sampling/progress through an interactive (non-loop) path; strict-mode
schema massaging for providers that demand it.
