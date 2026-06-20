# Pattern templating + the `/w1/` namespace (design spec)

> Status: **proposal / hand-off.** Written by the cosmic-fabric side as the
> consuming client; this is woollama-side work to take up. Targets the Rust
> `woollamad` (`woollama-server` crate). The executable contract lives in
> cosmic-fabric's `src/mock-woollamad` (extend it; cosmic-fabric's
> `test_integration.py` asserts against it).

## Why

cosmic-fabric wants woollama to **totally own prompt templating**, so it stops
needing `fabric --serve`. Today fabric does three jobs cosmic-fabric depends on:
stores the `scribe-*` patterns (markdown system prompts at
`~/.config/fabric/patterns/<name>/system.md`), **assembles** prompts (system prompt
+ `{{var}}` substitution + user input), and exposes them for discovery. We want
woollama to do all three: cosmic-fabric sends `(pattern, variables, model, input)`
and woollama renders + infers.

The blocker found in exploration: woollama **recipes** are *static* system strings
(no `{{var}}` substitution), bind a *fixed* inferencer, and there is no pattern
*source*. Those three limitations are the design targets below.

## The namespace decision: `/v1/` is OpenAI's; `/w1/` is woollama's

`/v1/*` exists to be **OpenAI-compatible** — every current endpoint
(`/v1/chat/completions`, `/v1/responses`, `/v1/conversations`, `/v1/models`) is a
real OpenAI route, which is the whole value of that prefix. Pattern templating is
**not** an OpenAI concept, so it must **not** live under `/v1/` (no `/v1/patterns`).

Introduce **`/w1/`** as woollama's own namespace — a deliberate parallel to `/v1/`
(ideally everything would be provider-prefixed: `/openai/v1/...`, `/woollama/...`,
but `/w1/` is the pragmatic, unambiguous choice). Rule going forward:

- **`/v1/*`** — OpenAI-compatible only. Unchanged. Pre-assembled prompts, raw
  inference, models/responses/conversations.
- **`/w1/*`** — woollama-native value-add. Pattern templating lives here.

This keeps OpenAI SDK clients working against `/v1` and gives woollama room to grow
its own surface without pretending it's OpenAI.

## The capability

**Patterns ARE recipes.** Reuse the `Recipe` struct + the existing dispatch; do not
introduce a parallel "pattern" concept. Two additions make recipes templated:

### 1. `Recipe::render(variables, model_override)` — the one new primitive
In `woollama-server/src/config.rs`:
```rust
impl Recipe {
    fn render(&self, variables: &Map<String, Value>, model_override: Option<&str>) -> Recipe
}
```
- Clone the recipe; for each `{{k}}` do a **dumb string replace** with `v` in
  `system` (byte-match fabric's substitution — `cosmic-fabric/src/core.py:359`,
  `sysp.replace("{{"+k+"}}", str(v))`). Leave unsupplied `{{x}}` tokens verbatim.
  **No tera/handlebars** — a new dep + it diverges from fabric's bytes.
- If `model_override` is `Some`, replace `inferencer`.
- Hand the rendered recipe to the **existing** `orchestrate_*` / `run_claude_*`
  paths — **do not touch `woollama-engine`** (parity-locked; `build_setup` consumes
  the recipe Value and is cross-language-synced). Rendering is a pure server-layer
  transform applied immediately before dispatch. This also covers claude-code free.

### 2. A fabric-pattern *source* (read-only directory scan)
In `config.rs`, a new config block:
```toml
[patterns]
dir = "~/.config/fabric/patterns"
default_inferencer = "ollama/qwen3:14b-iq4xs"
```
`load_patterns()`: for each `<dir>/<name>/system.md`, build
`Recipe { inferencer: default_inferencer, system: <file contents>, tools: [] }`;
merge into the recipes map `build_state` already holds. **`recipes.toml` wins** on a
name collision (hand-authored override beats auto-discovered). This is read-only
file parsing — **not** a `fabric --serve` dependency (which is the whole point).
`default_inferencer` is the fallback model when a call omits `model` (model is now
per-call, see below).

## The `/w1/` HTTP surface (3 endpoints)

### `GET /w1/patterns` — discovery
```jsonc
→ { "data": [ { "name": "scribe-summarize",
                "variables": ["depth", "language"],   // regex-scanned {{...}} tokens
                "source": "fabric" | "recipe" } ] }
```
Variable *names* come from scanning `{{...}}` in `system`. **Honesty constraint:**
fabric patterns carry no variable metadata — there are no defaults or value
enumerations to surface (just `<name>/system.md`). Defaults/choices are a **later**
optional overlay (frontmatter or `[patterns.<name>]` in config), not in v1.
(Patterns also remain in `/v1/models` as `woollama/<name>` for OpenAI-client
addressability — that does not change.)

### `POST /w1/patterns/{name}/render` — render-without-run (cosmic-fabric's `assemble`)
```jsonc
{ "input": "<user text>", "variables": { "depth": "ultra" } }
→ { "prompt": "<system prompt, {{vars}} substituted>\n\n<input>" }
```
For prompt-preview / agent hand-off. No model run.

### `POST /w1/patterns/{name}/run` — templated run + infer
```jsonc
{ "input": "<user text>",                       // or an OpenAI messages array
  "variables": { "depth": "ultra" },
  "model": "ollama/qwen3:14b-iq4xs",            // optional per-call override
  "stream": false,
  "options": { "temperature": 0.3 } }
→ an OpenAI chat-completion object (choices[0].message.content),
  or OpenAI SSE deltas when stream:true (choices[0].delta.content, [DONE]).
```
Internally: `recipe = recipes[name].render(variables, model)` then dispatch through
the **existing** orchestration/streaming path. Returning the OpenAI completion/SSE
shape means cosmic-fabric's existing `WoollamaClient.chat`/`chat_stream` parsers work
unchanged — only the URL differs.

> **Per-call model** (requirement 3): fabric patterns are model-agnostic
> (cosmic-fabric's per-run picker chooses ollama/qwen3 vs anthropic/sonnet). The
> recipe's bound `inferencer` is the default; `model` on the run call overrides it.
> The id cosmic-fabric sends is already woollama's inferencer namespace —
> `core.py:woollama_model()` produces `ollama/...`, `anthropic/...`.

## Implementation touch points (`woollama-server/src/`)
- **`config.rs`** — `Recipe::render`; the `[patterns]` config struct; `load_patterns()`
  + merge/precedence. (Largest change.)
- **`lib.rs`** — call `load_patterns()` in `build_state` and merge into `recipes`;
  add the 3 `/w1/...` routes + handlers (`render` calls `Recipe::render` and returns
  the prompt; `run` renders then reuses the existing dispatch; `patterns` lists). Keep
  patterns in `/v1/models`.
- **`woollama-engine/`** — **NOT touched** (parity-locked).

## Deferred (not in the MVP)
- **Vision** — `image_url` content parts → ollama multimodal. cosmic-fabric does
  vision via the `fabric -a` CLI today; replacing it is independent of templating.
- **Variable metadata overlay** — defaults/choices for `/w1/patterns` (frontmatter or
  `[patterns.<name>]`).
- **fabric advanced `/chat` features** — `context`/`strategy`/web-`search` are
  fabric-specific and dropped; `language` becomes a `{{language}}` variable; `session`
  is already woollama's (`/v1/responses`).

## Verification / the executable contract
The contract is `cosmic-fabric/src/mock-woollamad` — extend it to serve `GET
/w1/patterns`, `POST /w1/patterns/{name}/render`, `POST /w1/patterns/{name}/run`
(echo pattern+variables so the daemon tests can assert the pattern reached woollama).
cosmic-fabric's `test_integration.py` then proves the daemon path with **no fabric
present**. woollama-side: add `render` / `load_patterns` / `/w1/*` unit + conformance
tests; the parity suites for `woollama-engine` must stay green (engine untouched).

## What cosmic-fabric will call (the consumer side, for reference)
- discovery: `GET /w1/patterns` → the `patterns`/`surface` daemon ops.
- assemble: `POST /w1/patterns/<name>/render` → the `assemble` daemon op.
- run: `POST /w1/patterns/<name>/run` (with `variables` + `model`) → the `run` /
  `stream_run` paths, dropping `FAB.assemble_prompt`.

Once these three `/w1/` endpoints exist, cosmic-fabric can stop calling fabric for
assembly, discovery, and inference of plain pattern runs.
