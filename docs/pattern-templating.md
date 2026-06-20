# Pattern templating + the `/w1/` namespace (design spec)

> Status: **IMPLEMENTED** (was proposal / hand-off). Written by the cosmic-fabric
> side as the consuming client; built woollama-side in the Rust `woollamad`
> (`woollama-server` crate). The executable contract lives in cosmic-fabric's
> `src/mock-woollamad` (cosmic-fabric's `test_integration.py` asserts against it).
>
> **Corrections from the build** (this doc was written before implementation; two
> things changed — see the call-outs inline):
> 1. The fabric backend is configured in **`mcp.json`** (a `fabric` key), **NOT**
>    `[inferencers.fabric]`. fabric is not OpenAI-compatible and `woollama-engine`
>    is parity-locked; the engine's `inferencers.toml` loader requires every entry
>    to have a `base_url` and would **error** on a fabric entry. See "Fabric as a
>    managed backend" below.
> 2. Additional pattern backends (fabric, future providers) plug in via a
>    **`PatternBackend` trait** (`woollama-server/src/pattern_backend.rs`); native
>    recipes stay the built-in core. `lib.rs` handlers are backend-agnostic. See
>    "Implementation touch points".

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

## Fabric as a managed backend (woollama runs `fabric --serve` and routes to it)

**Requested explicitly — do not lose fabric's capabilities.** fabric has a lot worth
keeping: the full ~250-pattern **library**, fabric's real prompt **assembly**, named
**contexts**, prompt **strategies**, output **language**, model-side **web search**,
and **vision** (`fabric -a`). Rather than reimplement or drop these, woollama should be
able to **run a `fabric --serve` instance and route to it** as a backend. This is the
agreed *"fabric behind woollama"* architecture: cosmic-fabric talks only to woollama;
**woollama owns the fabric deployment** (today cosmic-fabric owns it — that supervisor
moves DOWN into woollama).

So woollama has **two pattern backends**, selected per pattern/call:
- **Native** — the directory scan + `Recipe::render` above. Lean, no fabric process,
  naive `{{var}}` substitution. The common offline path.
- **Fabric-routed** — woollama owns a `fabric --serve` and routes to it, gaining fabric's
  full machinery. The same `/w1/` surface; a different backend.

Design:
1. **A `fabric` backend** (a `PatternBackend`, NOT an engine inferencer). It is **NOT**
   OpenAI-compatible — it speaks fabric's REST (`POST /chat` SSE, `GET /patterns/names`,
   `GET /patterns/{name}`, `GET /models/names`).

   > **CORRECTION (as built):** config lives in **`mcp.json`** under a top-level `fabric`
   > key (mirroring `conversationStore`), **NOT** `[inferencers.fabric]`. The engine's
   > `inferencers.toml` loader requires every `[inferencers.*]` entry to have a `base_url`
   > and **errors** otherwise, and `woollama-engine` is parity-locked — so a fabric entry
   > there breaks config load. fabric is a server-layer backend, not an engine inferencer.
   > ```jsonc
   > // mcp.json
   > { "fabric": {
   >     "managed": true,                       // woollama spawns + supervises fabric --serve
   >     "default_model": "ollama/qwen3:14b-iq4xs", // fallback model (fabric patterns have no
   >                                            //   bound inferencer); enables woollama/<name>
   >                                            //   via /v1/chat/completions (no model slot there)
   >     "command": "fabric",                   // optional, default "fabric" (resolved on PATH)
   >     "address": "127.0.0.1:PORT"            // optional fixed bind; default a persisted free port
   > } }
   > // or route to an externally-run fabric:  { "fabric": { "url": "http://127.0.0.1:PORT" } }
   > ```
2. **woollama manages the fabric process** (`managed: true`): an `ensure_serve`-style
   supervisor *inside* woollama — pick a loopback port, `fabric --serve --address …`,
   poll readiness, reuse across restarts. (It is exactly the supervisor cosmic-fabric has
   today in `core.py:FabricClient.ensure_serve`; it relocated into woollama.) Or route to
   a configured `url` for an externally-run fabric.

   > **As built:** lifecycle is **reuse + graceful-kill** — the spawned fabric is detached
   > (no kill-on-drop) and its address is persisted to
   > `$XDG_RUNTIME_DIR/woollama.fabric-addr`, so a woollamad restart reuses the live fabric
   > instead of orphaning it; it is killed only on graceful shutdown. The provider→vendor map
   > fabric `/chat` needs is derived from fabric's own `/models/names` (not hardcoded).
3. **Patterns from fabric** — woollama sources the pattern list + assembly from the managed
   fabric (`/patterns/names`, `/patterns/{name}`), exposing fabric's full library under
   `/w1/patterns` and `woollama/<name>`. Alternative to / complement of the directory scan.
4. **Advanced features pass through.** `/w1/patterns/{name}/run` accepts optional
   `context` / `strategy` / `language` / `search`, forwarded to fabric's `/chat` when the
   pattern's backend is fabric. (Native patterns don't support them.) This is how woollama
   keeps these capabilities instead of dropping them.
5. **Vision** — route image runs to the managed fabric's `-a` until woollama has native
   multimodal; keeps vision working with zero cosmic-fabric change.

**Per-pattern backend selection:** a pattern resolves to native vs fabric by its source
(a `recipes.toml`/`[patterns]` recipe → native; a fabric-sourced pattern → fabric), or an
explicit `backend = "fabric" | "native"` on the run call. Sensible default: fabric when a
fabric provider is configured (full capabilities), native otherwise.

## Implementation touch points (`woollama-server/src/`) — as built
- **`config.rs`** — `Recipe::render` + the extracted `render_system()`; `scan_vars()`;
  `PatternSource`; the `[patterns]` dir-scan (`load_patterns()`); `load_fabric_config()`
  (reads the mcp.json `fabric` key).
- **`pattern_backend.rs`** (NEW — the plugin seam) — the `PatternBackend` trait
  (`id`/`list`/`has`/`render`/`run`/`v1_addressable`/`proxies`/`proxy`/`shutdown`) +
  `register_all()`, the single composition root that assembles configured backends. The
  trait speaks only woollama terms (name, variables, rendered system, OpenAI `Response`);
  **no backend-specific concept appears in it.** This is the model for ANY additional
  non-OpenAI system: add a module implementing the trait + one line in `register_all`.
- **`fabric.rs`** (NEW) — `impl PatternBackend for FabricBackend` + `register()`. ALL
  fabric-isms are confined here: the supervisor (`ensure_serve`, reuse + graceful-kill),
  the fabric REST client (`/chat` SSE, `/patterns/*`, `/models/names`), the provider→vendor
  map, the fabric `/chat` body shape, and the fabric-SSE ⇄ OpenAI translation. Advanced
  fields (`context`/`strategy`/`language`/`search`/options) pass through to fabric; vision
  passes through the `/fabric/*` proxy (`fabric -a`).
- **`lib.rs`** — backend-AGNOSTIC: `build_state` calls `pattern_backend::register_all()`;
  the `/w1/...` handlers + `/v1/models` + `/v1/chat/completions` iterate
  `AppState.pattern_backends` (native recipes win on a name collision, then registration
  order); the transparent proxy is mounted at `/{backend.id()}/*` (no backend-name literal).
- **`woollama-engine/`** — **NOT touched** (parity-locked; never sees a `{{var}}`).

## Deferred (not in the MVP)
- **Native multimodal** — `image_url` content parts → ollama multimodal. Until then,
  vision is **covered by the fabric backend** (`fabric -a`), so it isn't lost.
- **Variable metadata overlay** — defaults/choices for `/w1/patterns` (frontmatter or
  `[patterns.<name>]`).

> **Not dropped — provided by the fabric backend** (see above): the fabric advanced
> features (`context` / `strategy` / `language` / `search`), the full pattern library,
> and vision. The **native** pattern path doesn't implement them; route to the fabric
> backend when you need them. (`session` is already woollama's via `/v1/responses`.)
> The MVP can ship the native path first; the fabric backend can land alongside or just
> after — but it's a first-class goal, not an afterthought.

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
