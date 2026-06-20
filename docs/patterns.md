# Pattern templating — the `/w1/` namespace

woollama can **own prompt templating**: store parameterized system prompts, fill
their `{{variables}}`, and run them — so a client sends `(pattern, variables,
input)` and gets a completion back, without assembling prompts itself.

This lives under **`/w1/`**, woollama's own namespace — a deliberate parallel to
`/v1/`:

- **`/v1/*`** is OpenAI-compatible *only*. Every route is a real OpenAI route.
- **`/w1/*`** is woollama-native value-add. Templating is not an OpenAI concept,
  so it doesn't pretend to be one.

## Patterns *are* recipes

There's no separate "pattern" object. A **pattern is a recipe** (see
[Configuration → `recipes.toml`](configuration.md#recipestoml)) whose `system`
prompt may contain `{{var}}` tokens. The one new primitive is rendering:
substitute `{{k}}` with a value immediately before dispatch. `woollama-engine`
never sees a `{{var}}` — rendering is a pure server-layer step in front of the
existing orchestration path.

Substitution is a **dumb string replace** (`{{k}}` → value), byte-for-byte
compatible with fabric's substitution. Unsupplied tokens are left verbatim. There
is no template engine (no conditionals/loops) — that is deliberate.

Patterns come from three sources, all addressed the same way:

| Source | Where | `source` | Backend |
|---|---|---|---|
| Hand-authored recipes | `recipes.toml` `[recipes.*]` | `recipe` | native (engine) |
| A fabric-style directory scan | `recipes.toml` `[patterns]` → `<dir>/<name>/system.md` | `fabric` | native (engine) |
| A live fabric backend's library | mcp.json `fabric` → `fabric --serve` | `fabric` | fabric-routed |

**On a name collision, a `recipes.toml` recipe wins**, then registration order.
Every pattern is also addressable as `woollama/<name>` in `/v1/models` and
`/v1/chat/completions` (a fabric-backed pattern needs `fabric.default_model` to be
addressable there — see below).

## The three `/w1/` endpoints

### `GET /w1/patterns` — discovery

```jsonc
→ { "data": [
     { "name": "scribe-summarize",
       "variables": ["depth", "language"],   // {{...}} tokens scanned from the system prompt
       "source": "recipe" }                   // or "fabric"
   ] }
```

Variable *names* are scanned from `{{...}}` in the system prompt. (fabric-library
patterns are listed by name with `variables: []` — scanning the whole library on
every call is too costly; their variables resolve on render/run.)

### `POST /w1/patterns/{name}/render` — render without running

```jsonc
{ "input": "<user text>", "variables": { "depth": "ultra" } }
→ { "prompt": "<system prompt, {{vars}} substituted>\n\n<input>" }
```

For prompt preview or handing a fully-assembled prompt to another agent. No model
runs.

### `POST /w1/patterns/{name}/run` — render then infer

```jsonc
{ "input": "<user text>",                  // or an OpenAI messages array
  "variables": { "depth": "ultra" },
  "model": "ollama/qwen3:14b-iq4xs",       // optional per-call inferencer override
  "stream": false,
  "options": { "temperature": 0.3 } }      // merged into the inference request
→ an OpenAI chat-completion object (or OpenAI SSE deltas when stream:true)
```

The result is the standard OpenAI completion/SSE shape, so existing OpenAI
parsers work unchanged — only the URL differs.

- For a **native** pattern, `model` overrides the recipe's bound `inferencer`;
  omit it to use the recipe's own.
- For a **fabric-backed** pattern (which has no bound inferencer), `model` is
  required unless `fabric.default_model` is set.
- `options` (e.g. `temperature`) are merged into the inference request.

## MCP prompts get the same templating

Recipes are also exposed as **MCP prompts** on the `/mcp` surface. Their `{{var}}`
tokens are advertised as prompt **arguments**, and `prompts/get` renders the
pattern with the arguments you supply. So an MCP client (Claude Desktop, etc.)
gets the same parameterized templating as `/w1` — MCP covers *render*; *run* is
the `chat` verb.

## The fabric backend (`fabric --serve` behind woollama)

woollama can **own a fabric deployment** so clients get fabric's full machinery —
its ~250-pattern library, real prompt assembly, named contexts, prompt
strategies, output language, and model-side web search — without managing fabric
themselves. Enable it with the `fabric` key in mcp.json (see
[Configuration → fabric backend](configuration.md#fabric-backend)).

It adds two things:

1. **fabric's library on `/w1/`** — sourced from the running fabric, rendered and
   run through it (`/w1/patterns`, render, run all work for fabric patterns).
2. **A transparent proxy at `/fabric/*`** — fabric's REST API (`/chat` SSE,
   `/patterns/names`, `/patterns/{name}`, `/models/names`) is reverse-proxied
   verbatim. A client that already speaks fabric just points its base URL at
   `…/fabric/` and works unchanged — including advanced fields
   (`context`/`strategy`/`language`/`search`) and vision, which pass straight
   through.

**Two backends, one surface.** Native patterns are lean and offline (no fabric
process, naive `{{var}}`); fabric-routed patterns get fabric's full fidelity. Use
`/w1/` for the common path and `/fabric/` when you need fabric-specific features.

> The fabric backend is a **plugin** behind woollama's `PatternBackend` trait —
> it's the reference implementation for adding any non-OpenAI prompt/inference
> system. See [Extending woollama](extending.md).

See also: [Configuration](configuration.md) · [Extending woollama](extending.md).
