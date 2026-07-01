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
       "variables": [                          // one object per {{...}} token, in prompt order
         { "name": "depth",
           "default": "normal",                // optional — applied at render/run when unset
           "choices": ["normal", "ultra"],     // optional — surfaced for UIs; NOT enforced
           "description": "How deep to go" },  // optional
         { "name": "language" }                // a token with no overlay is just its name
       ],
       "source": "recipe" }                     // or "fabric"
   ] }
```

Each `{{...}}` token in the system prompt becomes a `variables` entry, in
first-seen order. The `name` is always present; `default`/`choices`/`description`
appear only when a recipe declares them (see the `[recipes.<name>.variables.<var>]`
overlay in [configuration.md](configuration.md#recipestoml)). Absent fields are
omitted — no `null` noise. (fabric-library patterns are listed by name with
`variables: []` — scanning the whole library on every call is too costly, and
fabric patterns carry no overlay; their variables resolve on render/run.)

The overlay is metadata, not enforcement: `choices` is advisory (a caller may pass
a value outside it), but a `default` **is** applied — if the caller omits a
variable that has one, `render` and `run` substitute the default before dispatch
(a caller-supplied value always wins; a variable with no default is left verbatim).

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
- **`input` as a messages array, for fabric patterns:** fabric's `/chat` takes a
  single `userInput` string, so woollama concatenates every `user` message's text
  (`\n\n`-joined). Non-user turns (assistant/system) are **not** sent — fabric
  patterns operate on raw content, and role scaffolding would change what the
  pattern sees. If you need full multi-turn context, talk to fabric directly via
  [`/fabric/*`](#the-fabric-backend). (Native recipes run through the engine and
  keep the messages array intact.)

### Vision (image input)

A `user` message whose `content` is an array can include an `image_url` part
(a `data:` URL or an `http(s)://` URL):

```jsonc
{ "input": [{ "role": "user", "content": [
    { "type": "text", "text": "What's in this image?" },
    { "type": "image_url", "image_url": { "url": "data:image/png;base64,iVBOR..." } }
  ]}],
  "model": "ollama/llama3.2-vision:latest"   // MUST be a vision-capable model
}
```

Both pattern kinds accept this — but by different routes, because a **vision-capable
`model` is required either way** (a text model, including a text `fabric.default_model`,
won't see the image):

- **Native recipes** (bound to, or `model`-overridden with, a vision model) —
  nothing special: the engine forwards the messages array **verbatim** to ollama's
  OpenAI-compatible endpoint, which accepts `image_url` (data URLs included). Text +
  images + `{{var}}`-rendered system all flow through. Works on `/w1/…/run` **and**
  via `/v1/chat/completions` as `woollama/<recipe>`. This is the plain OpenAI
  multimodal path.
- **fabric patterns** — fabric's REST `/chat` has no attachment field, so woollama
  routes image input to fabric's one-shot **CLI** (`fabric --pattern=… --attachment=…`,
  user text on stdin). Specifics of that path:
  - **Image source:** an `http(s)://` URL is passed to fabric as-is; a
    `data:<mime>;base64,…` URL (padded or unpadded) is decoded to a temp file
    (removed after the run). Any other `image_url` (a bare filesystem path, or an
    undecodable data-URL) is **rejected with a `400`** — the run fails loudly rather
    than silently answering text-only as if there were no image.
  - **⚠️ An `http(s)` image URL is fetched server-side by fabric** (woollama passes
    it straight to `-a`), so it carries the usual SSRF consideration — a caller can
    make fabric request an arbitrary URL. woollama binds loopback/UDS by default,
    which contains this to local callers; if you expose woollama beyond localhost,
    prefer `data:` images (self-contained, no server-side fetch).

Image size: `data:` images are capped at **20 MiB** decoded (a `400` past that), and
the request-body limit is raised to **32 MiB** (from axum's 2 MiB default) so real
photos aren't rejected as too large before the model sees them — this applies to
both the fabric and native paths.
  - **One image per run** — fabric's `-a` is single-attachment; extras are ignored
    (logged).
  - **Non-streaming** upstream: the CLI returns the whole answer at once. A
    `stream:true` request still gets back the OpenAI SSE *shape* (one content chunk +
    `[DONE]`), so streaming clients don't break.
  - Full fabric-native vision is also always available verbatim via
    [`/fabric/*`](#the-fabric-backend).

## MCP prompts get the same templating

Recipes are also exposed as **MCP prompts** on the `/mcp` surface. Their `{{var}}`
tokens are advertised as prompt **arguments** (carrying the variable
`description` from the metadata overlay), and `prompts/get` renders the pattern
with the arguments you supply — applying the same variable `default`s as the HTTP
surface for any argument you omit. So an MCP client (Claude Desktop, etc.) gets the
same parameterized templating as `/w1` — MCP covers *render*; *run* is the `chat`
verb.

## The fabric backend

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
