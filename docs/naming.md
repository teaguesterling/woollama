# Naming brainstorm — graduating "cosmic-fabric" to a general-purpose router

> **Superseded — historical.** This brainstorm is kept for the reasoning, but the
> decision was made *outside* its candidate list: the project shipped as
> **woollama** ("Web Over Ollama (and Llamas)" — she talks to llamas). The
> `bosun`/`conduit`/`loom` candidates below were not chosen. Also note
> `router-architecture.md` is now [`architecture.md`](architecture.md).

The architecture (`router-architecture.md`) has grown beyond cosmic-fabric's
original scope ("a COSMIC-native frontend for fabric"). It's now a model + tool
+ executor router that uses fabric as one possible pattern source. The COSMIC
+ fabric tie-in is too narrow.

## What the name needs to do

1. **Evoke routing / orchestration / coordination** — not inference, not chat,
   not UI
2. **Be one word, pronounceable, googleable** — for a public open-source
   project
3. **Stand alone without explanation** — works in `pip install <name>` and
   `<name> serve`
4. **Not collide with major existing projects** — checked against the leading
   AI-routing tools and the common dictionary projects
5. **Optionally fit the "Rigged Suite"** — the user's existing brand for the
   adjacent tool family (lackpy is "Part of the Rigged Suite"); if cos-fab
   joins this suite, a name from the rigging metaphor coheres

The function in one sentence: **it connects clients to AI inference and tool
sources, composing them into orchestrated calls**.

## Candidate clusters

### Cluster A — rigging / nautical (fits the Rigged Suite)

Metaphor: the user already has lackpy under "Rigged"; ship rigging is a
mature metaphor for "the assembled equipment that connects sails to crew to
direction." Words from rigging vocabulary that map to routing:

| candidate | what it means in rigging | why it fits |
|---|---|---|
| **boatswain** / **bosun** | the officer in charge of the deck, equipment, and crew | literally the coordinator/router on a ship; bosun is the short, pronounceable form (`bo'sn`) |
| **derrick** | a tower or crane for lifting/supporting | strong vertical structure; coordinates lift + load + position |
| **tackle** | a system of pulleys for mechanical advantage; also "to handle" (a problem) | dual meaning works; very rigging |
| **spar** | a pole supporting rigging | short, technical, single-syllable |
| **mast** | the central upright supporting all rigging | central, essential, evocative |
| **boom** | a horizontal spar that extends the sail | punchy; also means "expansive growth" |
| **halyard** | the line used to raise sails | dynamic ("raising things") |
| **windlass** | a winch for raising heavy loads | distinctive; uncommon enough to own |
| **capstan** | a vertical winch with bars for crew to push | distinctive; "many hands" implication |

### Cluster B — fabric / weaving (preserves the user's affection for "fabric")

Metaphor: cos-fab's existing fabric-themed name. Weaving is a real metaphor
for "composing many threads into a single coherent piece":

| candidate | meaning | why it fits |
|---|---|---|
| **loom** | the device that weaves threads into fabric | composes; strongest evocation; risk: name collisions (Loom Inc. video) |
| **weave** / **weft** / **warp** | the act and the threads | direct; weft and warp distinct |
| **shuttle** | the part of a loom that carries thread across the warp | "carries between sides"; routing |
| **skein** | a coil of yarn | smaller scope; less obvious as routing |
| **tapestry** | a complete woven piece | the result, not the doing — weaker |
| **lattice** | a crossing structure | adjacent; geometric |

### Cluster C — architectural / civic (gathering metaphors)

Metaphor: the router as a place where things converge.

| candidate | meaning | why it fits |
|---|---|---|
| **atrium** | central hall in a Roman house; all paths converge | beautiful; conveys gathering without grandeur |
| **forum** | Roman public square / marketplace | gathering; commerce; risk: too generic |
| **agora** | Greek equivalent of forum | distinctive sound; "agora-mcp" reads well |
| **portico** | colonnaded entrance | adjacency; less central |
| **rotunda** | circular hall | distinctive; "everything converges" |

### Cluster D — technical / network (the function directly)

Metaphor: pure routing.

| candidate | meaning | why it fits |
|---|---|---|
| **conduit** | a channel that carries something | classic router metaphor; clean |
| **plexus** | a network of intersecting things (nerves, blood vessels) | network/intersection; medical undertone |
| **manifold** | a pipe with multiple branches | distribution + multiple paths |
| **junction** | a place where things meet | generic but clear |
| **nexus** | a connection point | DEFINITELY taken (Sonatype Nexus); skip |
| **bridge** | connects two sides | too generic |

### Cluster E — music / orchestra (the orchestration metaphor)

Metaphor: composing many performers into one piece.

| candidate | meaning | why it fits |
|---|---|---|
| **baton** | the conductor's tool | small, sharp; conducting |
| **score** | the written composition | overloaded ("test scores") |
| **podium** | where the conductor stands | central, elevated |
| **chord** | many notes at once | composition; risk: "discord", electronic associations |
| **ensemble** | the group of performers | composes; long |

## My top picks

After several rounds of consideration, the names I'd actually want to own:

### 1. **bosun** ⭐ (Rigged Suite fit)
Pronounced "BOH-sn" (short for boatswain). The officer who coordinates the
deck. Three reasons it works:
- **Literal meaning matches the function**: the bosun is in charge of the
  ship's equipment, manages the crew, executes the captain's orders by
  routing them to the right hands. That's the router.
- **Fits the Rigged Suite naturally**: `lackpy` and `bosun` both sit
  comfortably in the nautical/rigging family without being heavy-handed
  about it.
- **Distinctive sound**: short, punchy, easy to type and pronounce; few
  collisions in the AI-tooling space.

Risk: people unfamiliar with the word may need a one-line explanation. The
word is well-known enough that it's not obscure — and "the bosun is the
coordinator" reads cleanly.

### 2. **conduit**
The cleanest pure-function name. A conduit carries something from one place
to another; that's literally the router. Crisp, clean, single syllable cluster.

Risk: there's a Conduit Server for Matrix (Rust) and a few minor uses; the
name has gravity but isn't owned in the AI-routing space.

### 3. **loom**
Strongest evocation among fabric-family options. A loom composes many threads
into one coherent thing — exactly what the router does with models + tools +
patterns. Preserves the user's stated affection for "cosmic-fabric" while
generalizing.

Risk: Loom Inc. (the video tool) is a well-known SaaS; this would compete
for search. Mitigation: niche audience for an AI router rarely searches for
"loom" generically.

### 4. **plexus**
Distinctive, technical, evokes network intersection. Less collision risk than
any of the above. Slight medical association (nerve plexus, brachial plexus)
is mostly positive — it reads as "intelligent network."

### 5. **atrium**
The "place where everything converges" framing. Beautiful name, hospitable
connotation. Less obviously a router than the others; might need explanation.

## Recommendation

**`bosun`** if you're committed to the Rigged Suite identity for this set of
tools. The naming coherence with lackpy alone is worth a lot for memorability
("the Rigged tools — lackpy and bosun"), and the literal meaning is on target.

**`conduit`** if you want each project's name to stand fully alone without
brand-family scaffolding. It's the most function-true name and the easiest
to explain in one sentence.

I'd not pick `loom` — the brand collision with Loom Inc. would make the next
five years of search/SEO an uphill fight, even in a niche.

## Other Rigged-suite additions to consider (for context)

If `bosun` lands, the suite could naturally grow:
- `lackpy` — the inference language (existing)
- `bosun` — the router
- `pinnace` (a small ship's boat) — would be a CLI client
- `helm` — would be a UI / control panel
- `ledger` — would be observability / audit
- `pilot` — a navigation / dispatch helper

(Names suggest themselves once you commit to the metaphor — that's an
argument for nautical naming, not against.)

## Outcome (done)

The name chosen was **woollama** (not on the shortlist above — it won on the
"Web Over Ollama (and Llamas)" backronym + the llama metaphor). This doc is kept
for the reasoning. The follow-through is complete: the repo is
`teaguesterling/woollama`, the architecture/design docs use the name throughout,
and cosmic-fabric consumes it as its router.
