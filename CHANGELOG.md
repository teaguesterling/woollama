# Changelog

## Unreleased

## v0.16.0 — 2026-08-18

**A request for a non-resident model queues behind the model swap instead of being refused.**
`woollama` 0.16.0 + `woollama-server` 0.15.0; `woollama-core` (0.9.0) and `woollama-engine`
(0.12.0) are unchanged.

### Features

- **A request for a non-resident model now queues behind the model swap instead of getting an
  immediate `503`.** On a capacity-bound device, serving B can mean evicting A — so when A was
  busy, nothing was evictable and B was refused outright, even though B was servable as soon as
  work already in flight drained. That pushed a cold-load-length retry decision onto every
  client, over a resource woollama was already sequencing. (#39)

  A busy model is still never evicted, and `503` + `Retry-After` is still the answer once the
  wait exceeds `queue_timeout` — the complaint was that it arrived *immediately*, not that it
  arrived. No new configuration: a swap is a cold load with an eviction in front of it, and
  `queue_timeout` already has to exceed a cold load.

  The non-obvious half is a **fairness hold**, and it is load-bearing rather than a refinement.
  `Gate::enter` enqueues before it loads, and eviction skips any model with a queued request, so
  under steady traffic for the resident model its queue never empties and a waiter would time
  out anyway — converting a fast failure into a slow one. While a swap is pending, arriving
  requests for the resident model are held *before* they enqueue, letting it drain. woollama
  never *chooses* to evict a model that is serving.

  **Verified on hardware**, same base and config, only this change differing:

  | | asker | holder |
  |---|---|---|
  | `main` | `503` after 0.0s | 4/4 ok |
  | this change | `200` after 32.6–32.7s | 5/5 ok |

  Reproduced 3/3 per arm on an NPU device with two 30B-class models that cannot coexist. The
  holder is not merely undamaged: the request that would have been refused is queued *through*
  the swap and the swap back, returning after 61.8s. In-flight work really is protected when
  woollama is the party doing the evicting.

  An earlier run appeared to show one holder request lost per swap. That run had no `pool_max`,
  where woollama never evicts and the device force-evicts instead — so none of these protections
  applied. It measured #47, not this. What bounds a swap is still not established: waits were
  observed to exceed `queue_timeout` and succeed.

- **An abandoned request no longer leaks its place in a model's queue.** A client disconnecting
  drops the handler future mid-await; the cleanup decremented the queued count on both `match`
  arms — every path the function can *return* by — but a dropped future takes none of them, so
  the count stayed raised forever and that model could never be evicted again. Replaced with an
  RAII ticket that releases on drop. Pre-existing, but the swap queueing above turns the symptom
  from "one model is never evicted" into "every later swap waits out `queue_timeout` and then
  `503`s" — which is indistinguishable from the bug that change exists to fix. Found by
  self-review, not by a failing test.

  Verified by mutation, not just by passing: removing the fairness hold fails the starvation test
  and nothing else; removing the post-hold `queue_max` check fails only its own test; removing
  the wait fails all four. The suite was run 40× to catch a race that
  a single green run had hidden.


## v0.15.0 — 2026-08-18

**CPython 3.14 is supported.** `woollama` 0.15.0 + `woollama-core` 0.9.0; `woollama-server`
(0.14.2) and `woollama-engine` (0.12.0) are unchanged.

### Features

- **CPython 3.14 is supported.** pyo3 0.23 refused to build against any interpreter newer than
  3.13, so 3.14 had no wheel *and* no working source build — `pip install woollama` failed
  outright, while `requires-python = ">=3.11"` advertised support. Upgraded pyo3,
  pyo3-async-runtimes and pythonize 0.23 → 0.29. The only source change needed was the GIL API
  rename (`Python::with_gil` → `Python::attach`, `py.allow_threads` → `py.detach`); the crate was
  already on the `Bound<'py, T>` API, which is the migration that would have hurt. (#43)

  Verified on 3.14.6: a cp314 wheel builds, the sdist builds from source, `woollama.core`
  imports, `InferenceError` still subclasses `Exception` (the reason abi3 remains unusable), and
  both suites pass — 42 conformance tests and 336 Python tests. `woollama`'s own dependencies
  (fastapi, httpx, mcp, uvicorn) all resolve on 3.14, so nothing else blocks the install.

  Note that prebuilt cp314 **wheels** additionally need the matrix change in the still-unapplied
  workflow patch. Until that lands, 3.14 installs by building from source, which needs a Rust
  toolchain and takes minutes — working, but not yet pleasant.

- No behavioural change on 3.11–3.13: clippy output is identical to the pre-upgrade baseline,
  and the Rust suites are unchanged (93 server, 16 engine).


## v0.14.4 — 2026-08-17

**Completes v0.14.3, which published only half of what it intended.** `woollama` 0.14.4 +
`woollama-core` 0.8.2. Use this rather than v0.14.3.

### Fixes

- **PyPI rejected the `woollama-core` 0.8.1 sdist, after the wheels had already uploaded.** The
  same maturin workspace-member hoisting bug behind #41, in a second place: PKG-INFO declared
  `License-File: LICENSE` — a path relative to the sdist root — while the file was packaged at
  `<sdist>/woollama-core/LICENSE`. PyPI validates that path and returned a 400. Because wheels
  upload before the sdist, 0.8.1 exists on PyPI as 21 wheels with **no sdist at all**, so the
  v0.14.3 fix never reached the people it was for: anyone without a prebuilt wheel. Declaring
  `license-files = ["LICENSE"]` explicitly (PEP 639) makes maturin place the file at the sdist
  root, so the metadata is true. `woollama` now floors on `woollama-core>=0.8.2` — 0.8.1 is
  unusable from source and 0.8.0 and earlier are unbuildable.
- `twine check` does **not** catch this; it passes the artifact PyPI rejects. The check has to
  confirm each declared `License-File` path exists, which is what the CI step below does.

### Note on v0.14.3

`woollama` 0.14.3 and the `woollama-core` 0.8.1 **wheels** are on PyPI and are fine on CPython
3.11–3.13. Only the sdist is missing, so source installs — the entire point of that release —
still fail there. v0.14.4 supersedes it.

## v0.14.3 — 2026-08-17

**Superseded by v0.14.4 — the sdist this release was about was rejected at upload.** `woollama`
0.14.3 + `woollama-core` 0.8.1 (wheels only); `woollama-server` stays at 0.14.2 and
`woollama-engine` at 0.12.0 (both unchanged).

### Fixes

- **The `woollama-core` sdist could not be built, in any release, on any Python.** maturin ships
  `.gitkeep` in wheels but filters it out of sdists, and `.gitkeep` was the only tracked file in
  `python/woollama/` — the empty PEP 420 namespace directory `python-source` points at. The
  directory came out empty, empty directories are not archived, and every source build died in
  metadata generation with `python-source is set to 'python' but the directory does not exist`.
  Prebuilt wheels covered every supported CPython, so nothing ever exercised the path. Python
  3.14 — for which pyo3 0.23 cannot build wheels — removed that cover and made `pip install
  woollama` fail outright. `woollama-core` 0.8.1 ships the file explicitly; `woollama` now
  requires `woollama-core>=0.8.1`, since the broken sdists remain on PyPI forever. (#41)
  Verified by installing the fixed sdist from source with `--no-binary :all:` and importing
  `woollama.core`, and by confirming the 0.8.0 sdist still fails that same check.

### Not yet gated

- The regression gate for the above — build the sdist, install it with `--no-binary :all:`,
  import `woollama.core`, and block the PyPI publish unless it passes — is written and verified
  but **not applied**: pushing `.github/workflows/` needs a token with `workflow` scope. Until
  it lands, "the sdist works" is a fact about this release, not a property CI enforces. That is
  precisely the gap that let this ship broken in every prior version.

### Known gaps

- **Python 3.14 is still unsupported, wheels or source.** An earlier draft of this entry claimed
  3.14 users could now install from source; that was wrong, and installing `woollama==0.14.4`
  under 3.14.6 disproves it. The failure has *moved* — metadata generation now succeeds and the
  build gets as far as compiling — but `pyo3-ffi` then refuses: `the configured Python interpreter
  version (3.14) is newer than PyO3's maximum supported version (3.13)`. A working sdist cannot
  help when the code in it will not compile. 3.14 needs the pyo3 0.23 → 0.29 upgrade in #43, and
  nothing short of it; abi3 can't sidestep this either, since `InferenceError` subclasses
  `PyException`.

  What the sdist fix does buy: every *other* environment we don't ship a wheel for — uncommon
  architectures, non-glibc/musl targets, distro packaging, and `--no-binary` install policies —
  can now build from source on 3.11–3.13, which was impossible in every previous release.

## v0.14.2 — 2026-08-17

**A probe no longer severs a running daemon's transports.** No functional change otherwise;
`woollama-engine` (0.12.0) and `woollama-core` (0.8.0) are unchanged.

### Fixes

- **`woollamad --help` no longer severs a running daemon's MCP transport.** `--version`, `-V`,
  `--help` and `-h` fell through to starting a full daemon, which overwrote the live daemon's
  discovery address file with its own ephemeral port and then, on exit, unlinked a socket it had
  explicitly **declined** to create. The healthy daemon was left bound to an orphaned inode: `ss`
  showed it listening, `ls` showed no file, path-based clients got `ECONNREFUSED`, and it reported
  healthy and logged nothing. Now they print and exit before anything reads config or touches the
  runtime directory.
- **A second daemon sharing a runtime directory refuses to start**, rather than degrading to
  TCP-only and clobbering the first's discovery address. Checked before binding anything, because
  the TCP bind is what writes that file. Sharing a runtime dir is a misconfiguration, and degrading
  there damages someone else rather than yourself.
- **A socket is only unlinked by the process that created it.** The guard existed at bind and not
  at cleanup, which is how a probe that correctly refused to steal the socket still deleted it.


## v0.14.1 — 2026-08-17

**Release fix — v0.14.0 reached PyPI but not crates.io.** `woollama-engine` gained public API in
this cycle (`missing_vars*`, `expand_env_with`) and was not bumped, so `cargo publish -p
woollama-server` resolved the *published* 0.11.0 and failed to compile. A path dependency hides
this locally: only publish resolves by version.

- `woollama-engine` → **0.12.0**, with its pins in `woollama-server` and `woollama-core` updated.
- A CI **publish dry-run** gate is written and verified but **not yet applied** — modifying
  `.github/workflows/` needs a token scope this session does not have. The patch is prepared: the
  server dry-run runs only when the pinned engine version is already on crates.io — precisely the
  forgot-to-bump case — so missing API becomes a compile error in CI rather than a failed release.
  Until it lands, this class of mistake is still only caught at publish time.

No functional change from v0.14.0; the crates.io line starts here.

## v0.14.0 — 2026-08-17

**Routing learns what a backend can do and what it is actually running, and the config
contract stops failing silently.** Every headline change here was found, reproduced, or
corrected against real hardware rather than a fixture. The `woollama-server` crate and the
`woollama` PyPI package move to 0.14.0; `woollama-engine` (0.11.0) and `woollama-core`
(0.8.0) are unchanged.

### Capability-aware routing (#20)

- **`<provider>/default` picks a model that can serve the endpoint.** `default` is never asked in
  the abstract — it is asked *at* an endpoint — so on a device holding an embedder, a reranker and
  a chat model, two of the three were never candidates. Capability comes from the backend where it
  publishes one (the `device` protocol carries it in the payload woollama already fetches, so
  discovery costs no extra call), and from `[inferencers.<name>.capabilities]` glob patterns where
  it does not.
- **Declare what a model *is*, not what may chat.** A positive allow-list requires naming every
  model that might ever legitimately serve the endpoint — on a shared device that means predicting
  what other consumers will load. Observed in production: a peer session loaded a chat model nobody
  had listed and every allow-list-shaped route failed. Exclusions survive models you did not
  foresee.
- **Pre-dispatch validation**: a request naming a model declared for another capability is refused
  with `400` naming what it *is*, rather than sent to a backend that may respond by taking the
  whole model service down. **`GET /v1/models`** annotates declared capability; absent never means
  "cannot".
- Precedence is **config → discovery → unknown**, and unknown is always eligible, so a backend
  that publishes nothing and has no declarations behaves exactly as before.

### `/v1/models` says which model will actually answer

- For a **management-capable** inferencer, each entry carries **`loaded`** — read through to the
  backend, sharing the coalescing window `<provider>/default` uses, so listing costs no extra round
  trip. Previously `/v1/models` was a catalogue with no readiness signal: it listed a model the
  backend was not running, and the only way to find out was a request that could `503` after a
  thirty-second load.
- A model that is **resident but undeclared** is listed too, flagged `undeclared`. It is routable
  and it is the one that will answer; omitting it hid the answer from a caller willing to use
  whatever is up.
- `loaded` is **omitted**, never `false`, for an inferencer with no pool or when the residency read
  fails. Not seeing is not the same as not loaded.
- **`loaded` is necessary, not sufficient.** It answers "is this in memory", not "will this call
  succeed" — a resident model may be unservable on the endpoint called, may crash on certain
  inputs, or may be evicted by another consumer between the check and the call. Documented on the
  field itself, so the next caller does not build a pre-flight check on it and get surprised the
  way a caller treating `/v1/models` as a readiness signal was.

### Config variables (#21)

- **`${VAR:-default}`** — POSIX `:-` semantics (unset *or* empty takes the fallback), in **both**
  implementations, because a config file is read by both and a Rust-only syntax would behave
  differently depending on which loaded it.
- **A bare `${VAR}` that is not set is now a hard load error.** woollamad refuses to start, naming
  every offending variable and pointing at `check-config`; the Python reference likewise refuses.
  Safe to do only because `:-` exists: without a way to say "absent is intended", refusing would
  break configurations that legitimately depend on an optional value — including the bundled
  default, which now declares its own fallback.
- A variable that is **set but empty** is not missing — `FOO=` is an explicit operator choice.
  Inside a server's `env` block it is still *warned* about, because there the consumer is a child
  process woollama cannot check for.
- The check runs on the **parsed** structure, so a `${VAR}` appearing only in documentation
  (`_`-prefixed keys in JSON and TOML, TOML comments) is prose and never fails a load. Applies to
  `mcp.json` and `inferencers.toml`.

### Fixes

- **The pool recovers when a model is unloaded underneath it (#38).** `ensure_loaded` short-circuits
  on its own belief before consulting the backend, so once that belief went stale — a crash that
  killed a model instance, an eviction by another consumer — *every* subsequent request for that
  model failed identically, forever. A 20-item batch lost six items to it, reporting nothing but
  per-item errors. Now an upstream `5xx` triggers a residency re-check, and a model the backend is
  no longer running stops being believed resident, so the next request reloads it. Deliberately
  **asks the backend rather than parsing its error**: whether a message means "unloaded" is vendor
  wording — but nor does it reconcile against the backend, because measurement showed the backend
  keeps reporting a crashed model as running for 2–5s while reaping catches up. A reconcile fired
  on the failure reads "all fine" and does nothing: the stale belief is the *backend's*, and reading
  through to the authority cannot help when the authority has not noticed. The failing model is
  marked for reload directly and the next request loads it unconditionally.

- **`default` no longer blames a config that is fine.** The fail-open warning fired whenever the
  candidate set came back empty, including when the residency read itself *failed* — telling an
  operator to fix a correct config while three models were resident and the real cause was a 401.


## v0.13.0 — 2026-08-17

**Downstream MCP servers reconnect on their own, per-server health is visible, and
pool state stops pretending woollama owns the device.** Two tracks landed together
plus two fixes that came out of running them against real hardware. The
`woollama-server` crate and the `woollama` PyPI package move to 0.13.0;
`woollama-engine` (0.11.0) and `woollama-core` (0.8.0) are unchanged.

### Downstream reconnect + introspection (Track 0, #23)

- **Reconnect.** A `url`- or `command`-form downstream that is unreachable at
  startup is retried on a per-server exponential backoff
  (`WOOLLAMA_MCP_RETRY_MAX_SECS`, default 60; `0` disables). A server whose
  *transport* cannot be built is reported `failed` and **not** retried — that is a
  config fault, and retrying produces noise until someone edits a file.
- **Per-server health.** `connected` / `retrying` (with attempts and last error) /
  `failed` (with reason), carried in the same snapshot as the tool set so a caller
  never sees health and tools from different instants. Distinct states because
  "absent" conflates a peer that will come back with one that never will.
- **`GET /v1/tools`** — each tool with its originating server, each server with its
  health. A downstream that is **down appears with its reason** rather than being
  omitted: absence and not-yet-connected are indistinguishable from outside, and a
  router showing neither looks healthy with its tools quietly gone. (Ports a route
  that existed only in the Python reference.)
- **Federation nesting cap** (`WOOLLAMA_MCP_MAX_NESTING`, default 2). Tool names
  gain a namespace level per federation hop; reconnect turns that from one level
  per *restart* into one per *refresh tick*, unbounded, in a mutual topology.
- Refresh is **background-only**, never request-triggered: `tools/list` serves the
  cached snapshot, so one router's request can never cause a fetch from the next.

### Pool state is a cache of the device, not a ledger (#26)

- **`<provider>/default` reads through to the backend** and lets it arbitrate.
  woollama is not the only consumer of a device's management API — the vendor's own
  UI loads models, and any caller needing endpoints woollama does not serve (images,
  embeddings, ASR) must drive it directly. Exclusivity is not achievable even in
  principle, so our view of residency is a cache. Concrete model ids and ordinary
  aliases need no device round trip.
- The freshness window (`WOOLLAMA_POOL_RESIDENCY_TTL_MS`, default 1000) is a
  **coalescing** window — a burst of `default` requests shares one query — not a
  staleness budget.
- **Inferencers sharing a `management_url` share one pool and one gate.** Separate
  gates enforced `parallel` once per route, **doubling** the concurrency the device
  actually saw; on hardware where `parallel = 1` exists to prevent a wedge, that is
  a safety property silently halved. Verified on real hardware: two routes,
  concurrent requests, serialized, no wedge.
- `parallel` documentation now claims the true thing — woollama sends at most N; the
  device may still receive more from other consumers.

### Fixes

- **The pool recovers when a model is unloaded underneath it (#38).** `ensure_loaded` short-circuits
  on its own belief before consulting the backend, so once that belief went stale — a crash that
  killed a model instance, an eviction by another consumer — *every* subsequent request for that
  model failed identically, forever. A 20-item batch lost six items to it, reporting nothing but
  per-item errors. Now an upstream `5xx` triggers a residency re-check, and a model the backend is
  no longer running stops being believed resident, so the next request reloads it. Deliberately
  **asks the backend rather than parsing its error**: whether a message means "unloaded" is vendor
  wording — but nor does it reconcile against the backend, because measurement showed the backend
  keeps reporting a crashed model as running for 2–5s while reaping catches up. A reconcile fired
  on the failure reads "all fine" and does nothing: the stale belief is the *backend's*, and reading
  through to the authority cannot help when the authority has not noticed. The failing model is
  marked for reload directly and the next request loads it unconditionally. found by running the above against real hardware

- **BUGFIX — `<provider>/default` was nondeterministic across restarts.**
  `reconcile` stamped every newly-discovered resident with the same `last_used`, so
  `snapshot`'s sort was a no-op and the winner came from `HashMap` iteration order —
  i.e. Rust's per-process hash seed. Measured **9 failures in 12** independent starts
  on a device holding a chat model plus an embedder and a reranker. Now: if an
  inferencer declares `models`, only those are candidates (residency is device-wide,
  not per-route), ordered deterministically with a resident `virtual.default` first.
  Fails open where no catalog is declared, and **warns once** when it does.
- **BUGFIX — the >64-char tool-name hash is now stable (#22).** It used std's
  `DefaultHasher`, whose algorithm is explicitly not guaranteed between Rust
  releases, so a toolchain upgrade could silently rename a tool — and a renamed tool
  stops resolving, on both the recipe allow-list and any client that cached it.
  Now FNV-1a with pinned test vectors. **This renames any tool currently on the
  hashed path**; federation is what pushes names there, so it only gets more
  expensive to change later.

### Also

- `mcp.json`'s `env` and the `check-config` subcommand shipped in v0.12.0; the
  review findings tracked in #8 are all resolved across both implementations.
- Known gap, recorded rather than overclaimed: a downstream that dies **after**
  connecting is not re-detected. Reconnect covers "down at startup, comes up later",
  not the reverse. Federation loop protection likewise remains deferred — mutual
  A→B→A cycles are reachable through ordinary restarts and unguarded.

## v0.12.0 — 2026-08-16

**`woollamad` can consume downstream MCP servers over Streamable HTTP.** An
`mcp.json` server is now either a `command` (stdio subprocess, as before) or a
`url`. Because woollama already *serves* `/mcp`, the `url` form is how one
woollamad consumes another — an inference-holding instance reaching a tools-only
instance's namespace, so the tools instance can hold zero provider API keys and a
compromise there exposes read-only corpora rather than the ability to spend money.
Rust-only: `woollamad` is the canonical router, and the Python reference (the
differential-test oracle) still requires `command`. The `woollama-server` crate
and the `woollama` PyPI package move to 0.12.0; `woollama-engine` (0.11.0) and
`woollama-core` (0.8.0) are unchanged.

> **Not yet validated against a real remote.** This ships verified between two
> loopback `woollamad` processes on one host, driven with a real Streamable-HTTP
> handshake. It has **not** been exercised across a LAN, through a reverse proxy,
> with TLS, or with a proxy-injected credential — a materially different failure
> surface. Acceptance against a real downstream is pending.

- **`url` + `headers` in `mcp.json`** (issue #19). `McpServerSpec` becomes an enum
  over transport; `connect_one` branches on the variant. Everything after
  `.serve(transport)` — peer handling, `list_all_tools`, the `wire_index` — is
  unchanged, so re-export, dispatch and the recipe allow-list work identically for
  an HTTP downstream.
- **Credentials use the existing `${VAR}` expansion**, not a second secrets path —
  the same shape as `api_key_env` for inferencers. Header values are **validated
  fail-closed at load**: `engine::expand_env` resolves an unset variable to the
  empty string, so `"Bearer ${TOKEN}"` with `TOKEN` unset would otherwise produce
  the literal header `Bearer ` — a well-formed request carrying *no credential*,
  which a permissive downstream accepts while everything reports healthy. An empty
  value or a bare `Authorization` scheme now invalidates that server. Sending
  `headers` to a non-loopback plain `http://` URL warns. (The general `expand_env`
  fail-open is filed as #21.)
- **Secrets stay out of `Debug`.** `HttpSpec` and `StdioSpec` implement `Debug` by
  hand, printing header/env *names* only and dropping the URL query string —
  `env` is a documented home for a provider key, `headers` for a bearer, and a URL
  can carry `?token=`. A derived `Debug` would put all three into any `{:?}`, log
  line, or panic message.
- **BUGFIX — `env` restored.** The Python reference parses `mcp.json`'s `env`
  (`config.py`) and forwards it to `StdioServerParameters.env` (`manager.py`); the
  Rust port dropped it silently, so a key documented in `docs/configuration.md`
  was ignored with no message. It is merged **over** the scrubbed base env —
  explicit entries win, but nothing arrives by inheritance, so the child-env scrub
  stays a floor. An operator can deliberately re-inject a provider key by naming
  it, which is explicit and greppable rather than silent.
- **BUGFIX — claude-code delegation forwards `env`.** It previously emitted only
  `{command, args}`, so after the fix above a stdio server would have worked in
  woollama's own loop and silently misbehaved under delegation, both paths
  reporting success.
- **A `url` server cannot be delegated to claude-code.** A recipe whose tools
  reference one is rejected with a 400 naming the server, rather than translated:
  the child `claude` process would connect to a network peer woollama never
  brokers, outside the allow-list boundary that makes delegation containable.
- **An invalid `mcp.json` entry is skipped, not fatal to its siblings.** One
  server's typo must not cost an operator the other eleven — `build_state`
  degrades a load error to an empty registry, so a whole-file abort would start
  the daemon "healthy" with zero MCP servers. Only unparseable JSON leaves woollama
  with no servers. A non-string `env` value is now an error rather than a silent
  drop, matching `headers`.
- **NEW — `woollamad check-config`.** Because a bad entry is skipped rather than
  fatal, its only runtime trace is a boot-log line. This subcommand validates
  `mcp.json`, `recipes.toml` and `inferencers.toml`, reports which servers are
  usable and which were skipped and why, and **exits non-zero** on any problem. It
  connects to nothing and binds nothing, so it is safe against a live deployment's
  config dir and suitable for gating a reload or a CI job.
- **Docs** — `docs/configuration.md` documents the `url` form and now splits a
  claim that was true only of the Python server: on a downstream that fails to
  start, `woollamad` logs and skips while the Python reference aborts startup.
- Also filed while building this: #21 (`expand_env` fails open), #22 (`wire_name`'s
  hash fallback uses `DefaultHasher`, whose algorithm is not stable across Rust
  releases — federation is what pushes tool names onto that path), #23 (`/v1/tools`
  exists only in the Python reference). `docs/roadmap.md` also records that
  federation **loop protection is deferred, not solved**: mutual A→B→A cycles are
  reachable through ordinary restarts and are unguarded.

## v0.11.0 — 2026-08-15

**Pluggable device-management protocols reach the Python reference server, and the
built-in preset is renamed `tiiny` → `device`.** The Python router (`python -m
woollama`) now matches the Rust `woollamad` on how it talks to a device: each
inferencer picks a `management_protocol`, protocols are defined in config, and the
built-in `device` preset is the back-compat default. The crates and the `woollama`
PyPI package move to 0.11.0; `woollama-core` is unchanged at 0.8.0 (only its
`woollama-engine` pin advances to 0.11.0 — the bridge is functionally the same).

- **`management_protocol` selector + `[management_protocols.<name>]` config** — a
  device backend is chosen per inferencer (default → built-in `device`). REST
  protocols declare `base`, default headers, and per-op `endpoints.<op>` tables
  (url / method / raw-string body / headers map + running path + id_field);
  `kind = "ollama"` selects the Ollama backend.
- **`DeviceBackend` seam** — `RestBackend` (config-parameterized, with the `device`
  preset) and `OllamaBackend` (`/api/ps` list, `/api/generate` + `keep_alive`
  load, `keep_alive: 0` unload) behind `DeviceModelManager`. `from_spec` does
  `{base}`/`{id}` templating, `${VAR}` expansion, and a case-insensitive header
  merge over the default `Authorization: Bearer`. An unknown protocol name skips
  just that inferencer's pool with a warning; the rest build normally.
- **BREAKING — `tiiny` → `device`.** The built-in preset name, the default
  `management_protocol` value, the reserved-names list, and the `RestBackend.tiiny`
  / `RestBackend::tiiny` method are renamed to `device` across both the Python
  server and the Rust crates, with **no back-compat alias**. Configs that named
  `management_protocol = "tiiny"` (or Rust callers of `RestBackend::tiiny`) must
  switch to `"device"`. The unset default is unchanged in behavior.
- **`_expand_env` parity fix** — matches the Rust `expand_env` exactly (unset
  `${VAR}` → empty, braceless `$VAR` left as-is), replacing `os.path.expandvars`.

## v0.10.0 — 2026-08-13

**Model pooling, request queuing & on-demand loading.** For an inferencer that
declares a `management_url`, woollama becomes *device-aware*: it loads models on
demand, serializes and queues requests around the backend's real concurrency
limit (`--parallel 1`), and resolves stable **virtual model names** — so a caller
never sees a bare not-loaded `503` or a hang. Fully additive: an inferencer with
no `management_url` behaves exactly as before. Wired into `/v1/chat/completions`;
the Rust `/v1/responses` core path is intentionally not pooled yet.

- **`resolver`** (pure) — `device/default` → the currently-loaded model, config
  `virtual` aliases, and real-id passthrough; plus queue-aware LRU eviction
  selection (never a busy model).
- **`pool.DeviceModelManager`** — async actor owning loaded-model state and device
  I/O (`/api/v1/models/{running,start,stop}`); de-dups concurrent same-id loads and
  evicts an idle LRU model to fit, with a re-check that never evicts a model
  mid-request.
- **`pool.Gate` / `Slot`** — per-model `asyncio.Semaphore` + FIFO queue. Backpressure
  returns `503` + `Retry-After` (never a wedge); device errors surface as `502`;
  streaming requests hold their slot for the stream's whole lifetime.
- New optional `[inferencers.<name>]` keys: `management_url`, `parallel`, `pool_max`,
  `queue_max`, `queue_timeout`, `virtual`. Existing configs are unaffected.
- Coverage: resolver 100%, pool 96%, router pooled paths 90%; 40+ new tests against
  a hermetic fake device (load-on-demand, concurrent-load de-dup, an eviction race,
  semaphore serialization, queue backpressure, streaming slot lifetime, lifespan).
- Build: declare `pydantic-settings`; cap `mcp<1.30` / `fastmcp<3.5` to the
  McpError-compatible line (fixes CI collection after `mcp 2.0.0` renamed
  `McpError` → `MCPError`).

## v0.9.0 — 2026-08-08

**Image + embedding pass-through endpoints.** woollama now proxies text-to-image
and text-embedding requests to a `<provider>/<model>` inferencer's own
OpenAI-compatible endpoints, mirroring the existing chat pass-through — so a single
OpenAI client pointed at woollama can do chat, images, and vectors.

- **`POST /v1/images/generations`** → forwards to the inferencer's
  `/v1/images/generations` (e.g. the device's `Z-Image-Turbo`), stripping the
  namespace prefix and adding auth. Always non-streaming, with a generous 300s read
  timeout for slow diffusion.
- **`POST /v1/embeddings`** → forwards to the inferencer's `/v1/embeddings` (e.g.
  `Qwen3-Embedding`), for local vectorization / RAG.
- `Inferencer.images_url()` / `embeddings_url()` alongside `chat_url()`; an unknown
  model namespace returns `400`, consistent with chat. Both verified end-to-end
  against the device; 4 new unit tests.

## v0.8.0 — 2026-07-19

**Surface authentication + fail-closed binding.** The HTTP surfaces (`/v1/*` and
the mounted `/mcp`) are now access-controlled: with no token configured, only
*local* peers (loopback TCP, the 0600 Unix socket) are served; a non-loopback
`WOOLLAMA_ADDRESS` refuses to start unless `WOOLLAMA_TOKEN` is set; with a token
set, every TCP request must send `Authorization: Bearer <token>` (the Unix
socket stays exempt — its file mode is the credential). The default loopback,
no-token workflow is unchanged.

- **Enforced by the shipping Rust daemon (`woollamad`), not just the Python
  oracle.** The surface auth + fail-closed bind land in `woollama-server`
  (`auth.rs`: bearer/loopback middleware applied to the TCP app; `check_bind_allowed`
  before the bind; the 0600 Unix-socket app carries no auth layer). The recipe
  allow-list was already enforced at dispatch in `woollama-engine`.
- **Unix socket no longer clobbered by a second daemon.** `binding.rs::bind_unix`
  now probes for a live peer before reclaiming the socket: a transient second
  `woollamad` serves TCP-only instead of stealing the primary's live socket (and
  its discovery files) — the discovery-clobber failure mode.
- **Recipe allow-list enforced at dispatch time (Python).** `Registry.dispatch`
  now takes the active allow-list and refuses a tool outside it; the recipe
  loop's `RegistryToolProvider` carries the recipe's `tools` list, so the
  boundary holds in Python independent of the core's offer-time filtering. The
  MCP aggregator surface (which re-exports every configured tool by design) is
  unchanged and gated by surface auth.
- **`mcp.json` `env` now reaches the spawned server** (via
  `StdioServerParameters.env`, merged over the SDK's safe default environment)
  instead of being parsed and dropped — no more `${VAR}`-into-argv workaround
  that leaked values into `ps`.
- **Downstream MCP tool calls are time-bounded** (`WOOLLAMA_TOOL_TIMEOUT`,
  default 180s): a hung server bounds the turn and no longer wedges the
  connection's worker for subsequent calls.
- **Managed-agents environments default to `limited` networking** (least
  privilege for the tool-less agent); `WOOLLAMA_AGENT_NETWORKING=unrestricted`
  restores the previous behavior.
- The durable conversation handle table (`conversations.json`) is written
  owner-only (0600).

## v0.7.0 — 2026-06-30

**Breaking:** `GET /w1/patterns` `variables` changes shape — bare name strings
become objects (`{name, default?, choices?, description?}`). Clients that read only
the variable `name` (e.g. cosmic-fabric's `WoollamaClient`) are unaffected; any
client that consumed `variables` as a `string[]` must update.

- **Vision (image input) for fabric patterns.** A `/w1/patterns/{name}/run` whose
  `input` carries an OpenAI `image_url` content part is dispatched to fabric's
  one-shot CLI (`fabric --attachment=…`, user text on stdin) — fabric's REST
  `/chat` has no attachment field. `http(s)://` image URLs pass through; `data:`
  URLs are decoded to a temp file (cleaned up after). Needs a vision-capable
  `model` (e.g. `ollama/llama3.2-vision`); one image per run (fabric `-a` is
  single-attachment); non-streaming (a `stream:true` request still gets the OpenAI
  SSE shape). As a byproduct, array-`content` messages no longer drop their text on
  the fabric REST path (previously `content` arrays were ignored entirely).
- **Native multimodal (`image_url`) confirmed.** A NATIVE recipe bound to (or
  `model`-overridden with) a vision model accepts `image_url` content with no
  special handling — the engine already forwards the messages array verbatim to
  ollama's OpenAI-compatible endpoint. Works on `/w1/…/run` and via
  `/v1/chat/completions` as `woollama/<recipe>`. Locked with a regression test; no
  engine change (it stays parity-locked).
- **Variable-metadata overlay for `/w1/` patterns.** Native recipes can annotate
  their `{{var}}` tokens with a `default`, `choices`, and `description` via an
  optional `[recipes.<name>.variables.<var>]` table in `recipes.toml`. `GET
  /w1/patterns` now surfaces `variables` as objects (`{name, default?, choices?,
  description?}`, absent fields omitted) instead of bare name strings; `default`s
  are applied wherever a recipe renders — `/w1/.../render`, `/w1/.../run`, and the
  MCP `prompts/get` surface — when the caller omits a variable (caller-supplied
  wins; `choices` is advisory, not enforced); `description` carries across to the
  MCP prompt argument. fabric-library patterns are unaffected (still `[]`).

## v0.6.0 — 2026-06-22

**Pattern templating (`/w1/`) + the fabric backend.** woollama can now own prompt
templating and front a full fabric deployment, behind a pluggable backend seam.

- **`/w1/` — woollama-native pattern templating.** A namespace parallel to `/v1/`:
  `GET /w1/patterns` (discovery), `POST /w1/patterns/{name}/render` (substitute
  `{{vars}}` without running), `POST /w1/patterns/{name}/run` (render then infer →
  an OpenAI completion/SSE). Patterns *are* recipes — a recipe whose `system`
  carries `{{var}}` tokens — plus an optional fabric-style `[patterns]` directory
  scan. Substitution is byte-compatible with fabric's (a dumb `{{k}}`→value
  replace); the engine never sees a `{{var}}`.
- **MCP prompts.** Recipes are exposed as MCP prompts on `/mcp`; their `{{var}}`
  tokens become prompt arguments and `prompts/get` renders them.
- **The fabric backend.** An optional `fabric` key in `mcp.json`: woollama spawns +
  supervises a `fabric --serve` (managed; reuse + graceful-kill) or routes to an
  external one (`url`). It **merges** fabric's ~250-pattern library into
  `/w1/patterns` (a `recipes.toml` recipe wins on a name collision) and
  **reverse-proxies fabric's REST verbatim at `/fabric/*`** (SSE, advanced
  `context`/`strategy`/`language`/`search`, and vision all pass through). On the
  `/w1` path, fabric's native SSE is translated to/from the OpenAI shape.
- **`PatternBackend` plugin seam.** Additional non-OpenAI prompt/inference systems
  plug in behind one trait + a single composition root; native recipes stay the
  built-in core, and the fabric backend is the reference impl. See
  `docs/extending.md`.
- **Self-healing fabric.** The pattern cache re-sources on a TTL
  (`WOOLLAMA_FABRIC_REFRESH_SECS`, default 60s — fabric hot-reloads its pattern
  dir) and after every respawn; a dead/hung **managed** fabric is respawned on the
  same address and the request retried once (single-flight, kill-before-rebind).
  `url` mode re-probes but never respawns a process it doesn't own.
- **Version family realigned to 0.6.0** across `woollama-engine`,
  `woollama-server`, `woollama-core`, and the `woollama` Python dist (the Rust
  crates had lagged at 0.5.0, the wheels at 0.5.3).

Docs: `docs/patterns.md` (the `/w1/` + `/fabric/` reference), `docs/extending.md`
(adding a backend), `docs/configuration.md` (the `fabric` key + resilience).

## v0.5.0 — 2026-06-14

**The Rust cutover.** woollama is now the Rust daemon **`woollamad`** (the
`woollama-server` crate), and it's **published**: `cargo install woollama-server`
installs the daemon, and `pip install woollama` pulls the pure-Python package plus
the native `woollama-core` engine wheel. The Python implementation in
`src/woollama/` is kept as the reference server and differential-test oracle, not
deleted. Authoritative live status is `docs/roadmap.md`.

*(Shipped across 0.5.0–0.5.3: 0.5.0 = crates.io publish of `woollama-engine` +
`woollama-server`; 0.5.1–0.5.3 = PyPI wheel publish of `woollama` +
`woollama-core` and fixes to the cross-platform wheel CI, notably the
manylinux-aarch64 build.)*

- **`woollamad`, the Rust router.** The full router surface ported to Rust on
  `woollama-engine` (pure engine) + `axum` + `rmcp`: OpenAI-compatible HTTP
  (`/v1/models`, `/v1/chat/completions` passthrough + recipe orchestration +
  streaming, `/v1/responses`, `/v1/conversations`), the MCP aggregator at `/mcp`
  and over stdio (`woollamad mcp`), the claude-code executor, stateful
  conversations (claude-resume / store-backed / managed-agents), and `/v1/models`
  discovery. Binds a unix socket (`$XDG_RUNTIME_DIR/woollama.sock`) + the loopback
  TCP port, same as the Python server.
- **Verified by a differential oracle.** The Python live integration suite runs
  against either implementation (`WOOLLAMA_TEST_CMD`), with `woollamad` the
  default target. Real behavioral divergences were caught and fixed (MCP
  capability advertisement, the `chat` tool's structured output, tool-level vs
  JSON-RPC error semantics).
- **Published.** crates.io: `woollama-engine` + `woollama-server` 0.5.0. PyPI:
  `woollama` + `woollama-core` 0.5.3, with cross-platform wheels (manylinux
  x86_64 + aarch64, musllinux, macOS x86_64 + arm64, Windows; cp311/312/313) +
  sdist, built by `.github/workflows/wheels.yml` (maturin-action).
- **TLS:** the engine's HTTP client moved from native-tls (OpenSSL) to **rustls**
  (system trust store via native-roots), so the native wheels build without a
  system OpenSSL and cross-compile to aarch64.

## v0.4.0 — 2026-06-10

Still the Python prototype (v1.0 is the Rust rewrite). The big shift: woollama is
now **embeddable as a library** — a server-free `woollama.core` other Python
projects import for model management — alongside an external conversation-store
family that makes non-claude models stateful without woollama ever owning bytes.
Authoritative live status is `docs/roadmap.md`.

- **Embeddable `woollama.core` library.** The model-management core — config +
  provider/model routing (`complete`/`complete_stream`, per-call `api_key`/
  `base_url`), `ModelRegistry`, recipes, and the recipe orchestration loop — is
  extracted into a server-free `woollama.core` subpackage so other projects embed
  it instead of running a sidecar. The FastAPI/MCP router now layers on top; the
  boundary is enforced by a test (importing `woollama.core` pulls in no
  FastAPI/uvicorn/MCP). The MCP↔OpenAI tool seam is explicit and lossless
  (`ToolProvider`/`ToolSpec`/`ToolResult` + a per-model renderer; carries MCP
  `isError`, fixing a silent tool-failure). Design: `docs/core-extraction.md`.
  **Note:** the old top-level module paths (`woollama.config`,
  `woollama.inferencers`, `woollama.recipes`, `woollama.ollama_native`) are gone —
  import from `woollama.core` (the server's public surface — the CLI + HTTP — is
  unchanged).
- **Pluggable conversation stores** (#2): non-claude models (ollama, cloud,
  recipes) become stateful through `/v1/responses` + `/v1/conversations` via an
  **external** store woollama is only a client to — it never owns transcript
  bytes (the conv-5 principle). Two reference providers prove the seam is
  transport-agnostic: an **MCP** store (`examples/mcp-convstore`) and a **REST**
  file store (`examples/rest-convstore`, persistent). Selected by the
  `conversationStore` key in `mcp.json` (a server name, or `{type:"mcp"|"http"}`);
  unset ⇒ stateless (no behavior change). A flaky store surfaces as a clean `502`.
- **Durable conversation handle table.** The `conversation_id → backend + native
  id` routing table is persisted (`$XDG_STATE_HOME/woollama/conversations.json`),
  so a client's conversation id keeps resolving across a woollama restart. Routing
  state only — never transcripts.
- **Attach by external key.** `POST /v1/conversations` and `/v1/responses` accept
  a caller-owned `key` (e.g. a session name): create-or-attach, idempotent — the
  caller drives turns by its own key and keeps no `key → id` map of its own.
- **Cloud models discoverable in `GET /v1/models`** (#3): each inferencer can opt
  in via `inferencers.toml` — a static `models = [...]` list and/or live
  `discover = true` that queries the provider's own `/v1/models`, filtered by
  `model_patterns` (fnmatch globs, e.g. `["claude-*", "gpt-4*"]`) so a huge
  catalog can be narrowed. Built-in cloud providers surface nothing until
  configured (no regression; ollama still auto-discovers its local catalog).
  Config now merges over built-ins field-by-field, so you can add `models` to
  `anthropic` without restating its `base_url`. Closes #3.

## v0.3.0 — 2026-06-07

Still the Python prototype (v1.0 is the Rust rewrite). Conversation-surface
release: a second state-owning backend (Managed Agents), the interactive
pause/answer path, streaming `/v1/responses`, ollama context-window control, and
the woollama side of a pluggable store backend. Authoritative live status is
`docs/roadmap.md`.

- **Streaming `/v1/responses`** (conv-1a streaming): a stateless `stream:true`
  turn now emits OpenAI **Responses SSE** (`response.created` →
  `output_text.delta`* → `response.completed`), sourcing deltas from a recipe
  (`orchestrate_events`, tool turns hidden) or a plain inferencer's chat SSE. The
  emitted frames validate against the `openai` SDK event models. Stateful
  streaming stays deferred (400). See `docs/conversations-api-design.md` §1.
- **Interactive `requires_action` path** (conv-8) via the managed-agents backend
  — without the tmux driver. The hosted agent carries an `ask_user` custom tool;
  when it's called the session pauses, woollama returns a Responses
  `status:"requires_action"` carrying the question, and continuing the
  conversation with the answer resumes it (`user.custom_tool_result`). The
  `requires_action` response is a documented superset of the OpenAI Responses
  shape. See `docs/conversations-api-design.md` §5.
- **Store-backed conversation backend** (#2, woollama-side mechanism): a
  store-only / BYO-inference backend that makes non-claude models (ollama, cloud,
  recipes) stateful by deferring the transcript to an external
  `ConversationStoreProvider` and doing assembly + stateless inference
  woollama-side — woollama never owns the bytes. Ships behind an **un-wired seam**
  (no provider by default, so those models stay stateless until one is
  registered); the provider contract is a *provisional proposal* to fabric. See
  `docs/conversations-api-design.md` §10. Stateful/store-backed ollama turns
  honor `num_ctx` too (request `options` thread through to `complete_stateless`,
  which routes ollama native — the #1↔#2 seam, closed and live-verified on the
  stateless `/v1/responses` path).
- **Ollama `num_ctx` honored** (#1): `ollama/<model>` passthrough requests that
  ask for a context size (`options.num_ctx`) now route to ollama's native
  `/api/chat` (which honors it) instead of the OpenAI-compat `/v1` endpoint
  (which silently ignores it), translating the request and the response (stream
  + non-stream) back to the OpenAI shape. Requests without `num_ctx`, and those
  with `tools`, stay on `/v1` unchanged. Live-verified: `/api/ps` reports the
  requested context.

## v0.2.0 — 2026-06-07

Still the Python prototype (v1.0 is the Rust rewrite — see
`docs/rust-transition.md`); these are committed slice-by-slice (see
`docs/build-log.md`). This resolves essentially every "queued for v0.2"
limitation listed under v0.1.0 below. Authoritative live status is
`docs/roadmap.md`.

### Surfaces

- **Streaming on both sides.** `stream:true` on `<provider>/<model>` relays the
  upstream SSE verbatim; on `woollama/<recipe>` it streams the answer as OpenAI
  SSE with the tool loop hidden (one async generator, `orchestrate_events`). The
  MCP `chat` tool emits a progress notification per tool call/result.
- **Stateful surface** (`docs/conversations-api-design.md`): `/v1/responses`
  (stateless subset + stateful) and `/v1/conversations` (create/list/get/delete
  + transcript `items`), in the OpenAI Responses/Conversations shape. woollama
  routes conversation *handles*; backends own the state.
- **MCP over Streamable HTTP** at `/mcp`, mounted on the same port as `/v1/*`,
  plus the stdio `woollama mcp` server. **Aggregator**: every downstream tool is
  re-exported namespaced, now carrying its `output_schema`; recipes become MCP
  prompts; a `chat` verb runs orchestration.
- **Unix socket** at `$XDG_RUNTIME_DIR/woollama.sock` (mode 0600) served
  alongside the loopback TCP port — the default for local MCP clients.
- `/v1/tools` introspection endpoint.

### Backends & routing

- **Multi-backend inferencer seam**: anthropic, openai, groq, together,
  openrouter built in, plus any OpenAI-compatible endpoint via `inferencers.toml`
  (e.g. self-hosted vLLM).
- **Claude Code** as a keyless inference backend (subscription auth), tool-less
  AND as an **executor** (tool delegation): a `claude-code` recipe with tools
  lets Claude own the agentic loop and call the recipe's allow-listed MCP tools
  itself, contained by a per-recipe `--mcp-config` + `--allowedTools`.
- **Conversation backends** (woollama routes handles; backends own state):
  - `claude-resume` (`claude --resume`, for `claude-code/<model>`) — the native
    Claude session owns the bytes; keyless/subscription.
  - `managed-agents` (Anthropic Managed Agents, for `claude-agent/<model>`) —
    Anthropic hosts the session + container; `ANTHROPIC_API_KEY` (paid). The
    first backend to implement transcript retrieval, so
    `/v1/conversations/{id}/items` serves it. (In the `agents` optional extra.)
  - Models with no state-owning backend are stateless (`store:false`). (A duckdb
    `stored` backend was briefly added and reverted — woollama does not store
    conversations in its own system; it routes handles to backends that own the
    state.)

### Platform

- **Multi-MCP-server discovery + unified tool registry** with long-lived
  connections (replaces per-request subprocess spawning).
- **File-driven config**: `mcp.json`, `recipes.toml`, `inferencers.toml`
  (`${VAR}` expansion; inferencers merge over built-ins).
- **Recipe allow-list** enforced as a security boundary (in the orchestration
  loop AND in delegation).
- **CI**: GitHub Actions runs `ruff check` + the hermetic suite on Python
  3.11/3.12; opt-in pre-commit hook mirrors the lint gate.
- **Documentation site**: MkDocs (Material) over the existing Markdown docs,
  published on ReadTheDocs at <https://woollama.readthedocs.io/>.

## v0.1.0 — 2026-05-31

First public version. **Working router; Python prototype, not production.**
Architecture validated end-to-end; v0.2 will harden, configure, and expand
the prototype. **v1.0 is a Rust rewrite** once the architecture stabilizes —
see `docs/rust-transition.md` for the explicit criteria.

### What v0.1 does

- **OpenAI-compatible HTTP surface** at `/v1/models` and `/v1/chat/completions`.
- **Model namespace routing**:
  - `ollama/<name>` — pure pass-through to local Ollama at `localhost:11434`
  - `woollama/<recipe>` — orchestrated chat-loop using the named recipe
- **One bundled example recipe** (`woollama/streamer`) demonstrating
  pattern + tools + inferencer composition.
- **MCP tool dispatch** via per-request stdio connection to the bundled
  hello server (`examples/mcp-hello/server.py`).
- **Ephemeral local-only binding** — random free loopback port at startup,
  persisted to `$XDG_RUNTIME_DIR/woollama.addr` for client discovery. Never
  binds to `0.0.0.0` without explicit `WOOLLAMA_ADDRESS` override.
- **Smoke tests** that don't require Ollama or network.

### Design ideas validated

- MCP + OpenAI compose as complementary standards without extension
- The model namespace (`<provider>/<name>`) is a sufficient addressing
  scheme for raw / pattern / variant / recipe model kinds
- Recipe orchestration is invisible to OpenAI clients — they get one final
  answer; the chat-loop happens inside the router
- Ephemeral local binding works for the OpenAI SDK out of the box (clients
  read the addr-file)

### Known limitations / queued for v0.2

- **No streaming** on either side (non-streaming round-trips only)
- **One hardcoded recipe** — real `~/.config/woollama/recipes.toml` to follow
- **One MCP server** — multi-server discovery + unified tool registry to follow
- **Ollama only** — Anthropic / OpenAI / vLLM via OpenAI-compat to follow
- **No Unix socket** transport — HTTP loopback only
- **woollama as MCP server** to its own clients is not yet implemented
- **No CI** — manual smoke tests; pytest config added but no GitHub Actions yet
- **Per-request MCP subprocess** is correct but slow; connection pooling to follow

### Origin

woollama is the rewrite of an architecture co-designed in [cosmic-fabric](
https://github.com/teaguesterling/cosmic-fabric), which remains as a frontend
client. The full design context lives in `docs/architecture.md` and
`docs/naming.md`.
