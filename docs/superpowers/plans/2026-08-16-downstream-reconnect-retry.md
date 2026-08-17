# Downstream Reconnect / Retry (Track 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A `url`-form downstream that is down at startup, or that goes away later, reconnects on its own — without the router ever presenting a reconnecting server as simply absent.

**Architecture:** `McpRegistry` gains interior mutability (one `RwLock` over a single immutable snapshot, so `servers` and `wire_index` can never be observed torn). A background task per configured server owns reconnect with capped exponential backoff and publishes a new snapshot on success. Refresh is **background-only, never request-triggered** — that is what preserves the property making federation safe today. A nesting cap bounds tool-name growth, which refresh would otherwise make unbounded. Per-server status becomes observable through a new `GET /v1/tools`, which also closes #23.

**Tech Stack:** Rust, `tokio` (RwLock + spawned tasks), `rmcp` 1.8, `axum` 0.8.

**Spec:** `docs/roadmap.md`, "Open tracks" item 0.

## Global Constraints

- **Rust-only.** Do not modify `src/woollama/` — `woollamad` is canonical, the Python tree is the differential-test oracle.
- **Do not modify `woollama-engine/`** (parity-locked) or `woollama-server/defaults/*` (`tests/defaults_sync.rs` pins them byte-identical to the Python package).
- **Never interpolate a header value or `env` value into a log line or error.** Both are documented homes for secrets.
- Lint gate is `cargo clippy --all-targets -- -D warnings`; suite is `cargo test -p woollama-server --features test-fixtures`.
- **Never assert on the absence of an error log as a proxy for success.** Assert on the named tool or the named state. A mechanism that silently produces nothing passes a log-absence check — this repo has now shipped that twice.

## Design decisions, with the reasoning that forced them

1. **Refresh is background-only. `list_tools` always serves the current snapshot.**
   The reason federation is safe today is that rosters are cached at connect and served from cache, so nothing recurses at request time (`mcp_surface.rs` `list_tools` → `reexport_tools` → cached `ServerConn.tools`). If a *request* could trigger a refresh, then A's `tools/list` → refresh → B's `tools/list` → refresh → A's… is live recursion across routers. Keeping refresh on a timer preserves the existing property for free and removes the need for any request-path hop counting.

2. **A nesting cap on re-export is MANDATORY, not a nicety.**
   Tool names nest one level per federation hop. Today a mutual A↔B topology gains a level only per *restart*. With a refresh timer it gains one **per refresh, without bound**:

   ```
   t1  A pulls B  →  A: [chat, mcp__B__chat]
   t2  B pulls A  →  B: [chat, mcp__A__chat, mcp__A__mcp__B__chat]
   t3  A pulls B  →  A: [..., mcp__B__mcp__A__mcp__B__chat]
   ```

   `wire_name` hashes past 64 chars, so this stays *correct* while becoming unreadable, and every hashed name is exposed to #22's unstable-hash problem. The cap is what stops a timer from producing an ever-growing roster. It is local, needs no protocol change, and is directly testable — unlike a `Via` chain, which cannot accumulate across startup-time connects anyway.

3. **Status is a first-class value, not a log line.**
   The v0.12.0 review established that a skipped server is announced only in the boot log, and `check-config` was the answer for *config* errors. A *connection* error is the retryable half and needs the runtime answer: distinct `Connected` / `Retrying` / `NeverConnected` states with last error and attempt count, surfaced on an endpoint. This is also the natural home for #23's `/v1/tools`.

4. **A server that fails to build its transport is not retried.**
   An unusable header or an unbuildable HTTP client is a config fault, not a world fault — retrying it forever would log noise until someone edits a file. Retry covers connect/handshake failures only.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `woollama-server/src/mcp_registry.rs` | Snapshot + `RwLock`, per-server status, reconnect task, nesting cap | Substantially modified |
| `woollama-server/src/lib.rs` | Spawn reconnect tasks after `build_state`; `GET /v1/tools` route | Modified |
| `woollama-server/tests/reconnect.rs` | **Create.** A downstream that appears late; degraded state visible while down | New test binary |
| `docs/configuration.md` | Retry knobs + `/v1/tools` | Modified |
| `docs/roadmap.md` | Move Track 0 to Shipped; record what the cap does and does not solve | Modified |

---

### Task 1: A swappable registry snapshot

Pure refactor of internal state — no retry yet, no behaviour change. Isolating it means the reconnect diff shows only reconnect.

**Files:** Modify `woollama-server/src/mcp_registry.rs`.

**Interfaces:**
- Produces:
  ```rust
  struct Snapshot { servers: HashMap<String, ServerConn>, wire_index: HashMap<String, (String, String)> }
  pub struct McpRegistry { inner: tokio::sync::RwLock<Arc<Snapshot>> }
  ```
  One lock over one immutable `Arc<Snapshot>`: readers clone the `Arc` and release the lock immediately, so no reader holds a lock across an `await`, and `servers`/`wire_index` can never be observed inconsistent with each other.

- [ ] **Step 1: Introduce the snapshot type and swap the accessors**

`resolve`, `tool`, `reexport_tools`, `call_server`, `call_raw` each take a read guard, clone what they need, and drop the guard before any `.await`. `connect` builds the first snapshot exactly as today.

- [ ] **Step 2: Run the suite — nothing may change**

Run: `cargo test -p woollama-server --features test-fixtures`
Expected: identical results to before. A refactor that flips a test is not a refactor.

- [ ] **Step 3: Lint + commit**

Run: `cargo clippy -p woollama-server --all-targets --features test-fixtures -- -D warnings`

---

### Task 2: Per-server status

**Files:** Modify `mcp_registry.rs`.

**Interfaces:**
- Produces:
  ```rust
  pub enum ServerHealth { Connected, Retrying { attempts: u32, last_error: String }, Failed { reason: String } }
  pub struct ServerStatus { pub name: String, pub transport: &'static str, pub health: ServerHealth, pub tools: usize }
  impl McpRegistry { pub async fn status(&self) -> Vec<ServerStatus> }
  ```
  `Failed` is terminal (transport could not be built — a config fault, see decision 4); `Retrying` is the retryable half.

- [ ] **Step 1: Write the failing test** — a dead `url` server reports `Retrying` with its error, not absence.
- [ ] **Step 2..4:** implement, verify, commit.

---

### Task 3: The reconnect loop

**Files:** Modify `mcp_registry.rs`, `lib.rs`. Create `tests/reconnect.rs`.

**Interfaces:**
- Consumes Tasks 1–2.
- Produces: `pub fn spawn_reconnect(reg: Arc<McpRegistry>, specs: HashMap<String, McpServerSpec>)`, called from `build_state`. Backoff: 1s doubling to a 60s cap, `WOOLLAMA_MCP_RETRY_MAX_SECS` to override, `0` disables retry entirely.

- [ ] **Step 1: Write the failing test.** Start a consumer whose `url` downstream is NOT yet listening; assert `status()` shows `Retrying`; then start the downstream; assert that within the backoff window the tool appears **by name** (`mcp__late__chat`), and status flips to `Connected`.
- [ ] **Step 2: Write the second failing test.** While down, the degraded state is *visible* — `status()` names the server with its last error. This is the test that matters: one asserting only eventual success also passes on a router permanently hiding a dead downstream.
- [ ] **Step 3..5:** implement, verify, commit.

---

### Task 4: The nesting cap

**Files:** Modify `mcp_registry.rs`.

- [ ] **Step 1: Write the failing test.** A downstream advertising an already-federated name (`mcp__x__mcp__y__z`) is not re-exported at a further level; a normal name is. Assert the cap by name, and assert the roster does not grow when the same over-nested roster is ingested twice.
- [ ] **Step 2..4:** implement (`WOOLLAMA_MCP_MAX_NESTING`, default 2), verify, commit.

---

### Task 5: `GET /v1/tools` (closes #23)

**Files:** Modify `lib.rs`, `docs/configuration.md`, `docs/roadmap.md`.

- [ ] **Step 1: Write the failing test.** The endpoint lists each tool with its originating server, and each server with its health — including a server that is `Retrying`, which must appear rather than be omitted.
- [ ] **Step 2..5:** implement, document, verify, commit.

## Acceptance

Retry is observable or it is not done. The check is not "reconnect eventually succeeds" but **"while the downstream is down, the degraded state is visible and names the server"** — the first passes on a router that permanently hides a dead downstream.
