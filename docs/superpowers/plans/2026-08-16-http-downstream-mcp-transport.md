# HTTP Downstream MCP Transport (issue #19) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `woollamad` consume downstream MCP servers over Streamable HTTP (`"url"` in `mcp.json`), not just stdio subprocesses — and restore the `env` key the Rust port silently dropped.

**Architecture:** `McpServerSpec` becomes an enum (`Stdio` | `Http`). `load_mcp_servers` parses `url` + `headers` alongside `command`/`args`/`env`. `connect_one` branches on the variant to build either a `TokioChildProcess` or a `StreamableHttpClientTransport`; everything downstream of `.serve(transport)` — peer handling, `list_all_tools`, the `wire_index` — is already transport-agnostic and does not change. Credentials ride the existing `${VAR}` expansion with a new fail-closed validation step.

**Tech Stack:** Rust, `rmcp` 1.8 (resolved from the `"1.7"` caret req; `transport-streamable-http-client` + `transport-streamable-http-client-reqwest` are already enabled and unused), `reqwest` 0.13 (unifies with rmcp's own reqwest 0.13.4), `axum` 0.8 (re-exports the `http` 1.5.0 header types — no new dependency).

**Spec:** `/home/teague/Projects/woollama/ISSUE-19-BRIEF.md`, as amended by "Spec amendments" below.

## Global Constraints

- **Rust-only.** Do not modify anything under `src/woollama/`. `woollamad` is canonical; the Python tree is the differential-test oracle (`docs/rust-transition.md`).
- **Do not add any test to `tests/*.py`.** The live integration suite runs against *either* implementation (`WOOLLAMA_TEST_CMD`); a pytest test for the `url` form would fail against a Python oracle that will never implement it. Every new test here is a Rust test.
- **Do not modify `woollama-server/defaults/mcp.json` or `defaults/recipes.toml`.** `tests/defaults_sync.rs` asserts they stay byte-identical to `src/woollama/defaults/`; changing them would require a Python-tree edit the first constraint forbids.
- **Do not modify `woollama-engine/`.** `config.rs:371` documents the engine as parity-locked, and `engine::expand_env` also feeds `inferencers.toml` — a change there would silently alter a second config surface.
- **Do not widen module visibility.** `config` and `mcp_registry` are private modules (`lib.rs:39,43`). Every test in this plan drives the public surface (`build_state`, `router`) instead. If you find yourself wanting `pub mod`, you are testing at the wrong level.
- **Never interpolate a header value into an error message or log line.** Header values carry bearer tokens.
- Lint gate is `cargo clippy -D warnings`; the suite is `cargo test --tests --features test-fixtures`.

## Spec amendments

Binding where they conflict with the brief:

1. **`env` is a parity regression, not an unimplemented key.** The brief says `env` is "never parsed" and the docs are wrong. In fact `src/woollama/config.py:136` parses it and `src/woollama/manager.py:89` forwards it to `StdioServerParameters.env`. `docs/configuration.md:38` correctly documents the oracle; `woollamad` dropped a working feature during the port. Restore Python's semantics; do not design a new key.
2. **"A failed server aborts startup" is divergent, not stale.** Python's `Registry.start_all` (`manager.py:149-151`) awaits `start()`, which re-raises at `:74`. Python genuinely aborts; Rust skips with a warning (`mcp_registry.rs:111`). The doc is true of the oracle. Split it per implementation rather than deleting the claim.
3. **The startup tool-list recursion does not exist.** The brief's decision 2 assumes `list_all_tools` can recurse across routers. It cannot: `mcp_surface.rs:118-126` serves `list_tools` from `registry.reexport_tools()` (`mcp_registry.rs:160-176`), which reads the `ServerConn.tools` vec cached once at connect (`:136`). No downstream call happens at request time. A mutual A↔B config is a startup ordering race over stale-or-empty rosters, bounded by `connect_timeout` and the concurrent skip-on-failure at `:97-117`.
4. **A self-loop guard would be unreachable code today, so this plan does not ship one.** `build_state()` — which connects every downstream — runs at `main.rs:13`; `axum::serve` starts at `:62`. The process is **not listening** while it connects. An instance whose `mcp.json` names its own address therefore fails with connection-refused and never reaches any inbound guard. A guard here would look responsible and could never fire. It becomes reachable exactly when reconnect-after-startup lands — see amendment 5.
5. **The hop cap AND the self-loop guard both belong to the future retry slice, for the same reason.** Both concern behavior *after* startup, and today there is no after-startup: `McpRegistry::connect` is called once (`lib.rs:111`) before the listener opens. Caching is what makes federation safe now, and startup-only connection is what makes self-loops unreachable now; a retry loop removes both properties at once. Task 4 records this so it is not orphaned.
6. **The recommended credential mechanism fails open.** `woollama-engine/src/lib.rs:417` resolves `${VAR}` with `unwrap_or_default()`, so an unset `${TOKEN}` expands to the empty string and `"Bearer ${TOKEN}"` becomes the literal header `Bearer ` — a well-formed request carrying no credential. Task 3 adds a fail-closed validation step in `load_mcp_servers` (never in the engine).
7. **The enum refactor touches claude-code delegation.** `lib.rs:536` reads `spec.command` / `spec.args` directly to build the `--mcp-config` handed to the `claude` CLI. That is the delegation containment boundary, not incidental code. Task 2 handles it.

## Out of scope

- **Retry / reconnect.** `McpRegistry` holds plain `HashMap`s behind an `Arc` with no interior mutability (`mcp_registry.rs:28-34`, `:206`); reconnect needs an `RwLock` through `resolve`, `tool`, `reexport_tools`, `call_server`, `call_raw`. Hot-path structural change. Scope note in Task 4.
- **A `Via` header, hop cap, instance identity, or self-loop guard.** See amendments 3–5.
- **A tool-name flattening rule.** `wire_name` hashes above 64 chars and `wire_index` is built from `wire_name()` output, so a federated name that overflows still resolves on dispatch. Three levels degrades to unreadable-but-correct; there is no correctness cliff to design against.
- **`expand_env`'s general fail-open** (engine, parity-locked) and **`wire_name`'s `DefaultHasher`** (algorithm not guaranteed stable across Rust releases). Filed as separate issues in Task 4.
- **Delegating a `url`-form server to the `claude` CLI.** Task 2 makes this an explicit error rather than writing a bearer token into a temp config file for a child process.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `woollama-server/src/config.rs` | Parse `mcp.json` into `McpServerSpec`; fail-closed header validation | Modify `:163-168` (struct → enum), `:404-424` (`load_mcp_servers`); add `validate_header_value` + unit tests |
| `woollama-server/src/mcp_registry.rs` | Build a transport per variant; everything after `.serve()` unchanged | Modify `connect_one` `:127-143`; add `merged_env`, `http_transport`; update `#[cfg(test)]` specs |
| `woollama-server/src/lib.rs` | claude-code delegation's `--mcp-config` subset | Modify `referenced_mcp_servers` `:520-539` |
| `woollama-server/tests/http_downstream.rs` | **Create.** One woollamad consuming another over the `url` form; header delivery | New test binary (separate file so global `WOOLLAMA_*` env can't race other files — the convention at `tests/mcp_surface.rs:7`) |
| `docs/configuration.md` | Document the `url` form; split the Rust/Python divergence | Modify the `mcpServers` section |
| `docs/roadmap.md` | Shipped row; the retry-slice scope note | Modify |

---

### Task 1: Restore the `env` key (stdio parity regression)

Independent of the transport work and shippable alone. `McpServerSpec` stays a struct here — Task 2 does the enum refactor, so a reviewer can judge the semantics fix separately from the shape change.

**Files:**
- Modify: `woollama-server/src/config.rs:163-168`, `:404-424`
- Modify: `woollama-server/src/mcp_registry.rs:127-143`
- Test: `#[cfg(test)]` modules in both files

**Interfaces:**
- Consumes: `scrubbed_env() -> HashMap<String, String>` (`mcp_registry.rs:39`); `engine::expand_env`, already applied to the `mcp.json` text at `config.rs:405`.
- Produces: `McpServerSpec { command: String, args: Vec<String>, env: HashMap<String, String> }`; `fn merged_env(&HashMap<String, String>) -> HashMap<String, String>` (private, `mcp_registry.rs`). Task 2 moves the three fields into `StdioSpec`.

- [ ] **Step 1: Write the failing config tests**

Add to the `#[cfg(test)] mod tests` block at the end of `woollama-server/src/config.rs`:

```rust
fn servers_from(json: &str) -> Result<HashMap<String, McpServerSpec>, String> {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("mcp.json"), json).unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", dir.path());
    let r = load_mcp_servers();
    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
    r
}

#[test]
fn mcp_server_env_block_is_parsed() {
    // The Python reference parses this key (config.py:136) and forwards it to the spawned
    // server (manager.py:89). woollamad dropped it in the port; this pins it restored.
    let servers =
        servers_from(r#"{"mcpServers": {"git": {"command": "git-mcp", "env": {"GIT_AUTHOR_NAME": "woollama"}}}}"#)
            .unwrap();
    assert_eq!(servers["git"].env.get("GIT_AUTHOR_NAME").map(String::as_str), Some("woollama"));
}

#[test]
fn mcp_server_without_env_block_gets_an_empty_map() {
    let servers = servers_from(r#"{"mcpServers": {"hello": {"command": "hi"}}}"#).unwrap();
    assert!(servers["hello"].env.is_empty(), "absent 'env' is an empty map, not an error");
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p woollama-server --features test-fixtures --lib mcp_server_env -- --test-threads=1`
Expected: FAIL to compile — `no field 'env' on type 'McpServerSpec'`.

(`--test-threads=1` throughout: these tests mutate process-global env.)

- [ ] **Step 3: Add the field and parse it**

In `woollama-server/src/config.rs`, replace the struct at `:163-168`:

```rust
/// A downstream MCP server to spawn (stdio). Matches Claude Code's mcp.json shape.
#[derive(Clone)]
pub struct McpServerSpec {
    pub command: String,
    pub args: Vec<String>,
    /// Extra environment for the spawned server, merged OVER the scrubbed base env
    /// (`mcp_registry::merged_env`). Restores parity with the Python reference, which parses
    /// this key (`config.py:136`) and hands it to `StdioServerParameters.env` (`manager.py:89`)
    /// rather than argv, where a secret would show up in `ps`.
    pub env: HashMap<String, String>,
}
```

In `load_mcp_servers`, replace the `out.insert(...)` at `:420` with:

```rust
            let env = s
                .get("env")
                .and_then(Value::as_object)
                .map(|o| {
                    o.iter()
                        .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string())))
                        .collect()
                })
                .unwrap_or_default();
            out.insert(name.clone(), McpServerSpec { command, args, env });
```

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test -p woollama-server --features test-fixtures --lib mcp_server_env -- --test-threads=1`
Expected: PASS (2 tests). The existing specs at `mcp_registry.rs:297-299` will now fail to compile — Step 5 fixes them.

- [ ] **Step 5: Write the failing merge tests**

In `woollama-server/src/mcp_registry.rs`, add `env: HashMap::new()` to the two existing spec literals at `:297` and `:299`, then add:

```rust
#[test]
#[cfg(unix)]
fn spec_env_merges_over_the_scrub_without_reopening_it() {
    std::env::set_var("ANTHROPIC_API_KEY", "leak-me");
    let mut spec_env = HashMap::new();
    spec_env.insert("GIT_AUTHOR_NAME".to_string(), "woollama".to_string());
    let merged = merged_env(&spec_env);
    // The half that makes the feature work.
    assert_eq!(merged.get("GIT_AUTHOR_NAME").map(String::as_str), Some("woollama"));
    // The half that rots if nobody pins it: the scrub is still a floor. Nothing arrives by
    // inheritance just because `env` exists.
    assert!(
        !merged.contains_key("ANTHROPIC_API_KEY"),
        "a non-allow-listed var must not reach the server just because `env` exists"
    );
}

#[test]
#[cfg(unix)]
fn spec_env_can_deliberately_reinject_a_provider_key() {
    let mut spec_env = HashMap::new();
    spec_env.insert("ANTHROPIC_API_KEY".to_string(), "on-purpose".to_string());
    // Explicit, in the operator's config, greppable — categorically different from inheriting
    // one silently. Pinned so nobody "hardens" it into a surprise.
    assert_eq!(merged_env(&spec_env).get("ANTHROPIC_API_KEY").map(String::as_str), Some("on-purpose"));
}
```

- [ ] **Step 6: Run to verify they fail**

Run: `cargo test -p woollama-server --features test-fixtures --lib spec_env -- --test-threads=1`
Expected: FAIL to compile — `cannot find function 'merged_env'`.

- [ ] **Step 7: Implement the merge**

In `woollama-server/src/mcp_registry.rs`, add after `scrubbed_env` (`:43`):

```rust
/// The scrubbed base env with the spec's `env` block merged OVER it — explicit entries win.
/// The scrub stays a floor: nothing reaches a tool server by inheritance, but an operator can
/// deliberately name a var (including a provider key) in `mcp.json`. Mirrors the Python
/// reference, where `StdioServerParameters.env` merges over the SDK's safe default env.
fn merged_env(spec_env: &HashMap<String, String>) -> HashMap<String, String> {
    let mut env = scrubbed_env();
    env.extend(spec_env.iter().map(|(k, v)| (k.clone(), v.clone())));
    env
}
```

And in `connect_one`, replace the `cmd.args(...)` line at `:132`:

```rust
        cmd.args(&spec.args).env_clear().envs(merged_env(&spec.env));
```

- [ ] **Step 8: Full suite**

Run: `cargo test --tests --features test-fixtures`
Expected: PASS, including the pre-existing `scrubbed_env_excludes_provider_secrets` and `hung_server_does_not_block_startup`.

- [ ] **Step 9: Lint**

Run: `cargo clippy --all-targets --features test-fixtures -- -D warnings`
Expected: no warnings.

- [ ] **Step 10: Commit**

```bash
git add woollama-server/src/config.rs woollama-server/src/mcp_registry.rs
git commit -m "fix: restore mcp.json 'env' support dropped in the Rust port

The Python reference parses 'env' (config.py:136) and forwards it to
StdioServerParameters.env (manager.py:89); woollamad dropped it silently
during the port, so a documented key was ignored with no message.

Merges over the scrubbed base env - explicit entries win, nothing arrives
by inheritance."
```

---

### Task 2: `McpServerSpec` becomes an enum, and delegation learns about it

Mostly a refactor, but it crosses the claude-code delegation boundary (`lib.rs:536` reads `spec.command`/`spec.args` to build the `--mcp-config` for the child `claude` process), so it is not purely mechanical. Isolating it means Task 3's diff shows only the HTTP path.

**Files:**
- Modify: `woollama-server/src/config.rs` (struct → enum, `load_mcp_servers`)
- Modify: `woollama-server/src/mcp_registry.rs` (`connect_one`, test specs)
- Modify: `woollama-server/src/lib.rs:520-539` (`referenced_mcp_servers`)

**Interfaces:**
- Consumes: `McpServerSpec { command, args, env }` from Task 1.
- Produces:
  ```rust
  pub enum McpServerSpec { Stdio(StdioSpec) }              // Http added in Task 3
  pub struct StdioSpec { pub command: String, pub args: Vec<String>, pub env: HashMap<String, String> }
  ```

- [ ] **Step 1: Change the type**

In `woollama-server/src/config.rs`, replace the Task 1 struct with:

```rust
/// A downstream MCP server. Matches Claude Code's mcp.json shape, extended with a `url` form
/// (issue #19) for a Streamable-HTTP endpoint instead of a spawned subprocess.
#[derive(Clone)]
pub enum McpServerSpec {
    Stdio(StdioSpec),
}

/// A downstream MCP server spawned as a child process, speaking MCP over stdio.
#[derive(Clone)]
pub struct StdioSpec {
    pub command: String,
    pub args: Vec<String>,
    /// Extra environment for the spawned server, merged OVER the scrubbed base env
    /// (`mcp_registry::merged_env`). Restores parity with the Python reference, which parses
    /// this key (`config.py:136`) and hands it to `StdioServerParameters.env` (`manager.py:89`)
    /// rather than argv, where a secret would show up in `ps`.
    pub env: HashMap<String, String>,
}
```

- [ ] **Step 2: Update the parser**

In `load_mcp_servers`:

```rust
            out.insert(name.clone(), McpServerSpec::Stdio(StdioSpec { command, args, env }));
```

Update the two Task 1 config tests to match the new shape:

```rust
    match &servers["git"] {
        McpServerSpec::Stdio(s) => {
            assert_eq!(s.env.get("GIT_AUTHOR_NAME").map(String::as_str), Some("woollama"))
        }
    }
```

(and the analogous `assert!(s.env.is_empty())` for the second test).

- [ ] **Step 3: Update `connect_one`**

In `woollama-server/src/mcp_registry.rs`, change the head of `connect_one` (`:127-134`) to:

```rust
    async fn connect_one(spec: &McpServerSpec) -> Result<ServerConn, String> {
        let running = match spec {
            McpServerSpec::Stdio(s) => {
                let mut cmd = tokio::process::Command::new(&s.command);
                // Scrub the child env: a downstream tool server must NOT inherit the daemon's
                // provider secrets (ANTHROPIC_API_KEY etc.). Mirrors the claude-code child scrub
                // and the Python MCP SDK's default-scrubbed stdio environment.
                cmd.args(&s.args).env_clear().envs(merged_env(&s.env));
                let transport = TokioChildProcess::new(cmd).map_err(|e| e.to_string())?;
                ().serve(transport).await.map_err(|e| e.to_string())?
            }
        };
        let peer = running.peer().clone();
```

Leave `:136-142` (`list_all_tools`, `std::mem::forget(running)`, `Ok(ServerConn { .. })`) exactly as they are — transport-agnostic, must not change.

Update the import to `use crate::config::{McpServerSpec, StdioSpec};` and the two test specs:

```rust
        specs.insert(
            "hung".to_string(),
            McpServerSpec::Stdio(StdioSpec { command: "sleep".into(), args: vec!["30".into()], env: HashMap::new() }),
        );
        specs.insert(
            "dead".to_string(),
            McpServerSpec::Stdio(StdioSpec { command: "false".into(), args: vec![], env: HashMap::new() }),
        );
```

- [ ] **Step 4: Write the failing delegation test**

`referenced_mcp_servers` builds the `--mcp-config` subset handed to the child `claude` process — the delegation containment boundary. Add to the `#[cfg(test)]` module in `woollama-server/src/lib.rs` (alongside `fnmatch_tests` / `address_tests` at `:1519`, `:1537`):

```rust
#[cfg(test)]
mod delegate_config_tests {
    use super::*;
    use crate::config::{McpServerSpec, StdioSpec};

    fn state_with(name: &str, spec: McpServerSpec) -> AppState {
        let mut mcp_specs = HashMap::new();
        mcp_specs.insert(name.to_string(), spec);
        AppState { mcp_specs, ..AppState::empty_for_test() }
    }

    #[test]
    fn delegate_config_forwards_the_env_block() {
        // A stdio server that needs `env` must behave the same in-loop and under delegation.
        // Dropping it here would make the tool work through woollama's own loop and
        // misbehave under claude-code delegation — a silent divergence.
        let mut env = HashMap::new();
        env.insert("GIT_AUTHOR_NAME".to_string(), "woollama".to_string());
        let state = state_with(
            "git",
            McpServerSpec::Stdio(StdioSpec { command: "git-mcp".into(), args: vec![], env }),
        );
        let out = referenced_mcp_servers(&state, &["git.log".to_string()]).unwrap();
        assert_eq!(out["git"]["env"]["GIT_AUTHOR_NAME"], "woollama");
    }
}
```

`AppState::empty_for_test()` does not exist yet. If constructing an `AppState` in a unit test proves impractical (it holds `Arc`s to a registry, pools, conversations, and managed-agents), **do not build a fake** — instead refactor `referenced_mcp_servers` to take `&HashMap<String, McpServerSpec>` rather than `&AppState`, which is all it uses, and test that directly:

```rust
fn referenced_mcp_servers(
    specs: &HashMap<String, config::McpServerSpec>,
    tools: &[String],
) -> Result<HashMap<String, Value>, EngineError>
```

updating its one caller at `:553` to `referenced_mcp_servers(&state.mcp_specs, &recipe.tools)?`. Prefer this — it is a smaller change than a test-only constructor and narrows the function to what it actually needs.

- [ ] **Step 5: Run to verify it fails**

Run: `cargo test -p woollama-server --features test-fixtures --lib delegate_config -- --test-threads=1`
Expected: FAIL — `env` is absent from the emitted JSON (the current line emits only `command` and `args`).

- [ ] **Step 6: Update `referenced_mcp_servers`**

Replace `lib.rs:520-539`:

```rust
/// The mcp.json subset for the servers a recipe's tools reference — the `--mcp-config` handed
/// to the child `claude` process under delegation. Errors if a referenced server isn't
/// configured.
///
/// A `url`-form server is REFUSED rather than translated. Claude Code's mcp.json can express an
/// HTTP server, but doing so would (a) have the child connect to the downstream directly,
/// outside woollama's allow-list boundary, and (b) require writing its bearer token into a temp
/// config file for another process. Neither is in scope for issue #19; fail loudly instead.
fn referenced_mcp_servers(
    specs: &HashMap<String, config::McpServerSpec>,
    tools: &[String],
) -> Result<HashMap<String, Value>, EngineError> {
    let mut servers = HashMap::new();
    for t in tools {
        let server = t.split_once('.').map(|(s, _)| s).unwrap_or(t.as_str());
        if servers.contains_key(server) {
            continue;
        }
        let Some(spec) = specs.get(server) else {
            return Err(EngineError::new(
                format!("recipe references MCP server '{server}' not in mcp.json config"),
                "invalid_request_error",
                400,
            ));
        };
        let entry = match spec {
            config::McpServerSpec::Stdio(s) => {
                json!({"command": s.command, "args": s.args, "env": s.env})
            }
        };
        servers.insert(server.to_string(), entry);
    }
    Ok(servers)
}
```

Update the caller at `:553` to `referenced_mcp_servers(&state.mcp_specs, &recipe.tools)?`.

- [ ] **Step 7: Run to verify it passes**

Run: `cargo test -p woollama-server --features test-fixtures --lib delegate_config -- --test-threads=1`
Expected: PASS.

- [ ] **Step 8: Full suite — no test result may change except the new one**

Run: `cargo test --tests --features test-fixtures`
Expected: PASS. A refactor that flips an existing test result is not a refactor — if one flips, stop and find out why. Pay particular attention to `tests/claude_code.rs`, which exercises the delegation path: adding `env` to the emitted config is a real behavior change and that suite is where it would show.

- [ ] **Step 9: Lint**

Run: `cargo clippy --all-targets --features test-fixtures -- -D warnings`
Expected: no warnings.

- [ ] **Step 10: Commit**

```bash
git add woollama-server/src/config.rs woollama-server/src/mcp_registry.rs woollama-server/src/lib.rs
git commit -m "refactor: McpServerSpec becomes an enum over transport

Isolates the shape change so the HTTP transport diff shows only the new
path. Delegation's --mcp-config now also forwards the 'env' block, so a
stdio server behaves the same in-loop and under claude-code delegation."
```

---

### Task 3: The `url` form — parse, validate, connect

**Files:**
- Modify: `woollama-server/src/config.rs` (add `HttpSpec`, the `url`/`headers` branch, `validate_header_value`)
- Modify: `woollama-server/src/mcp_registry.rs` (add `http_transport`, the `Http` match arm)
- Modify: `woollama-server/src/lib.rs` (`referenced_mcp_servers` gains the `Http` refusal arm)
- Create: `woollama-server/tests/http_downstream.rs`

**Interfaces:**
- Consumes: `McpServerSpec::Stdio(StdioSpec)` from Task 2; the public `woollama_server::build_state()` and `woollama_server::router(Arc<AppState>)`.
- Produces: `pub struct HttpSpec { pub url: String, pub headers: HashMap<String, String> }`; `McpServerSpec::Http(HttpSpec)`; `fn validate_header_value(server: &str, name: &str, value: &str) -> Result<(), String>` (private).

- [ ] **Step 1: Write the failing validation tests**

Add to `#[cfg(test)] mod tests` in `woollama-server/src/config.rs`:

```rust
#[test]
fn empty_header_value_is_rejected() {
    // engine::expand_env resolves an UNSET ${VAR} to "" (lib.rs:417, unwrap_or_default), so a
    // header sourced from a missing env var arrives here empty. Fail closed: a silently
    // credential-less request that a permissive downstream accepts is the worst outcome.
    let err = validate_header_value("shelf", "X-Api-Key", "").unwrap_err();
    assert!(err.contains("shelf") && err.contains("X-Api-Key"), "must name server and header: {err}");
}

#[test]
fn bare_auth_scheme_is_rejected() {
    // "Bearer ${SHELF_TOKEN}" with SHELF_TOKEN unset expands to the literal "Bearer " —
    // well-formed, no credential. This is the case that would otherwise look healthy.
    let err = validate_header_value("shelf", "Authorization", "Bearer ").unwrap_err();
    assert!(err.contains("Authorization"), "must name the header: {err}");
    assert!(validate_header_value("shelf", "Authorization", "Basic").is_err());
}

#[test]
fn a_populated_header_value_is_accepted() {
    assert!(validate_header_value("shelf", "Authorization", "Bearer sk-live-abc").is_ok());
    // A single-token value is legitimate when it isn't a bare auth scheme.
    assert!(validate_header_value("shelf", "X-Api-Key", "abc123").is_ok());
}

#[test]
fn a_rejection_never_echoes_the_header_value() {
    // Error text reaches logs and the operator's terminal; header values carry bearer tokens.
    let err = validate_header_value("shelf", "Authorization", "Bearer ").unwrap_err();
    assert!(!err.contains("Bearer "), "must not echo the value: {err}");
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p woollama-server --features test-fixtures --lib header_value -- --test-threads=1`
Expected: FAIL to compile — `cannot find function 'validate_header_value'`.

- [ ] **Step 3: Implement the validation**

Add to `woollama-server/src/config.rs`, above `load_mcp_servers`:

```rust
/// Auth schemes whose bare form (scheme, no credential) means an unset `${VAR}` ate the secret.
/// Lowercase; compared case-insensitively per RFC 7235.
const BARE_AUTH_SCHEMES: &[&str] = &["bearer", "basic", "digest", "token"];

/// Reject a header value that carries no credential.
///
/// `engine::expand_env` resolves an unset `${VAR}` to the empty string
/// (`woollama-engine/src/lib.rs:417`), so `"Bearer ${SHELF_TOKEN}"` with `SHELF_TOKEN` unset
/// becomes the literal `"Bearer "` — a well-formed request carrying no credential. Against a
/// downstream that requires auth you get a 401 and notice; against a permissive one, or one
/// whose auth is enforced by a proxy that isn't deployed yet, you connect unauthenticated and
/// everything reports healthy. Fail closed here rather than in `expand_env`, which is
/// parity-locked and also feeds `inferencers.toml`.
///
/// NEVER include `value` in the returned message — these carry bearer tokens.
fn validate_header_value(server: &str, name: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!(
            "mcp.json: server '{server}' header '{name}' is empty — an unset ${{VAR}} expands to \
             nothing, so this would send no credential"
        ));
    }
    let mut parts = value.split_whitespace();
    let scheme = parts.next().unwrap_or_default();
    if parts.next().is_none() && BARE_AUTH_SCHEMES.contains(&scheme.to_ascii_lowercase().as_str()) {
        return Err(format!(
            "mcp.json: server '{server}' header '{name}' is the bare auth scheme '{scheme}' with no \
             credential — an unset ${{VAR}} expands to nothing"
        ));
    }
    Ok(())
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test -p woollama-server --features test-fixtures --lib header_value -- --test-threads=1`
Expected: PASS (4 tests).

- [ ] **Step 5: Write the failing parse tests**

Add to `#[cfg(test)] mod tests` in `woollama-server/src/config.rs` (reusing `servers_from` from Task 1):

```rust
#[test]
fn url_form_parses_with_headers() {
    let servers = servers_from(
        r#"{"mcpServers": {"shelf": {"url": "http://127.0.0.1:9200/mcp",
             "headers": {"Authorization": "Bearer sk-abc"}}}}"#,
    )
    .unwrap();
    match &servers["shelf"] {
        McpServerSpec::Http(h) => {
            assert_eq!(h.url, "http://127.0.0.1:9200/mcp");
            assert_eq!(h.headers.get("Authorization").map(String::as_str), Some("Bearer sk-abc"));
        }
        _ => panic!("expected an Http spec"),
    }
}

#[test]
fn a_server_with_neither_command_nor_url_is_rejected() {
    let err = servers_from(r#"{"mcpServers": {"broken": {"args": ["x"]}}}"#).unwrap_err();
    assert!(err.contains("broken") && err.contains("url"), "must name the server and both forms: {err}");
}

#[test]
fn a_server_with_both_command_and_url_is_rejected() {
    // Ambiguous rather than merely redundant: silently preferring one would make the other half
    // of the config a lie.
    let err = servers_from(r#"{"mcpServers": {"both": {"command": "x", "url": "http://h/mcp"}}}"#).unwrap_err();
    assert!(err.contains("both"), "must name the server: {err}");
}

#[test]
fn an_unset_token_in_a_header_fails_the_load() {
    std::env::remove_var("WOOLLAMA_TEST_ABSENT_TOKEN");
    let err = servers_from(
        r#"{"mcpServers": {"shelf": {"url": "http://h/mcp",
             "headers": {"Authorization": "Bearer ${WOOLLAMA_TEST_ABSENT_TOKEN}"}}}}"#,
    )
    .unwrap_err();
    // The end-to-end version of the fail-open: config text -> expand_env -> validation.
    assert!(err.contains("Authorization"), "the unset-var case must fail the load: {err}");
}
```

- [ ] **Step 6: Run to verify they fail**

Run: `cargo test -p woollama-server --features test-fixtures --lib url_form a_server_with an_unset_token -- --test-threads=1`
Expected: FAIL — `no variant named 'Http'`.

- [ ] **Step 7: Add the variant and the parse branch**

In `woollama-server/src/config.rs`:

```rust
#[derive(Clone)]
pub enum McpServerSpec {
    Stdio(StdioSpec),
    Http(HttpSpec),
}

/// A downstream MCP server reached over Streamable HTTP at `url` (issue #19) — the form that
/// lets one woollamad consume another's `/mcp` surface. No child process, so the stdio env
/// scrub has no meaning here; credentials ride `headers` via the existing `${VAR}` expansion.
#[derive(Clone)]
pub struct HttpSpec {
    pub url: String,
    pub headers: HashMap<String, String>,
}
```

Replace the body of the `for (name, s) in servers` loop in `load_mcp_servers`:

```rust
        for (name, s) in servers {
            let url = s.get("url").and_then(Value::as_str).filter(|u| !u.is_empty());
            let command = s.get("command").and_then(Value::as_str).filter(|c| !c.is_empty());
            let spec = match (command, url) {
                (Some(_), Some(_)) => {
                    return Err(format!(
                        "mcp.json: server '{name}' sets both 'command' and 'url' — a server is \
                         either a stdio subprocess or an HTTP endpoint, not both"
                    ))
                }
                (None, None) => {
                    return Err(format!(
                        "mcp.json: server '{name}' needs either 'command' (stdio) or 'url' (HTTP)"
                    ))
                }
                (Some(command), None) => {
                    let args = s
                        .get("args")
                        .and_then(Value::as_array)
                        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                        .unwrap_or_default();
                    let env = s
                        .get("env")
                        .and_then(Value::as_object)
                        .map(|o| {
                            o.iter()
                                .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string())))
                                .collect()
                        })
                        .unwrap_or_default();
                    McpServerSpec::Stdio(StdioSpec { command: command.to_string(), args, env })
                }
                (None, Some(url)) => {
                    let mut headers = HashMap::new();
                    if let Some(o) = s.get("headers").and_then(Value::as_object) {
                        for (k, v) in o {
                            let Some(v) = v.as_str() else {
                                return Err(format!("mcp.json: server '{name}' header '{k}' must be a string"));
                            };
                            validate_header_value(name, k, v)?;
                            headers.insert(k.clone(), v.to_string());
                        }
                    }
                    McpServerSpec::Http(HttpSpec { url: url.to_string(), headers })
                }
            };
            out.insert(name.clone(), spec);
        }
```

- [ ] **Step 8: Add the delegation refusal arm**

The `match spec` in `referenced_mcp_servers` (`lib.rs`) is now non-exhaustive. Add:

```rust
            config::McpServerSpec::Http(_) => {
                return Err(EngineError::new(
                    format!(
                        "recipe references MCP server '{server}', which is a 'url' (HTTP) server — \
                         claude-code delegation cannot hand an HTTP downstream to the child process"
                    ),
                    "invalid_request_error",
                    400,
                ))
            }
```

- [ ] **Step 9: Run the lib tests**

Run: `cargo test -p woollama-server --features test-fixtures --lib -- --test-threads=1`
Expected: PASS, including the Task 1 `env` tests (the stdio branch still parses `env`).

- [ ] **Step 10: Write the failing federation test**

Create `woollama-server/tests/http_downstream.rs`. It drives only the public surface — no module-visibility changes.

```rust
//! Issue #19: consuming a downstream MCP server over Streamable HTTP (`"url"` in mcp.json).
//! The headline case is one woollamad consuming another — woollamad already SERVES `/mcp`
//! (lib.rs:261), so a single test exercises the transport and demonstrates tool federation.
//!
//! Separate test binary so the global WOOLLAMA_* env can't race other test files (same reason
//! as tests/mcp_surface.rs). Run with --test-threads=1: config discovery is process-global env.

use std::sync::Arc;

use axum::Router;
use rmcp::transport::streamable_http_client::StreamableHttpClientWorker;
use rmcp::transport::worker::WorkerTransport;
use rmcp::ServiceExt;

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

/// Build and serve a woollamad whose config dir contains exactly `mcp_json`.
/// The tempdir is leaked: `build_state` has already read it, and leaking is cheaper than
/// threading ownership through the test.
async fn spawn_woollamad(mcp_json: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("mcp.json"), mcp_json).unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", dir.path());
    // Keep each instance's durable handle table separate so two in-process routers don't share.
    std::env::set_var("WOOLLAMA_STATE_DIR", dir.path());
    let state = Arc::new(woollama_server::build_state().await);
    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
    std::env::remove_var("WOOLLAMA_STATE_DIR");
    std::mem::forget(dir);
    spawn(woollama_server::router(state)).await
}

async fn tool_names(base: &str) -> Vec<String> {
    let worker = StreamableHttpClientWorker::<reqwest::Client>::new_simple(format!("{base}/mcp"));
    let client = ().serve(WorkerTransport::spawn(worker)).await.unwrap();
    let tools = client.peer().list_all_tools().await.unwrap();
    tools.iter().map(|t| t.name.to_string()).collect()
}

#[tokio::test]
async fn woollamad_consumes_another_woollamad_over_http() {
    // B: a leaf with no downstreams of its own, like the real mcp-suite. Its /mcp still
    // advertises the built-in `chat` verb (mcp_surface.rs:123) — the stable thing to assert on
    // without depending on the ambient config dir.
    let b = spawn_woollamad(r#"{"mcpServers": {}}"#).await;
    // A: consumes B over the url form.
    let a = spawn_woollamad(&format!(r#"{{"mcpServers": {{"remote": {{"url": "{b}/mcp"}}}}}}"#)).await;

    let names = tool_names(&a).await;
    // Assert a NAMED federated tool, not a non-empty roster: a transport that connects and
    // silently returns nothing would pass the weaker check.
    assert!(
        names.contains(&"mcp__remote__chat".to_string()),
        "expected B's tools federated under 'remote', got {names:?}"
    );
}
```

- [ ] **Step 11: Run to verify it fails**

Run: `cargo test -p woollama-server --features test-fixtures --test http_downstream -- --test-threads=1 --nocapture`
Expected: FAIL — the assertion, with `woollamad: MCP server 'remote' failed to start, skipping: ...` on stderr. `connect` swallows the cause into stderr by design (`mcp_registry.rs:111`), so `--nocapture` is how you read it.

- [ ] **Step 12: Implement the HTTP transport**

In `woollama-server/src/mcp_registry.rs`, add the imports:

```rust
use rmcp::transport::streamable_http_client::{StreamableHttpClientTransport, StreamableHttpClientTransportConfig};

use crate::config::{HttpSpec, McpServerSpec, StdioSpec};
```

Add the transport builder:

```rust
/// Build the Streamable-HTTP client transport for a `url`-form downstream server.
///
/// `reqwest::Client` is rmcp's own HTTP client impl (Cargo.toml pins reqwest 0.13 to match), and
/// `axum::http` re-exports the same `http` 1.5 header types rmcp expects — no new dependency for
/// either.
fn http_transport(spec: &HttpSpec) -> Result<StreamableHttpClientTransport<reqwest::Client>, String> {
    use axum::http::{HeaderName, HeaderValue};

    let mut headers = HashMap::new();
    for (k, v) in &spec.headers {
        let name = HeaderName::try_from(k.as_str()).map_err(|e| format!("invalid header name '{k}': {e}"))?;
        // `InvalidHeaderValue`'s Display does not include the offending value, which is what we
        // want — these carry bearer tokens. Do not add `{v}` to this message.
        let value = HeaderValue::from_str(v).map_err(|e| format!("invalid header value for '{k}': {e}"))?;
        headers.insert(name, value);
    }
    let config = StreamableHttpClientTransportConfig::with_uri(spec.url.clone()).custom_headers(headers);
    Ok(StreamableHttpClientTransport::with_client(reqwest::Client::new(), config))
}
```

Add the match arm in `connect_one`, after the `Stdio` arm:

```rust
            McpServerSpec::Http(h) => {
                // No child process, so no env scrub applies. Everything after `.serve()` is
                // transport-agnostic and shared with the stdio path.
                let transport = http_transport(h)?;
                ().serve(transport).await.map_err(|e| e.to_string())?
            }
```

- [ ] **Step 13: Run the federation test**

Run: `cargo test -p woollama-server --features test-fixtures --test http_downstream -- --test-threads=1 --nocapture`
Expected: PASS.

- [ ] **Step 14: Write the failing header-delivery test**

The federation test proves the transport works; it does not prove configured headers reach the wire. Add to `woollama-server/tests/http_downstream.rs`:

```rust
use std::sync::Mutex;

#[tokio::test]
async fn configured_headers_reach_the_downstream_request() {
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    // A stub that records Authorization and then fails the handshake. The connect is EXPECTED to
    // fail — the claim under test is "the header reached the wire", which is exactly what a
    // silently credential-less transport would get wrong while looking healthy.
    let stub = Router::new().route(
        "/mcp",
        axum::routing::any(move |headers: axum::http::HeaderMap| {
            let sink = sink.clone();
            async move {
                if let Some(v) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
                    sink.lock().unwrap().push(v.to_string());
                }
                axum::http::StatusCode::NOT_IMPLEMENTED
            }
        }),
    );
    let base = spawn(stub).await;
    let _consumer = spawn_woollamad(&format!(
        r#"{{"mcpServers": {{"stub": {{"url": "{base}/mcp",
             "headers": {{"Authorization": "Bearer sk-test-value"}}}}}}}}"#
    ))
    .await;
    assert_eq!(
        seen.lock().unwrap().as_slice(),
        ["Bearer sk-test-value"],
        "the configured Authorization header must reach the downstream request exactly once"
    );
}
```

- [ ] **Step 15: Run it**

Run: `cargo test -p woollama-server --features test-fixtures --test http_downstream -- --test-threads=1`
Expected: PASS (2 tests). If the recorded vec is empty, the header is being dropped between `HttpSpec` and the transport — check that `custom_headers` is applied to the config *before* `with_client`.

- [ ] **Step 16: Full suite + lint**

Run: `cargo test --tests --features test-fixtures && cargo clippy --all-targets --features test-fixtures -- -D warnings`
Expected: PASS, no warnings. If clippy raises `large_enum_variant` on `McpServerSpec`, box the larger variant rather than allowing the lint.

- [ ] **Step 17: Commit**

```bash
git add woollama-server/src/config.rs woollama-server/src/mcp_registry.rs woollama-server/src/lib.rs woollama-server/tests/http_downstream.rs
git commit -m "feat: consume downstream MCP servers over Streamable HTTP

mcp.json gains a 'url' form alongside 'command', so one woollamad can
consume another's /mcp surface. Credentials ride the existing \${VAR}
expansion in 'headers'.

Header values are validated fail-closed: expand_env resolves an unset var
to the empty string, so 'Bearer \${TOKEN}' with TOKEN unset would otherwise
send a well-formed request carrying no credential.

A url-form server referenced by a claude-code recipe is refused rather than
translated - delegating it would put the child outside woollama's allow-list
boundary and write its bearer token to a temp file."
```

---

### Task 4: Docs, and the notes that must not be orphaned

**Files:**
- Modify: `docs/configuration.md`
- Modify: `docs/roadmap.md`

**Interfaces:** Consumes Tasks 1–3. Produces no code.

- [ ] **Step 1: Document the `url` form**

In `docs/configuration.md`, in the `mcpServers` section, after the existing table:

````markdown
A server entry is **either** a stdio subprocess (`command`) **or** a Streamable-HTTP endpoint
(`url`) — setting both is an error, as is setting neither.

| key | default | meaning |
|---|---|---|
| `url` | — | Streamable-HTTP MCP endpoint, e.g. another woollamad's `/mcp`. Mutually exclusive with `command`. |
| `headers` | `{}` | Headers sent with every request to `url`. Use `${VAR}` for credentials — never inline a token. |

```json
{
  "mcpServers": {
    "shelf": {
      "url": "http://mcp.example.lan:9200/mcp",
      "headers": { "Authorization": "Bearer ${SHELF_TOKEN}" }
    }
  }
}
```

Header values are validated at load: an empty value, or a bare auth scheme with nothing after
it, is a startup error naming the server and header. This is deliberate — `${VAR}` expansion
resolves an **unset** variable to the empty string, so `"Bearer ${SHELF_TOKEN}"` with
`SHELF_TOKEN` unset would otherwise produce the literal header `Bearer `: a well-formed request
carrying no credential, which a permissive downstream would accept while everything reported
healthy.

`env` has no meaning for a `url` server (there is no child process), and the child-env scrub
likewise applies only to the stdio form. A `url` server also cannot be handed to **claude-code
delegation** — a recipe whose tools reference one is rejected, rather than having the child
`claude` process connect to the downstream directly outside woollama's allow-list boundary.

> **The `url` form is `woollamad`-only.** The Python reference server (the differential-test
> oracle) requires `command` and will fail to load a config containing a `url` server. If you
> share one config dir between the two, keep `url` entries out of it.
````

- [ ] **Step 2: Split the Rust/Python divergence**

In `docs/configuration.md`, replace the note claiming a failed downstream server "aborts woollama startup":

```markdown
> **On a downstream server that fails to start** — the two implementations differ. `woollamad`
> (the canonical router) logs a warning and skips it, so the router comes up in a known-degraded
> state; the Python reference server aborts startup. If startup behavior matters to you, pin
> `command` to an absolute interpreter either way.
```

Leave the `env` row in the table as it is — Task 1 made `woollamad` match what it already documents.

- [ ] **Step 3: Record the retry-slice dependencies**

In `docs/roadmap.md`, under the not-yet section:

```markdown
- **Downstream reconnect/retry for `url`-form MCP servers.** A remote peer can come back where a
  failed `exec` won't. Three things belong to this slice specifically, and all three look like
  they belong somewhere else:
  - **Structure.** `McpRegistry` holds plain `HashMap`s behind an `Arc` with no interior
    mutability, so reconnect needs an `RwLock` through `resolve`/`tool`/`reexport_tools`/
    `call_server`/`call_raw` — a hot-path structural change, not an addition.
  - **The federation hop cap.** Tool federation is safe today *because rosters are cached at
    connect and served from cache* (`mcp_registry.rs:136`, `mcp_surface.rs:123`) — nothing
    recurses at request time. Dynamic roster refresh is exactly what removes that property: a
    refresh can cascade across routers at request time, making live recursion reachable for the
    first time. Design the hop cap here or not at all.
  - **A self-loop guard.** Likewise unreachable today: `build_state()` connects every downstream
    (`main.rs:13`) *before* `axum::serve` opens the listener (`:62`), so an instance naming its
    own address gets connection-refused and never reaches any inbound check. Reconnect-while-
    serving is what makes such a guard able to fire.
  - Retry must stay observable: never present a reconnecting server as present-with-zero-tools.
    Distinct connected / degraded-retrying / never-connected states with last error and attempt
    count. The test that matters is "while down, the degraded state is visible" — a test that
    only asserts eventual success also passes on a router permanently hiding a dead downstream.
```

- [ ] **Step 4: Add the shipped row**

In `docs/roadmap.md`'s "Shipped" table:

```markdown
| **HTTP downstream MCP transport** — `mcp.json` `url` form (Streamable HTTP) alongside `command`; one woollamad can consume another's `/mcp`. Credentials via `${VAR}` in `headers`, validated fail-closed. Restores the `env` key dropped in the port. Rust-only | `config.rs`, `mcp_registry.rs`, `lib.rs` | #19 |
```

- [ ] **Step 5: File the out-of-scope findings**

Open these (or hand the text to whoever owns the tracker). None belongs in #19:

1. **`engine::expand_env` fails open.** `woollama-engine/src/lib.rs:417` uses `unwrap_or_default()`, so any unset `${VAR}` silently becomes an empty string across `mcp.json` *and* `inferencers.toml`. #19 guards the one case it introduces (MCP headers); the general fix belongs in the engine, which is parity-locked, so it needs its own decision about whether unset-is-empty is intended.
2. **`wire_name` uses `DefaultHasher` for its >64-char fallback** (`mcp_registry.rs:64-68`). `std::collections::hash_map::DefaultHasher`'s algorithm is explicitly not guaranteed stable across Rust releases, so a hashed tool name can change under a toolchain upgrade — silently breaking any recipe allow-list entry or client-cached name that landed on the hashed path. Dormant today because almost nothing exceeds 64 chars; **federation is what pushes names over the line** (`mcp__mcp-suite__mcp__reader__parse_structured` is 45 chars at two levels), so it gets worse the longer #19 is deployed. Fix is a pinned hash (FNV or truncated SHA-256), no behavior change for names that already fit.
3. **`/v1/tools` exists only in the Python reference** (`src/woollama/router.py:255`); `woollamad` has no such route, but `README.md` lists it under "What works today" for the router generally. Either port it or correct the README — federation makes tool introspection more valuable, not less.

- [ ] **Step 6: Commit**

```bash
git add docs/configuration.md docs/roadmap.md
git commit -m "docs: mcp.json url form, and split the Rust/Python divergence

Documents the HTTP downstream transport and its fail-closed header
validation. Records that the hop cap AND a self-loop guard both belong to
the retry slice - both concern behavior after startup, and today there is
no after-startup: connect happens once, before the listener opens."
```

---

## Acceptance

**The builder does not declare this working.** Per the working agreement in the brief, the
`tiiny-85` session validates by pointing the device-management woollamad at the real
`mcp-suite` container over the `url` form and reporting what actually happens.

A green `cargo test` here proves the transport compiles and talks to another woollamad. That is
a different claim from "it consumes mcp-suite." Report the former; do not report the latter.

## Known divergence introduced by this plan

A `mcp.json` containing a `url` server is **not loadable by the Python reference server** —
`src/woollama/config.py:131` raises on a missing `command`. Anyone sharing one config dir
between `woollamad` and the Python oracle (including `WOOLLAMA_TEST_CMD="python -m woollama"
pytest -m integration`) will hit a startup error rather than a skip.

This is accepted, not overlooked: the brief scopes #19 Rust-only, and touching `src/woollama/`
is forbidden by the Global Constraints. Task 4 Step 1 documents it. **Flag it to Teague rather
than resolving it unilaterally** — the options (leave it; make Python skip `url` servers with a
warning; make Python error with a message naming the form as Rust-only) are a scope call, and
the latter two require a Python-tree edit this plan does not authorize.
