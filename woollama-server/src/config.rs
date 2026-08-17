//! Recipe + MCP-server config loading — ported from Python `woollama.config`.
//!
//! A user file in `config_dir()` is used if present, else the bundled default
//! (embedded from this crate's own `defaults/` so the crate is self-contained for a
//! crates.io publish). Those vendored copies are kept byte-identical to the Python
//! package's `src/woollama/defaults/` by `tests/defaults_sync.rs`. `${VAR}` in mcp.json
//! is expanded from the environment at load time.

use std::collections::HashMap;

use serde_json::Value;
use woollama_engine as engine;

const DEFAULT_RECIPES: &str = include_str!("../defaults/recipes.toml");
const DEFAULT_MCP: &str = include_str!("../defaults/mcp.json");

/// Where a recipe came from — surfaced in `GET /w1/patterns` as `"recipe"` (hand-authored
/// in recipes.toml) or `"fabric"` (auto-discovered from a `[patterns]` directory scan).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PatternSource {
    Recipe,
    Fabric,
}

impl PatternSource {
    pub fn as_str(self) -> &'static str {
        match self {
            PatternSource::Recipe => "recipe",
            PatternSource::Fabric => "fabric",
        }
    }
}

/// A composed recipe: a system prompt + an inferencer + an allow-list of namespaced
/// `<server>.<tool>` names (+ optional per-recipe inference params).
///
/// Recipes double as woollama's `/w1/` **patterns**: `system` may carry `{{var}}` tokens
/// that [`Recipe::render`] substitutes immediately before dispatch (the one templating
/// primitive — `woollama-engine` stays parity-locked and never sees a `{{var}}`).
#[derive(Clone)]
pub struct Recipe {
    pub inferencer: String,
    pub system: String,
    pub tools: Vec<String>,
    pub params: Option<Value>,
    pub source: PatternSource,
    /// Optional per-variable metadata, keyed by variable name, from
    /// `[recipes.<name>.variables.<var>]` in recipes.toml. A **server-layer overlay** only:
    /// it enriches `/w1/patterns` discovery (defaults/choices/description) and supplies
    /// `default`s at render time, but is never sent to the parity-locked engine
    /// (`to_value` ignores it). [`scan_vars`] stays authoritative for *which* `{{var}}`s
    /// exist and their order; an entry whose name isn't in `system` is simply unused.
    /// Native recipes only — fabric patterns carry no metadata (always empty).
    pub variables: HashMap<String, VarMeta>,
}

/// Author-supplied metadata for one `{{var}}` of a native recipe. Every field is optional;
/// absent fields are omitted from the `/w1/patterns` JSON (no `null` noise).
#[derive(Clone, Default)]
pub struct VarMeta {
    /// Value to substitute when the caller doesn't supply this variable. Caller-supplied
    /// always wins; a variable with no default and no caller value is left verbatim.
    pub default: Option<Value>,
    /// The allowed values, surfaced for discovery (so a UI can render a picker). **Not**
    /// server-enforced — a caller may still pass a value outside this list.
    pub choices: Option<Vec<Value>>,
    /// Human-readable description of the variable.
    pub description: Option<String>,
}

impl Recipe {
    /// The engine's recipe shape (what `build_setup` reads).
    pub fn to_value(&self) -> Value {
        let mut v = serde_json::json!({
            "inferencer": self.inferencer, "system": self.system, "tools": self.tools,
        });
        if let Some(p) = &self.params {
            v["params"] = p.clone();
        }
        v
    }

    /// Render a pattern for one call: clone the recipe, substitute each `{{k}}` in `system`
    /// with its value, and (if given) override the bound `inferencer` with `model_override`.
    ///
    /// Substitution is a **dumb string replace** — byte-for-byte fabric's
    /// `sysp.replace("{{"+k+"}}", str(v))` (`cosmic-fabric/src/core.py:359`). No template
    /// engine: a new dep would diverge from fabric's exact output. A non-string value uses
    /// its JSON rendering; unsupplied `{{x}}` tokens are left verbatim. This is a pure
    /// server-layer transform applied before the existing orchestration path runs.
    pub fn render(&self, variables: &serde_json::Map<String, Value>, model_override: Option<&str>) -> Recipe {
        Recipe {
            inferencer: model_override.map(String::from).unwrap_or_else(|| self.inferencer.clone()),
            system: render_system(&self.system, variables),
            tools: self.tools.clone(),
            params: self.params.clone(),
            source: self.source,
            variables: self.variables.clone(),
        }
    }

    /// Merge author-configured `default`s into the caller's variables for a native recipe:
    /// any variable with a configured default that the caller did **not** supply gets the
    /// default. Caller-supplied always wins; a variable with no default stays absent (and is
    /// left verbatim by [`render_system`]). The one place defaults are applied — both
    /// `/w1/patterns/{name}/render` and `/run` route through it so they can't diverge.
    pub fn apply_defaults(&self, variables: &serde_json::Map<String, Value>) -> serde_json::Map<String, Value> {
        let mut merged = variables.clone();
        for (name, meta) in &self.variables {
            if let Some(def) = &meta.default {
                merged.entry(name.clone()).or_insert_with(|| def.clone());
            }
        }
        merged
    }
}

/// Substitute `{{k}}` → value in a system prompt (the shared primitive behind
/// [`Recipe::render`] and the fabric-sourced render path, which has only a raw system string,
/// not a `Recipe`). Dumb string replace — byte-for-byte fabric's `sysp.replace`. A non-string
/// value uses its JSON rendering; unsupplied tokens stay verbatim.
pub fn render_system(system: &str, variables: &serde_json::Map<String, Value>) -> String {
    let mut out = system.to_string();
    for (k, v) in variables {
        let rep = match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        out = out.replace(&format!("{{{{{k}}}}}"), &rep);
    }
    out
}

/// The variable names a pattern exposes — every distinct `{{name}}` token scanned from a
/// system prompt, in first-seen order. A name is accepted only if it is non-empty and made
/// of identifier chars (`[A-Za-z0-9_.-]`), so prose like `{{ not a var }}` is ignored.
/// This is authoritative for *which* variables a pattern exposes and their order; native
/// recipes may additionally carry a [`Recipe::variables`] metadata overlay (defaults/
/// choices/description), keyed by these names. fabric patterns carry no overlay, so names
/// are all `/w1/patterns` surfaces for them.
pub fn scan_vars(system: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let bytes = system.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'{' && bytes[i + 1] == b'{' {
            if let Some(end) = system[i + 2..].find("}}") {
                let name = system[i + 2..i + 2 + end].trim();
                let ok = !name.is_empty()
                    && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'));
                if ok && !out.iter().any(|n| n == name) {
                    out.push(name.to_string());
                }
                i += 2 + end + 2;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// A parsed `mcp.json`: the usable servers, the per-server errors (entries that were skipped —
/// a bad entry never costs its siblings), and operator warnings that don't invalidate an entry.
pub type McpConfigLoad = (HashMap<String, McpServerSpec>, Vec<String>, Vec<String>);

/// A downstream MCP server. Matches Claude Code's mcp.json shape, extended with a `url` form
/// (issue #19) for a Streamable-HTTP endpoint instead of a spawned subprocess.
#[derive(Clone, Debug)]
pub enum McpServerSpec {
    Stdio(StdioSpec),
    Http(HttpSpec),
}

/// A downstream MCP server reached over Streamable HTTP at `url` (issue #19) — the form that lets
/// one woollamad consume another's `/mcp` surface. No child process, so the stdio env scrub has
/// no meaning here; credentials ride `headers` via the existing `${VAR}` expansion.
#[derive(Clone)]
pub struct HttpSpec {
    pub url: String,
    pub headers: HashMap<String, String>,
}

/// Hand-written so header VALUES never reach a log line, a panic message, or a `{:?}`.
/// They carry bearer tokens; the names are enough to debug a misconfiguration. The URL keeps only
/// its scheme+authority+path — a query string can itself carry a credential (`?token=…`).
impl std::fmt::Debug for HttpSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut names: Vec<&str> = self.headers.keys().map(String::as_str).collect();
        names.sort_unstable();
        let url = self.url.split(['?', '#']).next().unwrap_or("");
        f.debug_struct("HttpSpec").field("url", &url).field("headers", &names).finish()
    }
}

/// A downstream MCP server spawned as a child process, speaking MCP over stdio.
#[derive(Clone)]
pub struct StdioSpec {
    pub command: String,
    pub args: Vec<String>,
    /// Extra environment for the spawned server, merged OVER the scrubbed base env
    /// (`mcp_registry::merged_env`) — explicit entries win, but nothing arrives by
    /// inheritance. Restores parity with the Python reference, which parses this key
    /// (`config.py:136`) and hands it to `StdioServerParameters.env` (`manager.py:89`)
    /// rather than argv, where a secret would show up in `ps`.
    pub env: HashMap<String, String>,
}

/// Hand-written for the same reason as [`HttpSpec`]'s: `env` is a documented, deliberate home for
/// a provider key (see `mcp_registry::merged_env` and the test that pins it), so a derived
/// `Debug` would print secrets into any `{:?}`, log line, or panic message. Names only.
impl std::fmt::Debug for StdioSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut names: Vec<&str> = self.env.keys().map(String::as_str).collect();
        names.sort_unstable();
        f.debug_struct("StdioSpec")
            .field("command", &self.command)
            .field("args", &self.args)
            .field("env", &names)
            .finish()
    }
}

fn read_user_or_default(filename: &str, default: &str) -> String {
    let path = engine::config_dir().join(filename);
    std::fs::read_to_string(&path).unwrap_or_else(|_| default.to_string())
}

/// Resolve `WOOLLAMA_EXAMPLES_DIR` so the bundled-default `mcp.json`'s
/// `${WOOLLAMA_EXAMPLES_DIR}/mcp-*/server.py` references expand to a real path. We set the
/// process env (not just return a value) so the existing `engine::expand_env` picks it up —
/// the same approach as Python `config._expand_env`. Precedence:
///   1. an explicit `WOOLLAMA_EXAMPLES_DIR` (operator / config override) ALWAYS wins;
///   2. examples shipped ALONGSIDE the binary (`<exe-dir>/examples`) — the default for a
///      packaged install (the dir is 116K, so it ships with the binary);
///   3. the source checkout's `examples/` (`<crate>/../examples`) — dev runs, `cargo run`,
///      and the integration suite spawning `target/<profile>/woollama-server`.
///
/// If none exist it stays unset, so the bundled example servers are cleanly SKIPPED rather
/// than spawned from a bogus empty path (the bug the live oracle surfaced). Idempotent —
/// resolves to a deterministic value, safe to call once per `build_state`.
pub fn ensure_examples_dir() {
    // A candidate counts only if it actually holds the example servers — guards against
    // matching a bare `examples/` dir that means something else (notably cargo's reserved
    // `target/<profile>/examples`, where `cargo build --example` artifacts land).
    let is_examples = |p: &std::path::Path| p.join("mcp-hello").join("server.py").is_file();

    if std::env::var("WOOLLAMA_EXAMPLES_DIR").map(|v| !v.is_empty()).unwrap_or(false) {
        return; // (1) explicit override wins
    }
    // (2) shipped alongside the binary
    if let Ok(exe) = std::env::current_exe() {
        if let Some(cand) = exe.parent().map(|d| d.join("examples")) {
            if is_examples(&cand) {
                std::env::set_var("WOOLLAMA_EXAMPLES_DIR", cand);
                return;
            }
        }
    }
    // (3) source checkout (dev / cargo run / integration suite)
    let repo_examples = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().map(|p| p.join("examples"));
    if let Some(cand) = repo_examples {
        if is_examples(&cand) {
            std::env::set_var("WOOLLAMA_EXAMPLES_DIR", cand);
        }
    }
}

pub fn load_recipes() -> Result<HashMap<String, Recipe>, String> {
    let text = read_user_or_default("recipes.toml", DEFAULT_RECIPES);
    let v: Value = toml::from_str(&text).map_err(|e| format!("recipes.toml parse error: {e}"))?;
    let mut out = HashMap::new();
    if let Some(recipes) = v.get("recipes").and_then(Value::as_object) {
        for (name, r) in recipes {
            out.insert(
                name.clone(),
                Recipe {
                    inferencer: r.get("inferencer").and_then(Value::as_str).unwrap_or("").to_string(),
                    system: r.get("system").and_then(Value::as_str).unwrap_or("").to_string(),
                    tools: r
                        .get("tools")
                        .and_then(Value::as_array)
                        .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                        .unwrap_or_default(),
                    params: r.get("params").filter(|p| !p.is_null()).cloned(),
                    source: PatternSource::Recipe,
                    variables: parse_var_meta(r.get("variables")),
                },
            );
        }
    }
    Ok(out)
}

/// Parse a `[recipes.<name>.variables]` table (var-name → `{default, choices, description}`)
/// into the [`Recipe::variables`] overlay. Anything that isn't a table, or a var entry that
/// isn't a table, yields an empty map / skipped entry — a malformed overlay degrades to
/// name-only discovery rather than failing the whole recipe load.
fn parse_var_meta(v: Option<&Value>) -> HashMap<String, VarMeta> {
    let mut out = HashMap::new();
    if let Some(table) = v.and_then(Value::as_object) {
        for (name, meta) in table {
            let Some(meta) = meta.as_object() else { continue };
            out.insert(
                name.clone(),
                VarMeta {
                    default: meta.get("default").filter(|d| !d.is_null()).cloned(),
                    choices: meta.get("choices").and_then(Value::as_array).cloned(),
                    description: meta.get("description").and_then(Value::as_str).map(String::from),
                },
            );
        }
    }
    out
}

/// Expand a leading `~` / `~/` in a config path against `$HOME`. (`${VAR}` is handled
/// elsewhere by `engine::expand_env`; this only covers the home shorthand.)
fn expand_tilde(p: &str) -> std::path::PathBuf {
    if p == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home);
        }
    } else if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::Path::new(&home).join(rest);
        }
    }
    std::path::PathBuf::from(p)
}

/// Discover fabric-style patterns from the optional `[patterns]` block in recipes.toml:
/// ```toml
/// [patterns]
/// dir = "~/.config/fabric/patterns"
/// default_inferencer = "ollama/qwen3:14b-iq4xs"
/// ```
/// For each `<dir>/<name>/system.md`, build a `Recipe { system: <file>, inferencer:
/// default_inferencer, tools: [], source: Fabric }`. This is **read-only file parsing** —
/// no `fabric --serve` dependency. Opt-in: with no `[patterns]` block (the bundled default)
/// this returns empty. A missing/unreadable `dir` degrades to empty rather than erroring.
/// `recipes.toml` wins on a name collision — the caller merges with `or_insert`.
pub fn load_patterns() -> Result<HashMap<String, Recipe>, String> {
    let text = read_user_or_default("recipes.toml", DEFAULT_RECIPES);
    let v: Value = toml::from_str(&text).map_err(|e| format!("recipes.toml parse error: {e}"))?;
    let Some(p) = v.get("patterns").and_then(Value::as_object) else {
        return Ok(HashMap::new());
    };
    let dir_raw = p.get("dir").and_then(Value::as_str).unwrap_or("");
    if dir_raw.is_empty() {
        return Ok(HashMap::new());
    }
    let default_inferencer = p.get("default_inferencer").and_then(Value::as_str).unwrap_or("").to_string();
    let dir = expand_tilde(dir_raw);
    let mut out = HashMap::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Ok(out); // missing dir → no patterns (not fatal)
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Ok(system) = std::fs::read_to_string(path.join("system.md")) else {
            continue;
        };
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        out.insert(
            name.to_string(),
            Recipe {
                inferencer: default_inferencer.clone(),
                system,
                tools: Vec::new(),
                params: None,
                source: PatternSource::Fabric,
                variables: HashMap::new(),
            },
        );
    }
    Ok(out)
}

/// The external conversation store (issue #2), from the top-level `conversationStore`
/// key in mcp.json. None ⇒ non-claude models stay stateless.
pub enum ConvStoreConfig {
    Mcp { server: String },
    Http { url: String },
}

pub fn load_conversation_store() -> Result<Option<ConvStoreConfig>, String> {
    let text = engine::expand_env(&read_user_or_default("mcp.json", DEFAULT_MCP));
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("mcp.json parse error: {e}"))?;
    match v.get("conversationStore") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(ConvStoreConfig::Mcp { server: s.clone() })),
        Some(Value::Object(o)) => match o.get("type").and_then(Value::as_str) {
            Some("mcp") => {
                let server = o
                    .get("server")
                    .and_then(Value::as_str)
                    .ok_or("conversationStore type 'mcp' needs a string 'server'")?
                    .to_string();
                Ok(Some(ConvStoreConfig::Mcp { server }))
            }
            Some("http") => {
                let url = o
                    .get("url")
                    .and_then(Value::as_str)
                    .ok_or("conversationStore type 'http' needs a string 'url'")?
                    .to_string();
                Ok(Some(ConvStoreConfig::Http { url }))
            }
            other => Err(format!("unknown conversationStore type {other:?} (expected 'mcp' or 'http')")),
        },
        Some(_) => Err("'conversationStore' must be a string or an object with a 'type'".to_string()),
    }
}

/// The managed/routed fabric backend (Part 2), from the top-level `fabric` key in mcp.json.
/// `None` ⇒ no fabric backend (the default). Lives in mcp.json (a server-owned config file,
/// like `conversationStore`) — NOT `[inferencers.*]`: fabric is not OpenAI-compatible, and the
/// engine's `inferencers.toml` loader requires every entry to have a `base_url` (it would
/// error on a fabric entry), plus the engine is parity-locked.
pub struct FabricConfig {
    /// woollama spawns + supervises `fabric --serve` (loopback) when true and no `url` is set.
    pub managed: bool,
    /// Route to an externally-run fabric at this base URL instead of spawning one.
    pub url: Option<String>,
    /// The fabric binary to spawn in managed mode (default `"fabric"`; resolved against PATH).
    pub command: String,
    /// Optional fixed `host:port` to bind in managed mode (default: a persisted free loopback port).
    pub address: Option<String>,
    /// Optional fallback inferencer (e.g. `"ollama/qwen3"`) for fabric patterns when a run
    /// omits `model` — fabric patterns have no bound inferencer. Required for a fabric pattern
    /// to be runnable as `woollama/<name>` via `/v1/chat/completions` (which has no model slot
    /// of its own). When unset, such patterns aren't advertised in `/v1/models`.
    pub default_model: Option<String>,
}

pub fn load_fabric_config() -> Result<Option<FabricConfig>, String> {
    let text = engine::expand_env(&read_user_or_default("mcp.json", DEFAULT_MCP));
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("mcp.json parse error: {e}"))?;
    match v.get("fabric") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(o)) => Ok(Some(FabricConfig {
            managed: o.get("managed").and_then(Value::as_bool).unwrap_or(false),
            url: o.get("url").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string),
            command: o.get("command").and_then(Value::as_str).filter(|s| !s.is_empty()).unwrap_or("fabric").to_string(),
            address: o.get("address").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string),
            default_model: o.get("default_model").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string),
        })),
        Some(_) => Err("'fabric' must be an object (e.g. {\"managed\": true} or {\"url\": \"...\"})".to_string()),
    }
}

/// Auth schemes whose bare form (scheme, no credential) means an unset `${VAR}` ate the secret.
/// Lowercase; compared case-insensitively per RFC 7235.
const BARE_AUTH_SCHEMES: &[&str] = &["bearer", "basic", "digest", "token"];

/// Reject a header value that carries no credential.
///
/// `engine::expand_env` resolves an unset `${VAR}` to the empty string
/// (`woollama-engine/src/lib.rs:417`), so `"Bearer ${SHELF_TOKEN}"` with `SHELF_TOKEN` unset
/// becomes the literal `"Bearer "` — a well-formed request carrying no credential. Against a
/// downstream that requires auth you get a 401 and notice; against a permissive one, or one whose
/// auth is enforced by a proxy that isn't deployed yet, you connect unauthenticated and everything
/// reports healthy. Fail closed here rather than in `expand_env`, which is parity-locked and also
/// feeds `inferencers.toml`.
///
/// NEVER include `value` in the returned message — these carry bearer tokens.
fn validate_header_value(server: &str, name: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!(
            "mcp.json: server '{server}' header '{name}' is empty — an unset ${{VAR}} expands to \
             nothing, so this would send no credential"
        ));
    }
    // Gated on the header NAME: a lone `basic` or `token` is a plausible real value for an
    // unrelated header (`X-Cache-Mode: basic`), and rejecting it would be both wrong and
    // misleadingly worded.
    let is_auth = name.eq_ignore_ascii_case("authorization") || name.eq_ignore_ascii_case("proxy-authorization");
    let mut parts = value.split_whitespace();
    let scheme = parts.next().unwrap_or_default();
    if is_auth && parts.next().is_none() && BARE_AUTH_SCHEMES.contains(&scheme.to_ascii_lowercase().as_str()) {
        return Err(format!(
            "mcp.json: server '{server}' header '{name}' is the bare auth scheme '{scheme}' with no \
             credential — an unset ${{VAR}} expands to nothing"
        ));
    }
    Ok(())
}

/// `load_mcp_servers`, but returning the per-server diagnostics it would otherwise log and
/// discard. For `woollamad check-config`, which exists because those diagnostics are otherwise
/// visible only in the boot log: a *connection* failure may self-heal on the next request, but a
/// malformed entry stays malformed until a human edits the file, and the only evidence it was
/// ever configured is a startup line nobody re-reads.
pub fn diagnose_mcp_servers() -> Result<McpConfigLoad, String> {
    parse_mcp_servers(&engine::expand_env(&read_user_or_default("mcp.json", DEFAULT_MCP)))
}

pub fn load_mcp_servers() -> Result<HashMap<String, McpServerSpec>, String> {
    let (specs, errors, warnings) =
        parse_mcp_servers(&engine::expand_env(&read_user_or_default("mcp.json", DEFAULT_MCP)))?;
    for e in errors {
        eprintln!("woollamad: {e}, skipping");
    }
    for w in warnings {
        eprintln!("woollamad: {w}");
    }
    Ok(specs)
}

/// Parse already-`${VAR}`-expanded mcp.json text into `(specs, per-server errors)`.
///
/// A malformed *entry* is skipped and reported, never fatal to its siblings — `build_state`
/// degrades a load error to an EMPTY registry, so a whole-file abort would mean the daemon comes
/// up "healthy" with zero MCP servers, silently dropping every unrelated stdio tool. The blast
/// radius of a bad entry is that entry. Only unparseable JSON (nothing is recoverable) is `Err`.
/// Mirrors `McpRegistry::connect`'s own logged-and-skipped posture for a server that won't start.
///
/// Split out from `load_mcp_servers` so tests exercise parsing without touching
/// `WOOLLAMA_CONFIG_DIR` — that env var is process-global, so a test that sets it races every
/// other test that does (see the `load_patterns` test, which owns it).
fn parse_mcp_servers(text: &str) -> Result<McpConfigLoad, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("mcp.json parse error: {e}"))?;
    let mut out = HashMap::new();
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    if let Some(servers) = v.get("mcpServers").and_then(Value::as_object) {
        for (name, s) in servers {
            let url = s.get("url").and_then(Value::as_str).filter(|u| !u.is_empty());
            let command = s.get("command").and_then(Value::as_str).filter(|c| !c.is_empty());
            let spec = match (command, url) {
                (Some(_), Some(_)) => Err(format!(
                    "mcp.json: server '{name}' sets both 'command' and 'url' — a server is \
                     either a stdio subprocess or an HTTP endpoint, not both"
                )),
                (None, None) => Err(format!(
                    "mcp.json: server '{name}' needs either 'command' (stdio) or 'url' (HTTP)"
                )),
                (Some(command), None) => parse_stdio(name, s, command).map(McpServerSpec::Stdio),
                (None, Some(url)) => parse_http(name, s, url).map(|(h, warn)| {
                    warnings.extend(warn);
                    McpServerSpec::Http(h)
                }),
            };
            match spec {
                Ok(spec) => {
                    out.insert(name.clone(), spec);
                }
                // Collected, not returned: one bad entry must not cost the operator every other
                // server in the file (see this function's doc comment).
                Err(e) => errors.push(e),
            }
        }
    }
    Ok((out, errors, warnings))
}

fn parse_stdio(name: &str, s: &Value, command: &str) -> Result<StdioSpec, String> {
    let args = s
        .get("args")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let mut env = HashMap::new();
    if let Some(o) = s.get("env").and_then(Value::as_object) {
        for (k, v) in o {
            // Erroring rather than dropping: a silently discarded `"PORT": 8080` starts a server
            // missing a var the operator believes they set — the same "works one way, misbehaves
            // the other, everything reports healthy" divergence the delegation test exists to
            // prevent. `headers` already errors on this shape; `env` now matches.
            let Some(v) = v.as_str() else {
                return Err(format!(
                    "mcp.json: server '{name}' env '{k}' must be a string (JSON numbers and \
                     booleans are not environment values — quote it)"
                ));
            };
            env.insert(k.clone(), v.to_string());
        }
    }
    Ok(StdioSpec { command: command.to_string(), args, env })
}

/// Returns the spec plus an optional operator warning — returned rather than printed so the
/// WIRING is testable. A warning that is merely `eprintln`'d can be silently disconnected from
/// its trigger, and the predicate's own unit test would still pass.
fn parse_http(name: &str, s: &Value, url: &str) -> Result<(HttpSpec, Option<String>), String> {
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
    // Credentials in cleartext on every request, forever. Loopback is exempt (it never leaves the
    // host); anything else carrying a header over plain http is worth saying out loud, since the
    // config that does it looks entirely healthy.
    let warning = (!headers.is_empty() && !is_encrypted_or_local(url)).then(|| {
        format!(
            "MCP server '{name}' sends {} header(s) over plaintext http — credentials will \
             cross the network in the clear; prefer https",
            headers.len()
        )
    });
    Ok((HttpSpec { url: url.to_string(), headers }, warning))
}

/// `https`, or a loopback host where plaintext never leaves the machine.
fn is_encrypted_or_local(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    if lower.starts_with("https://") {
        return true;
    }
    let host = lower.strip_prefix("http://").unwrap_or(&lower);
    let host = host.split(['/', '?', '#']).next().unwrap_or("");
    let host = host.rsplit_once('@').map(|(_, h)| h).unwrap_or(host);
    let host = host.strip_prefix('[').and_then(|h| h.split_once(']')).map(|(h, _)| h).unwrap_or_else(|| {
        host.split_once(':').map(|(h, _)| h).unwrap_or(host)
    });
    host == "localhost" || host == "::1" || host.starts_with("127.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn recipe(system: &str) -> Recipe {
        Recipe {
            inferencer: "ollama/qwen3".into(),
            system: system.into(),
            tools: vec![],
            params: None,
            source: PatternSource::Recipe,
            variables: HashMap::new(),
        }
    }

    #[test]
    fn render_substitutes_each_var_like_fabric() {
        // Byte-match fabric's `sysp.replace("{{"+k+"}}", str(v))` — plain string values.
        let r = recipe("You are a {{tone}} summarizer. Depth: {{depth}}.");
        let mut vars = serde_json::Map::new();
        vars.insert("tone".into(), json!("terse"));
        vars.insert("depth".into(), json!("ultra"));
        let out = r.render(&vars, None);
        assert_eq!(out.system, "You are a terse summarizer. Depth: ultra.");
    }

    #[test]
    fn render_leaves_unsupplied_tokens_verbatim_and_overrides_model() {
        let r = recipe("{{greeting}}, {{name}}!");
        let mut vars = serde_json::Map::new();
        vars.insert("greeting".into(), json!("Hi"));
        let out = r.render(&vars, Some("anthropic/claude-sonnet-4-6"));
        assert_eq!(out.system, "Hi, {{name}}!", "unsupplied token stays verbatim");
        assert_eq!(out.inferencer, "anthropic/claude-sonnet-4-6", "model_override replaces inferencer");
    }

    #[test]
    fn render_non_string_value_uses_json_rendering() {
        let r = recipe("n={{n}} on={{on}}");
        let mut vars = serde_json::Map::new();
        vars.insert("n".into(), json!(3));
        vars.insert("on".into(), json!(true));
        assert_eq!(r.render(&vars, None).system, "n=3 on=true");
    }

    #[test]
    fn scan_vars_finds_distinct_tokens_in_order_and_ignores_prose() {
        let vars = scan_vars("{{depth}} then {{language}} then {{depth}} and {{ not a var }} {{ok_1.2-x}}");
        assert_eq!(vars, vec!["depth", "language", "ok_1.2-x"]);
    }

    #[test]
    fn scan_vars_empty_when_none() {
        assert!(scan_vars("plain prompt, no tokens").is_empty());
    }

    #[test]
    fn parse_var_meta_reads_default_choices_description() {
        // toml → serde_json::Value, the same path load_recipes uses.
        let v: Value = toml::from_str(
            r#"
            [tone]
            default = "neutral"
            choices = ["neutral", "terse"]
            description = "Writing tone"
            [depth]
            choices = [1, 2, 3]
            "#,
        )
        .unwrap();
        let m = parse_var_meta(Some(&v));
        let tone = m.get("tone").unwrap();
        assert_eq!(tone.default, Some(json!("neutral")));
        assert_eq!(tone.choices, Some(vec![json!("neutral"), json!("terse")]));
        assert_eq!(tone.description.as_deref(), Some("Writing tone"));
        let depth = m.get("depth").unwrap();
        assert_eq!(depth.default, None); // absent → None (no `null` noise downstream)
        assert_eq!(depth.choices, Some(vec![json!(1), json!(2), json!(3)]));
        // Malformed / missing overlay degrades to empty, never panics.
        assert!(parse_var_meta(None).is_empty());
        assert!(parse_var_meta(Some(&json!("not a table"))).is_empty());
    }

    #[test]
    fn apply_defaults_fills_only_unsupplied_with_a_configured_default() {
        let mut r = recipe("You are a {{tone}} {{role}} about {{topic}}.");
        r.variables.insert("tone".into(), VarMeta { default: Some(json!("neutral")), ..Default::default() });
        r.variables.insert("role".into(), VarMeta { default: Some(json!("assistant")), ..Default::default() });
        // `topic` has metadata but NO default → never auto-filled.
        r.variables.insert("topic".into(), VarMeta { description: Some("subject".into()), ..Default::default() });

        let mut supplied = serde_json::Map::new();
        supplied.insert("tone".into(), json!("wry")); // caller value must win over the default
        let merged = r.apply_defaults(&supplied);

        assert_eq!(merged.get("tone"), Some(&json!("wry")), "caller-supplied wins");
        assert_eq!(merged.get("role"), Some(&json!("assistant")), "unsupplied default filled");
        assert_eq!(merged.get("topic"), None, "no default ⇒ left unset (render keeps it verbatim)");
    }

    #[test]
    fn load_patterns_scans_dir_and_marks_fabric_source() {
        // Isolated temp tree: a config dir with a recipes.toml [patterns] block pointing at
        // a patterns dir holding one `<name>/system.md`.
        let base = std::env::temp_dir().join("woollama-load-patterns-test");
        let _ = std::fs::remove_dir_all(&base);
        let cfg = base.join("config");
        let pats = base.join("patterns");
        std::fs::create_dir_all(cfg.join("x")).unwrap();
        std::fs::create_dir_all(pats.join("scribe-summarize")).unwrap();
        std::fs::write(pats.join("scribe-summarize").join("system.md"), "Summarize {{depth}}.").unwrap();
        // a non-dir and a dir without system.md → both skipped
        std::fs::write(pats.join("loose.txt"), "ignore me").unwrap();
        std::fs::create_dir_all(pats.join("empty-pattern")).unwrap();
        std::fs::write(
            cfg.join("recipes.toml"),
            format!("[patterns]\ndir = \"{}\"\ndefault_inferencer = \"ollama/qwen3:14b-iq4xs\"\n", pats.display()),
        )
        .unwrap();

        std::env::set_var("WOOLLAMA_CONFIG_DIR", &cfg);
        let out = load_patterns().unwrap();
        std::env::remove_var("WOOLLAMA_CONFIG_DIR");

        assert_eq!(out.len(), 1, "only the dir with a system.md is a pattern");
        let r = out.get("scribe-summarize").expect("pattern discovered");
        assert_eq!(r.system, "Summarize {{depth}}.");
        assert_eq!(r.inferencer, "ollama/qwen3:14b-iq4xs");
        assert_eq!(r.source, PatternSource::Fabric);
        let _ = std::fs::remove_dir_all(&base);

        // Case 2 (SAME test fn — `WOOLLAMA_CONFIG_DIR` is process-global, so env-mutating
        // cases must run sequentially, not as parallel `#[test]`s; see orchestrate.rs).
        // No [patterns] block (the bundled default) → opt-out by default.
        let none = std::env::temp_dir().join("woollama-load-patterns-none");
        let _ = std::fs::remove_dir_all(&none);
        std::fs::create_dir_all(&none).unwrap();
        std::fs::write(none.join("recipes.toml"), "[recipes.hello]\ninferencer = \"ollama/x\"\nsystem = \"hi\"\n").unwrap();
        std::env::set_var("WOOLLAMA_CONFIG_DIR", &none);
        let out = load_patterns().unwrap();
        std::env::remove_var("WOOLLAMA_CONFIG_DIR");
        assert!(out.is_empty(), "no [patterns] block → no patterns");
        let _ = std::fs::remove_dir_all(&none);
    }

    /// Parse mcp.json text through the same `${VAR}` expansion `load_mcp_servers` applies,
    /// but WITHOUT `WOOLLAMA_CONFIG_DIR` — that var is process-global, so setting it here
    /// would race the `load_patterns` test that owns it.
    fn servers_from(json: &str) -> Result<HashMap<String, McpServerSpec>, String> {
        let (specs, errors, _warnings) = parse_mcp_servers(&engine::expand_env(json))?;
        // Tests that expect a rejection assert on the per-server error; a bad entry is skipped
        // rather than fatal (see `one_bad_entry_does_not_discard_the_healthy_servers`).
        match errors.into_iter().next() {
            Some(e) => Err(e),
            None => Ok(specs),
        }
    }

    #[test]
    fn one_bad_entry_does_not_discard_the_healthy_servers() {
        // The blast radius of a bad entry must be that ENTRY, not the file. `build_state` maps a
        // load error to an empty registry, so a whole-file abort means the daemon comes up
        // "healthy" with zero MCP servers — every pre-existing stdio tool gone and every recipe
        // referencing them failing. That is a far worse failure than the misconfiguration.
        let (specs, errors, _warnings) = parse_mcp_servers(&engine::expand_env(
            r#"{"mcpServers": {
                 "good": {"command": "hi"},
                 "bad": {"url": "http://h/mcp", "headers": {"Authorization": "Bearer "}}
               }}"#,
        ))
        .unwrap();
        assert!(specs.contains_key("good"), "a healthy server must survive a sibling's error");
        assert!(!specs.contains_key("bad"), "the invalid server must be skipped");
        assert_eq!(errors.len(), 1, "the operator must still be told, per server: {errors:?}");
        assert!(errors[0].contains("bad") && errors[0].contains("Authorization"), "{errors:?}");
    }

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
    fn a_server_must_be_exactly_one_of_stdio_or_http() {
        let err = servers_from(r#"{"mcpServers": {"broken": {"args": ["x"]}}}"#).unwrap_err();
        assert!(err.contains("broken") && err.contains("url"), "must name the server and both forms: {err}");

        // Ambiguous rather than merely redundant: silently preferring one would make the other
        // half of the operator's config a lie.
        let err = servers_from(r#"{"mcpServers": {"both": {"command": "x", "url": "http://h/mcp"}}}"#).unwrap_err();
        assert!(err.contains("both"), "must name the server: {err}");
    }

    #[test]
    fn an_unset_token_in_a_header_fails_the_load() {
        // End-to-end version of the fail-open: config text -> expand_env -> validation. The var
        // name is test-specific and never set, so this does not depend on ambient environment.
        let err = servers_from(
            r#"{"mcpServers": {"shelf": {"url": "http://h/mcp",
                 "headers": {"Authorization": "Bearer ${WOOLLAMA_TEST_ABSENT_TOKEN_X9}"}}}}"#,
        )
        .unwrap_err();
        assert!(err.contains("Authorization"), "the unset-var case must fail the load: {err}");
    }

    #[test]
    fn header_values_that_carry_no_credential_are_rejected() {
        // engine::expand_env resolves an UNSET ${VAR} to "" (lib.rs:417, unwrap_or_default), so
        // a header sourced from a missing env var arrives here empty or as a bare scheme. Fail
        // closed: a silently credential-less request that a permissive downstream accepts is the
        // worst outcome, because everything reports healthy.
        let err = validate_header_value("shelf", "X-Api-Key", "").unwrap_err();
        assert!(err.contains("shelf") && err.contains("X-Api-Key"), "must name server and header: {err}");

        // "Bearer ${SHELF_TOKEN}" with SHELF_TOKEN unset expands to the literal "Bearer ".
        let err = validate_header_value("shelf", "Authorization", "Bearer ").unwrap_err();
        assert!(err.contains("Authorization"), "must name the header: {err}");
        assert!(validate_header_value("shelf", "Authorization", "Basic").is_err());

        // Error text reaches logs and the operator's terminal; header values carry bearer tokens.
        assert!(!err.contains("Bearer "), "must not echo the value: {err}");
    }

    #[test]
    fn populated_header_values_are_accepted() {
        assert!(validate_header_value("shelf", "Authorization", "Bearer sk-live-abc").is_ok());
        // A single-token value is legitimate when it isn't a bare auth scheme.
        assert!(validate_header_value("shelf", "X-Api-Key", "abc123").is_ok());
        // The bare-scheme rule is gated on the header NAME: `basic` is a plausible real value for
        // an unrelated header, and rejecting it would be wrong AND misleadingly worded.
        assert!(validate_header_value("shelf", "X-Cache-Mode", "basic").is_ok());
        assert!(validate_header_value("shelf", "Proxy-Authorization", "Basic").is_err());
    }

    #[test]
    fn secrets_never_reach_a_debug_rendering() {
        // Both specs hold credentials — `env` is a documented home for a provider key, `headers`
        // for a bearer, and a URL query string can carry a token. A derived Debug on any of them
        // puts the secret into every `{:?}`, log line, and panic message.
        let mut env = HashMap::new();
        env.insert("ANTHROPIC_API_KEY".to_string(), "sk-secret-stdio".to_string());
        let stdio = format!("{:?}", StdioSpec { command: "x".into(), args: vec![], env });
        assert!(!stdio.contains("sk-secret-stdio"), "env value leaked into Debug: {stdio}");
        assert!(stdio.contains("ANTHROPIC_API_KEY"), "the NAME is what makes it debuggable: {stdio}");

        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer sk-secret-http".to_string());
        let http = format!("{:?}", HttpSpec { url: "https://h/mcp?token=sk-secret-query".into(), headers });
        assert!(!http.contains("sk-secret-http"), "header value leaked into Debug: {http}");
        assert!(!http.contains("sk-secret-query"), "url query credential leaked into Debug: {http}");
    }

    #[test]
    fn a_non_string_env_value_is_an_error_not_a_silent_drop() {
        // Dropping it starts a server missing a var the operator believes they set — the same
        // silent divergence `headers` already errors on.
        let err = servers_from(r#"{"mcpServers": {"s": {"command": "x", "env": {"PORT": 8080}}}}"#).unwrap_err();
        assert!(err.contains("PORT") && err.contains("string"), "{err}");
    }

    #[test]
    fn a_plaintext_credential_warns_and_still_loads() {
        // Exercises the WIRING, not just the predicate: a warning that is disconnected from its
        // trigger would still pass `plaintext_detection_exempts_loopback_only` below.
        let (specs, errors, warnings) = parse_mcp_servers(&engine::expand_env(
            r#"{"mcpServers": {"suite": {"url": "http://mcp.dobby.lan:9200/mcp",
                 "headers": {"Authorization": "Bearer sk-cleartext"}}}}"#,
        ))
        .unwrap();
        assert!(specs.contains_key("suite"), "a plaintext credential warns — it does not block");
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("suite") && warnings[0].contains("plaintext"), "{warnings:?}");
        assert!(!warnings[0].contains("sk-cleartext"), "the warning must not echo the credential");
    }

    #[test]
    fn https_and_loopback_and_headerless_do_not_warn() {
        let quiet = |json: &str| {
            let (_, errors, warnings) = parse_mcp_servers(&engine::expand_env(json)).unwrap();
            assert!(errors.is_empty(), "{errors:?}");
            assert!(warnings.is_empty(), "should not warn: {warnings:?}");
        };
        quiet(r#"{"mcpServers": {"a": {"url": "https://h/mcp", "headers": {"Authorization": "Bearer x"}}}}"#);
        quiet(r#"{"mcpServers": {"b": {"url": "http://127.0.0.1:9200/mcp", "headers": {"Authorization": "Bearer x"}}}}"#);
        // No credential to expose ⇒ nothing to warn about, even over plaintext.
        quiet(r#"{"mcpServers": {"c": {"url": "http://mcp.dobby.lan:9200/mcp"}}}"#);
    }

    #[test]
    fn plaintext_detection_exempts_loopback_only() {
        assert!(is_encrypted_or_local("https://mcp.example.lan:9200/mcp"));
        assert!(is_encrypted_or_local("http://127.0.0.1:9200/mcp"));
        assert!(is_encrypted_or_local("http://localhost/mcp"));
        assert!(is_encrypted_or_local("http://[::1]:9200/mcp"));
        // The case that must warn: a real network hop carrying a credential in the clear.
        assert!(!is_encrypted_or_local("http://mcp.dobby.lan:9200/mcp"));
        assert!(!is_encrypted_or_local("http://10.0.0.5/mcp"));
    }

    #[test]
    fn mcp_server_env_block_is_parsed() {
        // The Python reference parses this key (config.py:136) and hands it to the spawned
        // server via StdioServerParameters.env (manager.py:89). woollamad dropped it in the
        // Rust port, so a documented key was silently ignored; this pins it restored.
        let servers = servers_from(
            r#"{"mcpServers": {"git": {"command": "git-mcp", "env": {"GIT_AUTHOR_NAME": "woollama"}}}}"#,
        )
        .unwrap();
        let McpServerSpec::Stdio(s) = &servers["git"] else { panic!("expected a Stdio spec") };
        assert_eq!(s.env.get("GIT_AUTHOR_NAME").map(String::as_str), Some("woollama"));

        // Case 2 (SAME test fn — `WOOLLAMA_CONFIG_DIR` is process-global): an absent `env` is
        // an empty map, not an error.
        let servers = servers_from(r#"{"mcpServers": {"hello": {"command": "hi"}}}"#).unwrap();
        let McpServerSpec::Stdio(s) = &servers["hello"] else { panic!("expected a Stdio spec") };
        assert!(s.env.is_empty(), "absent 'env' is an empty map, not an error");
    }
}
