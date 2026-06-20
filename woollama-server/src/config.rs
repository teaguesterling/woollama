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
        let mut system = self.system.clone();
        for (k, v) in variables {
            let rep = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            system = system.replace(&format!("{{{{{k}}}}}"), &rep);
        }
        Recipe {
            inferencer: model_override.map(String::from).unwrap_or_else(|| self.inferencer.clone()),
            system,
            tools: self.tools.clone(),
            params: self.params.clone(),
            source: self.source,
        }
    }
}

/// The variable names a pattern exposes — every distinct `{{name}}` token scanned from a
/// system prompt, in first-seen order. A name is accepted only if it is non-empty and made
/// of identifier chars (`[A-Za-z0-9_.-]`), so prose like `{{ not a var }}` is ignored.
/// fabric patterns carry no variable metadata, so names are all `/w1/patterns` can honestly
/// surface (no defaults/choices — that's a later optional overlay).
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

/// A downstream MCP server to spawn (stdio). Matches Claude Code's mcp.json shape.
#[derive(Clone)]
pub struct McpServerSpec {
    pub command: String,
    pub args: Vec<String>,
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
                },
            );
        }
    }
    Ok(out)
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
        })),
        Some(_) => Err("'fabric' must be an object (e.g. {\"managed\": true} or {\"url\": \"...\"})".to_string()),
    }
}

pub fn load_mcp_servers() -> Result<HashMap<String, McpServerSpec>, String> {
    let text = engine::expand_env(&read_user_or_default("mcp.json", DEFAULT_MCP));
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("mcp.json parse error: {e}"))?;
    let mut out = HashMap::new();
    if let Some(servers) = v.get("mcpServers").and_then(Value::as_object) {
        for (name, s) in servers {
            let command = s
                .get("command")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("mcp.json: server '{name}' is missing 'command'"))?
                .to_string();
            let args = s
                .get("args")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            out.insert(name.clone(), McpServerSpec { command, args });
        }
    }
    Ok(out)
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
}
