//! Recipe + MCP-server config loading — ported from Python `woollama.config`.
//!
//! A user file in `config_dir()` is used if present, else the bundled default
//! (embedded from the Python package's `defaults/` so the two stay in sync). `${VAR}`
//! in mcp.json is expanded from the environment at load time.

use std::collections::HashMap;

use serde_json::Value;
use woollama_engine as engine;

const DEFAULT_RECIPES: &str = include_str!("../../src/woollama/defaults/recipes.toml");
const DEFAULT_MCP: &str = include_str!("../../src/woollama/defaults/mcp.json");

/// A composed recipe: a system prompt + an inferencer + an allow-list of namespaced
/// `<server>.<tool>` names (+ optional per-recipe inference params).
#[derive(Clone)]
pub struct Recipe {
    pub inferencer: String,
    pub system: String,
    pub tools: Vec<String>,
    pub params: Option<Value>,
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
                },
            );
        }
    }
    Ok(out)
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
