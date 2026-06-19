//! The downstream MCP registry — one rmcp **client** per configured server (spawned as
//! a child process over stdio), and the `RegistryToolProvider` that adapts it to the
//! engine's `ToolProvider` seam so the recipe loop can dispatch MCP tools.
//!
//! Mirrors Python `manager.Registry` / `RegistryToolProvider`. (The asyncio
//! queue-marshaling workaround the Python version needs doesn't apply here: rmcp's
//! `Peer` is a cheap, Send+Sync, clonable handle.)

use std::collections::HashMap;
use std::sync::Arc;

use rmcp::model::{CallToolRequestParams, CallToolResult, Tool};
use rmcp::service::RoleClient;
use rmcp::transport::TokioChildProcess;
use rmcp::{Peer, ServiceExt};
use serde_json::{json, Value};

use woollama_engine::{EngineError, ToolProvider};

use crate::config::McpServerSpec;

struct ServerConn {
    peer: Peer<RoleClient>,
    tools: Vec<Tool>,
}

/// All configured downstream MCP servers, connected and tool-listed.
pub struct McpRegistry {
    servers: HashMap<String, ServerConn>,
    /// Reverse map: advertised wire name (`mcp__server__tool`) -> (server, bare tool). Built
    /// once at connect so dispatch resolves the model's tool_call name unambiguously (no
    /// dot-splitting, and it works for the hashed >64-char fallback too).
    wire_index: HashMap<String, (String, String)>,
}

/// Allow-listed env for a spawned MCP server. Shares `claude_code::CHILD_ENV_ALLOW` (single
/// source of truth) so the downstream-server scrub can't drift from the claude-code one —
/// provider secrets in the daemon env (ANTHROPIC_API_KEY etc.) never reach a tool server.
fn scrubbed_env() -> HashMap<String, String> {
    std::env::vars()
        .filter(|(k, _)| crate::claude_code::CHILD_ENV_ALLOW.contains(&k.as_str()) || k.starts_with("LC_"))
        .collect()
}

/// Per-server connect timeout (handshake + initial tools/list). `WOOLLAMA_MCP_CONNECT_TIMEOUT_SECS`,
/// default 30s. Bounds startup so a hung downstream server can't wedge the daemon.
fn connect_timeout() -> std::time::Duration {
    let secs = std::env::var("WOOLLAMA_MCP_CONNECT_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30);
    std::time::Duration::from_secs(secs)
}

/// Map (server, tool) to a wire-safe tool name: `mcp__<server>__<tool>` — the same scheme
/// claude-code uses, and valid OpenAI/MCP function-name grammar (`[A-Za-z0-9_-]{1,64}`).
/// A dotted `server.tool` is rejected by strict OpenAI-compatible inferencers; a name that
/// would exceed 64 chars falls back to a deterministic hash (resolved via the reverse map).
fn wire_name(server: &str, tool: &str) -> String {
    let full = format!("mcp__{server}__{tool}");
    if full.len() <= 64 {
        full
    } else {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        full.hash(&mut h);
        format!("mcp__{:016x}", h.finish())
    }
}

/// Per-call timeout for a downstream tool invocation. `WOOLLAMA_MCP_CALL_TIMEOUT_SECS`,
/// default 120s. A hung downstream tool fails the one request instead of hanging it (and
/// leaking the request task) forever.
fn call_timeout() -> std::time::Duration {
    let secs = std::env::var("WOOLLAMA_MCP_CALL_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(120);
    std::time::Duration::from_secs(secs)
}

/// Invoke a downstream tool with the per-call timeout applied.
async fn call_with_timeout(peer: &Peer<RoleClient>, params: CallToolRequestParams) -> Result<CallToolResult, String> {
    let dur = call_timeout();
    match tokio::time::timeout(dur, peer.call_tool(params)).await {
        Ok(Ok(res)) => Ok(res),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err(format!("downstream tool call timed out after {}s", dur.as_secs())),
    }
}

impl McpRegistry {
    /// Connect to every configured server CONCURRENTLY, each bounded by a per-server timeout
    /// (best-effort: a server that fails to start OR hangs on the handshake is logged and
    /// skipped, so a single bad/slow server can neither take the router down nor block its
    /// startup). The timeout is `WOOLLAMA_MCP_CONNECT_TIMEOUT_SECS` (default 30s).
    pub async fn connect(specs: HashMap<String, McpServerSpec>) -> McpRegistry {
        let timeout = connect_timeout();
        let results = futures::future::join_all(
            specs
                .into_iter()
                .map(|(name, spec)| async move { (name, tokio::time::timeout(timeout, Self::connect_one(&spec)).await) }),
        )
        .await;
        let mut servers = HashMap::new();
        for (name, res) in results {
            match res {
                Ok(Ok(conn)) => {
                    servers.insert(name, conn);
                }
                Ok(Err(e)) => eprintln!("woollamad: MCP server '{name}' failed to start, skipping: {e}"),
                Err(_) => eprintln!(
                    "woollamad: MCP server '{name}' timed out after {}s connecting, skipping",
                    timeout.as_secs()
                ),
            }
        }
        let mut wire_index = HashMap::new();
        for (server, conn) in &servers {
            for t in &conn.tools {
                wire_index.insert(wire_name(server, &t.name), (server.clone(), t.name.to_string()));
            }
        }
        McpRegistry { servers, wire_index }
    }

    async fn connect_one(spec: &McpServerSpec) -> Result<ServerConn, String> {
        let mut cmd = tokio::process::Command::new(&spec.command);
        // Scrub the child env: a downstream tool server must NOT inherit the daemon's
        // provider secrets (ANTHROPIC_API_KEY etc.). Mirrors the claude-code child scrub
        // and the Python MCP SDK's default-scrubbed stdio environment.
        cmd.args(&spec.args).env_clear().envs(scrubbed_env());
        let transport = TokioChildProcess::new(cmd).map_err(|e| e.to_string())?;
        let running = ().serve(transport).await.map_err(|e| e.to_string())?;
        let peer = running.peer().clone();
        let tools = peer.list_all_tools().await.map_err(|e| e.to_string())?;
        // Keep the connection alive for the process lifetime: dropping the
        // RunningService would cancel its task and close the child. The router holds
        // these for as long as it runs, so leaking the handle is intentional here.
        // (A graceful-shutdown lifecycle can replace this later.)
        std::mem::forget(running);
        Ok(ServerConn { peer, tools })
    }

    /// Resolve an advertised wire name (`mcp__server__tool`) to (server peer, bare tool) via
    /// the reverse map built at connect — unambiguous, unlike splitting on a separator.
    fn resolve(&self, wire: &str) -> Option<(Peer<RoleClient>, String)> {
        let (server, bare) = self.wire_index.get(wire)?;
        let conn = self.servers.get(server)?;
        Some((conn.peer.clone(), bare.clone()))
    }

    fn tool(&self, namespaced: &str) -> Option<&Tool> {
        let (server, bare) = namespaced.split_once('.')?;
        self.servers.get(server)?.tools.iter().find(|t| t.name == bare)
    }

    /// Every downstream tool, re-exported namespaced `<server>.<tool>` with input +
    /// output schema MIRRORED — for woollama's own tools/list (the MCP aggregator).
    pub fn reexport_tools(&self) -> Vec<Tool> {
        let mut out = Vec::new();
        for (server, conn) in &self.servers {
            for t in &conn.tools {
                let mut nt = Tool::new(
                    wire_name(server, &t.name),
                    t.description.clone().unwrap_or_default(),
                    t.input_schema.clone(),
                );
                if let Some(os) = t.output_schema.clone() {
                    nt = nt.with_raw_output_schema(os);
                }
                out.push(nt);
            }
        }
        out
    }

    /// Call a tool by BARE name on a specific server (for the MCP conversation-store
    /// provider, whose tools — create_thread/etc. — aren't recipe-namespaced).
    pub async fn call_server(&self, server: &str, tool: &str, args: &Value) -> Result<CallToolResult, String> {
        let conn = self.servers.get(server).ok_or_else(|| format!("unknown server '{server}'"))?;
        let mut params = CallToolRequestParams::new(tool.to_string());
        if let Some(obj) = args.as_object() {
            params = params.with_arguments(obj.clone());
        }
        call_with_timeout(&conn.peer, params).await
    }

    /// Dispatch a namespaced tool and return the RAW `CallToolResult` (content +
    /// structured_content), for the MCP proxy passthrough (vs. the lossy text render
    /// `RegistryToolProvider::dispatch` does for the inference loop).
    pub async fn call_raw(&self, namespaced: &str, args: &Value) -> Result<CallToolResult, String> {
        let Some((peer, bare)) = self.resolve(namespaced) else {
            return Err(format!("unknown tool '{namespaced}'"));
        };
        let mut params = CallToolRequestParams::new(bare);
        if let Some(obj) = args.as_object() {
            params = params.with_arguments(obj.clone());
        }
        call_with_timeout(&peer, params).await
    }
}

/// Adapts an `McpRegistry` to the engine's `ToolProvider` seam.
pub struct RegistryToolProvider {
    pub reg: Arc<McpRegistry>,
}

#[async_trait::async_trait]
impl ToolProvider for RegistryToolProvider {
    fn tool_schemas(&self, allow: &[String]) -> Result<Vec<Value>, EngineError> {
        let mut out = Vec::new();
        for namespaced in allow {
            let Some(tool) = self.reg.tool(namespaced) else {
                eprintln!("woollamad: recipe references unknown tool '{namespaced}', skipping");
                continue;
            };
            // Recipe config is human-friendly `server.tool`; advertise the wire-safe
            // `mcp__server__tool` so the model emits a name we resolve via the reverse map.
            let Some((server, bare)) = namespaced.split_once('.') else { continue };
            out.push(json!({
                "type": "function",
                "function": {
                    "name": wire_name(server, bare),
                    "description": tool.description.as_deref().unwrap_or(""),
                    "parameters": Value::Object((*tool.input_schema).clone()),
                },
            }));
        }
        Ok(out)
    }

    async fn dispatch(&self, name: &str, args: &Value) -> (String, bool) {
        let Some((peer, bare)) = self.reg.resolve(name) else {
            return (format!("ERROR: unknown tool '{name}'"), false);
        };
        let mut params = CallToolRequestParams::new(bare);
        if let Some(obj) = args.as_object() {
            params = params.with_arguments(obj.clone());
        }
        match call_with_timeout(&peer, params).await {
            Ok(res) => render_result(&res),
            Err(e) => (format!("ERROR: {e}"), false),
        }
    }
}

/// Render a downstream `CallToolResult` to the `(content, ok)` the loop feeds back:
/// joined text blocks, else the structured payload as JSON; `is_error` → `ok=false`.
fn render_result(res: &CallToolResult) -> (String, bool) {
    let is_error = res.is_error.unwrap_or(false);
    let text: Vec<String> =
        res.content.iter().filter_map(|c| c.as_text().map(|t| t.text.clone())).collect();
    let mut body = if !text.is_empty() {
        text.join("\n")
    } else if let Some(sc) = &res.structured_content {
        serde_json::to_string(sc).unwrap_or_default()
    } else {
        String::new()
    };
    if is_error {
        body = if body.is_empty() { "[tool error]".to_string() } else { format!("[tool error] {body}") };
    }
    (body, !is_error)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(unix)]
    fn scrubbed_env_excludes_provider_secrets() {
        std::env::set_var("ANTHROPIC_API_KEY", "leak-me");
        std::env::set_var("OPENAI_API_KEY", "leak-me-2");
        let env = scrubbed_env();
        assert!(!env.contains_key("ANTHROPIC_API_KEY"), "provider key must not reach MCP servers");
        assert!(!env.contains_key("OPENAI_API_KEY"));
        if std::env::var_os("PATH").is_some() {
            assert!(env.contains_key("PATH"), "PATH must survive so the server interpreter resolves");
        }
        for k in env.keys() {
            assert!(
                crate::claude_code::CHILD_ENV_ALLOW.contains(&k.as_str()) || k.starts_with("LC_"),
                "leaked non-allow-listed var to an MCP server: {k}"
            );
        }
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn hung_server_does_not_block_startup() {
        std::env::set_var("WOOLLAMA_MCP_CONNECT_TIMEOUT_SECS", "1");
        let mut specs = HashMap::new();
        // `sleep` spawns but never speaks MCP -> the initialize handshake hangs -> timed out.
        specs.insert("hung".to_string(), McpServerSpec { command: "sleep".into(), args: vec!["30".into()] });
        // `false` exits immediately -> connect_one errors -> skipped.
        specs.insert("dead".to_string(), McpServerSpec { command: "false".into(), args: vec![] });
        let start = std::time::Instant::now();
        let reg = McpRegistry::connect(specs).await;
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "a hung downstream server must not block startup (took {elapsed:?})"
        );
        assert!(reg.servers.is_empty(), "hung + dead servers are skipped, not registered");
        std::env::remove_var("WOOLLAMA_MCP_CONNECT_TIMEOUT_SECS");
    }

    #[test]
    fn wire_name_is_valid_and_namespaced() {
        let w = wire_name("hello", "count_to");
        assert_eq!(w, "mcp__hello__count_to");
        let ok = |s: &str| s.len() <= 64 && !s.is_empty()
            && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        assert!(ok(&w), "must satisfy the OpenAI/MCP function-name grammar (no dots, <=64)");
        // An overlong combination falls back to a hashed, still-valid name.
        let long = wire_name(&"s".repeat(50), &"t".repeat(50));
        assert!(ok(&long) && long.starts_with("mcp__"), "overlong name must hash to a valid form");
    }

    #[test]
    fn call_timeout_honors_env_and_default() {
        std::env::remove_var("WOOLLAMA_MCP_CALL_TIMEOUT_SECS");
        assert_eq!(call_timeout().as_secs(), 120, "default per-call timeout");
        std::env::set_var("WOOLLAMA_MCP_CALL_TIMEOUT_SECS", "7");
        assert_eq!(call_timeout().as_secs(), 7, "env override");
        std::env::remove_var("WOOLLAMA_MCP_CALL_TIMEOUT_SECS");
    }
}
