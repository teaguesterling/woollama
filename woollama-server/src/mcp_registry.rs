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
}

impl McpRegistry {
    /// Connect to every configured server (best-effort: one that fails to start is
    /// logged and skipped, so a single bad server doesn't take the whole router down).
    pub async fn connect(specs: HashMap<String, McpServerSpec>) -> McpRegistry {
        let mut servers = HashMap::new();
        for (name, spec) in specs {
            match Self::connect_one(&spec).await {
                Ok(conn) => {
                    servers.insert(name.clone(), conn);
                }
                Err(e) => {
                    eprintln!("woollamad: MCP server '{name}' failed to start, skipping: {e}");
                }
            }
        }
        McpRegistry { servers }
    }

    async fn connect_one(spec: &McpServerSpec) -> Result<ServerConn, String> {
        let mut cmd = tokio::process::Command::new(&spec.command);
        cmd.args(&spec.args);
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

    /// Resolve a namespaced `<server>.<tool>` to (server peer, bare tool name).
    fn resolve(&self, namespaced: &str) -> Option<(Peer<RoleClient>, String)> {
        let (server, bare) = namespaced.split_once('.')?;
        let conn = self.servers.get(server)?;
        conn.tools.iter().find(|t| t.name == bare)?;
        Some((conn.peer.clone(), bare.to_string()))
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
                    format!("{server}.{}", t.name),
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
        conn.peer.call_tool(params).await.map_err(|e| e.to_string())
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
        peer.call_tool(params).await.map_err(|e| e.to_string())
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
            // The namespaced name flows out, so the model emits tool_calls we can route.
            out.push(json!({
                "type": "function",
                "function": {
                    "name": namespaced,
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
        match peer.call_tool(params).await {
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
