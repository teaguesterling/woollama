//! A tiny stdio MCP server used as a test fixture (CARGO_BIN_EXE_mcp_fixture). Serves
//! one tool, `count_to(n) -> "counted to n"`, so the orchestration integration test has
//! a real downstream MCP server to spawn — no Python / external deps.

use std::sync::Arc;

use rmcp::handler::server::ServerHandler;
use rmcp::model::*;
use rmcp::service::{RequestContext, RoleServer};
use rmcp::{serve_server, ErrorData as McpError};
use serde_json::{json, Value};

#[derive(Clone)]
struct Fixture;

impl ServerHandler for Fixture {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::default()
    }
    async fn list_tools(
        &self,
        _p: Option<PaginatedRequestParams>,
        _c: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let schema: Arc<JsonObject> = Arc::new(
            serde_json::from_value(json!({
                "type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]
            }))
            .unwrap(),
        );
        Ok(ListToolsResult::with_all_items(vec![Tool::new("count_to", "count to n", schema)]))
    }
    async fn call_tool(
        &self,
        req: CallToolRequestParams,
        _c: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let n = req
            .arguments
            .as_ref()
            .and_then(|a| a.get("n"))
            .and_then(Value::as_i64)
            .unwrap_or(0);
        Ok(CallToolResult::success(vec![Content::text(format!("counted to {n}"))]))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let running = serve_server(Fixture, (tokio::io::stdin(), tokio::io::stdout())).await?;
    running.waiting().await?;
    Ok(())
}
