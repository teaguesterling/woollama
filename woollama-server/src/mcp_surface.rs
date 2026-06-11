//! woollama AS an MCP server (slice 4b) — the outbound MCP surface, mirroring Python
//! `mcp_server.py`. An MCP client connecting to woollama sees:
//!   - the `chat` tool   — runs a recipe end-to-end, returns only the final answer.
//!   - re-exported tools — every downstream server's tools, namespaced `<server>.<tool>`,
//!                         with input + output schema mirrored, proxied to the registry.
//!   - recipe prompts    — one per recipe; get returns its system message.
//!
//! Served two ways from the SAME handler: stdio (`woollama-server mcp`) and a
//! Streamable-HTTP mount at `/mcp` (shared port). The `WoollamaMcp` handler holds an
//! `Arc<AppState>`, so the per-session factory shares the one downstream registry.

use std::sync::Arc;

use rmcp::handler::server::ServerHandler;
use rmcp::model::*;
use rmcp::service::{RequestContext, RoleServer};
use rmcp::ErrorData as McpError;
use serde_json::{json, Value};

use crate::AppState;

#[derive(Clone)]
pub struct WoollamaMcp {
    pub state: Arc<AppState>,
}

fn chat_tool() -> Tool {
    let schema: Arc<JsonObject> = Arc::new(
        serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "messages": {"type": "array", "description": "OpenAI-shaped chat messages"},
                "recipe": {"type": "string", "description": "recipe name to run"},
                "model": {"type": "string", "description": "optional woollama/<recipe> form"}
            },
            "required": ["messages"]
        }))
        .unwrap(),
    );
    Tool::new(
        "chat",
        "Run a woollama recipe end-to-end and return the final assistant message \
         (the inferencer<->tool loop stays hidden).",
        schema,
    )
}

impl WoollamaMcp {
    async fn run_chat(&self, arguments: Option<JsonObject>) -> Result<CallToolResult, McpError> {
        let args = arguments.unwrap_or_default();
        let recipe_name = args
            .get("recipe")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .or_else(|| {
                args.get("model")
                    .and_then(Value::as_str)
                    .and_then(|m| m.strip_prefix("woollama/"))
                    .map(String::from)
            })
            .ok_or_else(|| {
                McpError::invalid_params("chat requires a 'recipe' (or 'woollama/<recipe>' model)", None)
            })?;
        let Some(recipe) = self.state.recipes.get(&recipe_name) else {
            return Err(McpError::invalid_params(format!("unknown recipe '{recipe_name}'"), None));
        };
        let messages = args.get("messages").cloned().unwrap_or_else(|| json!([]));
        match crate::orchestrate_recipe(&self.state, recipe, &messages).await {
            Ok(resp) => {
                let text = resp["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(e) => Err(McpError::internal_error(e.message, None)),
        }
    }
}

impl ServerHandler for WoollamaMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::default()
    }

    async fn list_tools(
        &self,
        _p: Option<PaginatedRequestParams>,
        _c: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let mut tools = vec![chat_tool()];
        tools.extend(self.state.registry.reexport_tools());
        Ok(ListToolsResult::with_all_items(tools))
    }

    async fn call_tool(
        &self,
        req: CallToolRequestParams,
        _c: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        if req.name == "chat" {
            return self.run_chat(req.arguments).await;
        }
        // A re-exported downstream tool — proxy it (content + structured_content verbatim).
        let args = req.arguments.map(Value::Object).unwrap_or_else(|| json!({}));
        self.state
            .registry
            .call_raw(&req.name, &args)
            .await
            .map_err(|e| McpError::internal_error(format!("dispatch failed: {e}"), None))
    }

    async fn list_prompts(
        &self,
        _p: Option<PaginatedRequestParams>,
        _c: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, McpError> {
        let prompts = self
            .state
            .recipes
            .keys()
            .map(|name| {
                Prompt::new(name.clone(), Some(format!("woollama recipe '{name}' — its system prompt")), None)
            })
            .collect();
        Ok(ListPromptsResult::with_all_items(prompts))
    }

    async fn get_prompt(
        &self,
        req: GetPromptRequestParams,
        _c: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, McpError> {
        let Some(recipe) = self.state.recipes.get(req.name.as_str()) else {
            return Err(McpError::invalid_params(format!("unknown recipe '{}'", req.name), None));
        };
        Ok(GetPromptResult::new(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            recipe.system.clone(),
        )]))
    }
}
