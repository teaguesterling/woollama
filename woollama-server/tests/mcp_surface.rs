//! Slice 4b: woollama AS an MCP server, over the mounted `/mcp` Streamable-HTTP surface,
//! driven by a real rmcp client. Proves the aggregator end-to-end: re-exported downstream
//! tool (schema mirrored) + proxy passthrough (structured content), the `chat` orchestration
//! tool, recipe prompts, AND the shared-registry-across-sessions lifecycle (two concurrent
//! MCP sessions sharing the one downstream registry).
//!
//! Separate test binary so the global WOOLLAMA_* env can't race other test files.

use std::sync::Arc;

use axum::routing::post;
use axum::{Json, Router};
use rmcp::model::CallToolRequestParams;
use rmcp::transport::streamable_http_client::StreamableHttpClientWorker;
use rmcp::transport::worker::WorkerTransport;
use rmcp::ServiceExt;
use serde_json::{json, Value};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

async fn mcp_client(base: &str) -> rmcp::service::RunningService<rmcp::service::RoleClient, ()> {
    let url = format!("{base}/mcp");
    let worker = StreamableHttpClientWorker::<reqwest::Client>::new_simple(url);
    ().serve(WorkerTransport::spawn(worker)).await.unwrap()
}

#[tokio::test]
async fn woollama_mcp_surface_aggregates_and_orchestrates() {
    // Mock inferencer for the `chat` tool: tool_call → final (same shape as orchestrate.rs).
    let upstream = Router::new().route(
        "/v1/chat/completions",
        post(|Json(b): Json<Value>| async move {
            let has_tool = b
                .get("messages")
                .and_then(Value::as_array)
                .map(|ms| ms.iter().any(|m| m.get("role").and_then(Value::as_str) == Some("tool")))
                .unwrap_or(false);
            if has_tool {
                Json(json!({"choices": [{"message": {"role": "assistant", "content": "done counting"}}]}))
            } else {
                Json(json!({"choices": [{"message": {
                    "role": "assistant", "content": Value::Null,
                    "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "mcp__fix__count_to", "arguments": "{\"n\":3}"}}]
                }}]}))
            }
        }),
    );
    let upstream_url = spawn(upstream).await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(
        cfg.path().join("recipes.toml"),
        "[recipes.counter]\ninferencer=\"ollama/m\"\ntools=[\"fix.count_to\"]\nsystem=\"count helper\"\n",
    )
    .unwrap();
    let fixture = env!("CARGO_BIN_EXE_mcp_fixture");
    std::fs::write(
        cfg.path().join("mcp.json"),
        json!({"mcpServers": {"fix": {"command": fixture, "args": []}}}).to_string(),
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_OLLAMA_URL", &upstream_url);

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;

    let client = mcp_client(&base).await;

    // initialize: the server must ADVERTISE the tools + prompts capabilities it serves.
    // (A default ServerInfo reports neither — a capability-checking client like FastMCP
    // would then see an empty server even though list_tools/list_prompts work. The live
    // differential oracle caught exactly this; pin it here so it can't silently regress.)
    let caps = &client.peer_info().expect("server info on initialize").capabilities;
    assert!(caps.tools.is_some(), "tools capability must be advertised");
    assert!(caps.prompts.is_some(), "prompts capability must be advertised");

    // tools/list: the `chat` verb + the re-exported downstream tool with output_schema mirrored.
    let tools = client.list_tools(None).await.unwrap();
    let names: Vec<&str> = tools.tools.iter().map(|t| t.name.as_ref()).collect();
    assert!(names.contains(&"chat"), "chat tool present; got {names:?}");
    let ct = tools.tools.iter().find(|t| t.name == "mcp__fix__count_to").expect("re-exported tool");
    assert!(ct.output_schema.is_some(), "downstream output_schema must be mirrored");

    // Proxy passthrough: structured_content forwarded verbatim.
    let res = client
        .call_tool(CallToolRequestParams::new("mcp__fix__count_to").with_arguments(
            serde_json::from_value(json!({"n": 3})).unwrap(),
        ))
        .await
        .unwrap();
    assert_eq!(res.structured_content, Some(json!({"count": 3})));

    // prompts: one per recipe; get returns its system message.
    let prompts = client.list_prompts(None).await.unwrap();
    assert!(prompts.prompts.iter().any(|p| p.name == "counter"));
    let got = client
        .get_prompt(rmcp::model::GetPromptRequestParams::new("counter"))
        .await
        .unwrap();
    assert_eq!(got.messages.len(), 1);

    // chat: orchestrate the recipe through the registry; client sees only the final answer.
    let chat = client
        .call_tool(CallToolRequestParams::new("chat").with_arguments(
            serde_json::from_value(json!({
                "recipe": "counter", "messages": [{"role": "user", "content": "count to 3"}]
            }))
            .unwrap(),
        ))
        .await
        .unwrap();
    let text: String =
        chat.content.iter().filter_map(|c| c.as_text().map(|t| t.text.clone())).collect();
    assert_eq!(text, "done counting");
    // The answer ALSO rides structured_content under FastMCP's string-wrap convention, so a
    // FastMCP client's `result.data` resolves to the bare string (the live oracle's `.data`
    // assertion). chat advertises the matching wrap output_schema in tools/list.
    assert_eq!(
        chat.structured_content,
        Some(json!({ "result": "done counting" })),
        "chat result must carry FastMCP-wrapped structured_content"
    );
    let chat_tool = tools.tools.iter().find(|t| t.name == "chat").unwrap();
    let chat_out = chat_tool.output_schema.as_ref().expect("chat output_schema");
    assert_eq!(chat_out.get("x-fastmcp-wrap-result"), Some(&json!(true)), "wrap marker");

    // A bad recipe is a TOOL-level error (isError result → FastMCP `ToolError`), NOT a
    // JSON-RPC protocol error — matching the Python `chat` (which raises ValueError).
    let bad = client
        .call_tool(CallToolRequestParams::new("chat").with_arguments(
            serde_json::from_value(json!({
                "recipe": "_nope_", "messages": [{"role": "user", "content": "hi"}]
            }))
            .unwrap(),
        ))
        .await
        .expect("unknown recipe is a tool-level error, not a transport error");
    assert_eq!(bad.is_error, Some(true), "unknown recipe → isError result");
    let bad_text: String =
        bad.content.iter().filter_map(|c| c.as_text().map(|t| t.text.clone())).collect();
    assert!(bad_text.contains("unknown recipe"), "error text; got {bad_text:?}");

    // Lifecycle: a SECOND concurrent session shares the one downstream registry — both
    // sessions' proxy calls succeed against the single underlying connection.
    let client2 = mcp_client(&base).await;
    let (r1, r2) = tokio::join!(
        client.call_tool(
            CallToolRequestParams::new("mcp__fix__count_to")
                .with_arguments(serde_json::from_value(json!({"n": 1})).unwrap())
        ),
        client2.call_tool(
            CallToolRequestParams::new("mcp__fix__count_to")
                .with_arguments(serde_json::from_value(json!({"n": 2})).unwrap())
        ),
    );
    assert_eq!(r1.unwrap().structured_content, Some(json!({"count": 1})));
    assert_eq!(r2.unwrap().structured_content, Some(json!({"count": 2})));

    client.cancel().await.ok();
    client2.cancel().await.ok();
}
