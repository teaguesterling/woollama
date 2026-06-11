//! Slice 4a end-to-end: `woollama/<recipe>` orchestration over a REAL downstream MCP
//! server (the `mcp_fixture` bin, spawned as a child process) + a MOCK inferencer that
//! returns a tool_call then a final answer. Proves the recipe loop dispatches the MCP
//! tool through the registry and hides the loop from the client.
//!
//! Separate test binary from http.rs so the global WOOLLAMA_* env can't race.

use std::sync::Arc;

use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn woollama_recipe_orchestrates_through_mcp_registry() {
    // Mock inferencer: turn 1 (no tool result yet) → emit a tool_call for fix.count_to;
    // turn 2 (a tool message is present) → the final answer.
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
                    "tool_calls": [{
                        "id": "c1", "type": "function",
                        "function": {"name": "fix.count_to", "arguments": "{\"n\":3}"}
                    }]
                }}]}))
            }
        }),
    );
    let upstream_url = spawn(upstream).await;

    // Config: a recipe whose inferencer points at the mock and whose one tool lives on
    // the `fix` server = the compiled mcp_fixture binary (a real stdio MCP server).
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
    let c = reqwest::Client::new();

    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "woollama/counter", "messages": [{"role": "user", "content": "count to 3"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "orchestration should return 200");
    let body: Value = r.json().await.unwrap();
    // The client sees only the final answer — the internal tool loop is hidden.
    assert_eq!(body["choices"][0]["message"]["content"], "done counting");
    assert!(body["choices"][0]["message"]["tool_calls"].as_array().map_or(true, |a| a.is_empty())
        || body["choices"][0]["message"]["tool_calls"].is_null());

    // Same recipe over /v1/responses → Responses shape with the final text.
    let r = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "woollama/counter", "input": "count to 3"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let resp: Value = r.json().await.unwrap();
    assert_eq!(resp["output"][0]["content"][0]["text"], "done counting");

    // Unknown recipe → 404.
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "woollama/nope", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}
