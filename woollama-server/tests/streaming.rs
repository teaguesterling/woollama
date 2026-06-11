//! Slice 3b: the streaming family end-to-end against a MOCK upstream that speaks SSE
//! (/v1) and NDJSON (/api/chat). Covers: native num_ctx streaming (NDJSON→SSE),
//! stateless /v1/responses streaming (Responses event sequence), and streaming
//! orchestration of a woollama/<recipe> (chat.completion.chunk frames, tool loop hidden).
//!
//! Separate test binary so the global WOOLLAMA_* env can't race other files.

use std::sync::Arc;

use axum::body::Body;
use axum::response::Response;
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

fn sse(body: &'static str) -> Response {
    Response::builder().header("content-type", "text/event-stream").body(Body::from(body)).unwrap()
}

#[tokio::test]
async fn streaming_native_responses_and_orchestration() {
    let upstream = Router::new()
        .route(
            "/v1/chat/completions",
            post(|Json(b): Json<Value>| async move {
                let has_tool_msg = b
                    .get("messages")
                    .and_then(Value::as_array)
                    .map(|ms| ms.iter().any(|m| m.get("role").and_then(Value::as_str) == Some("tool")))
                    .unwrap_or(false);
                let has_tools = b.get("tools").and_then(Value::as_array).map_or(false, |a| !a.is_empty());
                if has_tool_msg {
                    // recipe turn 2: the final answer streamed.
                    sse("data: {\"choices\":[{\"delta\":{\"content\":\"done counting\"}}]}\n\ndata: [DONE]\n\n")
                } else if has_tools {
                    // recipe turn 1: a streamed tool_call (id/name in one chunk, args fragmented).
                    sse("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"fix.count_to\",\"arguments\":\"{\\\"n\\\":\"}}]}}]}\n\ndata: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"3}\"}}]}}]}\n\ndata: [DONE]\n\n")
                } else {
                    // plain inferencer stream (the responses test).
                    sse("data: {\"choices\":[{\"delta\":{\"content\":\"pong\"}}]}\n\ndata: [DONE]\n\n")
                }
            }),
        )
        .route(
            "/api/chat",
            post(|Json(_b): Json<Value>| async move {
                sse("{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n{\"message\":{\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}\n")
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
    let c = reqwest::Client::new();

    // 1. Native num_ctx streaming: NDJSON /api/chat → chat.completion.chunk SSE.
    let text = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "ollama/qwen", "stream": true, "messages": [],
                      "options": {"num_ctx": 16384}}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("\"role\":\"assistant\""), "native stream: role chunk; got {text}");
    assert!(text.contains("\"content\":\"hi\""), "native stream: content delta");
    assert!(text.contains("data: [DONE]"));

    // 2. Stateless /v1/responses streaming: Responses event sequence.
    let text = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "ollama/qwen", "input": "hi", "stream": true}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("event: response.created"));
    assert!(text.contains("event: response.output_text.delta"));
    assert!(text.contains("event: response.completed"));
    assert!(text.contains("\"delta\":\"pong\""), "responses stream: the delta; got {text}");

    // 3. Streaming orchestration: woollama/<recipe> → chat.completion.chunk, loop hidden.
    let text = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "woollama/counter", "stream": true,
                      "messages": [{"role": "user", "content": "count to 3"}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("\"content\":\"done counting\""), "orch stream: final content; got {text}");
    assert!(text.contains("\"finish_reason\":\"stop\""), "orch stream: one stop terminator");
    assert!(text.contains("data: [DONE]"));
    assert!(!text.contains("tool_calls"), "orch stream: the tool loop must stay hidden");
}
