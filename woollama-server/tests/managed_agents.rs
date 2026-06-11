//! Slice 7: the managed-agents backend WOOLLAMA-SIDE routing, against a MOCK Anthropic
//! Managed Agents server (ANTHROPIC_BASE_URL). Proves: claude-agent/* → managed-agents;
//! a turn returns text; the interactive requires_action PAUSE (agent asks) → status
//! requires_action + required_action; answering resumes → completed; /items served from
//! the event log; delete.
//!
//! NOTE: the mock implements the *simplified* protocol this client targets — the REAL
//! Anthropic wire format is validated only by the opt-in live @needs_anthropic test (paid).
//! This test gates the woollama routing, not the Anthropic wire shapes.
//!
//! Separate test binary so the global env can't race other files.

use std::sync::Arc;

use axum::extract::Path;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn managed_agents_routing_pause_resume_items() {
    let anthropic = Router::new()
        .route("/v1/environments", post(|| async { Json(json!({"id": "env1"})) }))
        .route("/v1/agents", post(|| async { Json(json!({"id": "agent1"})) }))
        .route("/v1/sessions", post(|| async { Json(json!({"id": "sess1"})) }))
        .route(
            "/v1/sessions/{id}/turns",
            post(|Path(_id): Path<String>, Json(b): Json<Value>| async move {
                if b.get("tool_use_id").is_some() {
                    let ans = b.get("answer").and_then(Value::as_str).unwrap_or("");
                    Json(json!({"text": format!("answered: {ans}"), "pending": Value::Null}))
                } else if b.get("input").and_then(Value::as_str).unwrap_or("").contains("ask") {
                    Json(json!({"text": "", "pending": {"id": "tu1", "input": {"question": "What is your name?"}}}))
                } else {
                    Json(json!({"text": "hello", "pending": Value::Null}))
                }
            }),
        )
        .route(
            "/v1/sessions/{id}/events",
            get(|Path(_id): Path<String>| async {
                Json(json!({"data": [
                    {"type": "user.message", "content": [{"type": "text", "text": "hi"}]},
                    {"type": "agent.message", "content": [{"type": "text", "text": "hello"}]}
                ]}))
            }),
        )
        .route("/v1/sessions/{id}", delete(|Path(_id): Path<String>| async { Json(json!({"deleted": true})) }));
    let anthropic_url = spawn(anthropic).await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("ANTHROPIC_BASE_URL", &anthropic_url);
    std::env::set_var("ANTHROPIC_API_KEY", "test-key");

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    // CREATE: claude-agent/* → managed-agents.
    let created = c
        .post(format!("{base}/v1/conversations"))
        .json(&json!({"model": "claude-agent/haiku", "title": "j"}))
        .send()
        .await
        .unwrap();
    assert_eq!(created.status(), 201);
    let conv: Value = created.json().await.unwrap();
    let cid = conv["id"].as_str().unwrap().to_string();
    assert_eq!(conv["backend"], "managed-agents");

    // A normal turn → completed, text from the (mock) hosted session.
    let r1: Value = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "claude-agent/haiku", "conversation": cid, "input": "hi"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r1["status"], "completed");
    assert_eq!(r1["output"][0]["content"][0]["text"], "hello");

    // A turn that makes the agent ask → requires_action + required_action.
    let r2: Value = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "claude-agent/haiku", "conversation": cid, "input": "please ask me a question"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r2["status"], "requires_action");
    assert_eq!(r2["required_action"]["type"], "ask_user");

    // Answering resumes → completed.
    let r3: Value = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "claude-agent/haiku", "conversation": cid, "input": "Alice"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r3["status"], "completed");
    assert_eq!(r3["output"][0]["content"][0]["text"], "answered: Alice");

    // /items SERVES the transcript from the event log (the managed-agents capability win).
    let items: Value = c.get(format!("{base}/v1/conversations/{cid}/items")).send().await.unwrap().json().await.unwrap();
    let roles: Vec<&str> = items["data"].as_array().unwrap().iter().filter_map(|i| i["role"].as_str()).collect();
    assert!(roles.contains(&"user") && roles.contains(&"assistant"));

    // DELETE tears down the session.
    let del: Value = c.delete(format!("{base}/v1/conversations/{cid}")).send().await.unwrap().json().await.unwrap();
    assert_eq!(del["deleted"], json!(true));
}
