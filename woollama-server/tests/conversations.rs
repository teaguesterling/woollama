//! Slice 6a: stateful conversations (claude-resume) over the HTTP surface, hermetic via
//! the fake `claude` CLI. Covers /v1/conversations CRUD, a stateful /v1/responses turn
//! attaching by conversation id, the durable handle table SURVIVING A RESTART (a fresh
//! AppState on the same WOOLLAMA_STATE_DIR resolves the handle), /items deferral, delete,
//! and the no-stateful-backend 501. Recall across turns is the opt-in LIVE test.
//!
//! Separate test binary so the global WOOLLAMA_* env can't race other files.

use std::sync::Arc;

use axum::Router;
use serde_json::{json, Value};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn stateful_conversations_claude_resume_and_restart() {
    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    let state_dir = tempfile::tempdir().unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_STATE_DIR", state_dir.path());
    std::env::set_var("WOOLLAMA_CLAUDE_BIN", env!("CARGO_BIN_EXE_fake_claude"));

    let s1 = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(s1)).await;
    let c = reqwest::Client::new();

    // CREATE: model picks the backend (claude-code → claude-resume).
    let created = c
        .post(format!("{base}/v1/conversations"))
        .json(&json!({"model": "claude-code/haiku", "title": "t", "metadata": {"k": "v"}}))
        .send()
        .await
        .unwrap();
    assert_eq!(created.status(), 201);
    let conv: Value = created.json().await.unwrap();
    let cid = conv["id"].as_str().unwrap().to_string();
    assert_eq!(conv["backend"], "claude-resume");
    assert_eq!(conv["title"], "t");

    // DISCOVER: in the list + GET /{id}.
    let list: Value = c.get(format!("{base}/v1/conversations")).send().await.unwrap().json().await.unwrap();
    assert!(list["data"].as_array().unwrap().iter().any(|x| x["id"] == json!(cid)));
    let got: Value = c.get(format!("{base}/v1/conversations/{cid}")).send().await.unwrap().json().await.unwrap();
    assert_eq!(got["id"], json!(cid));

    // DRIVE a turn attaching by conversation id → 200, carries the conversation id.
    let r1: Value = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "claude-code/haiku", "conversation": cid, "input": "hi"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r1["conversation"]["id"], json!(cid));
    assert_eq!(r1["output"][0]["content"][0]["text"], "fake-answer");

    // A second turn resumes the same handle.
    let r2 = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "claude-code/haiku", "conversation": cid, "input": "more"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 200);

    // /items defers for claude-resume → 501.
    let items = c.get(format!("{base}/v1/conversations/{cid}/items")).send().await.unwrap();
    assert_eq!(items.status(), 501);

    // RESTART SURVIVAL: a fresh AppState on the SAME state dir resolves the handle.
    let s2 = Arc::new(woollama_server::build_state().await);
    let base2 = spawn(woollama_server::router(s2)).await;
    let got2 = c.get(format!("{base2}/v1/conversations/{cid}")).send().await.unwrap();
    assert_eq!(got2.status(), 200, "the handle must survive a restart");
    let r3 = c
        .post(format!("{base2}/v1/responses"))
        .json(&json!({"model": "claude-code/haiku", "conversation": cid, "input": "again"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r3.status(), 200);

    // DELETE → gone (404 + absent from the list).
    let del: Value = c.delete(format!("{base2}/v1/conversations/{cid}")).send().await.unwrap().json().await.unwrap();
    assert_eq!(del["deleted"], json!(true));
    assert_eq!(c.get(format!("{base2}/v1/conversations/{cid}")).send().await.unwrap().status(), 404);

    // A model with no state-owning backend → 501 on create.
    let no = c
        .post(format!("{base2}/v1/conversations"))
        .json(&json!({"model": "ollama/qwen"}))
        .send()
        .await
        .unwrap();
    assert_eq!(no.status(), 501);
}
