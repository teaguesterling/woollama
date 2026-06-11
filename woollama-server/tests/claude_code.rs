//! Slice 5: a `claude-code/<model>` recipe runs through the executor, end-to-end over
//! the HTTP surface, with a FAKE `claude` CLI (WOOLLAMA_CLAUDE_BIN → fake_claude bin).
//! Real-CLI behavior + the lockdown actually holding at runtime are the opt-in live
//! tests (plain terminal); the argv/lockdown/boundary are unit-tested in claude_code.rs.
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
async fn claude_code_recipe_runs_via_fake_cli() {
    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(
        cfg.path().join("recipes.toml"),
        "[recipes.cc]\ninferencer=\"claude-code/haiku\"\ntools=[]\nsystem=\"be brief\"\n",
    )
    .unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_CLAUDE_BIN", env!("CARGO_BIN_EXE_fake_claude"));

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    // /v1/chat/completions — the tool-less claude-code recipe returns the CLI's answer.
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "woollama/cc", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Value>().await.unwrap()["choices"][0]["message"]["content"], "fake-answer");

    // /v1/responses — same recipe through the executor.
    let r = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "woollama/cc", "input": "hi"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Value>().await.unwrap()["output"][0]["content"][0]["text"], "fake-answer");

    // streaming — claude-code is non-streaming, so the whole answer is one delta.
    let text = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "woollama/cc", "stream": true,
                      "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("\"content\":\"fake-answer\""), "stream content; got {text}");
    assert!(text.contains("data: [DONE]"));
}
