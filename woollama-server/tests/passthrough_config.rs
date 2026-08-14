//! Task 1 (rust-parity-port): the chat passthrough must resolve providers through the
//! config-merged `state.inferencers` Registry, not the built-ins-only `engine::get_inferencer`.
//! Proves a config-only inferencer (no built-in of the same name) is reachable on
//! `POST /v1/chat/completions`.
//!
//! Separate test binary so the global WOOLLAMA_* env can't race other files.

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
async fn passthrough_reaches_config_only_inferencer() {
    // Mock upstream for the config-only "device" inferencer.
    let upstream = Router::new().route(
        "/chat/completions",
        post(|| async { Json(json!({"choices": [{"message": {"content": "ok"}}]})) }),
    );
    let upstream_url = spawn(upstream).await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        format!("[inferencers.device]\nbase_url=\"{u}\"\n", u = upstream_url),
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "device/somemodel", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();

    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "ok");
}
