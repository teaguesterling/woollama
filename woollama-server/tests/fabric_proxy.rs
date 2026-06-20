//! The `/fabric/*` transparent reverse-proxy (Part 2, commit A). A mock fabric REST server
//! stands in for `fabric --serve`; woollamad is configured to ROUTE to it (`fabric.url` —
//! external mode, no process spawn). Proves requests/responses pass through verbatim,
//! including fabric's native (non-OpenAI) SSE.
//!
//! Single test fn (one binary) so the global WOOLLAMA_* env can't race.

use std::sync::Arc;

use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn fabric_proxy_passes_through_verbatim() {
    // Mock fabric: its real REST surface (patterns/models + the non-OpenAI /chat SSE).
    let fabric = Router::new()
        .route("/patterns/names", get(|| async { Json(json!(["summarize", "analyze"])) }))
        .route(
            "/models/names",
            get(|| async { Json(json!({"models": ["m"], "vendors": {"Ollama": ["m"], "Anthropic": ["x"]}})) }),
        )
        .route("/patterns/summarize", get(|| async { Json(json!({"Pattern": "You are {{tone}}."})) }))
        .route(
            "/chat",
            post(|Json(b): Json<Value>| async move {
                // Echo the pattern name back inside fabric's native SSE envelope.
                let pat = b["prompts"][0]["patternName"].as_str().unwrap_or("").to_string();
                let sse = format!(
                    "data: {}\n\ndata: {}\n\n",
                    json!({"type": "content", "content": format!("ran:{pat}")}),
                    json!({"type": "complete"})
                );
                ([("content-type", "text/event-stream")], sse).into_response()
            }),
        );
    let fabric_url = spawn(fabric).await;

    // woollamad routed at the mock fabric (external mode).
    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), json!({"fabric": {"url": fabric_url}}).to_string()).unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());

    let state = Arc::new(woollama_server::build_state().await);
    assert!(state.fabric.is_some(), "fabric backend must connect to the mock");
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    // GET passthrough: /fabric/patterns/names → fabric's list, verbatim.
    let r = c.get(format!("{base}/fabric/patterns/names")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Value>().await.unwrap(), json!(["summarize", "analyze"]));

    // POST passthrough with fabric's native SSE streamed through untouched.
    let r = c
        .post(format!("{base}/fabric/chat"))
        .json(&json!({"prompts": [{"patternName": "summarize", "userInput": "x"}], "model": "m"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(
        r.headers().get("content-type").unwrap().to_str().unwrap(),
        "text/event-stream",
        "fabric's content-type is preserved"
    );
    let text = r.text().await.unwrap();
    assert!(text.contains(r#""type":"content""#), "fabric SSE event type passes through: {text}");
    assert!(text.contains("ran:summarize"), "fabric saw the pattern name: {text}");
    assert!(text.contains(r#""type":"complete""#), "fabric's complete marker passes through");
}
