//! Fabric library sourced into the `/w1/` surface (Part 2 B/C) via the generic
//! `PatternBackend` trait. A mock fabric stands in for `fabric --serve` (external-url mode).
//! Proves discovery, render (fetch fabric system + substitute), and run (build fabric's `/chat`
//! body with the derived vendor + translate fabric's native SSE → OpenAI), plus the
//! recipes-win-on-collision rule.
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
async fn fabric_library_on_w1_surface() {
    let fabric = Router::new()
        .route("/patterns/names", get(|| async { Json(json!(["summarize", "shared"])) }))
        .route("/models/names", get(|| async { Json(json!({"vendors": {"Ollama": ["m"], "Anthropic": ["x"]}})) }))
        .route("/patterns/summarize", get(|| async { Json(json!({"Pattern": "Fabric summarize {{tone}}."})) }))
        .route(
            "/chat",
            post(|Json(b): Json<Value>| async move {
                // Echo the assembled fabric request back through fabric's native SSE so the
                // translation + vendor-map + variable plumbing are all assertable.
                let p = &b["prompts"][0];
                let line = format!(
                    "vendor={}|model={}|pat={}|tone={}|input={}",
                    p["vendor"].as_str().unwrap_or(""),
                    p["model"].as_str().unwrap_or(""),
                    p["patternName"].as_str().unwrap_or(""),
                    p["variables"]["tone"].as_str().unwrap_or(""),
                    p["userInput"].as_str().unwrap_or(""),
                );
                let sse = format!(
                    "data: {}\n\ndata: {}\n\n",
                    json!({"type": "content", "content": line}),
                    json!({"type": "complete"})
                );
                ([("content-type", "text/event-stream")], sse).into_response()
            }),
        );
    let fabric_url = spawn(fabric).await;

    let cfg = tempfile::tempdir().unwrap();
    // A native recipe named "shared" must WIN over fabric's "shared" pattern.
    std::fs::write(cfg.path().join("recipes.toml"), "[recipes.shared]\ninferencer=\"ollama/m\"\nsystem=\"native shared\"\n").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), json!({"fabric": {"url": fabric_url}}).to_string()).unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    // 1) discovery: fabric's "summarize" appears (source fabric); "shared" appears ONCE as a
    //    native recipe (recipes win on collision).
    let body: Value = c.get(format!("{base}/w1/patterns")).send().await.unwrap().json().await.unwrap();
    let entries = body["data"].as_array().unwrap();
    let summarize = entries.iter().find(|p| p["name"] == "summarize").expect("fabric summarize listed");
    assert_eq!(summarize["source"], "fabric");
    let shared: Vec<&Value> = entries.iter().filter(|p| p["name"] == "shared").collect();
    assert_eq!(shared.len(), 1, "collision listed once");
    assert_eq!(shared[0]["source"], "recipe", "native recipe wins the name");

    // fabric patterns are also addressable as woollama/<name> in /v1/models.
    let models: Value = c.get(format!("{base}/v1/models")).send().await.unwrap().json().await.unwrap();
    assert!(
        models["data"].as_array().unwrap().iter().any(|m| m["id"] == "woollama/summarize"),
        "fabric pattern in /v1/models"
    );

    // 2) render a fabric pattern: woollama fetches its system + substitutes locally.
    let r: Value = c
        .post(format!("{base}/w1/patterns/summarize/render"))
        .json(&json!({"input": "the news", "variables": {"tone": "terse"}}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["prompt"], "Fabric summarize terse.\n\nthe news");

    // 3) run a fabric pattern (non-stream): vendor derived (ollama→Ollama), bare model, variables
    //    + input reach fabric, fabric's SSE is accumulated into one OpenAI completion.
    let r: Value = c
        .post(format!("{base}/w1/patterns/summarize/run"))
        .json(&json!({"input": "hi", "variables": {"tone": "wry"}, "model": "ollama/m2"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        r["choices"][0]["message"]["content"],
        "vendor=Ollama|model=m2|pat=summarize|tone=wry|input=hi"
    );

    // 4) run (stream): fabric SSE → OpenAI chat.completion.chunk + [DONE].
    let text = c
        .post(format!("{base}/w1/patterns/summarize/run"))
        .json(&json!({"input": "hi", "variables": {"tone": "wry"}, "model": "ollama/m2", "stream": true}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("chat.completion.chunk"), "OpenAI SSE shape: {text}");
    assert!(text.contains("vendor=Ollama"), "fabric content translated through: {text}");
    assert!(text.contains("data: [DONE]"), "terminated with [DONE]");

    // 5) fabric run WITHOUT a model → 400 (fabric patterns have no bound inferencer).
    let r = c
        .post(format!("{base}/w1/patterns/summarize/run"))
        .json(&json!({"input": "hi"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400, "fabric run requires a model");
}
