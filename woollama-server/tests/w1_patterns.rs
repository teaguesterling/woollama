//! `/w1/` pattern-templating surface end-to-end: discovery, render-without-run, and a
//! templated run that dispatches through the EXISTING orchestration path to a MOCK
//! inferencer. The mock echoes back the system message (and the model it received), so we
//! can prove `{{var}}` substitution + per-call model override actually reached inference.
//!
//! Single test fn (one binary) so the global WOOLLAMA_* env can't race across cases.

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
async fn w1_patterns_discover_render_and_run() {
    // Mock inferencer: echo the system prompt back as the answer + report the model it got.
    let upstream = Router::new().route(
        "/v1/chat/completions",
        post(|Json(b): Json<Value>| async move {
            let system = b["messages"][0]["content"].as_str().unwrap_or("").to_string();
            let model = b["model"].as_str().unwrap_or("").to_string();
            // `temperature` is how we prove run's `options` merged into the upstream body.
            let temp = b.get("temperature").cloned().unwrap_or(Value::Null);
            Json(json!({"choices": [{"message": {
                "role": "assistant", "content": format!("{system}|model={model}|temp={temp}")
            }}]}))
        }),
    );
    let upstream_url = spawn(upstream).await;

    // A recipe whose system carries a {{tone}} token → it doubles as a /w1 pattern.
    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(
        cfg.path().join("recipes.toml"),
        "[recipes.summarize]\ninferencer=\"ollama/m\"\nsystem=\"You are a {{tone}} summarizer.\"\n",
    )
    .unwrap();
    std::fs::write(cfg.path().join("mcp.json"), json!({"mcpServers": {}}).to_string()).unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_OLLAMA_URL", &upstream_url);

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    // 1) discovery — name, scanned variables, source.
    let r = c.get(format!("{base}/w1/patterns")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let entry = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "summarize")
        .expect("summarize pattern listed");
    assert_eq!(entry["variables"], json!(["tone"]));
    assert_eq!(entry["source"], "recipe");

    // 2) render-without-run — substitute {{tone}} + append input, no model run.
    let r = c
        .post(format!("{base}/w1/patterns/summarize/render"))
        .json(&json!({"input": "the news", "variables": {"tone": "terse"}}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["prompt"], "You are a terse summarizer.\n\nthe news");

    // 3) templated run — substitution reaches the inferencer (echoed system proves it).
    let r = c
        .post(format!("{base}/w1/patterns/summarize/run"))
        .json(&json!({"input": "the news", "variables": {"tone": "terse"}}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(content, "You are a terse summarizer.|model=m|temp=null", "got: {content}");

    // 4) per-call model + options overrides — model replaces the bound inferencer; options
    // (temperature) merge into the upstream request body.
    let r = c
        .post(format!("{base}/w1/patterns/summarize/run"))
        .json(&json!({
            "input": "x", "variables": {"tone": "wry"},
            "model": "ollama/m2", "options": {"temperature": 0.3}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(content, "You are a wry summarizer.|model=m2|temp=0.3", "override changes model + options reach upstream");

    // 5) unknown pattern → 404 on both render and run.
    let r = c
        .post(format!("{base}/w1/patterns/nope/render"))
        .json(&json!({"input": "x"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
    let r = c
        .post(format!("{base}/w1/patterns/nope/run"))
        .json(&json!({"input": "x"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}
