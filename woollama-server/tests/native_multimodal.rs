//! Tier 3 — native (engine) multimodal: an OpenAI `image_url` content part on a NATIVE recipe run
//! reaches the inferencer VERBATIM. Native recipes go through the parity-locked engine, which
//! forwards user-message `content` untouched to the OpenAI-compat `/chat/completions` — so image
//! input "just works" for a recipe bound to (or overridden with) a vision model, with no special
//! handling. This test pins that: a mock inferencer echoes back what it received, proving the
//! `{{var}}`-rendered system AND the image part both arrive. (Real vision — llama3.2-vision — is
//! proven end-to-end manually: red PNG → "Red." on both `/w1/run` and `/v1/chat/completions`.)
//!
//! Single test fn (one binary) so the process-global WOOLLAMA_* env can't race.

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
async fn native_recipe_forwards_image_url_content_verbatim() {
    // Mock inferencer: reflect the FIRST system message + the image_url it saw, so the test can
    // assert both the rendered system prompt and the verbatim image part reached inference.
    let upstream = Router::new().route(
        "/v1/chat/completions",
        post(|Json(b): Json<Value>| async move {
            let msgs = b["messages"].as_array().cloned().unwrap_or_default();
            let system = msgs
                .iter()
                .find(|m| m["role"] == "system")
                .and_then(|m| m["content"].as_str())
                .unwrap_or("(none)")
                .to_string();
            // Find the image_url URL anywhere in a user message's array content.
            let img = msgs
                .iter()
                .filter(|m| m["role"] == "user")
                .filter_map(|m| m["content"].as_array())
                .flatten()
                .find(|p| p["type"] == "image_url")
                .and_then(|p| p["image_url"]["url"].as_str())
                .unwrap_or("(no-image)")
                .to_string();
            Json(json!({"choices": [{"message": {
                "role": "assistant", "content": format!("system={system}|img={img}")
            }}]}))
        }),
    );
    let upstream_url = spawn(upstream).await;

    // A native recipe bound to an ollama model, with a {{tone}} var — so we also prove render and
    // image content coexist (system is substituted; the image is untouched).
    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(
        cfg.path().join("recipes.toml"),
        "[recipes.seer]\ninferencer=\"ollama/m\"\nsystem=\"You are a {{tone}} image describer.\"\n",
    )
    .unwrap();
    std::fs::write(cfg.path().join("mcp.json"), json!({"mcpServers": {}}).to_string()).unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_OLLAMA_URL", &upstream_url);

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    let data_url = "data:image/png;base64,iVBORw0KGgoAAAANS=";
    let multimodal = json!([{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": data_url}}
    ]}]);

    // 1) Native recipe via /w1/run: {{tone}} rendered into the system AND the image forwarded.
    let r = c
        .post(format!("{base}/w1/patterns/seer/run"))
        .json(&json!({"input": multimodal, "variables": {"tone": "terse"}}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(
        content,
        format!("system=You are a terse image describer.|img={data_url}"),
        "rendered system + verbatim image both reached inference; got: {content}"
    );

    // 2) Same recipe addressed as `woollama/seer` on the OpenAI `/v1/chat/completions` surface —
    // image_url must survive there too (OpenAI-client addressability).
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "woollama/seer", "messages": multimodal}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.contains(&format!("img={data_url}")), "image forwarded on /v1 too; got: {content}");

    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
    std::env::remove_var("WOOLLAMA_OLLAMA_URL");
}
