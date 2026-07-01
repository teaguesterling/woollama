//! Tier 3 — vision via the fabric CLI (`fabric -a`). A `/w1/patterns/{name}/run` whose `input`
//! carries an OpenAI `image_url` part can't ride fabric's REST `/chat` (no attachment field), so
//! woollama shells out to the one-shot `fabric` CLI: userInput on stdin, `--attachment=` for the
//! image, plain-text stdout → an OpenAI completion.
//!
//! Hermetic: fabric runs in external-`url` mode (a mock REST answers discovery so `has()` sees the
//! pattern), and `command` points at the `fake_fabric` fixture, which echoes its `--pattern` /
//! `--attachment` / stdin so we can assert the argv assembly + stdin reached the subprocess. The
//! REAL CLI + a real vision model (llama3.2-vision) are exercised manually (red square → "Red").
//!
//! Single test fn (one binary) so the process-global WOOLLAMA_* env can't race.

use std::sync::Arc;

use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn fabric_vision_routes_image_input_through_the_cli() {
    // Mock fabric REST: just enough for discovery so `vision-pat` is a known pattern. The vision
    // run never touches `/chat` — it goes to the CLI fixture.
    let fabric = Router::new()
        .route("/patterns/names", get(|| async { Json(json!(["vision-pat"])) }))
        .route("/models/names", get(|| async { Json(json!({"vendors": {"Ollama": ["llama-vision"]}})) }));
    let fabric_url = spawn(fabric).await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(
        cfg.path().join("mcp.json"),
        json!({"fabric": {
            "url": fabric_url,
            "default_model": "ollama/llama-vision",
            // The CLI vision path shells out to this — the hermetic stand-in for `fabric`.
            "command": env!("CARGO_BIN_EXE_fake_fabric"),
        }})
        .to_string(),
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    let content = |b: &Value| b["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();

    // 1) data-URL image → decoded to a temp file; text on stdin. fake_fabric echoes all three.
    let png_data_url = "data:image/png;base64,iVBORw0KGgo="; // arbitrary valid base64
    let r = c
        .post(format!("{base}/w1/patterns/vision-pat/run"))
        .json(&json!({
            "input": [{"role": "user", "content": [
                {"type": "text", "text": "what color?"},
                {"type": "image_url", "image_url": {"url": png_data_url}}
            ]}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "vision run ok");
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    let text = content(&body);
    // fake_fabric prints: `VISION pattern=<p> att=<path> input=<stdin>`
    assert!(text.contains("pattern=vision-pat"), "pattern reached the CLI; got: {text}");
    assert!(text.contains("input=what color?"), "user text piped on stdin; got: {text}");
    assert!(text.contains("att=") && text.contains(".png"), "data-URL written to a .png temp file; got: {text}");
    // And the temp attachment is cleaned up (NamedTempFile drop) — extract the path and confirm.
    let att = text.split("att=").nth(1).and_then(|s| s.split(" input=").next()).unwrap_or("").to_string();
    assert!(!att.is_empty(), "attachment path echoed");
    assert!(!std::path::Path::new(&att).exists(), "temp attachment removed after the run: {att}");

    // 2) http(s) image URL → passed through verbatim as the attachment (no temp file).
    let r = c
        .post(format!("{base}/w1/patterns/vision-pat/run"))
        .json(&json!({
            "input": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
            ]}],
            "variables": {"role": "expert"}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert!(content(&body).contains("att=https://example.com/cat.jpg"), "URL passed through; got: {}", content(&body));

    // 3) stream:true → the OpenAI SSE shape (not a surprise JSON body), with the content chunk.
    let r = c
        .post(format!("{base}/w1/patterns/vision-pat/run"))
        .json(&json!({
            "input": [{"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
            ]}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert!(
        r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").contains("text/event-stream"),
        "stream:true returns SSE"
    );
    let sse = r.text().await.unwrap();
    assert!(sse.contains("chat.completion.chunk"), "OpenAI chunk frames");
    assert!(sse.contains("pattern=vision-pat"), "content carried in the stream");
    assert!(sse.trim_end().ends_with("[DONE]"), "terminated with [DONE]");

    // 4) An image_url that can't be used (undecodable data-URL) → 400, NOT a silent text answer.
    let r = c
        .post(format!("{base}/w1/patterns/vision-pat/run"))
        .json(&json!({
            "input": [{"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%notb64%%%"}}
            ]}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400, "undecodable image fails loudly instead of falling through to text");
    let body: Value = r.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap_or("").contains("image_url"), "clear error");

    // 5) A >2 MiB image body must NOT 413 — the router raises axum's 2 MiB default so real vision
    // photos fit. ~3 MiB of valid base64 ('A'*N, N a multiple of 4 → no padding needed).
    let big_b64 = "A".repeat(3 * 1024 * 1024);
    let r = c
        .post(format!("{base}/w1/patterns/vision-pat/run"))
        .json(&json!({
            "input": [{"role": "user", "content": [
                {"type": "text", "text": "big"},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{big_b64}")}}
            ]}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "a >2MiB image body is accepted (body limit raised), not 413");
    assert!(content(&r.json().await.unwrap()).contains("att=") , "large image reached the CLI path");

    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
}
