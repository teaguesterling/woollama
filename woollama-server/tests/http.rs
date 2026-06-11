//! Slice 2 integration tests: the axum skeleton end-to-end against a MOCK upstream
//! (no live Ollama). Proves binding, `/v1/models`, and `/v1/chat/completions`
//! passthrough (bare-model rewrite + response relay), plus the deferral 501/400s.

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

/// Serve `router` on an ephemeral loopback port; return its base URL.
async fn spawn(router: axum::Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn skeleton_serves_models_and_passthrough() {
    // A mock "ollama" /v1 endpoint that records the forwarded request body.
    let seen: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let rec = seen.clone();
    let upstream = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move |axum::Json(b): axum::Json<Value>| {
            let rec = rec.clone();
            async move {
                *rec.lock().unwrap() = Some(b);
                axum::Json(json!({
                    "id": "x", "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "pong"}}]
                }))
            }
        }),
    );
    let upstream_url = spawn(upstream).await;
    // ollama's base_url is derived from $WOOLLAMA_OLLAMA_URL (+ /v1) by the engine.
    std::env::set_var("WOOLLAMA_OLLAMA_URL", &upstream_url);

    let base = spawn(woollama_server::router()).await;
    let c = reqwest::Client::new();

    // GET /v1/models → 200, OpenAI list shape.
    let r = c.get(format!("{base}/v1/models")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let m: Value = r.json().await.unwrap();
    assert_eq!(m["object"], "list");
    assert!(m["data"].is_array());

    // Passthrough: ollama/qwen forwards the BARE model and relays the response.
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "ollama/qwen", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "pong");
    let fwd = seen.lock().unwrap().clone().expect("upstream was called");
    assert_eq!(fwd["model"], "qwen"); // bare, not namespaced
    assert_eq!(fwd["stream"], false);

    // woollama/<recipe> → 501 (orchestration is slice 4).
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "woollama/streamer", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 501);

    // Unknown namespace → 400.
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "bogus/x", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);

    // Streaming passthrough is deferred to slice 3 → 501.
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "ollama/qwen", "messages": [], "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 501);
}

#[test]
fn resolve_tcp_target_default_and_override() {
    // One test (not two) so the WOOLLAMA_ADDRESS mutations can't race under parallel
    // execution. Default → loopback + free port; explicit → honored (incl. 0.0.0.0).
    std::env::remove_var("WOOLLAMA_ADDRESS");
    assert_eq!(woollama_server::resolve_tcp_target(), ("127.0.0.1".to_string(), 0));
    std::env::set_var("WOOLLAMA_ADDRESS", "0.0.0.0:8080");
    assert_eq!(woollama_server::resolve_tcp_target(), ("0.0.0.0".to_string(), 8080));
    std::env::remove_var("WOOLLAMA_ADDRESS");
}
