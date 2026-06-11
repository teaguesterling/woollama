//! Integration tests: the axum surface end-to-end against a MOCK upstream (no live
//! Ollama). One consolidated test (so the global `WOOLLAMA_OLLAMA_URL` can't race under
//! parallel execution) drives every slice-2/3 path: `/v1/models`, passthrough
//! (non-stream + streaming + native num_ctx), and stateless `/v1/responses`.

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::response::{IntoResponse, Response};
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
async fn http_surface_passthrough_native_and_responses() {
    let v1_seen: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let native_seen: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));

    // Mock "ollama": /v1/chat/completions (JSON, or SSE when stream) + native /api/chat.
    let v1 = v1_seen.clone();
    let native = native_seen.clone();
    let upstream = Router::new()
        .route(
            "/v1/chat/completions",
            post(move |Json(b): Json<Value>| {
                let v1 = v1.clone();
                async move {
                    let streaming = b.get("stream").and_then(Value::as_bool).unwrap_or(false);
                    *v1.lock().unwrap() = Some(b);
                    if streaming {
                        Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(Body::from(
                                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n",
                            ))
                            .unwrap()
                    } else {
                        Json(json!({
                            "object": "chat.completion",
                            "choices": [{"message": {"role": "assistant", "content": "pong"}}]
                        }))
                        .into_response()
                    }
                }
            }),
        )
        .route(
            "/api/chat",
            post(move |Json(b): Json<Value>| {
                let native = native.clone();
                async move {
                    *native.lock().unwrap() = Some(b);
                    Json(json!({
                        "message": {"role": "assistant", "content": "native-pong"},
                        "done": true, "done_reason": "stop",
                        "prompt_eval_count": 5, "eval_count": 3
                    }))
                }
            }),
        );
    let upstream_url = spawn(upstream).await;
    std::env::set_var("WOOLLAMA_OLLAMA_URL", &upstream_url);

    let base = spawn(woollama_server::router()).await;
    let c = reqwest::Client::new();

    // /v1/models → 200 list.
    let r = c.get(format!("{base}/v1/models")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Value>().await.unwrap()["object"], "list");

    // Non-stream passthrough → bare model, relayed.
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "ollama/qwen", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Value>().await.unwrap()["choices"][0]["message"]["content"], "pong");
    let fwd = v1_seen.lock().unwrap().clone().unwrap();
    assert_eq!(fwd["model"], "qwen");
    assert_eq!(fwd["stream"], false);

    // Native num_ctx → /api/chat, translated back to chat.completion.
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "ollama/qwen", "messages": [{"role": "user", "content": "hi"}],
                      "options": {"num_ctx": 16384}}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["content"], "native-pong");
    assert_eq!(body["usage"]["total_tokens"], 8);
    let nfwd = native_seen.lock().unwrap().clone().unwrap();
    assert_eq!(nfwd["model"], "qwen");
    assert_eq!(nfwd["options"]["num_ctx"], 16384);

    // Streaming passthrough → SSE relayed verbatim.
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "ollama/qwen", "messages": [], "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
    let text = r.text().await.unwrap();
    assert!(text.contains("\"content\":\"hi\""));
    assert!(text.contains("data: [DONE]"));

    // Stateless /v1/responses → Responses shape with the assistant text.
    let r = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "ollama/qwen", "input": "hi"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let resp: Value = r.json().await.unwrap();
    assert_eq!(resp["object"], "response");
    assert_eq!(resp["status"], "completed");
    assert_eq!(resp["output"][0]["content"][0]["text"], "pong");

    // Deferrals: woollama/<recipe> → 501; unknown ns → 400; stateful responses → 501.
    let r = c.post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "woollama/streamer", "messages": []})).send().await.unwrap();
    assert_eq!(r.status(), 501);
    let r = c.post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "bogus/x", "messages": []})).send().await.unwrap();
    assert_eq!(r.status(), 400);
    let r = c.post(format!("{base}/v1/responses"))
        .json(&json!({"model": "ollama/qwen", "input": "hi", "store": true})).send().await.unwrap();
    assert_eq!(r.status(), 501);
}

#[test]
fn resolve_tcp_target_default_and_override() {
    std::env::remove_var("WOOLLAMA_ADDRESS");
    assert_eq!(woollama_server::resolve_tcp_target(), ("127.0.0.1".to_string(), 0));
    std::env::set_var("WOOLLAMA_ADDRESS", "0.0.0.0:8080");
    assert_eq!(woollama_server::resolve_tcp_target(), ("0.0.0.0".to_string(), 8080));
    std::env::remove_var("WOOLLAMA_ADDRESS");
}
