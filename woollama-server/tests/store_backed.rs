//! Slice 6b: store-backed statefulness — a NON-claude (ollama) model becomes stateful via
//! an external REST conversation store. The store (a mock) owns the transcript; woollama
//! assembles prior + new and runs stateless inference. The mock inferencer returns the
//! MESSAGE COUNT it received, so turn 2's "3" proves the store-held prior (2 msgs) was
//! reassembled. Also checks /items served from the store, and delete.
//!
//! Separate test binary so the global WOOLLAMA_* env can't race other files.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::extract::Path;
use axum::routing::{post, put};
use axum::{Json, Router};
use serde_json::{json, Value};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn store_backed_ollama_conversation_journey() {
    // Mock REST store: in-memory thread → messages.
    let threads: Arc<Mutex<HashMap<String, Vec<Value>>>> = Arc::new(Mutex::new(HashMap::new()));
    let (t1, t2, t3, t4) = (threads.clone(), threads.clone(), threads.clone(), threads.clone());
    let store = Router::new().route(
        "/threads/{id}",
        put(move |Path(id): Path<String>| {
            let t = t1.clone();
            async move {
                t.lock().unwrap().entry(id).or_default();
                axum::http::StatusCode::OK
            }
        })
        .get(move |Path(id): Path<String>| {
            let t = t2.clone();
            async move { Json(json!(t.lock().unwrap().get(&id).cloned().unwrap_or_default())) }
        })
        .patch(move |Path(id): Path<String>, Json(body): Json<Value>| {
            let t = t3.clone();
            async move {
                if let Some(v) = body.as_array() {
                    t.lock().unwrap().entry(id).or_default().extend(v.clone());
                }
                axum::http::StatusCode::OK
            }
        })
        .delete(move |Path(id): Path<String>| {
            let t = t4.clone();
            async move {
                t.lock().unwrap().remove(&id);
                axum::http::StatusCode::OK
            }
        }),
    );
    let store_url = spawn(store).await;

    // Mock inferencer: returns the number of messages it received (so turn 2 proves the
    // prior was reassembled).
    let inferencer = Router::new().route(
        "/v1/chat/completions",
        post(|Json(b): Json<Value>| async move {
            let n = b.get("messages").and_then(Value::as_array).map(|a| a.len()).unwrap_or(0);
            Json(json!({"choices": [{"message": {"role": "assistant", "content": n.to_string()}}]}))
        }),
    );
    let inferencer_url = spawn(inferencer).await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(
        cfg.path().join("mcp.json"),
        json!({"mcpServers": {}, "conversationStore": {"type": "http", "url": store_url}}).to_string(),
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_OLLAMA_URL", &inferencer_url);

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    // CREATE: a non-claude model is now stateful (store-backed).
    let created = c
        .post(format!("{base}/v1/conversations"))
        .json(&json!({"model": "ollama/m", "title": "j"}))
        .send()
        .await
        .unwrap();
    assert_eq!(created.status(), 201);
    let cid = created.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_string();

    // Turn 1: store is fresh → 1 message → "1".
    let r1: Value = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "ollama/m", "conversation": cid, "input": "remember banana"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r1["output"][0]["content"][0]["text"], "1");

    // Turn 2: store-held prior (2 msgs) + the new user msg = 3 → "3" proves reassembly.
    let r2: Value = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "ollama/m", "conversation": cid, "input": "what did I say?"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r2["output"][0]["content"][0]["text"], "3");

    // /items SERVES the transcript from the store (2 user + 2 assistant).
    let items: Value = c.get(format!("{base}/v1/conversations/{cid}/items")).send().await.unwrap().json().await.unwrap();
    let roles: Vec<&str> = items["data"].as_array().unwrap().iter().filter_map(|i| i["role"].as_str()).collect();
    assert_eq!(roles.iter().filter(|r| **r == "user").count(), 2);
    assert!(roles.contains(&"assistant"));

    // DELETE removes the conversation AND tells the store to drop the thread.
    let del: Value = c.delete(format!("{base}/v1/conversations/{cid}")).send().await.unwrap().json().await.unwrap();
    assert_eq!(del["deleted"], json!(true));
    assert!(threads.lock().unwrap().is_empty(), "delete should drop the store thread");
}
