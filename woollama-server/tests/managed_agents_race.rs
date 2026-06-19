//! Concurrency regression: two simultaneous resume requests on the SAME `awaiting_input`
//! managed-agents conversation must NOT both re-resolve the pending custom_tool_use_id
//! (double-resume). The fix re-reads the conversation row under the per-conv lock, so the
//! second turn sees the first's committed `idle` state and runs a fresh turn instead.
//!
//! Without the fix, both requests act on the stale pre-lock snapshot and POST
//! `user.custom_tool_result` twice. The mock counts those POSTs; we assert exactly one.
//!
//! Separate test binary so the global env can't race other files.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::Path;
use axum::response::Response;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

fn sse(events: &[Value]) -> Response {
    let body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from(body))
        .unwrap()
}

#[tokio::test]
async fn concurrent_resumes_do_not_double_resolve() {
    // Count user.custom_tool_result POSTs (the thing under test); order the stream by call #.
    let resumes: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let stream_calls: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let resumes_post = resumes.clone();
    let stream_get = stream_calls.clone();

    let anthropic = Router::new()
        .route("/v1/environments", post(|| async { Json(json!({"id": "env1"})) }))
        .route("/v1/agents", post(|| async { Json(json!({"id": "agent1"})) }))
        .route("/v1/sessions", post(|| async { Json(json!({"id": "sess1"})) }))
        .route(
            "/v1/sessions/{id}/events",
            post(move |Path(_id): Path<String>, Json(b): Json<Value>| {
                let resumes = resumes_post.clone();
                async move {
                    if b["events"][0]["type"].as_str() == Some("user.custom_tool_result") {
                        resumes.fetch_add(1, Ordering::SeqCst);
                    }
                    Json(json!({"ok": true}))
                }
            }),
        )
        .route(
            "/v1/sessions/{id}/events/stream",
            get(move |Path(_id): Path<String>| {
                let calls = stream_get.clone();
                async move {
                    // Call 0 = the initial "ask" turn -> requires_action; later calls (the
                    // resumes) -> end_turn, with a delay so both resume requests snapshot the
                    // awaiting_input state before either commits.
                    let n = calls.fetch_add(1, Ordering::SeqCst);
                    if n == 0 {
                        sse(&[
                            json!({"type": "agent.custom_tool_use", "id": "evt1",
                                   "name": "ask_user", "input": {"question": "name?"}}),
                            json!({"type": "session.status_idle",
                                   "stop_reason": {"type": "requires_action", "event_ids": ["evt1"]}}),
                        ])
                    } else {
                        tokio::time::sleep(Duration::from_millis(150)).await;
                        sse(&[
                            json!({"type": "agent.message", "content": [{"type": "text", "text": "done"}]}),
                            json!({"type": "session.status_idle", "stop_reason": {"type": "end_turn"}}),
                        ])
                    }
                }
            }),
        )
        .route("/v1/sessions/{id}", delete(|Path(_id): Path<String>| async { Json(json!({"deleted": true})) }));
    let anthropic_url = spawn(anthropic).await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("ANTHROPIC_BASE_URL", &anthropic_url);
    std::env::set_var("ANTHROPIC_API_KEY", "test-key");

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    // Create + drive the agent to ask -> conversation goes awaiting_input (pending evt1).
    let cid = c
        .post(format!("{base}/v1/conversations"))
        .json(&json!({"model": "claude-agent/haiku"}))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();
    let ask: Value = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "claude-agent/haiku", "conversation": cid, "input": "please ask me a question"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ask["status"], "requires_action", "setup: conversation should be awaiting input");

    // Fire TWO resume requests concurrently at the same awaiting_input conversation.
    let (c1, c2) = (c.clone(), c.clone());
    let (b1, b2) = (base.clone(), base.clone());
    let (id1, id2) = (cid.clone(), cid.clone());
    let r1 = tokio::spawn(async move {
        c1.post(format!("{b1}/v1/responses"))
            .json(&json!({"model": "claude-agent/haiku", "conversation": id1, "input": "Alice"}))
            .send().await.unwrap().status()
    });
    let r2 = tokio::spawn(async move {
        c2.post(format!("{b2}/v1/responses"))
            .json(&json!({"model": "claude-agent/haiku", "conversation": id2, "input": "Bob"}))
            .send().await.unwrap().status()
    });
    let (s1, s2) = (r1.await.unwrap(), r2.await.unwrap());
    assert!(s1.is_success() && s2.is_success(), "both responses should succeed ({s1}, {s2})");

    // The pending custom_tool_use_id must be resolved EXACTLY ONCE across the two requests.
    assert_eq!(
        resumes.load(Ordering::SeqCst),
        1,
        "concurrent resumes double-resolved the pending custom_tool_use_id"
    );
}
