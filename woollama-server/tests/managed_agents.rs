//! Slice 7: the managed-agents backend WOOLLAMA-SIDE routing, against a MOCK Anthropic
//! Managed Agents server (ANTHROPIC_BASE_URL). Proves: claude-agent/* → managed-agents;
//! a turn returns text; the interactive requires_action PAUSE (agent asks) → status
//! requires_action + required_action; answering resumes → completed; /items served from
//! the event log; delete.
//!
//! The mock implements the REAL event+SSE protocol shape, reconciled against the official
//! `anthropic` SDK 0.109 and LIVE-VERIFIED against api.anthropic.com (2026-06-18):
//! - send a turn: `POST /v1/sessions/{id}/events` ({events:[user.message]})
//! - read the answer: `GET /v1/sessions/{id}/events/stream` (SSE; the DEDICATED stream route)
//! - history (items): `GET /v1/sessions/{id}/events` ({data:[...all events...]})
//! - resume: `POST /v1/sessions/{id}/events` ({events:[user.custom_tool_result]})
//!
//! The stream route is distinct from the list route — getting that wrong was the one bug the
//! live reconciliation caught. This gates the woollama ROUTING and the SSE parsing against the
//! verified shapes; the live API itself is the opt-in @needs_anthropic test.
//!
//! Separate test binary so the global env can't race other files.

use std::sync::{Arc, Mutex};

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
async fn managed_agents_routing_pause_resume_items() {
    // The mock's "what does the next streamed turn produce" state, set by the POST /events
    // and consumed by the SSE GET /events/stream — mirrors the real send-event-then-stream split.
    let next: Arc<Mutex<String>> = Arc::new(Mutex::new("hello".to_string()));
    let next_post = next.clone();
    let next_get = next.clone();

    let anthropic = Router::new()
        .route("/v1/environments", post(|| async { Json(json!({"id": "env1"})) }))
        .route("/v1/agents", post(|| async { Json(json!({"id": "agent1"})) }))
        .route("/v1/sessions", post(|| async { Json(json!({"id": "sess1"})) }))
        .route(
            "/v1/sessions/{id}/events",
            // POST = send an event (decides what the NEXT stream produces);
            // GET  = history list ({data:[...]}) — NOT the SSE stream (that's /events/stream).
            post(move |Path(_id): Path<String>, Json(b): Json<Value>| {
                let next = next_post.clone();
                async move {
                    let ev = &b["events"][0];
                    let kind = match ev.get("type").and_then(Value::as_str) {
                        Some("user.custom_tool_result") => {
                            let ans = ev["content"][0]["text"].as_str().unwrap_or("");
                            format!("answered: {ans}")
                        }
                        Some("user.message")
                            if ev["content"][0]["text"].as_str().unwrap_or("").contains("ask") =>
                        {
                            "ask".to_string()
                        }
                        _ => "hello".to_string(),
                    };
                    *next.lock().unwrap() = kind;
                    Json(json!({"ok": true}))
                }
            })
            .get(|Path(_id): Path<String>| async {
                // history (list events) — the envelope the live API returns.
                Json(json!({"data": [
                    {"type": "user.message", "content": [{"type": "text", "text": "hi"}]},
                    {"type": "agent.message", "content": [{"type": "text", "text": "hello"}]}
                ]}))
            }),
        )
        .route(
            "/v1/sessions/{id}/events/stream",
            get(move |Path(_id): Path<String>| {
                let next = next_get.clone();
                async move {
                    let kind = next.lock().unwrap().clone();
                    if kind == "ask" {
                        sse(&[
                            json!({"type": "agent.custom_tool_use", "id": "evt_tool_1",
                                   "name": "ask_user", "input": {"question": "What is your name?"}}),
                            json!({"type": "session.status_idle",
                                   "stop_reason": {"type": "requires_action", "event_ids": ["evt_tool_1"]}}),
                        ])
                    } else {
                        let text = if kind == "hello" { "hello".to_string() } else { kind };
                        sse(&[
                            json!({"type": "agent.message", "content": [{"type": "text", "text": text}]}),
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

    // CREATE: claude-agent/* → managed-agents.
    let created = c
        .post(format!("{base}/v1/conversations"))
        .json(&json!({"model": "claude-agent/haiku", "title": "j"}))
        .send()
        .await
        .unwrap();
    assert_eq!(created.status(), 201);
    let conv: Value = created.json().await.unwrap();
    let cid = conv["id"].as_str().unwrap().to_string();
    assert_eq!(conv["backend"], "managed-agents");

    // A normal turn → completed, text streamed from the (mock) hosted session.
    let r1: Value = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "claude-agent/haiku", "conversation": cid, "input": "hi"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r1["status"], "completed");
    assert_eq!(r1["output"][0]["content"][0]["text"], "hello");

    // A turn that makes the agent ask → requires_action + required_action.
    let r2: Value = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "claude-agent/haiku", "conversation": cid, "input": "please ask me a question"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r2["status"], "requires_action");
    assert_eq!(r2["required_action"]["type"], "ask_user");

    // Answering resumes → completed (via a user.custom_tool_result event).
    let r3: Value = c
        .post(format!("{base}/v1/responses"))
        .json(&json!({"model": "claude-agent/haiku", "conversation": cid, "input": "Alice"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r3["status"], "completed");
    assert_eq!(r3["output"][0]["content"][0]["text"], "answered: Alice");

    // /items SERVES the transcript from the event log (the managed-agents capability win).
    let items: Value = c.get(format!("{base}/v1/conversations/{cid}/items")).send().await.unwrap().json().await.unwrap();
    let roles: Vec<&str> = items["data"].as_array().unwrap().iter().filter_map(|i| i["role"].as_str()).collect();
    assert!(roles.contains(&"user") && roles.contains(&"assistant"));

    // DELETE tears down the session.
    let del: Value = c.delete(format!("{base}/v1/conversations/{cid}")).send().await.unwrap().json().await.unwrap();
    assert_eq!(del["deleted"], json!(true));
}
