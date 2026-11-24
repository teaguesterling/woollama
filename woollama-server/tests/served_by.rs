//! Every routed reply says which route and which model served it.
//!
//! A reply's body carries only the bare model id the backend chose, so a client that asked for
//! `device/default` cannot tell what answered. Callers have had to compare device clock times to
//! prove which device served an embedding. woollamad is the one component that knows both facts —
//! the route it picked and the model `default` resolved to — so it says them, in headers that no
//! client has to parse the body for:
//!
//!   x-woollama-route   the inferencer namespace (a chained hop prefixes its own: `hop/device`)
//!   x-woollama-model   the model id sent to the backend, after `default` and aliases resolve
//!
//! Properties, each failing against a woollamad without the feature:
//!   1. pooled `default` names the RESOLVED model, not the literal "default"
//!   2. embeddings carry the headers too
//!   3. a stream carries them (headers go out before the first chunk)
//!   4. a woollamad that relays to another woollamad passes the inner answer through, so a
//!      client behind a forwarder still learns the real model
//!
//! Separate test binary: `WOOLLAMA_CONFIG_DIR` is process-global and would race other files.

use std::sync::Arc;

use axum::extract::Path as AxPath;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

const RESIDENT: &str = "Qwen/Chat-A";

async fn handle_get(AxPath(rest): AxPath<String>) -> Response {
    if rest != "running" {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response();
    }
    Json(json!({"object": "list", "running": [RESIDENT, "Qwen/Emb"], "pending": []})).into_response()
}

async fn handle_post(AxPath(_rest): AxPath<String>) -> Response {
    Json(json!({"ok": true})).into_response()
}

async fn handle_chat(Json(body): Json<Value>) -> Response {
    let model = body["model"].as_str().unwrap_or("").to_string();
    if body["stream"].as_bool().unwrap_or(false) {
        let chunk = json!({"choices": [{"delta": {"content": "hi"}, "index": 0}], "model": model});
        let sse = format!("data: {chunk}\n\ndata: [DONE]\n\n");
        return ([("content-type", "text/event-stream")], sse).into_response();
    }
    Json(json!({"model": model, "choices": [{"message": {"role": "assistant", "content": "hi"}}]}))
        .into_response()
}

async fn handle_embeddings(Json(body): Json<Value>) -> Response {
    Json(json!({"model": body["model"], "data": [{"embedding": [0.1, 0.2], "index": 0}]})).into_response()
}

async fn spawn_router(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

fn write_cfg(dir: &std::path::Path, inferencers: &str) {
    std::fs::write(dir.join("recipes.toml"), "").unwrap();
    std::fs::write(dir.join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::fs::write(dir.join("inferencers.toml"), inferencers).unwrap();
}

fn served(r: &reqwest::Response) -> (Option<String>, Option<String>) {
    let h = |n: &str| r.headers().get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
    (h("x-woollama-route"), h("x-woollama-model"))
}

fn some(route: &str, model: &str) -> (Option<String>, Option<String>) {
    (Some(route.to_string()), Some(model.to_string()))
}

#[tokio::test]
async fn replies_name_the_route_and_the_resolved_model() {
    let device = Router::new()
        .route("/api/v1/models/{*rest}", get(handle_get).post(handle_post))
        .route("/chat/completions", post(handle_chat))
        .route("/embeddings", post(handle_embeddings));
    let device_url = spawn_router(device).await;

    // Inner woollamad: one pooled device route.
    let inner_cfg = tempfile::tempdir().unwrap();
    write_cfg(
        inner_cfg.path(),
        &format!("[inferencers.device]\nbase_url=\"{u}\"\nmanagement_url=\"{u}\"\n", u = device_url),
    );
    std::env::set_var("WOOLLAMA_CONFIG_DIR", inner_cfg.path());
    let inner = spawn_router(woollama_server::router(Arc::new(woollama_server::build_state().await))).await;

    let c = reqwest::Client::new();
    let chat = |base: &str, model: &str, stream: bool| {
        c.post(format!("{base}/v1/chat/completions"))
            .json(&json!({"model": model, "stream": stream, "messages": [{"role": "user", "content": "hi"}]}))
            .send()
    };

    // 1. pooled `default` names the model it resolved to.
    let r = chat(&inner, "device/default", false).await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(served(&r), some("device", RESIDENT), "default must name the resolved model");

    // 2. embeddings.
    let r = c
        .post(format!("{inner}/v1/embeddings"))
        .json(&json!({"model": "device/Qwen/Emb", "input": ["a"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(served(&r), some("device", "Qwen/Emb"));

    // 3. a stream carries the headers.
    let r = chat(&inner, "device/default", true).await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(served(&r), some("device", RESIDENT), "a stream must carry the headers");
    let _ = r.text().await;

    // 4. a forwarder (plain relay to the inner woollamad) passes the inner answer through.
    let outer_cfg = tempfile::tempdir().unwrap();
    write_cfg(outer_cfg.path(), &format!("[inferencers.hop]\nbase_url=\"{inner}/v1\"\n"));
    std::env::set_var("WOOLLAMA_CONFIG_DIR", outer_cfg.path());
    let outer = spawn_router(woollama_server::router(Arc::new(woollama_server::build_state().await))).await;

    let r = chat(&outer, "hop/device/default", false).await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(served(&r), some("hop/device", RESIDENT), "a hop must pass the inner route and model through");

    let r = chat(&outer, "hop/device/default", true).await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(served(&r), some("hop/device", RESIDENT), "a streamed hop must pass them through too");
    let _ = r.text().await;
}
