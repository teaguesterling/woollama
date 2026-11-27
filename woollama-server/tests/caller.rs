//! A hop passes the caller on, so the woollamad nearest the device logs who really sent a request.
//!
//! Behind a forwarding woollamad, every request arrives from the forwarder. Without this, the
//! device-side access line can name only the forwarder, and a busy caller hides behind it.
//!
//! Properties, each failing against a woollamad without the feature:
//!   1. a declared `x-woollama-caller` reaches the backend unchanged, through a hop
//!   2. with no declaration, the client's User-Agent product reaches it as `ua:<product>`
//!
//! Separate test binary: `WOOLLAMA_CONFIG_DIR` is process-global and would race other files.

use std::sync::{Arc, Mutex};

use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{extract::State, Json, Router};
use serde_json::{json, Value};

type Seen = Arc<Mutex<Vec<Option<String>>>>;

async fn handle_chat(State(seen): State<Seen>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    let caller = headers.get("x-woollama-caller").and_then(|v| v.to_str().ok()).map(str::to_string);
    seen.lock().unwrap().push(caller);
    Json(json!({"model": body["model"], "choices": [{"message": {"role": "assistant", "content": "hi"}}]}))
        .into_response()
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

async fn woollamad(cfg: &tempfile::TempDir, inferencers: &str) -> String {
    write_cfg(cfg.path(), inferencers);
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    spawn_router(woollama_server::router(Arc::new(woollama_server::build_state().await))).await
}

#[tokio::test]
async fn a_hop_passes_the_caller_to_the_backend() {
    let seen: Seen = Arc::default();
    let backend = spawn_router(Router::new().route("/chat/completions", post(handle_chat)).with_state(seen.clone())).await;

    let inner_cfg = tempfile::tempdir().unwrap();
    let inner = woollamad(&inner_cfg, &format!("[inferencers.backend]\nbase_url=\"{backend}\"\n")).await;
    let outer_cfg = tempfile::tempdir().unwrap();
    let outer = woollamad(&outer_cfg, &format!("[inferencers.hop]\nbase_url=\"{inner}/v1\"\n")).await;

    let body = json!({"model": "hop/backend/m", "messages": [{"role": "user", "content": "hi"}]});
    let c = reqwest::Client::new();

    // 1. a declared caller, through the hop.
    let r = c
        .post(format!("{outer}/v1/chat/completions"))
        .header("x-woollama-caller", "kb-indexer")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap_or_default());

    // 2. no declaration: the User-Agent product.
    let r = c
        .post(format!("{outer}/v1/chat/completions"))
        .header("user-agent", "pi-agent/0.9 (linux)")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    let seen = seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        vec![Some("kb-indexer".to_string()), Some("ua:pi-agent/0.9".to_string())],
        "the backend must see the original caller, not the hop"
    );
}
