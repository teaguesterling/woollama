//! Task 3 (rust-parity-port): `POST /v1/images/generations` + `POST /v1/embeddings`
//! passthrough — siblings of the chat passthrough non-stream path. Proves: the namespace
//! prefix is resolved through the config-merged `state.inferencers` Registry (a config-only
//! "device" inferencer, never a built-in), the prefix is stripped from the forwarded `model`,
//! the upstream response is relayed verbatim, and an unknown provider 400s.
//!
//! Separate test binary so the global WOOLLAMA_* env can't race other files. A single
//! #[tokio::test] (mirroring passthrough_config.rs / discovery.rs) — WOOLLAMA_CONFIG_DIR is a
//! process-global env var, so multiple tests in one binary would race.

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

#[derive(Clone, Default)]
struct Seen {
    images_model: Arc<Mutex<Option<String>>>,
    embeddings_model: Arc<Mutex<Option<String>>>,
}

#[tokio::test]
async fn images_and_embeddings_passthrough() {
    let seen = Seen::default();

    let upstream = Router::new()
        .route(
            "/images/generations",
            post({
                let seen = seen.clone();
                move |State(_): State<()>, Json(body): Json<Value>| {
                    let seen = seen.clone();
                    async move {
                        *seen.images_model.lock().unwrap() = body["model"].as_str().map(str::to_string);
                        Json(json!({"data": [{"b64_json": "x"}]}))
                    }
                }
            }),
        )
        .route(
            "/embeddings",
            post({
                let seen = seen.clone();
                move |State(_): State<()>, Json(body): Json<Value>| {
                    let seen = seen.clone();
                    async move {
                        *seen.embeddings_model.lock().unwrap() = body["model"].as_str().map(str::to_string);
                        Json(json!({"data": [{"embedding": [0.1, 0.2]}]}))
                    }
                }
            }),
        )
        .with_state(());
    let upstream_url = spawn(upstream).await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        format!("[inferencers.device]\nbase_url=\"{u}\"\n", u = upstream_url),
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    // --- images: prefix stripped, upstream response relayed ---
    let r = c
        .post(format!("{base}/v1/images/generations"))
        .json(&json!({"model": "device/img", "prompt": "a cat"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body, json!({"data": [{"b64_json": "x"}]}));
    assert_eq!(seen.images_model.lock().unwrap().as_deref(), Some("img"));

    // --- embeddings: prefix stripped, upstream response relayed ---
    let r = c
        .post(format!("{base}/v1/embeddings"))
        .json(&json!({"model": "device/embed", "input": "hello"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body, json!({"data": [{"embedding": [0.1, 0.2]}]}));
    assert_eq!(seen.embeddings_model.lock().unwrap().as_deref(), Some("embed"));

    // --- unknown provider -> 400, for both routes ---
    let r = c
        .post(format!("{base}/v1/images/generations"))
        .json(&json!({"model": "nope/x", "prompt": "a cat"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);

    let r = c
        .post(format!("{base}/v1/embeddings"))
        .json(&json!({"model": "nope/x", "input": "hello"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
}
