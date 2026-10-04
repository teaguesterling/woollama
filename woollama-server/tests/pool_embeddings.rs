//! `POST /v1/embeddings` goes through the pool gate, like chat does.
//!
//! Until this was wired, embeddings were the ONE way to reach a management-capable
//! inferencer without the gate: no load-on-demand, no in-flight permit, no queue. The
//! consequence is not abstract — on the Tiiny, a bulk embedding client running beside a
//! chat client is the documented condition that wedges the NPU, and recovery requires
//! stopping every model on the device, not just the embedder.
//!
//! Two properties, both of which fail against a bare passthrough:
//!   1. load-on-demand — an embedding request starts the model if it is not resident
//!   2. serialization  — with `parallel=1`, two concurrent embedding requests do not
//!      overlap at the backend
//!
//! `FakeDevice` is a trimmed duplicate of `pool_gate.rs`'s fixture (same management
//! endpoints) with an `/embeddings` route that records observed concurrency. Separate
//! test binary: `WOOLLAMA_CONFIG_DIR` is process-global and would race other files.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

#[derive(Default)]
struct DeviceInner {
    running: std::collections::HashSet<String>,
    calls: Vec<(String, String)>,
    /// Concurrency actually observed at the embeddings backend.
    in_flight: u32,
    max_in_flight: u32,
    embed_models: Vec<String>,
}

#[derive(Clone)]
struct DeviceState {
    inner: Arc<StdMutex<DeviceInner>>,
}

async fn handle_get(State(st): State<DeviceState>, AxPath(rest): AxPath<String>) -> Response {
    if rest != "running" {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response();
    }
    let inner = st.inner.lock().unwrap();
    let mut running: Vec<String> = inner.running.iter().cloned().collect();
    running.sort();
    (StatusCode::OK, Json(json!({"object": "list", "running": running, "pending": []}))).into_response()
}

async fn handle_post(State(st): State<DeviceState>, AxPath(rest): AxPath<String>) -> Response {
    if let Some(id) = rest.strip_suffix("/start") {
        let mut inner = st.inner.lock().unwrap();
        inner.calls.push(("start".to_string(), id.to_string()));
        inner.running.insert(id.to_string());
        return (StatusCode::OK, Json(json!({"ok": true}))).into_response();
    }
    if let Some(id) = rest.strip_suffix("/stop") {
        let mut inner = st.inner.lock().unwrap();
        inner.calls.push(("stop".to_string(), id.to_string()));
        inner.running.remove(id);
        return (StatusCode::OK, Json(json!({"ok": true}))).into_response();
    }
    (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response()
}

/// Records peak concurrency, then holds long enough that an ungated second request
/// would certainly overlap. The sleep is the test's whole sensitivity: without the
/// gate both requests sit in here together and `max_in_flight` reaches 2.
async fn handle_embeddings(State(st): State<DeviceState>, Json(body): Json<Value>) -> Response {
    {
        let mut inner = st.inner.lock().unwrap();
        inner.embed_models.push(body["model"].as_str().unwrap_or("").to_string());
        inner.in_flight += 1;
        let cur = inner.in_flight;
        if cur > inner.max_in_flight {
            inner.max_in_flight = cur;
        }
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    st.inner.lock().unwrap().in_flight -= 1;
    (StatusCode::OK, Json(json!({"data": [{"embedding": [0.1, 0.2], "index": 0}]}))).into_response()
}

async fn spawn_router(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn embeddings_load_on_demand_and_serialize() {
    let inner = Arc::new(StdMutex::new(DeviceInner::default())); // nothing resident
    let device = Router::new()
        .route("/api/v1/models/{*rest}", get(handle_get).post(handle_post))
        .route("/embeddings", post(handle_embeddings))
        .with_state(DeviceState { inner: inner.clone() });
    let device_url = spawn_router(device).await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    // `parallel` is left at its default of 1 — that default is what the serialization
    // assertion below is testing, so setting it explicitly would weaken the test.
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        format!("[inferencers.device]\nbase_url=\"{u}\"\nmanagement_url=\"{u}\"\n", u = device_url),
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn_router(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    // Two concurrent embedding requests for the same model, which is not resident.
    let one = {
        let (c, base) = (c.clone(), base.clone());
        tokio::spawn(async move {
            c.post(format!("{base}/v1/embeddings"))
                .json(&json!({"model": "device/Qwen/Qwen3-Embedding-0.6B", "input": ["a"]}))
                .send()
                .await
                .unwrap()
                .status()
        })
    };
    let two = {
        let (c, base) = (c.clone(), base.clone());
        tokio::spawn(async move {
            c.post(format!("{base}/v1/embeddings"))
                .json(&json!({"model": "device/Qwen/Qwen3-Embedding-0.6B", "input": ["b"]}))
                .send()
                .await
                .unwrap()
                .status()
        })
    };
    assert_eq!(one.await.unwrap(), 200);
    assert_eq!(two.await.unwrap(), 200);

    let got = inner.lock().unwrap();

    // 1. serialization: parallel=1 means the backend never saw two at once. A bare
    //    passthrough reaches 2 here, which is the wedge condition on a real device.
    //    Asserted FIRST deliberately: it is the property this change exists for, and an
    //    earlier assertion failing would leave it unexercised in the negative control.
    assert_eq!(
        got.max_in_flight, 1,
        "parallel=1 must serialize embeddings; backend saw {} concurrent",
        got.max_in_flight
    );

    // 2. load-on-demand: the gate brought the embedder up before forwarding.
    assert!(
        got.calls.contains(&("start".to_string(), "Qwen/Qwen3-Embedding-0.6B".to_string())),
        "embeddings should load the model on demand; calls={:?}",
        got.calls
    );

    // The prefix is still stripped before forwarding (unchanged passthrough behaviour).
    assert!(
        got.embed_models.iter().all(|m| m == "Qwen/Qwen3-Embedding-0.6B"),
        "provider prefix should be stripped; saw {:?}",
        got.embed_models
    );
    assert_eq!(got.embed_models.len(), 2);
}
