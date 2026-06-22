//! Tier 2a — the fabric backend's name cache stays fresh while fabric hot-reloads its pattern
//! dir. A mock fabric (external-`url` mode) grows its `/patterns/names` from `[seed]` to
//! `[seed, added]` after connect; with the refresh TTL forced to 0, woollama's traffic-driven
//! re-source must pick `added` up WITHOUT a restart, and then ROUTE it (render dispatches via the
//! refreshed `has`). url mode never respawns — the managed kill+respawn path is verified live,
//! not here (no fabric binary in a hermetic test).
//!
//! Single test fn (one binary) so the process-global WOOLLAMA_* env can't race.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn fabric_name_cache_refreshes_on_hot_reload() {
    // `/patterns/names` returns [seed] for the first 2 calls (connect does ready + fetch_names),
    // then [seed, added] — i.e. fabric "gained" a pattern after woollama connected.
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let fabric = Router::new()
        .route(
            "/patterns/names",
            get(|| async {
                let n = CALLS.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Json(json!(["seed"]))
                } else {
                    Json(json!(["seed", "added"]))
                }
            }),
        )
        .route("/models/names", get(|| async { Json(json!({"vendors": {"Ollama": ["m"]}})) }))
        .route("/patterns/seed", get(|| async { Json(json!({"Pattern": "Seed system."})) }))
        .route("/patterns/added", get(|| async { Json(json!({"Pattern": "Added system."})) }))
        .route(
            "/chat",
            post(|| async {
                let sse = format!(
                    "data: {}\n\ndata: {}\n\n",
                    json!({"type": "content", "content": "ok"}),
                    json!({"type": "complete"})
                );
                ([("content-type", "text/event-stream")], sse).into_response()
            }),
        );
    let fabric_url = spawn(fabric).await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(
        cfg.path().join("mcp.json"),
        json!({"fabric": {"url": fabric_url, "default_model": "ollama/m"}}).to_string(),
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    // Refresh on every read so the test doesn't have to wait out a 60s TTL.
    std::env::set_var("WOOLLAMA_FABRIC_REFRESH_SECS", "0");

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    let names = |body: &Value| -> Vec<String> {
        body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["name"].as_str().unwrap().to_string())
            .collect()
    };

    // Initially only [seed] is cached (connect's two /patterns/names calls both returned [seed]).
    let first: Value = c.get(format!("{base}/w1/patterns")).send().await.unwrap().json().await.unwrap();
    let first = names(&first);
    assert!(first.contains(&"seed".to_string()), "seed present from connect");
    assert!(!first.contains(&"added".to_string()), "added not yet cached (eventual, not on the triggering call)");

    // Each read kicks a detached re-source; poll until `added` shows (it must, fabric now lists it).
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut saw_added = false;
    while Instant::now() < deadline {
        let b: Value = c.get(format!("{base}/w1/patterns")).send().await.unwrap().json().await.unwrap();
        let ns = names(&b);
        assert!(ns.contains(&"seed".to_string()), "seed never disappears");
        if ns.contains(&"added".to_string()) {
            saw_added = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(saw_added, "the hot-reloaded `added` pattern was re-sourced into the cache");

    // And it's now ROUTABLE: render dispatches through the refreshed `has` and fetches its system.
    let r = c
        .post(format!("{base}/w1/patterns/added/render"))
        .json(&json!({"input": "BODY"}))
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success(), "newly-sourced pattern routes (has() sees it)");
    let body: Value = r.json().await.unwrap();
    let prompt = body["prompt"].as_str().unwrap();
    assert!(prompt.contains("Added system."), "fabric system fetched for the new pattern");
    assert!(prompt.contains("BODY"), "input appended");

    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
    std::env::remove_var("WOOLLAMA_FABRIC_REFRESH_SECS");
}
