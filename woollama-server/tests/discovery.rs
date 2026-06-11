//! Slice 8: GET /v1/models discovery. Against a mock provider /v1/models, proves: live
//! `discover` (ollama lists its catalog namespaced), `model_patterns` filtering, static
//! `models`, and `woollama/<recipe>` entries.
//!
//! Separate test binary so the global WOOLLAMA_* env can't race other files.

use std::sync::Arc;

use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn v1_models_discovery() {
    // A mock provider /v1/models catalog (used by every discover-enabled inferencer here).
    let upstream = Router::new().route(
        "/v1/models",
        get(|| async {
            Json(json!({"object": "list", "data": [
                {"id": "qwen3:14b"}, {"id": "keep-this"}, {"id": "drop-that"}
            ]}))
        }),
    );
    let upstream_url = spawn(upstream).await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "[recipes.r1]\ninferencer=\"ollama/m\"\ntools=[]\nsystem=\"x\"\n").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    // A static-models provider and a pattern-filtered discover provider (both point their
    // /v1/models at the mock).
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        format!(
            "[inferencers.myprovider]\nbase_url=\"{u}/v1\"\nmodels=[\"static-1\",\"static-2\"]\n\
             [inferencers.filtered]\nbase_url=\"{u}/v1\"\ndiscover=true\nmodel_patterns=[\"keep-*\"]\n",
            u = upstream_url
        ),
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_OLLAMA_URL", &upstream_url);

    let state = Arc::new(woollama_server::build_state().await);
    let base = spawn(woollama_server::router(state)).await;
    let c = reqwest::Client::new();

    let m: Value = c.get(format!("{base}/v1/models")).send().await.unwrap().json().await.unwrap();
    assert_eq!(m["object"], "list");
    let ids: Vec<&str> = m["data"].as_array().unwrap().iter().filter_map(|x| x["id"].as_str()).collect();

    // ollama (discover, no patterns) → its whole catalog, namespaced.
    assert!(ids.contains(&"ollama/qwen3:14b"), "ollama discovery; got {ids:?}");
    assert!(ids.contains(&"ollama/keep-this") && ids.contains(&"ollama/drop-that"));
    // filtered (discover + model_patterns) → only matching ids.
    assert!(ids.contains(&"filtered/keep-this"), "pattern keep");
    assert!(!ids.contains(&"filtered/drop-that"), "pattern must filter out non-matches");
    // static models (no discovery).
    assert!(ids.contains(&"myprovider/static-1") && ids.contains(&"myprovider/static-2"));
    // recipes.
    assert!(ids.contains(&"woollama/r1"), "recipe listed");
}
