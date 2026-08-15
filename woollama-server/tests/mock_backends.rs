//! Self-tests for the `tests/common` mock management-backend fixture (Deliverable 2 of
//! the pluggable-device-management-protocols test fixture). Driven with plain `reqwest`
//! — deliberately NOT any woollama-server production type, since the `rest`/`ollama`
//! `DeviceBackend` adapters this fixture exists to support don't exist yet.

mod common;

use common::{spawn_ollama, spawn_rest, IdLoc, RestMockConfig, RunningShape};

// --- rest, path-style ids (device-like) --------------------------------------------

fn device_like_cfg() -> RestMockConfig {
    RestMockConfig {
        running_route: "/api/v1/models/running".to_string(),
        start_route: "/api/v1/models/{id}/start".to_string(),
        stop_route: "/api/v1/models/{id}/stop".to_string(),
        running_shape: RunningShape::Strings { field: "running".to_string() },
        id_loc: IdLoc::Path,
    }
}

#[tokio::test]
async fn rest_path_style_set_loaded_reflects_in_running() {
    let device = spawn_rest(device_like_cfg());
    device.set_loaded(&["A", "B"]);

    let client = reqwest::Client::new();
    let resp = client.get(format!("{}/api/v1/models/running", device.base_url)).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let running: Vec<String> =
        body["running"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(running, vec!["A".to_string(), "B".to_string()]);
}

#[tokio::test]
async fn rest_path_style_top_level_array_shape() {
    let mut cfg = device_like_cfg();
    cfg.running_shape = RunningShape::Strings { field: String::new() };
    let device = spawn_rest(cfg);
    device.set_loaded(&["A"]);

    let client = reqwest::Client::new();
    let resp = client.get(format!("{}/api/v1/models/running", device.base_url)).send().await.unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body, serde_json::json!(["A"]));
}

#[tokio::test]
async fn rest_path_style_start_and_stop_are_recorded_and_mutate_state() {
    let device = spawn_rest(device_like_cfg());
    let client = reqwest::Client::new();

    let resp = client.post(format!("{}/api/v1/models/foo/start", device.base_url)).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(device.loaded(), vec!["foo".to_string()]);

    let starts = device.requests_to("/start");
    assert_eq!(starts.len(), 1);
    assert_eq!(starts[0].method, "POST");
    assert_eq!(starts[0].path, "/api/v1/models/foo/start");

    let resp = client.post(format!("{}/api/v1/models/foo/stop", device.base_url)).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert!(device.loaded().is_empty());
    assert_eq!(device.requests_to("/stop").len(), 1);
}

#[tokio::test]
async fn rest_path_style_fail_start_returns_500() {
    let device = spawn_rest(device_like_cfg());
    device.set_fail_start(true);

    let client = reqwest::Client::new();
    let resp = client.post(format!("{}/api/v1/models/foo/start", device.base_url)).send().await.unwrap();
    assert_eq!(resp.status(), 500);
    assert!(device.loaded().is_empty(), "failed start must not register the model as loaded");
}

#[tokio::test]
async fn rest_path_style_fail_stop_returns_500() {
    let device = spawn_rest(device_like_cfg());
    device.set_loaded(&["foo"]);
    device.set_fail_stop(true);

    let client = reqwest::Client::new();
    let resp = client.post(format!("{}/api/v1/models/foo/stop", device.base_url)).send().await.unwrap();
    assert_eq!(resp.status(), 500);
    assert_eq!(device.loaded(), vec!["foo".to_string()], "failed stop must not unregister the model");
}

#[tokio::test]
async fn rest_path_style_slash_bearing_id_round_trips() {
    let device = spawn_rest(device_like_cfg());
    let client = reqwest::Client::new();
    let id = "Qwen/Coder";

    let resp = client.post(format!("{}/api/v1/models/{id}/start", device.base_url)).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(device.loaded(), vec![id.to_string()]);

    let resp = client.get(format!("{}/api/v1/models/running", device.base_url)).send().await.unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["running"], serde_json::json!([id]));

    let resp = client.post(format!("{}/api/v1/models/{id}/stop", device.base_url)).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert!(device.loaded().is_empty());

    // Recorded requests preserve the raw (unencoded) slash-bearing path.
    let starts = device.requests_to("/start");
    assert_eq!(starts[0].path, "/api/v1/models/Qwen/Coder/start");
}

// --- rest, body-style ids (LM-Studio-like) ----------------------------------------

fn lmstudio_like_cfg() -> RestMockConfig {
    RestMockConfig {
        running_route: "/api/v0/models".to_string(),
        start_route: "/api/v0/models/load".to_string(),
        stop_route: "/api/v0/models/unload".to_string(),
        running_shape: RunningShape::Objects { field: "data".to_string(), id_key: "id".to_string() },
        id_loc: IdLoc::Body { field: "model".to_string() },
    }
}

#[tokio::test]
async fn rest_body_style_object_array_running_shape() {
    let device = spawn_rest(lmstudio_like_cfg());
    device.set_loaded(&["Qwen/Coder", "Llama/8B"]);

    let client = reqwest::Client::new();
    let resp = client.get(format!("{}/api/v0/models", device.base_url)).send().await.unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    let ids: Vec<String> = body["data"].as_array().unwrap().iter().map(|v| v["id"].as_str().unwrap().to_string()).collect();
    assert_eq!(ids, vec!["Llama/8B".to_string(), "Qwen/Coder".to_string()]);
}

#[tokio::test]
async fn rest_body_style_load_by_body_field_and_custom_header_is_captured() {
    let device = spawn_rest(lmstudio_like_cfg());
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/api/v0/models/load", device.base_url))
        .header("X-Custom-Auth", "secret-token")
        .json(&serde_json::json!({"model": "X"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(device.loaded(), vec!["X".to_string()]);

    let loads = device.requests_to("/load");
    assert_eq!(loads.len(), 1);
    assert_eq!(loads[0].body, r#"{"model":"X"}"#);
    assert_eq!(loads[0].headers.get("x-custom-auth").map(String::as_str), Some("secret-token"));
}

#[tokio::test]
async fn rest_body_style_unload_by_body_field() {
    let device = spawn_rest(lmstudio_like_cfg());
    device.set_loaded(&["X"]);
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/api/v0/models/unload", device.base_url))
        .json(&serde_json::json!({"model": "X"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(device.loaded().is_empty());
}

// --- ollama ------------------------------------------------------------------------

#[tokio::test]
async fn ollama_ps_lists_loaded_as_name_and_model() {
    let device = spawn_ollama();
    device.set_loaded(&["llama3", "qwen3"]);

    let client = reqwest::Client::new();
    let resp = client.get(format!("{}/api/ps", device.base_url)).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let models = body["models"].as_array().unwrap();
    assert_eq!(models.len(), 2);
    for m in models {
        assert_eq!(m["name"], m["model"]);
    }
}

#[tokio::test]
async fn ollama_generate_with_keep_alive_loads_model() {
    let device = spawn_ollama();
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/api/generate", device.base_url))
        .json(&serde_json::json!({"model": "qwen3", "keep_alive": "30m"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(device.loaded(), vec!["qwen3".to_string()]);

    let calls = device.requests_to("/api/generate");
    assert_eq!(calls.len(), 1);
    let recorded_body: serde_json::Value = serde_json::from_str(&calls[0].body).unwrap();
    assert_eq!(recorded_body["model"], "qwen3");
    assert_eq!(recorded_body["keep_alive"], "30m");
}

#[tokio::test]
async fn ollama_generate_with_zero_keep_alive_unloads_model() {
    let device = spawn_ollama();
    device.set_loaded(&["qwen3"]);
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/api/generate", device.base_url))
        .json(&serde_json::json!({"model": "qwen3", "keep_alive": 0}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(device.loaded().is_empty());

    let calls = device.requests_to("/api/generate");
    let recorded_body: serde_json::Value = serde_json::from_str(&calls[0].body).unwrap();
    assert_eq!(recorded_body["model"], "qwen3");
    assert_eq!(recorded_body["keep_alive"], 0);
}

#[tokio::test]
async fn ollama_generate_string_zero_keep_alive_also_unloads() {
    let device = spawn_ollama();
    device.set_loaded(&["qwen3"]);
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/api/generate", device.base_url))
        .json(&serde_json::json!({"model": "qwen3", "keep_alive": "0"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(device.loaded().is_empty());
}

// --- recording sanity ----------------------------------------------------------

#[tokio::test]
async fn recording_captures_lowercased_headers_and_verbatim_body() {
    let device = spawn_rest(device_like_cfg());
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/api/v1/models/foo/start", device.base_url))
        .header("X-Weird-CASE-Header", "value1")
        .body("not-json-but-recorded-verbatim")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let reqs = device.requests();
    let start_req = reqs.iter().find(|r| r.path == "/api/v1/models/foo/start").unwrap();
    assert_eq!(start_req.headers.get("x-weird-case-header").map(String::as_str), Some("value1"));
    assert!(!start_req.headers.contains_key("X-Weird-CASE-Header"), "header keys must be lowercased");
    assert_eq!(start_req.body, "not-json-but-recorded-verbatim");
}
