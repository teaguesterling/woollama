//! Task 4: `OllamaBackend` + `management_protocol = "ollama"` resolution in
//! `PoolRegistry::from_registry`.
//!
//! Test A drives the built-in `"ollama"` protocol (no configured `keep_alive`) through
//! `ensure_loaded`, asserting the warm-up `/api/generate` call carries the model id and
//! omits `keep_alive` entirely, and that `list_loaded`/`snapshot` reflect `/api/ps`.
//! Test B forces an eviction (`pool_max = 1`, load a second model) and asserts the
//! victim gets `/api/generate` with `keep_alive: 0`. Test C covers the config
//! `[management_protocols.x] kind = "ollama"` path with a configured `keep_alive`,
//! asserting it's forwarded on the load body.

mod common;

use std::collections::HashMap;

use common::spawn_ollama;
use woollama_engine as engine;
use woollama_server::pool::PoolRegistry;

fn device_inferencer(name: &str, management_url: String, management_protocol: Option<String>) -> engine::Inferencer {
    engine::Inferencer {
        name: name.to_string(),
        base_url: "http://device.example/v1".to_string(),
        api_key_env: None,
        extra_body: serde_json::json!({}),
        models: Vec::new(),
        discover: false,
        model_patterns: Vec::new(),
        management_url: Some(management_url),
        management_protocol,
        parallel: 1,
        pool_max: None,
        queue_max: None,
        queue_timeout: 30.0,
        virtual_models: Default::default(),
    }
}

// --- Test A: built-in "ollama" protocol, no configured keep_alive -----------------

#[tokio::test]
async fn from_registry_resolves_builtin_ollama_and_drives_ensure_loaded() {
    let device = spawn_ollama();

    let mut reg = engine::Registry::new();
    reg.insert(device_inferencer("device", device.base_url.clone(), Some("ollama".to_string())));

    let protocols: HashMap<String, engine::ProtocolSpec> = HashMap::new();
    let pools = PoolRegistry::from_registry(&reg, &protocols);
    let (manager, _gate) = pools.get("device").expect("pool built for 'device'");

    manager.ensure_loaded("qwen3:14b", None).await.expect("ensure_loaded should succeed");

    let loads = device.requests_to("/api/generate");
    assert_eq!(loads.len(), 1, "exactly one warm-up generate request");
    assert_eq!(loads[0].method, "POST");
    let body: serde_json::Value = serde_json::from_str(&loads[0].body).expect("valid JSON body");
    assert_eq!(body.get("model").and_then(serde_json::Value::as_str), Some("qwen3:14b"));
    assert!(
        body.get("keep_alive").is_none(),
        "no configured keep_alive => the field must be omitted so Ollama's own default applies, got {body}"
    );

    assert_eq!(device.loaded(), vec!["qwen3:14b".to_string()]);
    assert_eq!(manager.snapshot(), vec!["qwen3:14b".to_string()]);
}

// --- Test B: eviction unloads the victim with keep_alive: 0 -----------------------

#[tokio::test]
async fn ensure_loaded_evicts_victim_with_keep_alive_zero() {
    let device = spawn_ollama();

    let mut reg = engine::Registry::new();
    reg.insert(device_inferencer("device", device.base_url.clone(), Some("ollama".to_string())));

    let protocols: HashMap<String, engine::ProtocolSpec> = HashMap::new();
    let pools = PoolRegistry::from_registry(&reg, &protocols);
    let (manager, _gate) = pools.get("device").expect("pool built for 'device'");

    manager.ensure_loaded("qwen3:14b", Some(1)).await.expect("first load should succeed");
    manager.ensure_loaded("llama3:8b", Some(1)).await.expect("second load should evict the first");

    let unloads = device.requests_to("/api/generate");
    let evict = unloads
        .iter()
        .find(|r| {
            let v: serde_json::Value = serde_json::from_str(&r.body).unwrap();
            v.get("model").and_then(serde_json::Value::as_str) == Some("qwen3:14b")
                && v.get("keep_alive").and_then(serde_json::Value::as_f64) == Some(0.0)
        })
        .expect("victim must receive a generate call with keep_alive: 0");
    let victim_body: serde_json::Value = serde_json::from_str(&evict.body).unwrap();
    assert_eq!(victim_body.get("keep_alive"), Some(&serde_json::json!(0)));

    assert_eq!(device.loaded(), vec!["llama3:8b".to_string()]);
    assert_eq!(manager.snapshot(), vec!["llama3:8b".to_string()]);
}

// --- Test C: config-defined kind = "ollama" with a configured keep_alive -----------

#[tokio::test]
async fn from_registry_resolves_config_ollama_and_forwards_keep_alive() {
    let device = spawn_ollama();

    let mut protocols = HashMap::new();
    protocols
        .insert("ollama-custom".to_string(), engine::ProtocolSpec::Ollama { keep_alive: Some("5m".to_string()) });

    let mut reg = engine::Registry::new();
    reg.insert(device_inferencer("device", device.base_url.clone(), Some("ollama-custom".to_string())));

    let pools = PoolRegistry::from_registry(&reg, &protocols);
    let (manager, _gate) = pools.get("device").expect("pool built for 'device'");

    manager.ensure_loaded("qwen3:14b", None).await.expect("ensure_loaded should succeed");

    let loads = device.requests_to("/api/generate");
    assert_eq!(loads.len(), 1, "exactly one warm-up generate request");
    let body: serde_json::Value = serde_json::from_str(&loads[0].body).expect("valid JSON body");
    assert_eq!(body.get("model").and_then(serde_json::Value::as_str), Some("qwen3:14b"));
    assert_eq!(
        body.get("keep_alive").and_then(serde_json::Value::as_str),
        Some("5m"),
        "configured keep_alive must be forwarded on the load body"
    );
}
