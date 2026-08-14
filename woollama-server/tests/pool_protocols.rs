//! Task 3: config-driven `RestBackend::from_spec` + `management_protocol` resolution
//! in `PoolRegistry::from_registry`.
//!
//! Test A drives a fully config-defined ("custom") REST protocol end to end through
//! `ensure_loaded`, asserting the mock observed the configured start call (method,
//! path, body, header) and that the manager's view of loaded models matches the mock.
//! Test B is the back-compat path (no `management_protocol` => tiiny preset). Test C
//! asserts an unresolvable `management_protocol` name is isolated to the offending
//! inferencer — `from_registry` skips (and warns about) just that one inferencer,
//! while a sibling inferencer with a valid protocol is still pooled normally (review
//! fix: this used to fail the WHOLE registry, degrading every device inferencer to
//! the no-auto-load relay on a single typo). Test E covers the sibling review fix for
//! `RestBackend::list_loaded`: a `path` that resolves to a present-but-non-array value
//! is a loud `PoolError::Device`, distinct from the back-compat "absent key => empty
//! set" case.

mod common;

use std::collections::{BTreeMap, HashMap};

use common::{spawn_rest, IdLoc, RestMockConfig, RunningShape};
use woollama_engine as engine;
use woollama_server::pool::{PoolError, PoolRegistry};

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

// --- Test A: custom, config-defined REST protocol ---------------------------------

#[tokio::test]
async fn from_registry_resolves_custom_protocol_and_drives_ensure_loaded() {
    let device = spawn_rest(RestMockConfig {
        running_route: "/status".to_string(),
        start_route: "/models/load".to_string(),
        stop_route: "/models/unload".to_string(),
        running_shape: RunningShape::Objects { field: "data".to_string(), id_key: "id".to_string() },
        id_loc: IdLoc::Body { field: "model".to_string() },
    });

    let running = engine::EndpointSpec {
        url: "{base}/status".to_string(),
        method: None,
        body: None,
        headers: BTreeMap::new(),
        path: Some("data".to_string()),
        id_field: Some("id".to_string()),
    };
    let start = engine::EndpointSpec {
        url: "{base}/models/load".to_string(),
        method: Some("POST".to_string()),
        body: Some(r#"{"model": "{id}"}"#.to_string()),
        headers: BTreeMap::from([("X-Custom-Auth".to_string(), "secret-token".to_string())]),
        path: None,
        id_field: None,
    };
    let stop = engine::EndpointSpec {
        url: "{base}/models/unload".to_string(),
        method: Some("POST".to_string()),
        body: Some(r#"{"model": "{id}"}"#.to_string()),
        headers: BTreeMap::new(),
        path: None,
        id_field: None,
    };
    let mut protocols = HashMap::new();
    protocols.insert("custom".to_string(), engine::ProtocolSpec::Rest { running, start, stop });

    let mut reg = engine::Registry::new();
    reg.insert(device_inferencer("device", device.base_url.clone(), Some("custom".to_string())));

    let pools = PoolRegistry::from_registry(&reg, &protocols);
    let (manager, _gate) = pools.get("device").expect("pool built for 'device'");

    manager.ensure_loaded("m1", None).await.expect("ensure_loaded should succeed");

    let loads = device.requests_to("/models/load");
    assert_eq!(loads.len(), 1, "exactly one start request");
    assert_eq!(loads[0].method, "POST");
    assert_eq!(loads[0].path, "/models/load");
    assert_eq!(loads[0].body, r#"{"model": "m1"}"#);
    assert_eq!(loads[0].headers.get("x-custom-auth").map(String::as_str), Some("secret-token"));

    assert_eq!(device.loaded(), vec!["m1".to_string()]);
    assert_eq!(manager.snapshot(), vec!["m1".to_string()]);
}

// --- Test B: back-compat (no management_protocol => tiiny preset) -----------------

#[tokio::test]
async fn from_registry_back_compat_defaults_to_tiiny() {
    let device = spawn_rest(RestMockConfig {
        running_route: "/api/v1/models/running".to_string(),
        start_route: "/api/v1/models/{id}/start".to_string(),
        stop_route: "/api/v1/models/{id}/stop".to_string(),
        running_shape: RunningShape::Strings { field: "running".to_string() },
        id_loc: IdLoc::Path,
    });

    let mut reg = engine::Registry::new();
    reg.insert(device_inferencer("device", device.base_url.clone(), None));

    let protocols: HashMap<String, engine::ProtocolSpec> = HashMap::new();
    let pools = PoolRegistry::from_registry(&reg, &protocols);
    let (manager, _gate) = pools.get("device").expect("pool built for 'device'");

    manager.ensure_loaded("m1", None).await.expect("ensure_loaded should succeed");
    assert_eq!(device.loaded(), vec!["m1".to_string()]);
    assert_eq!(manager.snapshot(), vec!["m1".to_string()]);
}

// --- Test C: unknown protocol name is isolated to the offending inferencer --------

/// A typo'd `management_protocol` on one inferencer must not disable pooling for
/// every other device inferencer. `from_registry` skips (with a warning) only the
/// offending inferencer — its provider is absent (`get()` => `None`) from the
/// resulting `PoolRegistry` — while a sibling inferencer with a resolvable protocol
/// (here, the default `"tiiny"`) is still pooled normally.
#[test]
fn from_registry_isolates_unknown_protocol_name_to_its_own_inferencer() {
    let mut reg = engine::Registry::new();
    reg.insert(device_inferencer("bad-device", "http://device.example:8800".to_string(), Some("nope".to_string())));
    reg.insert(device_inferencer("good-device", "http://device.example:8801".to_string(), None));

    let protocols: HashMap<String, engine::ProtocolSpec> = HashMap::new();
    let pools = PoolRegistry::from_registry(&reg, &protocols);

    assert!(
        pools.get("bad-device").is_none(),
        "an inferencer with an unresolvable management_protocol must be skipped, not error out the whole registry"
    );
    assert!(
        pools.get("good-device").is_some(),
        "a sibling inferencer with a resolvable management_protocol must still get a pool"
    );
}

// --- Test D: endpoint header overrides default auth case-insensitively ------------

/// `default_headers` (from `api_key_env` via `auth_headers()`) always keys the
/// bearer token `Authorization` (capitalized). A config author overriding it on an
/// endpoint may naturally reach for the lowercase HTTP-convention spelling
/// `authorization` — that must still win, sending exactly ONE `authorization` header
/// line (the endpoint's), never both. Uses `RecordedRequest::header_count`, which
/// (unlike the last-value-wins `headers` map) preserves every header line as
/// received, so a genuine wire-level duplicate is caught deterministically instead
/// of depending on which of two same-named `HashMap` entries happens to iterate
/// last.
#[tokio::test]
async fn from_registry_endpoint_header_overrides_default_auth_case_insensitively() {
    std::env::set_var("WOOLLAMA_TEST_DEVICE_TOKEN", "default-secret");

    let device = spawn_rest(RestMockConfig {
        running_route: "/status".to_string(),
        start_route: "/models/{id}/start".to_string(),
        stop_route: "/models/{id}/stop".to_string(),
        running_shape: RunningShape::Strings { field: "running".to_string() },
        id_loc: IdLoc::Path,
    });

    let running = engine::EndpointSpec {
        url: "{base}/status".to_string(),
        method: None,
        body: None,
        headers: BTreeMap::new(),
        path: Some("running".to_string()),
        id_field: None,
    };
    let start = engine::EndpointSpec {
        url: "{base}/models/{id}/start".to_string(),
        method: Some("POST".to_string()),
        body: None,
        // Lowercase key, deliberately differing in case from the `Authorization`
        // header `auth_headers()` produces — must still override, not coexist.
        headers: BTreeMap::from([("authorization".to_string(), "endpoint-secret".to_string())]),
        path: None,
        id_field: None,
    };
    let stop = engine::EndpointSpec {
        url: "{base}/models/{id}/stop".to_string(),
        method: Some("POST".to_string()),
        body: None,
        headers: BTreeMap::new(),
        path: None,
        id_field: None,
    };
    let mut protocols = HashMap::new();
    protocols.insert("custom-auth".to_string(), engine::ProtocolSpec::Rest { running, start, stop });

    let mut inf = device_inferencer("device", device.base_url.clone(), Some("custom-auth".to_string()));
    inf.api_key_env = Some("WOOLLAMA_TEST_DEVICE_TOKEN".to_string());
    let mut reg = engine::Registry::new();
    reg.insert(inf);

    let pools = PoolRegistry::from_registry(&reg, &protocols);
    let (manager, _gate) = pools.get("device").expect("pool built for 'device'");

    manager.ensure_loaded("m1", None).await.expect("ensure_loaded should succeed");

    let starts = device.requests_to("/start");
    assert_eq!(starts.len(), 1, "exactly one start request");
    assert_eq!(
        starts[0].header_count("authorization"),
        1,
        "exactly one authorization header line must reach the device — not both the \
         default Bearer and the endpoint override"
    );
    assert_eq!(
        starts[0].headers.get("authorization").map(String::as_str),
        Some("endpoint-secret"),
        "the endpoint's own header must win over the default Bearer auth"
    );
}

// --- Test E: a present-but-non-array running path is a clear device error ---------

/// A config `path` that resolves to a PRESENT-but-non-array value (e.g. a typo that
/// lands on an object or scalar instead of the loaded-models array) must surface as a
/// clear `PoolError::Device` naming the path — not silently fall through to "no
/// models running" (which would otherwise surface downstream as a much more
/// confusing `load_timeout`-expiry "not running" error on the very next load).
#[tokio::test]
async fn ensure_loaded_errors_clearly_when_running_path_resolves_to_non_array() {
    let device = spawn_rest(RestMockConfig {
        running_route: "/status".to_string(),
        start_route: "/models/{id}/start".to_string(),
        stop_route: "/models/{id}/stop".to_string(),
        running_shape: RunningShape::NonArray {
            field: "data".to_string(),
            value: serde_json::json!({ "oops": "not an array" }),
        },
        id_loc: IdLoc::Path,
    });

    let running = engine::EndpointSpec {
        url: "{base}/status".to_string(),
        method: None,
        body: None,
        headers: BTreeMap::new(),
        path: Some("data".to_string()),
        id_field: None,
    };
    let start = engine::EndpointSpec {
        url: "{base}/models/{id}/start".to_string(),
        method: Some("POST".to_string()),
        body: None,
        headers: BTreeMap::new(),
        path: None,
        id_field: None,
    };
    let stop = engine::EndpointSpec {
        url: "{base}/models/{id}/stop".to_string(),
        method: Some("POST".to_string()),
        body: None,
        headers: BTreeMap::new(),
        path: None,
        id_field: None,
    };
    let mut protocols = HashMap::new();
    protocols.insert("bad-shape".to_string(), engine::ProtocolSpec::Rest { running, start, stop });

    let mut reg = engine::Registry::new();
    reg.insert(device_inferencer("device", device.base_url.clone(), Some("bad-shape".to_string())));

    let pools = PoolRegistry::from_registry(&reg, &protocols);
    let (manager, _gate) = pools.get("device").expect("pool built for 'device'");

    let err = manager.ensure_loaded("m1", None).await.expect_err("non-array running path must error");
    match err {
        PoolError::Device(msg) => {
            assert!(msg.contains("data"), "error should name the offending path, got: {msg}");
        }
        other => panic!("expected PoolError::Device, got {other:?}"),
    }
}
