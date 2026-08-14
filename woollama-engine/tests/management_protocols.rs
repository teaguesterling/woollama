//! Task 2: the `management_protocol` selector field on `Inferencer` plus the typed
//! `[management_protocols.<name>]` parser (`load_management_protocols`). A protocol
//! block describes HOW to talk to a device's management API (REST endpoints, or
//! ollama's native keep_alive knob); `management_protocol` on an inferencer just
//! names which block applies.
//!
//! Separate test binary so the global WOOLLAMA_CONFIG_DIR env can't race other files.

use std::collections::BTreeMap;

use woollama_engine::{load_management_protocols, EndpointSpec, ProtocolSpec, Registry};

/// One WOOLLAMA_CONFIG_DIR-touching test per file (see inferencer_pooling.rs) — all
/// assertions live in one #[test] so env mutation stays sequential, not racing across
/// parallel test threads.
#[test]
fn protocol_field_and_parser() {
    std::env::set_var("MYBOX_TOKEN", "secret-123");

    // --- happy path: selector field + rest/ollama protocol parsing ---
    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        r#"
[inferencers.dev]
base_url = "http://dev/v1"
management_url = "http://dev:8800"
management_protocol = "mybox"

[management_protocols.mybox]
kind = "rest"

[management_protocols.mybox.endpoints.running]
url = "http://dev:8800/api/ps"
path = "models"
id_field = "id"

[management_protocols.mybox.endpoints.running.headers]
X-Token = "${MYBOX_TOKEN}"

[management_protocols.mybox.endpoints.start]
url = "http://dev:8800/api/start"
method = "POST"
body = '{"model": "{model}"}'

[management_protocols.mybox.endpoints.stop]
url = "http://dev:8800/api/stop"
method = "POST"

[management_protocols.oll]
kind = "ollama"
keep_alive = "5m"
"#,
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());

    let reg = Registry::from_config().unwrap();
    let dev = reg.resolve("dev").expect("dev inferencer");
    assert_eq!(dev.management_protocol, Some("mybox".to_string()));

    let plain = reg.resolve("anthropic").expect("anthropic builtin");
    assert_eq!(plain.management_protocol, None, "builtin has no protocol selector by default");

    let protocols = load_management_protocols().unwrap();
    assert_eq!(protocols.len(), 2);

    match protocols.get("mybox").expect("mybox protocol") {
        ProtocolSpec::Rest { running, start, stop } => {
            assert_eq!(
                running,
                &EndpointSpec {
                    url: "http://dev:8800/api/ps".to_string(),
                    method: None,
                    body: None,
                    headers: BTreeMap::from([("X-Token".to_string(), "secret-123".to_string())]),
                    path: Some("models".to_string()),
                    id_field: Some("id".to_string()),
                }
            );
            assert_eq!(
                start,
                &EndpointSpec {
                    url: "http://dev:8800/api/start".to_string(),
                    method: Some("POST".to_string()),
                    body: Some(r#"{"model": "{model}"}"#.to_string()),
                    headers: BTreeMap::new(),
                    path: None,
                    id_field: None,
                }
            );
            assert_eq!(
                stop,
                &EndpointSpec {
                    url: "http://dev:8800/api/stop".to_string(),
                    method: Some("POST".to_string()),
                    body: None,
                    headers: BTreeMap::new(),
                    path: None,
                    id_field: None,
                }
            );
        }
        other => panic!("expected Rest, got {other:?}"),
    }

    match protocols.get("oll").expect("oll protocol") {
        ProtocolSpec::Ollama { keep_alive } => assert_eq!(keep_alive, &Some("5m".to_string())),
        other => panic!("expected Ollama, got {other:?}"),
    }

    // --- error case: unknown kind ---
    let cfg_bad_kind = tempfile::tempdir().unwrap();
    std::fs::write(
        cfg_bad_kind.path().join("inferencers.toml"),
        r#"
[management_protocols.weird]
kind = "carrier-pigeon"
"#,
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg_bad_kind.path());
    let err = load_management_protocols().unwrap_err();
    assert!(err.message.contains("weird"), "error should name the offending protocol: {}", err.message);
    assert!(err.message.contains("kind"), "error should name the offending key: {}", err.message);

    // --- error case: rest block missing endpoints.stop ---
    let cfg_missing_stop = tempfile::tempdir().unwrap();
    std::fs::write(
        cfg_missing_stop.path().join("inferencers.toml"),
        r#"
[management_protocols.incomplete]
kind = "rest"

[management_protocols.incomplete.endpoints.running]
url = "http://dev:8800/api/ps"
path = "models"

[management_protocols.incomplete.endpoints.start]
url = "http://dev:8800/api/start"
"#,
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg_missing_stop.path());
    let err = load_management_protocols().unwrap_err();
    assert!(err.message.contains("incomplete"), "error should name the offending protocol: {}", err.message);
    assert!(err.message.contains("stop"), "error should name the offending key: {}", err.message);

    // --- missing file / absent section => empty map ---
    let cfg_empty = tempfile::tempdir().unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg_empty.path());
    assert!(load_management_protocols().unwrap().is_empty(), "missing inferencers.toml => empty map");

    std::fs::write(cfg_empty.path().join("inferencers.toml"), "[inferencers.dev]\nbase_url = \"http://dev/v1\"\n").unwrap();
    assert!(
        load_management_protocols().unwrap().is_empty(),
        "inferencers.toml with no [management_protocols] section => empty map"
    );

    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
    std::env::remove_var("MYBOX_TOKEN");
}
