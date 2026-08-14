//! Task 2: the six model-pooling config fields (mirrors the Python `Inferencer`
//! dataclass) plus `images_url`/`embeddings_url`. A config-defined inferencer picks
//! up all six from `inferencers.toml`; a built-in with no overrides keeps the
//! documented defaults.
//!
//! Separate test binary so the global WOOLLAMA_CONFIG_DIR env can't race other files.

#[test]
fn pooling_fields_from_config_and_builtin_defaults() {
    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        r#"
[inferencers.device]
base_url = "http://dev/v1"
management_url = "http://dev:8800"
parallel = 2
pool_max = 3
queue_max = 8
queue_timeout = 45
virtual = { default = "Qwen/Coder", coder = "Qwen/Coder" }
"#,
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());

    let reg = woollama_engine::Registry::from_config().unwrap();

    let device = reg.resolve("device").expect("device inferencer");
    assert_eq!(device.management_url, Some("http://dev:8800".to_string()));
    assert_eq!(device.parallel, 2);
    assert_eq!(device.pool_max, Some(3));
    assert_eq!(device.queue_max, Some(8));
    assert_eq!(device.queue_timeout, 45.0);
    assert_eq!(device.virtual_models["default"], "Qwen/Coder");
    assert_eq!(device.virtual_models["coder"], "Qwen/Coder");
    assert_eq!(device.images_url(), "http://dev/v1/images/generations");
    assert_eq!(device.embeddings_url(), "http://dev/v1/embeddings");

    let anthropic = reg.resolve("anthropic").expect("anthropic builtin");
    assert_eq!(anthropic.management_url, None);
    assert_eq!(anthropic.parallel, 1);
    assert_eq!(anthropic.pool_max, None);
    assert_eq!(anthropic.queue_max, None);
    assert_eq!(anthropic.queue_timeout, 30.0);
    assert!(anthropic.virtual_models.is_empty());

    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
}
