//! `load_timeout` is refused at config load when it is not a usable wait. The pool turns it
//! into a `Duration`, and `Duration::from_secs_f64` panics on a negative, non-finite or huge
//! value; a value that is not a number would silently fall back to 120 s. Either way the
//! operator learns about it from a crash or a 502, long after the config was written. The
//! error names the inferencer and the key, so the fix is obvious.
//!
//! Separate test binary so the global WOOLLAMA_CONFIG_DIR env can't race other files. One
//! test, so the cases run in sequence.

fn load(value: &str) -> Result<woollama_engine::Registry, String> {
    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        format!("[inferencers.device]\nbase_url = \"http://dev/v1\"\nload_timeout = {value}\n"),
    )
    .unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    woollama_engine::Registry::from_config().map_err(|e| format!("{e:?}"))
}

#[test]
fn an_unusable_load_timeout_is_refused_at_config_load() {
    for bad in ["-1", "0", "nan", "inf", "1e30", "86401", "\"300\""] {
        let err = match load(bad) {
            Ok(_) => panic!("load_timeout = {bad} was accepted"),
            Err(e) => e,
        };
        assert!(
            err.contains("device") && err.contains("load_timeout"),
            "load_timeout = {bad}: the error must name the inferencer and the key: {err}"
        );
    }
    for good in ["0.5", "120", "300", "86400"] {
        let reg = load(good).unwrap_or_else(|e| panic!("load_timeout = {good} was refused: {e}"));
        assert_eq!(reg.resolve("device").unwrap().load_timeout, good.parse::<f64>().unwrap());
    }
}
