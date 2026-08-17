//! A config the operator must fix STOPS the daemon, rather than degrading it.
//!
//! This exists because the first implementation of that rule refused nothing. `load_mcp_servers`
//! returned an error and every startup caller degraded it to empty, so one typo'd variable in a
//! twelve-server `mcp.json` started woollamad with **zero** MCP tools, `conversationStore` silently
//! back to `None` (statelessness restored for non-claude models), and no fabric backend — while
//! reporting itself listening and healthy. A "hard failure" that degrades is neither.
//!
//! So these drive the real binary and assert the **process** outcome. Nothing at the library level
//! can make that claim: `build_state` still degrades by design, and the refusal lives in `main`.

use std::path::Path;
use std::process::Command;

/// Write a config dir and run `woollamad check-config` against it → (exit code, combined output).
fn check_config(dir: &Path, files: &[(&str, &str)]) -> (i32, String) {
    for (name, body) in files {
        std::fs::write(dir.join(name), body).unwrap();
    }
    let out = Command::new(env!("CARGO_BIN_EXE_woollamad"))
        .arg("check-config")
        .env("WOOLLAMA_CONFIG_DIR", dir)
        // Pinned so the bundled default's `${WOOLLAMA_EXAMPLES_DIR:-...}` resolves the same way
        // regardless of the developer's environment.
        .env("WOOLLAMA_EXAMPLES_DIR", "/nonexistent-for-test")
        .output()
        .expect("run check-config");
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), text)
}

/// Try to start the daemon; return (exit code, output).
///
/// BOUNDED deliberately. `Command::output()` waits for exit, and a daemon that (wrongly) starts
/// never exits — so a regression here would HANG CI rather than fail it. Confirmed by mutation:
/// stubbing out the refusal made this hang until killed. A test whose failure mode is a stall is
/// most of the way to no test at all.
fn start_daemon(dir: &Path, files: &[(&str, &str)]) -> (i32, String) {
    for (name, body) in files {
        std::fs::write(dir.join(name), body).unwrap();
    }
    let run = dir.join("run");
    std::fs::create_dir_all(&run).unwrap();
    let log = dir.join("daemon.log");
    let mut child = Command::new(env!("CARGO_BIN_EXE_woollamad"))
        .env("WOOLLAMA_CONFIG_DIR", dir)
        .env("XDG_RUNTIME_DIR", &run)
        .env("WOOLLAMA_STATE_DIR", &run)
        .env("WOOLLAMA_EXAMPLES_DIR", "/nonexistent-for-test")
        // Port 0 would still bind; a refusal must happen before any listener exists.
        .env("WOOLLAMA_ADDRESS", "127.0.0.1:0")
        .stdout(std::process::Stdio::from(std::fs::File::create(&log).unwrap()))
        .stderr(std::process::Stdio::from(
            std::fs::OpenOptions::new().append(true).open(&log).unwrap(),
        ))
        .spawn()
        .expect("spawn woollamad");

    for _ in 0..100 {
        if let Some(status) = child.try_wait().expect("try_wait") {
            let text = std::fs::read_to_string(&log).unwrap_or_default();
            return (status.code().unwrap_or(-1), text);
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let _ = child.kill();
    let _ = child.wait();
    let text = std::fs::read_to_string(&log).unwrap_or_default();
    panic!("woollamad did not exit within 10s — it STARTED instead of refusing. Output:\n{text}")
}

#[test]
fn a_missing_variable_stops_the_daemon_rather_than_degrading_it() {
    let dir = tempfile::tempdir().unwrap();
    // A file with a HEALTHY server alongside the broken one: the degradation this guards against
    // silently dropped the healthy one too, which is what made it so much worse than the typo.
    let (code, text) = start_daemon(
        dir.path(),
        &[
            ("recipes.toml", ""),
            (
                "mcp.json",
                r#"{"mcpServers": {"good": {"command": "true"},
                                    "bad": {"command": "${WOOLLAMA_TEST_TYPO}/x"}}}"#,
            ),
        ],
    );
    assert_eq!(code, 1, "the daemon must refuse, not start degraded. Output:\n{text}");
    assert!(text.contains("WOOLLAMA_TEST_TYPO"), "must name the variable: {text}");
    assert!(text.contains("check-config"), "must point at the tool that verifies a fix: {text}");
    assert!(
        !text.contains("listening"),
        "it must not announce itself before refusing — that is the failure being guarded: {text}"
    );
}

#[test]
fn check_config_refuses_the_same_config_and_exits_nonzero() {
    // check-config is what an operator gates a reload on, so it must agree with the daemon.
    // An earlier version warned only at startup, leaving the verification tool silent.
    let dir = tempfile::tempdir().unwrap();
    let (code, text) = check_config(
        dir.path(),
        &[
            ("recipes.toml", ""),
            ("mcp.json", r#"{"mcpServers": {"a": {"command": "${WOOLLAMA_TEST_TYPO}/x"}}}"#),
        ],
    );
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("WOOLLAMA_TEST_TYPO"), "{text}");
    assert!(text.contains(":-"), "the error must point at the escape hatch: {text}");
}

#[test]
fn a_declared_fallback_starts_normally() {
    // The other half: `:-` is what makes refusing safe, so it had better work end to end.
    let dir = tempfile::tempdir().unwrap();
    let (code, text) = check_config(
        dir.path(),
        &[
            ("recipes.toml", ""),
            ("mcp.json", r#"{"mcpServers": {"a": {"command": "${WOOLLAMA_TEST_TYPO:-/bin/true}"}}}"#),
        ],
    );
    assert_eq!(code, 0, "a declared fallback must load: {text}");
    assert!(text.contains("config OK"), "{text}");
}

#[test]
fn inferencers_toml_is_checked_too_and_its_comments_are_not() {
    let dir = tempfile::tempdir().unwrap();

    // A missing variable in inferencers.toml is refused — a silently-empty `base_url` is at
    // least as bad as a missing MCP server.
    let (code, text) = check_config(
        dir.path(),
        &[
            ("recipes.toml", ""),
            ("mcp.json", r#"{"mcpServers": {}}"#),
            ("inferencers.toml", "[inferencers.x]\nbase_url = \"${WOOLLAMA_TEST_TYPO}/v1\"\n"),
        ],
    );
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("WOOLLAMA_TEST_TYPO"), "{text}");
    assert!(!text.contains("inferencers.toml: /"), "the path must not be printed twice: {text}");

    // A COMMENT or a `_`-prefixed key mentioning a variable is documentation, not configuration.
    // Scanning raw text made woollamad refuse its own bundled config; the TOML walker skipped
    // comments but not `_` keys, so the same file loaded under Python and failed here.
    let (code, text) = check_config(
        dir.path(),
        &[(
            "inferencers.toml",
            "# see ${WOOLLAMA_TEST_TYPO} for details\n\
             _doc = \"also set ${WOOLLAMA_TEST_TYPO}\"\n\
             [inferencers.x]\nbase_url = \"http://h/v1\"\n",
        )],
    );
    assert_eq!(code, 0, "documentation must not fail a load: {text}");
}
