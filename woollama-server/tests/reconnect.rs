//! Track 0: a downstream that is down at startup reconnects on its own — and while it is down,
//! the router says so.
//!
//! The second half is the point. A test asserting only "reconnect eventually succeeds" also
//! passes on a router that permanently hides a dead downstream, because absence and
//! not-yet-connected look identical from outside. So each test below asserts a NAMED state or a
//! NAMED tool, never the absence of an error.

use std::collections::HashMap;
use std::sync::Arc;

use woollama_server::{spawn_reconnect, McpRegistry, McpServerSpec, ServerHealth, StdioSpec};

/// A stdio spec pointing at `path`, which need not exist yet — `connect_one` resolves the command
/// on every attempt, so making it appear mid-test is exactly the "downstream comes up late" case.
fn stdio_spec(path: &std::path::Path) -> McpServerSpec {
    McpServerSpec::Stdio(StdioSpec {
        command: path.to_string_lossy().to_string(),
        args: vec![],
        env: HashMap::new(),
    })
}

async fn wait_until<F: Fn() -> bool>(secs: u64, f: F) -> bool {
    for _ in 0..(secs * 10) {
        if f() {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    false
}

/// Both cases live in ONE test fn: `WOOLLAMA_MCP_RETRY_MAX_SECS` is process-global, so a
/// parallel `#[tokio::test]` that sets it disables retry underneath its neighbour. (That is
/// exactly how this first failed — the reconnect case saw attempts stuck at 1.)
#[tokio::test]
async fn reconnect_picks_up_a_late_downstream_and_can_be_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let late = dir.path().join("late-server");

    let mut specs = HashMap::new();
    specs.insert("late".to_string(), stdio_spec(&late));

    // Startup: the command does not exist yet.
    let reg = Arc::new(McpRegistry::connect(specs.clone()).await);

    // FIRST, and more important than the reconnect: while it is down, the operator can see it,
    // by name, with a reason. Not "the tool list is short" — an actual reported state.
    let status = reg.status();
    assert_eq!(status.len(), 1, "the configured server must be reported even though it never connected");
    assert_eq!(status[0].name, "late");
    match &status[0].health {
        ServerHealth::Retrying { last_error, .. } => {
            assert!(!last_error.is_empty(), "a retrying server must carry WHY it isn't up")
        }
        other => panic!("expected Retrying while the downstream is absent, got {other:?}"),
    }
    assert_eq!(status[0].tools, 0);

    spawn_reconnect(reg.clone(), specs);

    // Now the downstream appears.
    std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_mcp_fixture"), &late).unwrap();

    assert!(
        wait_until(20, || reg.all_connected()).await,
        "the router should reconnect once the downstream exists; status: {:?}",
        reg.status()
    );

    // Assert the recovered TOOL by name — a reconnect that attached an empty roster would still
    // flip the health flag.
    let names: Vec<String> = reg.reexport_tools().iter().map(|t| t.name.to_string()).collect();
    assert!(
        names.contains(&"mcp__late__count_to".to_string()),
        "the late downstream's tool must be dispatchable after reconnect, got {names:?}"
    );
    assert_eq!(reg.status()[0].health, ServerHealth::Connected);
    assert_eq!(reg.status()[0].tools, 1);

    // --- Case 2 (SAME test fn, see the note above): retry disabled ---
    // An operator may prefer a downstream to stay down until someone looks at it. Pinned because
    // a retry loop that cannot be turned off is its own operational hazard.
    std::env::set_var("WOOLLAMA_MCP_RETRY_MAX_SECS", "0");
    let never = dir.path().join("never");
    let mut off_specs = HashMap::new();
    off_specs.insert("never".to_string(), stdio_spec(&never));
    let off = Arc::new(McpRegistry::connect(off_specs.clone()).await);
    spawn_reconnect(off.clone(), off_specs);
    std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_mcp_fixture"), &never).unwrap();
    // Ample time for a retry loop to fire if one were running (backoff starts at 1s).
    tokio::time::sleep(std::time::Duration::from_secs(4)).await;
    std::env::remove_var("WOOLLAMA_MCP_RETRY_MAX_SECS");
    assert!(
        !off.all_connected(),
        "retry was disabled; the router must NOT have reconnected: {:?}",
        off.status()
    );
}
