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

/// Serializes every test in this binary that touches process-global `WOOLLAMA_*` config.
/// Setting one of those vars is visible to EVERY concurrently-running test in the process — this
/// file has already lost two tests to that, in both directions.
static ENV: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

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
    //
    // The env var is held for as short a window as possible and under ENV, because it is
    // process-global: an earlier version kept it set across a 4-second sleep, silently disabling
    // retry for whatever else was running.
    let _env = ENV.lock().await;
    std::env::set_var("WOOLLAMA_MCP_RETRY_MAX_SECS", "0");
    let never = dir.path().join("never");
    let mut off_specs = HashMap::new();
    off_specs.insert("never".to_string(), stdio_spec(&never));
    let off = Arc::new(McpRegistry::connect(off_specs.clone()).await);
    spawn_reconnect(off.clone(), off_specs);
    let status = off.status();
    std::env::remove_var("WOOLLAMA_MCP_RETRY_MAX_SECS");
    drop(_env);

    // With retry off, nothing will ever try again — so reporting `retrying` would be exactly the
    // conflation ServerHealth exists to prevent. It must say so terminally, and say why.
    match &status[0].health {
        ServerHealth::Failed { reason } => assert!(
            reason.contains("retry disabled"),
            "a server that will never be retried must say so: {reason}"
        ),
        other => panic!("retry is disabled; expected Failed, got {other:?}"),
    }
    assert!(!off.all_connected());
}

/// The nesting cap, against a REAL downstream advertising an already-federated name — not
/// against the depth arithmetic. A cap that computed correctly but was never consulted by
/// `reexport_tools` would pass a predicate test; this one would fail.
///
/// Also exercises the `env` block end-to-end: the advertised name is driven by an env var the
/// spec forwards to the spawned server.
#[tokio::test]
async fn re_export_stops_at_the_nesting_cap() {
    let fixture = env!("CARGO_BIN_EXE_mcp_fixture");
    let spec = |tool: &str| {
        let mut env = HashMap::new();
        env.insert("MCP_FIXTURE_TOOL_NAME".to_string(), tool.to_string());
        McpServerSpec::Stdio(StdioSpec { command: fixture.to_string(), args: vec![], env })
    };

    let mut specs = HashMap::new();
    // depth 0 -> re-exports to depth 1: kept.
    specs.insert("plain".to_string(), spec("count_to"));
    // depth 2 -> would re-export to depth 3: dropped at the default cap of 2.
    specs.insert("deep".to_string(), spec("mcp__b__mcp__a__count_to"));

    let reg = McpRegistry::connect(specs).await;
    let names: Vec<String> = reg.reexport_tools().iter().map(|t| t.name.to_string()).collect();

    assert!(
        names.contains(&"mcp__plain__count_to".to_string()),
        "an ordinary downstream tool must still be re-exported, got {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.contains("mcp__deep__")),
        "a tool already at the cap must not gain another federation level, got {names:?}"
    );
    // Both servers connected — the cap drops a TOOL, it does not drop the server or mark it
    // unhealthy. Conflating the two would make a capped roster look like a connection failure.
    assert!(reg.all_connected(), "the cap must not affect health: {:?}", reg.status());
}

/// `GET /v1/tools` (issue #23) — and specifically that a downstream which is DOWN appears in it.
///
/// The easy version of this endpoint lists what's connected, which is worse than useless during
/// an incident: a router with a dead downstream would render identically to one with no
/// downstream configured. The assertion that matters is the retrying server being present, named,
/// with its reason.
#[tokio::test]
async fn v1_tools_lists_tool_origins_and_reports_a_downstream_that_is_down() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("cfg");
    std::fs::create_dir_all(&cfg).unwrap();
    let fixture = env!("CARGO_BIN_EXE_mcp_fixture");
    std::fs::write(
        cfg.join("mcp.json"),
        format!(
            r#"{{"mcpServers": {{
                 "up":   {{"command": "{fixture}"}},
                 "down": {{"command": "{}/definitely-not-here"}}
               }}}}"#,
            dir.path().display()
        ),
    )
    .unwrap();

    let _env = ENV.lock().await;
    std::env::set_var("WOOLLAMA_CONFIG_DIR", &cfg);
    std::env::set_var("WOOLLAMA_STATE_DIR", &cfg);
    // NOTE: deliberately does NOT touch WOOLLAMA_MCP_RETRY_MAX_SECS. That var is process-global,
    // and setting it here disabled retry underneath the reconnect test running in parallel — the
    // same race this repo has now hit four times. Instead 'down' points at a path that will never
    // exist, so retry can never succeed and the assertions are stable with retry left ON.
    let state = Arc::new(woollama_server::build_state().await);
    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
    std::env::remove_var("WOOLLAMA_STATE_DIR");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = woollama_server::router(state);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let body: serde_json::Value = reqwest::get(format!("http://{addr}/v1/tools"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // A tool names the server it came from — the only way to read a federated namespace without
    // driving an MCP handshake by hand.
    let data = body["data"].as_array().expect("data array");
    let found = data.iter().find(|t| t["tool"] == "count_to").expect("the live server's tool");
    assert_eq!(found["server"], "up");
    assert_eq!(found["name"], "mcp__up__count_to");
    assert!(body["tools"].as_array().unwrap().iter().any(|n| n == "up.count_to"),
        "the Python reference's `<server>.<tool>` shape is preserved: {}", body["tools"]);

    let servers = body["servers"].as_array().expect("servers array");
    let up = servers.iter().find(|s| s["name"] == "up").expect("live server reported");
    assert_eq!(up["health"], "connected");
    assert_eq!(up["tools"], 1);

    // THE point of the endpoint: the dead one is present, not omitted, and says why.
    let down = servers
        .iter()
        .find(|s| s["name"] == "down")
        .expect("a downstream that is DOWN must still be reported — absence is not a status");
    assert_eq!(down["health"], "retrying");
    assert_eq!(down["tools"], 0);
    assert!(
        down["last_error"].as_str().map(|e| !e.is_empty()).unwrap_or(false),
        "a retrying server must carry WHY: {down}"
    );
}
