//! Issue #19: consuming a downstream MCP server over Streamable HTTP (`"url"` in mcp.json).
//! The headline case is one woollamad consuming another — woollamad already SERVES `/mcp`
//! (lib.rs), so a single test exercises the transport and demonstrates tool federation.
//!
//! Separate test binary so the global `WOOLLAMA_*` env can't race other test files (the
//! convention stated in tests/mcp_surface.rs). Within this file, `CONFIG_ENV` serializes the
//! window where `WOOLLAMA_CONFIG_DIR` is set — it is process-global, so two concurrent
//! `build_state()` calls would otherwise read each other's config dir.

use std::sync::{Arc, Mutex};

use axum::Router;
use rmcp::transport::streamable_http_client::StreamableHttpClientWorker;
use rmcp::transport::worker::WorkerTransport;
use rmcp::ServiceExt;

/// Serializes `WOOLLAMA_CONFIG_DIR` mutation. `tokio::sync::Mutex` rather than `std`'s because
/// the guard is held across `build_state().await`.
static CONFIG_ENV: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

/// Build and serve a woollamad whose config dir contains exactly `mcp_json`, and return its
/// base URL. The tempdir is leaked: `build_state` has already read it, and leaking is cheaper
/// than threading ownership through the test.
async fn spawn_woollamad(mcp_json: &str) -> String {
    let state = {
        let _guard = CONFIG_ENV.lock().await;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("mcp.json"), mcp_json).unwrap();
        std::env::set_var("WOOLLAMA_CONFIG_DIR", dir.path());
        // Keep each instance's durable handle table separate so two in-process routers don't
        // share one conversations.json.
        std::env::set_var("WOOLLAMA_STATE_DIR", dir.path());
        let state = Arc::new(woollama_server::build_state().await);
        std::env::remove_var("WOOLLAMA_CONFIG_DIR");
        std::env::remove_var("WOOLLAMA_STATE_DIR");
        std::mem::forget(dir);
        state
    };
    spawn(woollama_server::router(state)).await
}

/// Every tool name a woollamad advertises on its own `/mcp` surface, via a real rmcp client.
async fn tool_names(base: &str) -> Vec<String> {
    let worker = StreamableHttpClientWorker::<reqwest::Client>::new_simple(format!("{base}/mcp"));
    let client = ().serve(WorkerTransport::spawn(worker)).await.unwrap();
    let tools = client.peer().list_all_tools().await.unwrap();
    tools.iter().map(|t| t.name.to_string()).collect()
}

#[tokio::test]
async fn woollamad_consumes_another_woollamad_over_http() {
    // B: a leaf with no downstreams of its own, like the real mcp-suite. Its /mcp still
    // advertises the built-in `chat` verb — the stable thing to assert on without depending on
    // whatever the ambient config dir happens to hold.
    let b = spawn_woollamad(r#"{"mcpServers": {}}"#).await;
    assert!(tool_names(&b).await.contains(&"chat".to_string()), "leaf must advertise its own chat verb");

    // A: consumes B over the url form.
    let a = spawn_woollamad(&format!(r#"{{"mcpServers": {{"remote": {{"url": "{b}/mcp"}}}}}}"#)).await;

    let names = tool_names(&a).await;
    // Assert a NAMED federated tool, not a non-empty roster: a transport that connects and
    // silently returns nothing would pass the weaker check.
    assert!(
        names.contains(&"mcp__remote__chat".to_string()),
        "expected B's tools federated under 'remote', got {names:?}"
    );
}

#[tokio::test]
async fn configured_headers_reach_the_downstream_request() {
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    // A stub that records Authorization and then fails the handshake. The connect is EXPECTED to
    // fail — the claim under test is "the header reached the wire", which is exactly what a
    // silently credential-less transport would get wrong while looking healthy.
    let stub = Router::new().route(
        "/mcp",
        axum::routing::any(move |headers: axum::http::HeaderMap| {
            let sink = sink.clone();
            async move {
                if let Some(v) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
                    sink.lock().unwrap().push(v.to_string());
                }
                axum::http::StatusCode::NOT_IMPLEMENTED
            }
        }),
    );
    let base = spawn(stub).await;
    let _consumer = spawn_woollamad(&format!(
        r#"{{"mcpServers": {{"stub": {{"url": "{base}/mcp",
             "headers": {{"Authorization": "Bearer sk-test-value"}}}}}}}}"#
    ))
    .await;

    let recorded = seen.lock().unwrap().clone();
    assert!(
        recorded.contains(&"Bearer sk-test-value".to_string()),
        "the configured Authorization header must reach the downstream request, saw {recorded:?}"
    );
}

/// `woollamad check-config` on a config dir holding exactly `mcp_json` → (exit code, stdout+stderr).
/// Drives the real binary, because the exit code IS the contract an operator scripts against.
fn check_config(mcp_json: &str) -> (i32, String) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("mcp.json"), mcp_json).unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_woollamad"))
        .arg("check-config")
        .env("WOOLLAMA_CONFIG_DIR", dir.path())
        .output()
        .expect("run woollamad check-config");
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), text)
}

#[test]
fn check_config_exits_nonzero_on_a_skipped_server() {
    // A malformed entry is SKIPPED at boot so one typo can't cost the operator every other tool
    // server — but a skip is otherwise announced only in the boot log. This subcommand is where
    // that becomes actionable: an operator can gate a reload on it, and CI can gate a config
    // change on it. If this ever exits 0 on a bad entry, the skip is silent again.
    let (code, text) = check_config(
        r#"{"mcpServers": {
             "good": {"command": "hi"},
             "bad": {"url": "http://h/mcp", "headers": {"Authorization": "Bearer "}}
           }}"#,
    );
    assert_eq!(code, 1, "a skipped server must fail the check: {text}");
    assert!(text.contains("bad"), "must name the offending server: {text}");
    assert!(text.contains("good"), "must still report the usable server: {text}");
    assert!(!text.contains("config OK"), "{text}");
}

#[test]
fn check_config_exits_zero_on_a_healthy_config() {
    let (code, text) = check_config(r#"{"mcpServers": {"hello": {"command": "hi"}}}"#);
    assert_eq!(code, 0, "a valid config must pass: {text}");
    assert!(text.contains("config OK"), "{text}");
}

#[tokio::test]
async fn a_url_server_that_is_not_listening_is_skipped_not_fatal() {
    // Matches the stdio posture (mcp_registry.rs: failed server logged and skipped). A dead
    // downstream must degrade the router, not take it down — and the router must still serve.
    let dead = {
        // Bind and immediately drop, so the port is almost certainly unused.
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let a = l.local_addr().unwrap();
        drop(l);
        format!("http://{a}")
    };
    let a = spawn_woollamad(&format!(r#"{{"mcpServers": {{"gone": {{"url": "{dead}/mcp"}}}}}}"#)).await;
    let names = tool_names(&a).await;
    assert!(names.contains(&"chat".to_string()), "router must still serve its own verb: {names:?}");
    assert!(
        !names.iter().any(|n| n.starts_with("mcp__gone__")),
        "an unreachable downstream must contribute no tools: {names:?}"
    );
}
