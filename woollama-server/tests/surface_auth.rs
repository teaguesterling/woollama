//! Surface auth end-to-end against the REAL TCP surface (the gap the audit flagged: the Python
//! auth tests are unit-level and never exercise the shipping binary's wiring). This serves the
//! actual `router()` with the same auth layer + `ConnectInfo` that main.rs applies to the TCP
//! app, then drives it over loopback TCP with reqwest.
//!
//! The UDS exemption is structural (main.rs applies NO auth layer to the unix-socket app) and is
//! covered by cosmic-fabric's gated `RealWoollama` suite (UDS, no token) — it can't be exercised
//! hermetically here without a UDS HTTP client. Single test fn: WOOLLAMA_TOKEN is process-global.

use std::net::SocketAddr;
use std::sync::Arc;

use serde_json::json;

#[tokio::test]
async fn tcp_surface_enforces_bearer_auth_uniformly() {
    // Isolated config so we don't touch the user's real mcp.json (which would spawn managed fabric).
    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("mcp.json"), json!({"mcpServers": {}}).to_string()).unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::remove_var("WOOLLAMA_TOKEN");

    let state = Arc::new(woollama_server::build_state().await);
    // EXACT main.rs TCP wiring: auth layer + peer ConnectInfo.
    let app = woollama_server::router(state)
        .layer(axum::middleware::from_fn(woollama_server::auth::require_surface_auth));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
            .await
            .unwrap()
    });
    let c = reqwest::Client::new();

    // 1) No token configured → a loopback client is served (the default, keyless-local behavior).
    std::env::remove_var("WOOLLAMA_TOKEN");
    let r = c.get(format!("{base}/v1/models")).send().await.unwrap();
    assert_eq!(r.status(), 200, "no token: loopback peer is served");

    // 2) Token configured → EVEN a loopback peer must present it (uniform; no loopback exemption).
    std::env::set_var("WOOLLAMA_TOKEN", "s3cr3t-abc");
    let r = c.get(format!("{base}/v1/models")).send().await.unwrap();
    assert_eq!(r.status(), 401, "token set: loopback with no header → 401");
    let www = r.headers().get("www-authenticate").and_then(|v| v.to_str().ok()).unwrap_or("");
    assert!(www.contains("Bearer"), "401 carries WWW-Authenticate: Bearer (got {www:?})");
    let body: serde_json::Value = r.json().await.unwrap();
    assert_eq!(body["error"]["type"], "authentication_error");

    let r = c.get(format!("{base}/v1/models")).bearer_auth("nope").send().await.unwrap();
    assert_eq!(r.status(), 401, "wrong token → 401");

    let r = c.get(format!("{base}/v1/models")).bearer_auth("s3cr3t-abc").send().await.unwrap();
    assert_eq!(r.status(), 200, "correct token → served");

    // The mounted /mcp surface is behind the same layer (auth runs before routing).
    let r = c.post(format!("{base}/mcp")).send().await.unwrap();
    assert_eq!(r.status(), 401, "token set: /mcp is also gated");

    std::env::remove_var("WOOLLAMA_TOKEN");
    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
}
