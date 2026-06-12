//! Part A of the cutover: woollamad's LOCAL BINDING — the unix socket + addr-file surface
//! ported from Python `woollama.binding`. Proves the same `router(state)` serves over a
//! 0600 unix socket, that the socket is bound at `$XDG_RUNTIME_DIR/woollama.sock`, and that
//! the TCP discovery address is persisted to `woollama.addr`. This is the Rust evidence the
//! rust-port plan owed for the unix-socket surface (the old Python in-process test only
//! exercised `binding.py`).
//!
//! Separate test binary so the global WOOLLAMA_*/XDG_RUNTIME_DIR env can't race other files.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;

use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn spawn_tcp(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

/// Minimal dependency-free HTTP/1.1 GET over a unix socket; returns the raw response.
async fn http_get_unix(sock: &Path, path: &str) -> String {
    let mut stream = tokio::net::UnixStream::connect(sock).await.expect("connect UDS");
    let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    String::from_utf8_lossy(&buf).into_owned()
}

#[tokio::test]
async fn woollamad_serves_over_unix_socket_and_persists_addr() {
    // Mock ollama /v1/models so the discovery-enabled endpoint returns hermetically.
    let upstream = Router::new().route(
        "/v1/models",
        get(|| async { Json(json!({"object": "list", "data": [{"id": "m1"}]})) }),
    );
    let upstream_url = spawn_tcp(upstream).await;

    // Isolate the runtime dir → the socket + addr-file land in a temp dir, not the user's.
    let rt = tempfile::tempdir().unwrap();
    std::env::set_var("XDG_RUNTIME_DIR", rt.path());

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(
        cfg.path().join("recipes.toml"),
        "[recipes.r1]\ninferencer=\"ollama/m\"\ntools=[]\nsystem=\"x\"\n",
    )
    .unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_OLLAMA_URL", &upstream_url);

    // Mirror what main.rs does: persist the TCP addr, bind the unix socket, serve the app.
    woollama_server::binding::persist_addr("127.0.0.1", 54321);
    let sock = woollama_server::binding::sock_path();
    let unix = woollama_server::binding::bind_unix(&sock).expect("unix socket must bind");

    let state = Arc::new(woollama_server::build_state().await);
    let app = woollama_server::router(state);
    tokio::spawn(async move { axum::serve(unix, app).await.unwrap() });

    // The router answers /v1/models over the UNIX SOCKET.
    let resp = http_get_unix(&sock, "/v1/models").await;
    assert!(resp.contains("200 OK"), "expected 200 over UDS; got: {}", &resp[..resp.len().min(120)]);
    assert!(resp.contains("\"object\""), "models JSON over UDS; got tail: {resp}");
    assert!(resp.contains("ollama/m1") || resp.contains("woollama/r1"), "catalog over UDS");

    // The socket is mode 0600 (a connectable socket can spend the router's API keys).
    let mode = std::fs::metadata(&sock).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "unix socket must be 0600");

    // The TCP address is persisted to the addr-file for discovery.
    let addr = std::fs::read_to_string(woollama_server::binding::addr_path()).unwrap();
    assert_eq!(addr.trim(), "127.0.0.1:54321", "addr-file holds host:port");
}
