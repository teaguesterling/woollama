//! The woollama router service binary. Loads config + connects downstream MCP servers,
//! binds TCP (`WOOLLAMA_ADDRESS` or loopback), and serves the axum app. See
//! docs/rust-router-port.md.

use std::sync::Arc;

#[tokio::main]
async fn main() {
    let (host, port) = woollama_server::resolve_tcp_target();
    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .unwrap_or_else(|e| panic!("bind {host}:{port}: {e}"));
    let addr = listener.local_addr().expect("local_addr");
    let state = Arc::new(woollama_server::build_state().await);
    eprintln!("woollama-server {} listening on http://{addr}", env!("CARGO_PKG_VERSION"));
    axum::serve(listener, woollama_server::router(state)).await.expect("serve");
}
