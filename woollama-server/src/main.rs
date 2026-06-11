//! `woollamad` — the woollama router daemon. Loads config + connects downstream MCP servers,
//! then either serves the MCP surface over stdio (`woollamad mcp`) or binds TCP
//! and serves the axum HTTP app. See docs/rust-router-port.md.

use std::sync::Arc;

#[tokio::main]
async fn main() {
    let state = Arc::new(woollama_server::build_state().await);

    // `woollamad mcp` → serve the MCP surface over stdio (for an MCP client's
    // mcp.json). stdout is the protocol channel; the banner/logs go to stderr.
    if std::env::args().nth(1).as_deref() == Some("mcp") {
        if let Err(e) = woollama_server::serve_mcp_stdio(state).await {
            eprintln!("woollamad mcp: {e}");
            std::process::exit(1);
        }
        return;
    }

    let (host, port) = woollama_server::resolve_tcp_target();
    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .unwrap_or_else(|e| panic!("bind {host}:{port}: {e}"));
    let addr = listener.local_addr().expect("local_addr");
    eprintln!("woollamad {} listening on http://{addr}", env!("CARGO_PKG_VERSION"));
    axum::serve(listener, woollama_server::router(state)).await.expect("serve");
}
