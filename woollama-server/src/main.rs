//! `woollamad` — the woollama router daemon. Loads config + connects downstream MCP servers,
//! then either serves the MCP surface over stdio (`woollamad mcp`) or binds TWO local surfaces
//! — a TCP loopback (OpenAI-compatible HTTP) and a unix socket (the default for local MCP
//! clients) — serving the same axum app on both. See docs/rust-router-port.md.

use std::sync::Arc;

use woollama_server::binding;

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

    // TCP loopback (or the WOOLLAMA_ADDRESS override). Persist the real host:port for discovery.
    let (host, port) = woollama_server::resolve_tcp_target();
    let tcp = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .unwrap_or_else(|e| panic!("bind {host}:{port}: {e}"));
    let addr = tcp.local_addr().expect("local_addr");
    binding::persist_addr(&addr.ip().to_string(), addr.port());

    // Unix socket alongside TCP — best-effort (degrades to TCP-only). The default transport
    // for local MCP clients (the panel, the CLI).
    let sock_path = binding::sock_path();
    let unix = binding::bind_unix(&sock_path);

    eprintln!("woollamad {} listening:", env!("CARGO_PKG_VERSION"));
    if unix.is_some() {
        eprintln!("  unix socket:   {} (default for local MCP clients)", sock_path.display());
    }
    eprintln!("  HTTP loopback: http://{addr}");
    eprintln!("  addr file:     {}", binding::addr_path().display());

    // Serve the SAME app on both listeners (axum 0.8 implements Listener for TcpListener AND
    // UnixListener). Two serve tasks; exit on either dying or on Ctrl-C, then clean up the socket.
    let app = woollama_server::router(state);
    let app_tcp = app.clone();
    let tcp_task = tokio::spawn(async move { axum::serve(tcp, app_tcp).await });
    let unix_task = unix.map(|u| {
        let app_unix = app.clone();
        tokio::spawn(async move { axum::serve(u, app_unix).await })
    });

    if let Some(unix_task) = unix_task {
        tokio::select! {
            _ = tcp_task => {}
            _ = unix_task => {}
            _ = shutdown_signal() => {}
        }
    } else {
        tokio::select! {
            _ = tcp_task => {}
            _ = shutdown_signal() => {}
        }
    }
    binding::cleanup_unix(&sock_path);
}

/// Resolve on SIGINT (Ctrl-C) or SIGTERM (systemd/`kill`) so we can remove the socket file
/// on a clean stop. (Unlink-before-bind already self-heals a leftover on the next start, so
/// this is hygiene, not correctness — a SIGKILL still leaves the stale socket, harmlessly.)
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(_) => {
            // No SIGTERM handler available — fall back to Ctrl-C only.
            let _ = tokio::signal::ctrl_c().await;
            return;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}
