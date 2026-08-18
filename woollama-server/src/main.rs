//! `woollamad` — the woollama router daemon. Loads config + connects downstream MCP servers,
//! then either serves the MCP surface over stdio (`woollamad mcp`) or binds TWO local surfaces
//! — a TCP loopback (OpenAI-compatible HTTP) and a unix socket (the default for local MCP
//! clients) — serving the same axum app on both. See docs/rust-router-port.md.

use std::net::SocketAddr;
use std::sync::Arc;

use woollama_server::binding;

#[tokio::main]
async fn main() {
    // Informational flags print and exit, BEFORE anything reads config or touches the runtime
    // dir. They previously fell through and started a full daemon — so `woollamad --version`
    // never exited, and on its way out clobbered a running daemon's discovery.
    let arg1 = std::env::args().nth(1);
    match arg1.as_deref() {
        Some("--version" | "-V") => {
            println!("woollamad {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        Some("--help" | "-h") => {
            println!(
                "woollamad {}\n\n\
                 USAGE:\n  \
                 woollamad                 serve the OpenAI + MCP surfaces (loopback TCP + unix socket)\n  \
                 woollamad mcp             serve the MCP surface over stdio\n  \
                 woollamad check-config    validate config and exit non-zero on any problem\n  \
                 woollamad --version       print the version\n\n\
                 Config lives in $WOOLLAMA_CONFIG_DIR (default $XDG_CONFIG_HOME/woollama).\n\
                 See https://woollama.readthedocs.io/",
                env!("CARGO_PKG_VERSION")
            );
            return;
        }
        _ => {}
    }

    // `woollamad check-config` → validate config, report, exit. Runs BEFORE build_state: it must
    // not connect downstreams or bind anything, so it is safe to run against a live deployment's
    // config dir before a reload.
    if arg1.as_deref() == Some("check-config") {
        std::process::exit(woollama_server::check_config());
    }

    // Refuse a config the operator must fix, rather than starting degraded. Starting with zero
    // MCP servers and silently-restored statelessness while reporting healthy is worse than not
    // starting: the failure is invisible until something downstream misbehaves.
    if let Some(e) = woollama_server::fatal_config_error() {
        eprintln!("woollamad: {e}");
        eprintln!("woollamad: refusing to start. Run `woollamad check-config` to verify a fix.");
        std::process::exit(1);
    }

    let state = Arc::new(woollama_server::build_state().await);

    // `woollamad mcp` → serve the MCP surface over stdio (for an MCP client's
    // mcp.json). stdout is the protocol channel; the banner/logs go to stderr.
    if arg1.as_deref() == Some("mcp") {
        if let Err(e) = woollama_server::serve_mcp_stdio(state).await {
            eprintln!("woollamad mcp: {e}");
            std::process::exit(1);
        }
        return;
    }

    // BEFORE binding anything: if a live peer holds our unix socket, another daemon already owns
    // this runtime dir. Continuing would overwrite ITS address file with our ephemeral port, so
    // every client discovering by that file gets a dead address — the healthy daemon's discovery
    // broken by ours, with no error on either side. Checked here rather than after the TCP bind
    // because the TCP bind is what writes the address file.
    let sock_path = binding::sock_path();
    if std::os::unix::net::UnixStream::connect(&sock_path).is_ok() {
        eprintln!(
            "woollamad: another woollamad already owns {} — refusing to start rather than \
             overwrite its discovery address. Use a different XDG_RUNTIME_DIR to run a second one.",
            sock_path.display()
        );
        std::process::exit(1);
    }

    // TCP loopback (or the WOOLLAMA_ADDRESS override). Persist the real host:port for discovery.
    let (host, port) = woollama_server::resolve_tcp_target();
    // Fail closed: refuse to bind a non-loopback address without an auth token, rather than
    // silently exposing an unauthenticated surface to the network. No-op for the default
    // (loopback) bind and for any token-bearing deployment.
    if let Err(e) = woollama_server::auth::check_bind_allowed(&host) {
        eprintln!("woollamad: {e}");
        std::process::exit(1);
    }
    let tcp = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .unwrap_or_else(|e| panic!("bind {host}:{port}: {e}"));
    let addr = tcp.local_addr().expect("local_addr");
    binding::persist_addr(&addr.ip().to_string(), addr.port());

    // Unix socket alongside TCP — best-effort (degrades to TCP-only). The default transport
    // for local MCP clients (the panel, the CLI).
    let unix = binding::bind_unix(&sock_path);
    // Whether WE created the socket. Only the creator may unlink it on shutdown.
    let unix_owned = unix.is_some();

    eprintln!("woollamad {} listening:", env!("CARGO_PKG_VERSION"));
    if unix.is_some() {
        eprintln!("  unix socket:   {} (default for local MCP clients)", sock_path.display());
    }
    eprintln!("  HTTP loopback: http://{addr}");
    eprintln!("  addr file:     {}", binding::addr_path().display());

    // Serve the SAME app on both listeners (axum 0.8 implements Listener for TcpListener AND
    // UnixListener). Two serve tasks; exit on either dying or on Ctrl-C, then clean up the socket.
    let backends = state.pattern_backends.clone(); // for graceful shutdown (e.g. kill managed fabric)
    let app = woollama_server::router(state);
    // TCP surface is access-controlled: apply the auth layer and serve with peer ConnectInfo so
    // the layer can see the client IP. The UDS app carries NO auth layer — the mode-0600 socket is
    // the credential — so local MCP clients (panel/CLI/cosmic-fabric) stay exempt.
    let app_tcp =
        app.clone().layer(axum::middleware::from_fn(woollama_server::auth::require_surface_auth));
    let tcp_task = tokio::spawn(async move {
        axum::serve(tcp, app_tcp.into_make_service_with_connect_info::<SocketAddr>()).await
    });
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
    binding::cleanup_unix(&sock_path, unix_owned);
    // Graceful shutdown: let each pattern backend release resources (e.g. kill a managed
    // `fabric --serve` we spawned; a reused/external one has no child handle → no-op).
    for backend in &backends {
        backend.shutdown().await;
    }
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
