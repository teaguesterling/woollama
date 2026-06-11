//! The woollama router service binary. Binds TCP (`WOOLLAMA_ADDRESS` or loopback) and
//! serves the axum app from `woollama_server::router()`. See docs/rust-router-port.md.

#[tokio::main]
async fn main() {
    let (host, port) = woollama_server::resolve_tcp_target();
    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .unwrap_or_else(|e| panic!("bind {host}:{port}: {e}"));
    let addr = listener.local_addr().expect("local_addr");
    eprintln!("woollama-server {} listening on http://{addr}", env!("CARGO_PKG_VERSION"));
    axum::serve(listener, woollama_server::router()).await.expect("serve");
}
