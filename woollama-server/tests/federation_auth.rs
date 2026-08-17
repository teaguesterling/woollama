//! Federation across an ENFORCED authentication boundary — two real `woollamad` processes.
//!
//! `tests/http_downstream.rs` proves the transport in-process against a leaf with no auth, and
//! proves the `Authorization` header reaches the wire against a stub this repo wrote. Neither
//! shows a credential being *enforced* by a real woollamad, or what happens when the consumer
//! holds the wrong one — and that is the half of a real deployment most likely to be wrong.
//!
//! Separate processes are required, not stylistic: `WOOLLAMA_TOKEN` is read from the process
//! environment, so two in-process routers cannot hold different tokens and the negative case is
//! unreachable in-process.
//!
//! Mirrors the shape reported for the first real deployment: a tools-only instance published to
//! loopback that still demands a bearer on every request (so the bind address is not
//! load-bearing for auth), consumed by a router that reads the same secret from the environment.
//! It does NOT cover the LAN hop, a reverse proxy, TLS, or containers.

use std::path::Path;
use std::process::{Child, Command};

use rmcp::transport::streamable_http_client::{StreamableHttpClientTransport, StreamableHttpClientTransportConfig};
use rmcp::ServiceExt;

const TOKEN: &str = "s3cr3t-suite-token";

struct Daemon {
    child: Child,
    base: String,
    log: std::path::PathBuf,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Start `woollamad` on an ephemeral loopback port with its own config/runtime dirs, and wait
/// until it publishes its address. Reading the addr file (rather than pre-picking a port) keeps
/// this free of bind races.
// The child is reaped by `Daemon::drop` (kill + wait); clippy can't see through the struct.
#[allow(clippy::zombie_processes)]
fn spawn_daemon(dir: &Path, mcp_json: &str, token: &str) -> Daemon {
    let cfg = dir.join("cfg");
    let run = dir.join("run");
    std::fs::create_dir_all(&cfg).unwrap();
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(cfg.join("mcp.json"), mcp_json).unwrap();

    let log_path = dir.join("daemon.log");
    let log = std::fs::File::create(&log_path).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_woollamad"))
        .env("WOOLLAMA_CONFIG_DIR", &cfg)
        .env("XDG_RUNTIME_DIR", &run)
        .env("WOOLLAMA_STATE_DIR", &run)
        .env("WOOLLAMA_TOKEN", token)
        .stdout(std::process::Stdio::null())
        .stderr(log)
        .spawn()
        .expect("spawn woollamad");

    let addr_file = run.join("woollama.addr");
    for _ in 0..200 {
        if let Ok(s) = std::fs::read_to_string(&addr_file) {
            let s = s.trim();
            if !s.is_empty() {
                return Daemon { child, base: format!("http://{s}"), log: log_path };
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!("woollamad never published an address; log:\n{}", read_log(&log_path));
}

fn read_log(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_default()
}

/// HTTP status for a GET, with an optional bearer. Raw socket: this crate has no blocking HTTP
/// client and one is not worth a dev-dependency for two probes.
fn get(base: &str, path: &str, token: Option<&str>) -> u16 {
    use std::io::{Read, Write};
    let hostport = base.strip_prefix("http://").unwrap_or(base).to_string();
    let auth = token.map(|t| format!("Authorization: Bearer {t}\r\n")).unwrap_or_default();
    let mut s = std::net::TcpStream::connect(&hostport).expect("connect");
    s.write_all(format!("GET {path} HTTP/1.1\r\nHost: {hostport}\r\nConnection: close\r\n{auth}\r\n").as_bytes())
        .unwrap();
    let mut buf = Vec::new();
    let _ = s.read_to_end(&mut buf);
    String::from_utf8_lossy(&buf).split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0)
}

/// Every tool a daemon advertises on its own `/mcp`, via a real rmcp client presenting `token`.
/// Asserting on NAMES rather than on the absence of an error log: a federation that silently
/// produced nothing would pass a log-absence check.
async fn tool_names(base: &str, token: &str) -> Vec<String> {
    let config = StreamableHttpClientTransportConfig::with_uri(format!("{base}/mcp")).auth_header(token);
    let transport = StreamableHttpClientTransport::with_client(reqwest::Client::new(), config);
    let client = ().serve(transport).await.expect("mcp handshake");
    let tools = client.peer().list_all_tools().await.expect("tools/list");
    tools.iter().map(|t| t.name.to_string()).collect()
}

#[tokio::test]
async fn federation_works_through_an_enforced_credential_and_fails_closed_without_one() {
    let root = tempfile::tempdir().unwrap();

    // The tools-only leaf: no downstreams of its own, bearer required on every request
    // INCLUDING loopback, so the bind address is not what protects it.
    let leaf = spawn_daemon(&root.path().join("leaf"), r#"{"mcpServers": {}}"#, TOKEN);
    assert_eq!(
        get(&leaf.base, "/v1/models", None),
        401,
        "the leaf must actually enforce its token — otherwise the rest of this test proves nothing"
    );
    assert_eq!(get(&leaf.base, "/v1/models", Some(TOKEN)), 200, "the correct token must be accepted");

    let mcp_json = format!(
        r#"{{"mcpServers": {{"suite": {{"url": "{}/mcp",
             "headers": {{"Authorization": "Bearer ${{WOOLLAMA_TOKEN}}"}}}}}}}}"#,
        leaf.base
    );

    // The consumer, holding the right secret.
    let router = spawn_daemon(&root.path().join("router"), &mcp_json, TOKEN);
    assert_eq!(get(&router.base, "/v1/models", Some(TOKEN)), 200, "consumer must come up");
    let names = tool_names(&router.base, TOKEN).await;
    assert!(
        names.contains(&"mcp__suite__chat".to_string()),
        "the leaf's tools must federate through the enforced credential, got {names:?}; log:\n{}",
        read_log(&router.log)
    );

    // The consumer, holding the WRONG secret. This is the case that cannot be reached in-process.
    let bad = spawn_daemon(&root.path().join("bad"), &mcp_json, "wrong-token");
    assert_eq!(
        get(&bad.base, "/v1/models", Some("wrong-token")),
        200,
        "a downstream that rejects our credential must leave the router RUNNING (degraded), not dead"
    );
    let bad_names = tool_names(&bad.base, "wrong-token").await;
    assert!(
        !bad_names.iter().any(|n| n.starts_with("mcp__suite__")),
        "SECURITY: a router holding the wrong credential must federate NOTHING, got {bad_names:?}"
    );
    let bad_log = read_log(&bad.log);
    assert!(
        bad_log.contains("MCP server 'suite' failed to start"),
        "a rejected credential must be reported, naming the server — a silent auth failure is the \
         worst outcome, since the router looks healthy with the tools quietly absent. Log:\n{bad_log}"
    );
}
