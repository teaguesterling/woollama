//! Ephemeral local-only binding — the Rust port of Python `woollama.binding`.
//!
//! `woollamad` serves the SAME router on TWO coexisting local surfaces (docs/architecture.md
//! §"Binding"):
//!   * **Unix socket** — `$XDG_RUNTIME_DIR/woollama.sock`, the default for local MCP clients
//!     (the panel, the CLI). Mode 0600: a connectable socket is as good as the API keys the
//!     router holds, and `$XDG_RUNTIME_DIR` is already a 0700 per-user dir.
//!   * **HTTP loopback** — `127.0.0.1` on a free port (or the `WOOLLAMA_ADDRESS` override),
//!     whose real `host:port` is persisted to `$XDG_RUNTIME_DIR/woollama.addr` for discovery.
//!
//! The socket path is deterministic, so its mere existence is the discovery artifact; only the
//! random TCP port needs the `.addr` file. UDS binding is best-effort — if it fails (unwritable
//! runtime dir, or a path over the ~108-char `sun_path` limit) we log and serve TCP-only,
//! exactly like the Python loader, rather than failing to start.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use tokio::net::UnixListener;

/// `$XDG_RUNTIME_DIR` (0700, per-user) or `/tmp` as a fallback — mirrors `_runtime_dir`.
pub fn runtime_dir() -> PathBuf {
    std::env::var("XDG_RUNTIME_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

/// `$XDG_RUNTIME_DIR/woollama.sock`.
pub fn sock_path() -> PathBuf {
    runtime_dir().join("woollama.sock")
}

/// `$XDG_RUNTIME_DIR/woollama.addr`.
pub fn addr_path() -> PathBuf {
    runtime_dir().join("woollama.addr")
}

/// Bind a stream Unix socket at `path`, mode 0600. **Probes for a live peer before reclaiming**:
/// if another process is already accepting on this socket, we do NOT unlink it (a transient second
/// `woollamad` must not steal the primary's live socket — that breaks the running daemon's UDS
/// clients and clobbers discovery). Only a *dead*/stale socket is reclaimed. Returns `None` (and
/// logs) if a live peer holds it or any step fails, so the caller degrades to TCP-only. Mirrors the
/// probe-before-reclaim pattern the fabric backend uses (`fabric.rs::ready`).
pub fn bind_unix(path: &Path) -> Option<UnixListener> {
    // A successful connect means a live peer is accepting here (a listening socket completes the
    // connect into its backlog even without an active accept()). Refuse to clobber it.
    if std::os::unix::net::UnixStream::connect(path).is_ok() {
        eprintln!(
            "woollamad: {} is already served by a live peer; serving TCP-only (not stealing the socket)",
            path.display()
        );
        return None;
    }
    // No live peer — clear the stale socket file, if any (FileNotFound is fine), then bind.
    if let Err(e) = std::fs::remove_file(path) {
        if e.kind() != std::io::ErrorKind::NotFound {
            eprintln!("woollamad: unix socket unavailable (stale unlink: {e}); serving TCP-only");
            return None;
        }
    }
    let listener = match UnixListener::bind(path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("woollamad: unix socket unavailable ({e}); serving TCP-only");
            return None;
        }
    };
    // Explicit 0600 — the parent ($XDG_RUNTIME_DIR) is already 0700, so there is no
    // world-connectable window in practice, but don't rely on that alone.
    if let Err(e) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
        eprintln!("woollamad: unix socket chmod 0600 failed ({e}); serving TCP-only");
        let _ = std::fs::remove_file(path);
        return None;
    }
    Some(listener)
}

/// Persist `host:port` to the addr-file for client discovery (best-effort, silent-fail —
/// mirrors `_persist` + the `open_sockets` call). Writes a trailing newline like the Python.
pub fn persist_addr(host: &str, port: u16) {
    let path = addr_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    // Bracket an IPv6 host so the `host:port` form is unambiguous (`[::1]:8080`, not `::1:8080`).
    let host_fmt = if host.contains(':') { format!("[{host}]") } else { host.to_string() };
    let _ = std::fs::write(&path, format!("{host_fmt}:{port}\n"));
}

/// Remove the Unix socket file on shutdown (FileNotFound is fine). Mirrors `cleanup`.
pub fn cleanup_unix(path: &Path) {
    let _ = std::fs::remove_file(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test] // bind_unix binds a tokio UnixListener → needs a reactor
    async fn bind_unix_does_not_steal_a_live_socket_but_reclaims_a_dead_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("woollama.sock");

        // A LIVE peer owns the socket (a real std listening socket — the kernel completes a
        // connect() into its backlog, so the liveness probe sees it even with no active
        // accept()). A second bind_unix must refuse (serve TCP-only), NOT unlink it.
        let primary = std::os::unix::net::UnixListener::bind(&path).unwrap();
        assert!(bind_unix(&path).is_none(), "must NOT clobber a live peer's socket");
        assert!(path.exists(), "the live socket file survives");
        drop(primary); // the primary goes away, leaving a STALE socket file behind (std doesn't unlink)

        // A DEAD/stale socket (file present, nothing listening) IS reclaimed.
        assert!(path.exists(), "stale socket file still on disk");
        let reclaimed = bind_unix(&path);
        assert!(reclaimed.is_some(), "a dead socket is reclaimed");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "reclaimed socket is 0600");
    }
}
