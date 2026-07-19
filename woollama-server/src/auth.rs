//! Surface access control for `woollamad`'s **TCP** HTTP surface (`/v1/*` and the
//! mounted `/mcp`). A Rust port of Python `woollama.auth` (see docs/security.md).
//!
//! The router holds provider API keys and can dispatch local MCP tools, so its TCP
//! surface is access-controlled, not open:
//!
//!   * **No token configured** (the default): only *loopback* TCP peers are served,
//!     and the bind layer refuses to bind a non-loopback address at all
//!     ([`check_bind_allowed`]) — fail-closed at startup AND per-request (the
//!     per-request check also covers a reverse proxy re-exposing a loopback bind).
//!   * **`WOOLLAMA_TOKEN` configured**: every TCP request must present
//!     `Authorization: Bearer <token>` (constant-time compared), loopback included —
//!     a token-bearing deployment is *uniformly* authenticated. This is what makes an
//!     explicit non-loopback `WOOLLAMA_ADDRESS` acceptable.
//!
//! **Unix-socket peers are exempt by construction**: this middleware is applied ONLY
//! to the TCP-served app (see main.rs); the UDS app carries no auth layer, because the
//! mode-0600 socket file *is* the credential. So [`authorize`] only ever decides TCP
//! requests, and setting a token never breaks the local UDS clients (panel/CLI/
//! cosmic-fabric).

use std::net::{IpAddr, SocketAddr};

use axum::{
    extract::{ConnectInfo, Request},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

pub const ENV_TOKEN: &str = "WOOLLAMA_TOKEN";

/// The configured surface token, or `None` (unset/empty ⇒ no token).
pub fn configured_token() -> Option<String> {
    match std::env::var(ENV_TOKEN) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// True iff `host` *provably* refers to loopback: `localhost`, `127.0.0.0/8`, `::1`,
/// or an IPv4-mapped loopback (`::ffff:127.x.x.x`). Any other or unparseable host is
/// treated as NOT loopback (fail closed — we never resolve names).
pub fn is_loopback_host(host: &str) -> bool {
    if host == "localhost" {
        return true;
    }
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(v4)) => v4.is_loopback(),
        Ok(IpAddr::V6(v6)) => {
            v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
        Err(_) => false,
    }
}

/// Fail closed at startup: a non-loopback bind target requires a configured token.
/// Returns `Err(reason)` instead of letting an unauthenticated surface bind beyond
/// loopback. A loopback target, or any target with a token set, is allowed.
pub fn check_bind_allowed(host: &str) -> Result<(), String> {
    if is_loopback_host(host) || configured_token().is_some() {
        return Ok(());
    }
    Err(format!(
        "refusing to bind non-loopback address {host:?} without an auth token: set {ENV_TOKEN} \
         (clients then send 'Authorization: Bearer <token>'), or bind loopback (unset \
         WOOLLAMA_ADDRESS / use 127.0.0.1)"
    ))
}

/// Constant-time byte compare — accumulates differences with no content short-circuit.
/// A length mismatch returns early (matching Python's `secrets.compare_digest`, which
/// also leaks length); adequate for a bearer-token comparison.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// The per-request decision for a TCP peer. Returns `None` when authorized, else a
/// short refusal reason.
///
/// Token set ⇒ EVERY TCP request (loopback included) must present the bearer token.
/// No token ⇒ loopback-only. (UDS peers never reach here — the TCP app is the only one
/// carrying this layer.)
pub fn authorize(peer_host: &str, authorization: Option<&str>) -> Option<String> {
    if let Some(token) = configured_token() {
        let supplied = authorization.and_then(|h| h.strip_prefix("Bearer ")).unwrap_or("");
        if ct_eq(supplied.as_bytes(), token.as_bytes()) {
            return None;
        }
        return Some("missing or invalid bearer token".to_string());
    }
    if is_loopback_host(peer_host) {
        return None;
    }
    Some(format!(
        "no {ENV_TOKEN} is configured, so only local (loopback / unix socket) clients are served"
    ))
}

/// axum middleware applying [`authorize`] to every request on the TCP app, BEFORE
/// routing (so the mounted `/mcp` service is covered too). It inspects only the request
/// (peer address + `Authorization` header) and either short-circuits a 401 or calls
/// `next` — it never touches the response body, so SSE streams pass through unbuffered.
pub async fn require_surface_auth(req: Request, next: Next) -> Response {
    // `ConnectInfo<SocketAddr>` is injected by `into_make_service_with_connect_info` on the
    // TCP serve. Absent ⇒ we can't identify the peer ⇒ fail closed (should not happen on the
    // TCP path, but denying is the safe default).
    let peer = req.extensions().get::<ConnectInfo<SocketAddr>>().map(|ci| ci.0.ip().to_string());
    let authorization =
        req.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).map(str::to_string);
    let reason = match peer {
        Some(host) => authorize(&host, authorization.as_deref()),
        None => Some("could not determine peer address".to_string()),
    };
    match reason {
        None => next.run(req).await,
        Some(reason) => {
            let body = serde_json::json!({
                "error": {"message": format!("unauthorized: {reason}"), "type": "authentication_error"}
            });
            (StatusCode::UNAUTHORIZED, [(header::WWW_AUTHENTICATE, "Bearer")], axum::Json(body))
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Env-mutating cases live in ONE test fn: WOOLLAMA_TOKEN is process-global, so these
    // must not race a parallel test in this binary. The pure `is_loopback_host` cases have
    // their own env-free test.
    struct TokenGuard;
    impl TokenGuard {
        fn set(v: &str) -> Self {
            std::env::set_var(ENV_TOKEN, v);
            TokenGuard
        }
        fn clear() -> Self {
            std::env::remove_var(ENV_TOKEN);
            TokenGuard
        }
    }
    impl Drop for TokenGuard {
        fn drop(&mut self) {
            std::env::remove_var(ENV_TOKEN);
        }
    }

    #[test]
    fn loopback_host_classification() {
        for ok in ["localhost", "127.0.0.1", "127.5.9.3", "::1", "::ffff:127.0.0.1"] {
            assert!(is_loopback_host(ok), "{ok} should be loopback");
        }
        for no in ["0.0.0.0", "192.168.1.10", "10.0.0.1", "::ffff:8.8.8.8", "example.com", ""] {
            assert!(!is_loopback_host(no), "{no} should NOT be loopback");
        }
    }

    #[test]
    fn authorize_and_bind_gate_token_matrix() {
        // --- no token: loopback-only ---
        let _g = TokenGuard::clear();
        assert_eq!(authorize("127.0.0.1", None), None, "loopback peer served without a token");
        assert!(authorize("192.168.1.5", None).is_some(), "non-loopback peer refused without a token");
        assert!(check_bind_allowed("127.0.0.1").is_ok(), "loopback bind ok without a token");
        assert!(check_bind_allowed("0.0.0.0").is_err(), "non-loopback bind refused without a token");

        // --- token set: UNIFORM (loopback included must present it) ---
        let _g = TokenGuard::set("s3cr3t");
        assert_eq!(authorize("127.0.0.1", Some("Bearer s3cr3t")), None, "correct token → ok");
        assert!(
            authorize("127.0.0.1", None).is_some(),
            "token set: even a LOOPBACK peer must present it (no loopback exemption)"
        );
        assert!(authorize("127.0.0.1", Some("Bearer wrong")).is_some(), "wrong token → refused");
        assert!(authorize("127.0.0.1", Some("s3cr3t")).is_some(), "missing 'Bearer ' prefix → refused");
        assert!(authorize("192.168.1.5", Some("Bearer s3cr3t")).is_none(), "token makes non-loopback ok");
        assert!(check_bind_allowed("0.0.0.0").is_ok(), "non-loopback bind ok WITH a token");
    }
}
