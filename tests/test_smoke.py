"""Smoke tests — verify the package imports and key invariants hold without
needing Ollama, network, or GPU. CI-friendly."""
from __future__ import annotations


def test_version_is_string():
    import woollama
    assert isinstance(woollama.__version__, str)
    assert woollama.__version__.count(".") == 2  # X.Y.Z


def test_app_is_fastapi():
    """The router app exists and is wired."""
    from woollama.router import app
    # Sanity: it has the two routes we expose
    paths = {route.path for route in app.routes}
    assert "/v1/models" in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/responses" in paths        # stateful surface (conv-1a)
    assert "/v1/conversations" in paths     # discovery surface (conv-2)


def test_recipes_have_streamer(monkeypatch, tmp_path):
    """The bundled single-server recipe is intact."""
    # Point config dir at an empty location → falls back to bundled defaults
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    from woollama.core import recipes
    recipes.reload()
    r = recipes.get("streamer")
    assert r is not None
    assert r["inferencer"].startswith("ollama/")
    # Tools are now namespaced as <server>.<tool>
    assert "hello.count_to" in r["tools"]
    assert "streamer" in recipes.names()


def test_recipes_have_cross_server(monkeypatch, tmp_path):
    """The cross-server recipe references tools from both bundled servers."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    from woollama.core import recipes
    recipes.reload()
    r = recipes.get("textcounter")
    assert r is not None
    assert "textops.word_count" in r["tools"]
    assert "hello.count_to" in r["tools"]


def test_registry_namespacing():
    """The registry's tool lookup parses `<server>.<tool>` correctly."""
    import pytest

    from woollama.manager import Registry
    reg = Registry()
    with pytest.raises(KeyError, match="namespaced"):
        reg.lookup_tool("count_to")  # bare name should reject
    with pytest.raises(KeyError, match="unknown server"):
        reg.lookup_tool("nonexistent.count_to")


def test_recipes_unknown_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    from woollama.core import recipes
    recipes.reload()
    assert recipes.get("does-not-exist") is None


def test_binding_addr_file_path_under_xdg_runtime_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    from woollama.binding import addr_file_path, sock_file_path
    assert addr_file_path() == str(tmp_path / "woollama.addr")
    assert sock_file_path() == str(tmp_path / "woollama.sock")


def test_binding_resolve_tcp_target_default_and_override(monkeypatch):
    from woollama.binding import resolve_tcp_target
    monkeypatch.delenv("WOOLLAMA_ADDRESS", raising=False)
    assert resolve_tcp_target() == ("127.0.0.1", 0)   # loopback, free port
    monkeypatch.setenv("WOOLLAMA_ADDRESS", "127.0.0.1:54321")
    assert resolve_tcp_target() == ("127.0.0.1", 54321)
    monkeypatch.setenv("WOOLLAMA_ADDRESS", "0.0.0.0")  # host only → free port
    assert resolve_tcp_target() == ("0.0.0.0", 0)


def test_open_sockets_binds_both_and_persists_real_port(tmp_path, monkeypatch):
    """open_sockets binds a UDS + a loopback TCP socket, and the persisted .addr
    is the ACTUAL bound port (read back from the socket — no resolve/bind race)."""
    import socket as _socket
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("WOOLLAMA_ADDRESS", raising=False)
    from woollama import binding
    listeners = binding.open_sockets()
    try:
        families = {s.family for s in listeners.sockets}
        assert _socket.AF_UNIX in families and _socket.AF_INET in families
        assert listeners.sock_path == str(tmp_path / "woollama.sock")
        # The persisted port matches what the TCP socket is really bound to.
        tcp = next(s for s in listeners.sockets if s.family == _socket.AF_INET)
        assert listeners.tcp_port == tcp.getsockname()[1]
        assert (tmp_path / "woollama.addr").read_text().strip() == \
            f"127.0.0.1:{listeners.tcp_port}"
        # The socket file IS the discovery artifact (deterministic path), not a
        # text file — it exists and is a socket, and discover_sock() finds it.
        import stat
        assert stat.S_ISSOCK((tmp_path / "woollama.sock").stat().st_mode)
        assert binding.discover_sock() == str(tmp_path / "woollama.sock")
    finally:
        for s in listeners.sockets:
            s.close()
        binding.cleanup(listeners)


def test_open_sockets_unix_socket_is_mode_0600(tmp_path, monkeypatch):
    """The Unix socket must not be world-connectable — a connectable socket can
    spend the router's API keys (binding.py's whole security premise)."""
    import stat
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("WOOLLAMA_ADDRESS", raising=False)
    from woollama import binding
    listeners = binding.open_sockets()
    try:
        mode = stat.S_IMODE((tmp_path / "woollama.sock").stat().st_mode)
        assert mode == 0o600, oct(mode)
    finally:
        for s in listeners.sockets:
            s.close()
        binding.cleanup(listeners)


def test_open_sockets_unlinks_stale_socket_file(tmp_path, monkeypatch):
    """A leftover socket file from a dead run must not block startup."""
    import socket as _socket
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("WOOLLAMA_ADDRESS", raising=False)
    stale = tmp_path / "woollama.sock"
    stale.write_text("not a socket")     # stale leftover
    from woollama import binding
    listeners = binding.open_sockets()
    try:
        uds = next(s for s in listeners.sockets if s.family == _socket.AF_UNIX)
        assert uds.getsockname() == str(stale)   # rebound at the same path
    finally:
        for s in listeners.sockets:
            s.close()
        binding.cleanup(listeners)
    assert not stale.exists()            # cleanup removed it
