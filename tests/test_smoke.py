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


def test_recipes_have_streamer():
    """The bundled single-server recipe is intact."""
    from woollama import recipes
    r = recipes.get("streamer")
    assert r is not None
    assert r["inferencer"].startswith("ollama/")
    # Tools are now namespaced as <server>.<tool>
    assert "hello.count_to" in r["tools"]
    assert "streamer" in recipes.names()


def test_recipes_have_cross_server():
    """The cross-server recipe references tools from both bundled servers."""
    from woollama import recipes
    r = recipes.get("textcounter")
    assert r is not None
    assert "textops.word_count" in r["tools"]
    assert "hello.count_to" in r["tools"]


def test_registry_namespacing():
    """The registry's tool lookup parses `<server>.<tool>` correctly."""
    from woollama.manager import Registry
    import pytest
    reg = Registry()
    with pytest.raises(KeyError, match="namespaced"):
        reg.lookup_tool("count_to")  # bare name should reject
    with pytest.raises(KeyError, match="unknown server"):
        reg.lookup_tool("nonexistent.count_to")


def test_recipes_unknown_returns_none():
    from woollama import recipes
    assert recipes.get("does-not-exist") is None


def test_binding_addr_file_path_under_xdg_runtime_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    from woollama.binding import addr_file_path
    assert addr_file_path() == str(tmp_path / "woollama.addr")


def test_binding_resolve_writes_addr_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("WOOLLAMA_ADDRESS", raising=False)
    from woollama.binding import resolve_bind
    host, port = resolve_bind()
    assert host == "127.0.0.1"
    assert 1024 < port < 65536
    written = (tmp_path / "woollama.addr").read_text().strip()
    assert written == f"{host}:{port}"


def test_binding_override_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("WOOLLAMA_ADDRESS", "127.0.0.1:54321")
    from woollama.binding import resolve_bind
    host, port = resolve_bind()
    assert host == "127.0.0.1"
    assert port == 54321
