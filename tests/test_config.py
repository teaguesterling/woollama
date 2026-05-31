"""Tests for the config loaders — both fallback-to-defaults and user-override
behavior, plus error reporting on malformed files."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# --- defaults fallback ------------------------------------------------------

def test_mcp_defaults_when_no_user_config(monkeypatch, tmp_path):
    """No user mcp.json → loads the bundled default."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    from woollama import config
    servers = config.load_mcp_servers()
    assert "hello" in servers
    assert "textops" in servers
    # Bundled defaults reference packaged examples; the path should be
    # resolved (no ${VAR} remaining)
    args = servers["hello"]["args"]
    assert "${" not in args[0]
    assert args[0].endswith("server.py")


def test_recipes_defaults_when_no_user_config(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    from woollama import config
    recipes = config.load_recipes()
    assert "streamer" in recipes
    assert "textcounter" in recipes
    assert recipes["streamer"]["tools"] == ["hello.count_to"]


# --- user override ----------------------------------------------------------

def test_user_mcp_overrides_default(monkeypatch, tmp_path):
    """User's mcp.json is loaded instead of bundled defaults."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    user_cfg = {"mcpServers": {
        "custom": {"command": "echo", "args": ["hello"]},
    }}
    (tmp_path / "mcp.json").write_text(json.dumps(user_cfg))
    from woollama import config
    servers = config.load_mcp_servers()
    assert list(servers.keys()) == ["custom"]
    assert servers["custom"]["command"] == "echo"


def test_user_recipes_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    user_toml = """
[recipes.tiny]
inferencer = "ollama/test-model"
tools = []
system = "Be brief."
"""
    (tmp_path / "recipes.toml").write_text(user_toml)
    from woollama import config
    recipes = config.load_recipes()
    assert list(recipes.keys()) == ["tiny"]
    assert recipes["tiny"]["inferencer"] == "ollama/test-model"


# --- env var substitution ---------------------------------------------------

def test_env_substitution_in_mcp_json(monkeypatch, tmp_path):
    """${VAR} is expanded at load time so user configs can reference paths."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("MY_SCRIPT_DIR", "/some/where")
    user_cfg = {"mcpServers": {
        "x": {"command": "python", "args": ["${MY_SCRIPT_DIR}/srv.py"]},
    }}
    (tmp_path / "mcp.json").write_text(json.dumps(user_cfg))
    from woollama import config
    servers = config.load_mcp_servers()
    assert servers["x"]["args"] == ["/some/where/srv.py"]


# --- error reporting --------------------------------------------------------

def test_malformed_mcp_json_raises_with_source(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mcp.json").write_text("{not json")
    from woollama import config
    with pytest.raises(ValueError, match="parse error"):
        config.load_mcp_servers()


def test_recipe_missing_required_field_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text(
        "[recipes.broken]\ninferencer = 'ollama/x'\n# tools and system missing\n"
    )
    from woollama import config
    with pytest.raises(ValueError, match="missing 'tools'"):
        config.load_recipes()


def test_server_missing_command_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mcp.json").write_text(json.dumps({
        "mcpServers": {"bad": {"args": []}}   # no command
    }))
    from woollama import config
    with pytest.raises(ValueError, match="missing 'command'"):
        config.load_mcp_servers()


# --- config_dir resolution --------------------------------------------------

def test_config_dir_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path / "custom"))
    from woollama import config
    assert config.config_dir() == tmp_path / "custom"


def test_config_dir_xdg_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("WOOLLAMA_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    from woollama import config
    assert config.config_dir() == tmp_path / "xdg" / "woollama"
