"""Tests for the config loaders — both fallback-to-defaults and user-override
behavior, plus error reporting on malformed files."""
from __future__ import annotations

import json

import pytest

# --- path resolution --------------------------------------------------------

def test_examples_dir_resolves(tmp_path):
    """`_examples_dir()` must point at the REAL repo examples/ (the bundled default
    mcp.json's ${WOOLLAMA_EXAMPLES_DIR} spawns servers from it). Move-sensitive:
    this guards against a relocation of config.py silently shifting the parent walk
    (which is exactly what broke when config.py moved into woollama/core/)."""
    from woollama.core import config
    d = config._examples_dir()
    assert (d / "mcp-hello" / "server.py").is_file(), \
        f"_examples_dir() resolved to {d}, which has no mcp-hello/server.py"


# --- defaults fallback ------------------------------------------------------

def test_mcp_defaults_when_no_user_config(monkeypatch, tmp_path):
    """No user mcp.json → loads the bundled default."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    from woollama.core import config
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
    from woollama.core import config
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
    from woollama.core import config
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
    from woollama.core import config
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
    from woollama.core import config
    servers = config.load_mcp_servers()
    assert servers["x"]["args"] == ["/some/where/srv.py"]


# --- error reporting --------------------------------------------------------

def test_malformed_mcp_json_raises_with_source(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mcp.json").write_text("{not json")
    from woollama.core import config
    with pytest.raises(ValueError, match="parse error"):
        config.load_mcp_servers()


def test_recipe_missing_required_field_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text(
        "[recipes.broken]\ninferencer = 'ollama/x'\n# tools and system missing\n"
    )
    from woollama.core import config
    with pytest.raises(ValueError, match="missing 'tools'"):
        config.load_recipes()


def test_server_missing_command_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mcp.json").write_text(json.dumps({
        "mcpServers": {"bad": {"args": []}}   # no command
    }))
    from woollama.core import config
    with pytest.raises(ValueError, match="missing 'command'"):
        config.load_mcp_servers()


# --- conversation store selection (issue #2; config-driven, not an env var) ---

def test_conversation_store_default_none(monkeypatch, tmp_path):
    # Bundled defaults name no store → non-claude models stay stateless.
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    from woollama.core import config
    assert config.load_conversation_store() is None


def test_conversation_store_string_is_mcp_shorthand(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mcp.json").write_text(json.dumps({
        "conversationStore": "convstore",
        "mcpServers": {"convstore": {"command": "python", "args": ["s.py"]}},
    }))
    from woollama.core import config
    assert config.load_conversation_store() == {"type": "mcp", "server": "convstore"}


def test_conversation_store_typed_mcp(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mcp.json").write_text(json.dumps({
        "conversationStore": {"type": "mcp", "server": "convstore"},
        "mcpServers": {"convstore": {"command": "python", "args": ["s.py"]}},
    }))
    from woollama.core import config
    assert config.load_conversation_store() == {"type": "mcp", "server": "convstore"}


def test_conversation_store_typed_http(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mcp.json").write_text(json.dumps({
        "conversationStore": {"type": "http", "url": "http://127.0.0.1:9000"},
        "mcpServers": {},
    }))
    from woollama.core import config
    assert config.load_conversation_store() == {
        "type": "http", "url": "http://127.0.0.1:9000"}


def test_conversation_store_http_missing_url_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mcp.json").write_text(json.dumps({
        "conversationStore": {"type": "http"},   # no url
        "mcpServers": {},
    }))
    from woollama.core import config
    with pytest.raises(ValueError, match="type 'http' needs a string 'url'"):
        config.load_conversation_store()


def test_conversation_store_unknown_type_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mcp.json").write_text(json.dumps({
        "conversationStore": {"type": "smoke-signals"},
        "mcpServers": {},
    }))
    from woollama.core import config
    with pytest.raises(ValueError, match="unknown conversationStore type"):
        config.load_conversation_store()


# --- type guards: each malformed-shape contract has its own clear error ------

def test_mcp_servers_not_object_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mcp.json").write_text(json.dumps({"mcpServers": ["x"]}))
    from woollama.core import config
    with pytest.raises(ValueError, match="'mcpServers' must be an object"):
        config.load_mcp_servers()


def test_mcp_server_entry_not_object_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mcp.json").write_text(json.dumps({"mcpServers": {"bad": "echo"}}))
    from woollama.core import config
    with pytest.raises(ValueError, match="server 'bad' must be an object"):
        config.load_mcp_servers()


def test_malformed_recipes_toml_raises_with_source(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text("[recipes.x\nbroken = ")
    from woollama.core import config
    with pytest.raises(ValueError, match="parse error"):
        config.load_recipes()


def test_recipes_not_table_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text("recipes = 5\n")
    from woollama.core import config
    with pytest.raises(ValueError, match="'recipes' must be a table"):
        config.load_recipes()


def test_recipe_entry_not_table_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text("[recipes]\nx = 'not a table'\n")
    from woollama.core import config
    with pytest.raises(ValueError, match="recipe 'x' must be a table"):
        config.load_recipes()


def test_recipe_tools_not_list_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text(
        "[recipes.x]\ninferencer='ollama/m'\ntools='nope'\nsystem='s'\n")
    from woollama.core import config
    with pytest.raises(ValueError, match="'tools' must be a list"):
        config.load_recipes()


def test_malformed_inferencers_toml_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "inferencers.toml").write_text("[inferencers.x\nbroken")
    from woollama.core import config
    with pytest.raises(ValueError, match="parse error"):
        config.load_inferencers()


def test_inferencers_not_table_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "inferencers.toml").write_text("inferencers = 1\n")
    from woollama.core import config
    with pytest.raises(ValueError, match="'inferencers' must be a table"):
        config.load_inferencers()


def test_inferencer_entry_not_table_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "inferencers.toml").write_text("[inferencers]\nx = 'str'\n")
    from woollama.core import config
    with pytest.raises(ValueError, match="'x' must be a table"):
        config.load_inferencers()


# --- config_dir resolution --------------------------------------------------

def test_config_dir_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path / "custom"))
    from woollama.core import config
    assert config.config_dir() == tmp_path / "custom"


def test_config_dir_xdg_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("WOOLLAMA_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    from woollama.core import config
    assert config.config_dir() == tmp_path / "xdg" / "woollama"
