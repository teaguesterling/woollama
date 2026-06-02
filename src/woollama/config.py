"""Configuration loading for woollama.

Two config files, both optional:

  $XDG_CONFIG_HOME/woollama/mcp.json     - which MCP servers to spawn
  $XDG_CONFIG_HOME/woollama/recipes.toml - the orchestration bundles

When a user config file doesn't exist, the packaged default ships at
`src/woollama/defaults/<file>` is loaded instead — so `pip install woollama
&& woollama` works out of the box with the bundled hello + textops example
servers and the two example recipes.

Env var substitution: `${VAR}` is expanded at load time using `os.path
.expandvars`. The loader also sets `WOOLLAMA_EXAMPLES_DIR` to the absolute
path of the package's `examples/` directory so the bundled defaults can
reference the bundled example servers portably.

Override the config search dir entirely with `$WOOLLAMA_CONFIG_DIR`.
"""
from __future__ import annotations

import json
import logging
import os
import tomllib
from importlib.resources import files
from pathlib import Path

from . import recipes as recipes_module

log = logging.getLogger("woollama.config")


def config_dir() -> Path:
    """Where user config lives. Precedence:
      1. `$WOOLLAMA_CONFIG_DIR` (explicit override)
      2. `$XDG_CONFIG_HOME/woollama`
      3. `~/.config/woollama`
    """
    if override := os.environ.get("WOOLLAMA_CONFIG_DIR"):
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "woollama"


def _examples_dir() -> Path:
    """The package's `examples/` directory, used for bundled defaults."""
    return Path(__file__).resolve().parent.parent.parent / "examples"


def _expand_env(text: str) -> str:
    """Apply env-var substitution. We set WOOLLAMA_EXAMPLES_DIR before
    calling so the bundled defaults can reference packaged examples."""
    os.environ["WOOLLAMA_EXAMPLES_DIR"] = str(_examples_dir())
    return os.path.expandvars(text)


def _read_user_or_default(filename: str) -> tuple[str, str]:
    """Return (source_label, text_content). Prefers user config; falls back
    to the packaged default."""
    user_path = config_dir() / filename
    if user_path.is_file():
        return (str(user_path), user_path.read_text())
    default = files("woollama.defaults").joinpath(filename).read_text()
    return (f"<bundled defaults: {filename}>", default)


# ----- mcp.json loading -----------------------------------------------------

def load_mcp_servers() -> dict[str, dict]:
    """Return `{name: {command, args, env}}` for every configured MCP server.
    Shape matches Claude Code's mcp.json `mcpServers` block."""
    source, text = _read_user_or_default("mcp.json")
    text = _expand_env(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"mcp.json parse error in {source}: {e}") from e
    servers = data.get("mcpServers") or {}
    if not isinstance(servers, dict):
        raise ValueError(f"mcp.json {source}: 'mcpServers' must be an object")
    out: dict[str, dict] = {}
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            raise ValueError(f"mcp.json {source}: server '{name}' must be an object")
        if "command" not in entry:
            raise ValueError(f"mcp.json {source}: server '{name}' is missing 'command'")
        out[name] = {
            "command": entry["command"],
            "args": entry.get("args") or [],
            "env": entry.get("env") or {},
        }
    log.info("loaded %d MCP server(s) from %s: %s",
             len(out), source, list(out.keys()))
    return out


# ----- recipes.toml loading -------------------------------------------------

def load_recipes() -> dict[str, recipes_module.Recipe]:
    """Return `{name: Recipe}` from the user's recipes.toml or the bundled
    default. The TOML shape is `[recipes.<name>]` with `inferencer`, `tools`
    (namespaced array), and `system` fields."""
    source, text = _read_user_or_default("recipes.toml")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"recipes.toml parse error in {source}: {e}") from e
    raw = data.get("recipes") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"recipes.toml {source}: 'recipes' must be a table")
    out: dict[str, recipes_module.Recipe] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"recipes.toml {source}: recipe '{name}' must be a table")
        for required in ("inferencer", "tools", "system"):
            if required not in entry:
                raise ValueError(
                    f"recipes.toml {source}: recipe '{name}' missing '{required}'")
        if not isinstance(entry["tools"], list):
            raise ValueError(
                f"recipes.toml {source}: recipe '{name}': 'tools' must be a list")
        out[name] = {
            "inferencer": entry["inferencer"],
            "tools": list(entry["tools"]),
            "system": entry["system"].strip(),
        }
    log.info("loaded %d recipe(s) from %s: %s",
             len(out), source, list(out.keys()))
    return out


# ----- inferencers.toml loading ---------------------------------------------

def load_inferencers() -> dict[str, dict]:
    """User-defined OpenAI-compat inferencers from `$config/inferencers.toml`
    (optional). Returns `{name: {base_url, api_key_env, extra_body}}`.

    Unlike recipes/mcp.json, this is MERGED OVER the built-in providers (a same
    name overrides a built-in) rather than replacing them — inferencers are an
    infrastructure registry you extend (add vLLM / a self-hosted endpoint /
    override a base_url), not user content you own wholesale. See
    `inferencers._registry`. `${VAR}` is expanded in values (e.g.
    `base_url = "${VLLM_URL}/v1"`); `api_key_env` is the NAME of an env var.

    TOML shape:
        [inferencers.<name>]
        base_url   = "https://host/v1"     # required (OpenAI-compatible base)
        api_key_env = "SOME_API_KEY"       # optional; omit for no-auth (local)
        extra_body = { temperature = 0 }   # optional; merged into each request
    """
    path = config_dir() / "inferencers.toml"
    if not path.is_file():
        return {}
    text = _expand_env(path.read_text())
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"inferencers.toml parse error in {path}: {e}") from e
    raw = data.get("inferencers") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"inferencers.toml {path}: 'inferencers' must be a table")
    out: dict[str, dict] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"inferencers.toml {path}: '{name}' must be a table")
        if "base_url" not in entry:
            raise ValueError(f"inferencers.toml {path}: '{name}' missing 'base_url'")
        out[name] = {
            "base_url": entry["base_url"],
            "api_key_env": entry.get("api_key_env"),
            "extra_body": entry.get("extra_body") or {},
        }
    log.info("loaded %d user inferencer(s) from %s: %s",
             len(out), path, list(out.keys()))
    return out
