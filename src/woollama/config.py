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
from dataclasses import dataclass, field
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


def state_dir() -> Path:
    """Where woollama's durable runtime state lives (e.g. the conversation handle
    table that must survive a restart). Precedence:
      1. `$WOOLLAMA_STATE_DIR` (explicit override)
      2. `$XDG_STATE_HOME/woollama`
      3. `~/.local/state/woollama`
    Distinct from `config_dir()` (user-authored config) — this is app-managed."""
    if override := os.environ.get("WOOLLAMA_STATE_DIR"):
        return Path(override).expanduser()
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(base) / "woollama"


def _examples_dir() -> Path:
    """The repo's `examples/` directory (dev checkout only — used by the bundled
    default mcp.json's `${WOOLLAMA_EXAMPLES_DIR}`). This file lives at
    `src/woollama/config.py`, so the repo root is the 3rd parent up. Guarded
    by `test_examples_dir_resolves` so a future move can't silently break it."""
    return Path(__file__).resolve().parents[2] / "examples"


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


def load_conversation_store() -> dict | None:
    """The conversation store to use (issue #2), from the top-level
    `conversationStore` key in `mcp.json`. Returns a typed descriptor or `None`
    (the default ⇒ non-claude models stay stateless). Accepted forms:

      - `"convstore"`                         → `{"type": "mcp", "server": "convstore"}`
        (string shorthand: names an `mcpServers` entry)
      - `{"type": "mcp", "server": "name"}`   → an MCP conversation-store server
      - `{"type": "http", "url": "http://…"}` → a REST conversation-store endpoint

    The router validates that an `mcp` server actually exists in `mcpServers` (it
    warns + stays stateless if not). Config-driven, not an env var, so the wiring
    lives with the rest of woollama's file config."""
    source, text = _read_user_or_default("mcp.json")
    text = _expand_env(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"mcp.json parse error in {source}: {e}") from e
    raw = data.get("conversationStore")
    if raw is None:
        return None
    if isinstance(raw, str):                       # shorthand → mcp server name
        return {"type": "mcp", "server": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"mcp.json {source}: 'conversationStore' must be a "
                         f"string or an object with a 'type'")
    kind = raw.get("type")
    if kind == "mcp":
        server = raw.get("server")
        if not isinstance(server, str):
            raise ValueError(f"mcp.json {source}: conversationStore type 'mcp' "
                             f"needs a string 'server' (an mcpServers key)")
        return {"type": "mcp", "server": server}
    if kind == "http":
        url = raw.get("url")
        if not isinstance(url, str):
            raise ValueError(f"mcp.json {source}: conversationStore type 'http' "
                             f"needs a string 'url'")
        return {"type": "http", "url": url}
    raise ValueError(f"mcp.json {source}: unknown conversationStore type "
                     f"{kind!r} (expected 'mcp' or 'http')")


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
        params = entry.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(
                f"recipes.toml {source}: recipe '{name}': 'params' must be a table")
        out[name] = {
            "inferencer": entry["inferencer"],
            "tools": list(entry["tools"]),
            "system": entry["system"].strip(),
            "params": params,
        }
    log.info("loaded %d recipe(s) from %s: %s",
             len(out), source, list(out.keys()))
    return out


# ----- inferencers.toml loading ---------------------------------------------

def load_inferencers() -> dict[str, dict]:
    """User-defined OpenAI-compat inferencers from `$config/inferencers.toml`
    (optional). Returns `{name: {base_url, api_key_env, extra_body}}`.

    Unlike recipes/mcp.json, this is MERGED OVER the built-in providers FIELD BY
    FIELD (a same name overlays only the keys it sets) rather than replacing them
    — inferencers are an infrastructure registry you extend (add vLLM, override a
    base_url, or add `models` to a built-in cloud provider without restating its
    base_url). The merge + the "new provider needs base_url" check live in
    `inferencers._registry`; this function only parses. `${VAR}` is expanded in
    values; `api_key_env` is the NAME of an env var.

    TOML shape (all fields optional when extending a built-in; base_url required
    for a NEW provider):
        [inferencers.<name>]
        base_url       = "https://host/v1"      # OpenAI-compatible base
        api_key_env    = "SOME_API_KEY"         # omit for no-auth (local)
        extra_body     = { temperature = 0 }    # merged into each request
        models         = ["id1", "id2"]         # surface these in /v1/models (#3)
        discover       = true                   # live-query the provider's /v1/models
        model_patterns = ["claude-*", "gpt-4*"] # fnmatch filter for `discover`

    Returns `{name: {only the keys present in the TOML}}` — absent keys are
    omitted (not defaulted) so the registry merge can tell "unset" from "set".
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
        spec: dict = {}
        for key in ("base_url", "api_key_env", "extra_body"):
            if key in entry:
                spec[key] = entry[key]
        for list_key in ("models", "model_patterns"):
            if list_key in entry:
                if not isinstance(entry[list_key], list):
                    raise ValueError(
                        f"inferencers.toml {path}: '{name}.{list_key}' must be a list")
                spec[list_key] = [str(v) for v in entry[list_key]]
        if "discover" in entry:
            if not isinstance(entry["discover"], bool):
                raise ValueError(
                    f"inferencers.toml {path}: '{name}.discover' must be true/false")
            spec["discover"] = entry["discover"]
        if "management_url" in entry:
            spec["management_url"] = str(entry["management_url"])
        if "management_protocol" in entry:
            spec["management_protocol"] = str(entry["management_protocol"])
        for int_key in ("parallel", "pool_max", "queue_max"):
            if int_key in entry:
                v = entry[int_key]
                if isinstance(v, bool) or not isinstance(v, int):
                    raise ValueError(
                        f"inferencers.toml {path}: '{name}.{int_key}' must be an integer")
                spec[int_key] = v
        if "queue_timeout" in entry:
            v = entry["queue_timeout"]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"inferencers.toml {path}: '{name}.queue_timeout' must be a number")
            spec["queue_timeout"] = float(v)
        if "virtual" in entry:
            v = entry["virtual"]
            if not isinstance(v, dict):
                raise ValueError(
                    f"inferencers.toml {path}: '{name}.virtual' must be a table")
            spec["virtual"] = {str(k): str(val) for k, val in v.items()}
        out[name] = spec
    log.info("loaded %d user inferencer(s) from %s: %s",
             len(out), path, list(out.keys()))
    return out


# ----- management_protocols parsing (from inferencers.toml) -----------------
#
# Mirrors the Rust `EndpointSpec`/`ProtocolSpec` + `load_management_protocols`
# in woollama-engine/src/lib.rs — the behavior authority for this shape. Pure
# parsing only: no `{base}`/`{id}` templating and no further `${VAR}` expansion
# beyond the whole-file _expand_env pass already applied below (that's Task 3's
# runtime concern, when a backend actually issues requests).

@dataclass(frozen=True)
class EndpointSpec:
    """One HTTP call of a `rest` management protocol. `path`/`id_field` apply
    only to the `running` endpoint: `path` is a dotted lookup into the JSON
    response for the array/object of currently-loaded models, `id_field` the
    key holding each model's id when that array holds objects rather than bare
    strings."""
    url: str
    method: str | None = None
    body: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    path: str | None = None
    id_field: str | None = None


@dataclass(frozen=True)
class RestProtocolSpec:
    """Three explicit HTTP calls: list loaded models, start one, stop one."""
    running: EndpointSpec
    start: EndpointSpec
    stop: EndpointSpec
    kind: str = "rest"


@dataclass(frozen=True)
class OllamaProtocolSpec:
    """ollama's native `keep_alive` knob (no explicit start/stop calls)."""
    kind: str = "ollama"
    keep_alive: str | None = None


ProtocolSpec = RestProtocolSpec | OllamaProtocolSpec


def _protocol_err(path: Path, name: str, key: str, msg: str) -> ValueError:
    return ValueError(f"inferencers.toml {path}: management_protocols.{name}.{key}: {msg}")


def _parse_endpoint_spec(
    path: Path, proto_name: str, op: str, endpoints: dict, *, require_path: bool,
) -> EndpointSpec:
    """Parse one `[management_protocols.<name>.endpoints.<op>]` table.
    `require_path` is set only for the `running` endpoint (`path` is
    meaningless for `start`/`stop`)."""
    key = f"endpoints.{op}"
    eobj = endpoints.get(op)
    if not isinstance(eobj, dict):
        raise _protocol_err(path, proto_name, key, "required table")
    url = eobj.get("url")
    if url is None:
        raise _protocol_err(path, proto_name, f"{key}.url", "required")
    if not isinstance(url, str):
        raise _protocol_err(path, proto_name, f"{key}.url", "must be a string")
    method = eobj.get("method")
    if method is not None and not isinstance(method, str):
        raise _protocol_err(path, proto_name, f"{key}.method", "must be a string")
    body = eobj.get("body")
    if body is not None and not isinstance(body, str):
        raise _protocol_err(path, proto_name, f"{key}.body", "must be a string")
    headers: dict[str, str] = {}
    headers_raw = eobj.get("headers")
    if headers_raw is not None:
        if not isinstance(headers_raw, dict):
            raise _protocol_err(path, proto_name, f"{key}.headers", "must be a table of strings")
        for hk, hv in headers_raw.items():
            if not isinstance(hv, str):
                raise _protocol_err(path, proto_name, f"{key}.headers.{hk}", "must be a string")
            headers[hk] = hv
    if require_path:
        p = eobj.get("path")
        if not isinstance(p, str):
            raise _protocol_err(path, proto_name, f"{key}.path", "required")
        id_field = eobj.get("id_field")
        if id_field is not None and not isinstance(id_field, str):
            raise _protocol_err(path, proto_name, f"{key}.id_field", "must be a string")
    else:
        p = None
        id_field = None
    return EndpointSpec(url=url, method=method, body=body, headers=headers,
                        path=p, id_field=id_field)


def load_management_protocols() -> dict[str, ProtocolSpec]:
    """User-defined device management protocols from `[management_protocols.<name>]`
    in `$config/inferencers.toml` (same file as `[inferencers.*]`, optional). A
    protocol is selected per-inferencer via `Inferencer.management_protocol`;
    `None` means "use the built-in tiiny backend" (resolved by Task 3).

    Missing file, or a file with no `[management_protocols]` section at all,
    both yield `{}` — this section is entirely optional. `${VAR}` is expanded
    (same whole-file pass as `load_inferencers`); no `{base}`/`{id}` templating
    is applied here — that's a runtime concern for whatever actually issues the
    HTTP calls.

    TOML shape:
        [management_protocols.<name>]
        kind = "rest"                        # or "ollama"

        [management_protocols.<name>.endpoints.running]  # kind = "rest" only
        url = "http://host/api/loaded"
        path = "models"                      # required: dotted JSON lookup
        id_field = "id"                      # optional

        [management_protocols.<name>.endpoints.start]
        url = "http://host/api/load"
        method = "POST"                      # optional
        body = '{"model": "{id}"}'           # optional, templated at call time
        headers = { Authorization = "Bearer ${TOKEN}" }  # optional

        [management_protocols.<name>.endpoints.stop]
        url = "http://host/api/unload"

        # or, for kind = "ollama":
        [management_protocols.<name>]
        kind = "ollama"
        keep_alive = "5m"                    # optional
    """
    path = config_dir() / "inferencers.toml"
    if not path.is_file():
        return {}
    text = _expand_env(path.read_text())
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"inferencers.toml parse error in {path}: {e}") from e
    raw = data.get("management_protocols")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"inferencers.toml {path}: 'management_protocols' must be a table")
    out: dict[str, ProtocolSpec] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"inferencers.toml {path}: management_protocols.'{name}' must be a table")
        kind = entry.get("kind")
        if kind is None:
            raise _protocol_err(path, name, "kind", "required")
        if not isinstance(kind, str):
            raise _protocol_err(path, name, "kind", "must be a string")
        if kind == "rest":
            endpoints = entry.get("endpoints")
            if not isinstance(endpoints, dict):
                raise _protocol_err(path, name, "endpoints",
                                    "required table for kind = \"rest\"")
            running = _parse_endpoint_spec(path, name, "running", endpoints, require_path=True)
            start = _parse_endpoint_spec(path, name, "start", endpoints, require_path=False)
            stop = _parse_endpoint_spec(path, name, "stop", endpoints, require_path=False)
            out[name] = RestProtocolSpec(running=running, start=start, stop=stop)
        elif kind == "ollama":
            keep_alive = entry.get("keep_alive")
            if keep_alive is not None and not isinstance(keep_alive, str):
                raise _protocol_err(path, name, "keep_alive", "must be a string")
            out[name] = OllamaProtocolSpec(keep_alive=keep_alive)
        else:
            raise _protocol_err(path, name, "kind",
                                f"must be 'rest' or 'ollama' (got {kind!r})")
    log.info("loaded %d management protocol(s) from %s: %s",
             len(out), path, list(out.keys()))
    return out
