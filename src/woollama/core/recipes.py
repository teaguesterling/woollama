"""Recipes — the "pre-packaged system prompt + tools + inferencer" bundle.

Loaded from `$XDG_CONFIG_HOME/woollama/recipes.toml` (or the bundled
default at `woollama/defaults/recipes.toml` if the user file doesn't
exist). See `config.load_recipes` and `defaults/recipes.toml` for the
schema.

Tool names are namespaced as `<server>.<tool>`, matching the multi-MCP
unified registry in `manager.Registry`. The router parses the namespace
at dispatch time and routes to the owning server.
"""
from __future__ import annotations

from typing import TypedDict


class Recipe(TypedDict):
    """A composed addressable thing: a system prompt + an inferencer + an
    allow-list of namespaced tools the model may use."""

    inferencer: str        # "<provider>/<model>" — only "ollama/X" in v0.1
    system: str            # system prompt prepended to the conversation
    tools: list[str]       # `<server>.<tool>` names — see manager.Registry


# Populated lazily on first access; reloaded if `reload()` is called.
_LOADED: dict[str, Recipe] | None = None


def _load() -> dict[str, Recipe]:
    global _LOADED
    if _LOADED is None:
        # Imported here to avoid a circular import between config <-> recipes.
        from . import config
        _LOADED = config.load_recipes()
    return _LOADED


def reload() -> None:
    """Force a re-read of recipes.toml on next access. Useful for tests
    and for a future `woollama config reload` op."""
    global _LOADED
    _LOADED = None


def get(name: str) -> Recipe | None:
    return _load().get(name)


def names() -> list[str]:
    return list(_load().keys())
