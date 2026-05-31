"""Recipes — the "pre-packaged system prompt + tools + inferencer" bundle.

In v0.1 recipes are a hardcoded dict here. In v0.2+ they move to a real
configuration file (`~/.config/woollama/recipes.toml` or similar) — the
shape stays the same.

A recipe is addressable as a model name `woollama/<name>` in the OpenAI
surface. When a client sets `model="woollama/streamer"`, the router fetches
this recipe and orchestrates the full chat-loop transparently — the client
sees only the final answer."""
from __future__ import annotations

from typing import TypedDict


class Recipe(TypedDict):
    """A composed addressable thing: a system prompt + an inferencer + an
    allow-list of tools the model may use. The router builds the messages
    array (system + user turns), runs the chat-loop against the inferencer,
    dispatches tool_calls to MCP servers, returns the final assistant text.
    """

    inferencer: str        # "<provider>/<model>" — only "ollama/X" supported in v0.1
    system: str            # the system prompt prepended to the conversation
    tools: list[str]       # tool names from connected MCP servers (allow-list)


BUILTIN: dict[str, Recipe] = {
    "streamer": {
        "inferencer": "ollama/qwen3:14b-iq4xs",
        "system": (
            "You are a counting assistant. When the user asks you to count to "
            "a number, use the count_to tool with n set to that number. After "
            "the tool returns, confirm the count completed in one short sentence."
        ),
        "tools": ["count_to"],
    },
}


def get(name: str) -> Recipe | None:
    """Look up a recipe by name. Future: layer user-defined recipes over
    the builtins, so users can override `streamer` or add their own."""
    return BUILTIN.get(name)


def names() -> list[str]:
    return list(BUILTIN.keys())
