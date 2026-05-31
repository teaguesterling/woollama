"""Recipes — the "pre-packaged system prompt + tools + inferencer" bundle.

In v0.1 recipes are a hardcoded dict here. In v0.2 they move to a real
configuration file — the shape stays the same.

Tool names are **namespaced** as `<server>.<tool>`, matching woollama's
multi-MCP-server registry. The router parses the namespace at dispatch time
and routes the tool call to the owning manager."""
from __future__ import annotations

from typing import TypedDict


class Recipe(TypedDict):
    """A composed addressable thing: a system prompt + an inferencer + an
    allow-list of namespaced tools the model may use."""

    inferencer: str        # "<provider>/<model>" — only "ollama/X" in v0.1
    system: str            # system prompt prepended to the conversation
    tools: list[str]       # `<server>.<tool>` names — see manager.Registry


BUILTIN: dict[str, Recipe] = {
    # Single-server recipe — uses one tool from one server.
    "streamer": {
        "inferencer": "ollama/qwen3:14b-iq4xs",
        "system": (
            "You are a counting assistant. When the user asks you to count to "
            "a number, use the hello.count_to tool with n set to that number. "
            "After the tool returns, confirm the count completed in one short "
            "sentence."
        ),
        "tools": ["hello.count_to"],
    },
    # Cross-server recipe — combines tools from BOTH bundled example servers,
    # proving the multi-server unified registry composes in a single chat-loop.
    "textcounter": {
        "inferencer": "ollama/qwen3:14b-iq4xs",
        "system": (
            "You are a text-processing helper. When the user gives you text, "
            "use textops.word_count to count its words, then use "
            "hello.count_to to count to that number. Report both the word "
            "count and that the counting completed, in one short sentence."
        ),
        "tools": ["textops.word_count", "hello.count_to"],
    },
}


def get(name: str) -> Recipe | None:
    return BUILTIN.get(name)


def names() -> list[str]:
    return list(BUILTIN.keys())
