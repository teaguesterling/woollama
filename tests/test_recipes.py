"""woollama.recipes — the Recipe TypedDict + make_recipe helper (Python, server-side;
not part of the Rust engine). Construct recipes in code without TOML."""
from __future__ import annotations

from woollama.recipes import make_recipe


def test_make_recipe_defaults_and_params():
    assert make_recipe("ollama/x") == {
        "inferencer": "ollama/x", "system": "", "tools": [], "params": {}}
    r = make_recipe("ollama/x", "sys", tools=("a.b",), params={"temperature": 0.2})
    assert r == {"inferencer": "ollama/x", "system": "sys",
                 "tools": ["a.b"], "params": {"temperature": 0.2}}
