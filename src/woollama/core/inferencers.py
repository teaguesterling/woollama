"""Inferencer registry — the OpenAI-compat backend seam.

An *inferencer* is an OpenAI-compatible chat-completions backend addressed as
`<provider>/<model>` (e.g. `ollama/qwen3:14b`, `anthropic/claude-sonnet-4-6`).
woollama is the router: it resolves the provider to a base URL + auth and POSTs
the same OpenAI-shaped request there. "Inference backends speak OpenAI-compat
natively, so we don't ship wrappers — the router talks to their OpenAI endpoints
directly" (docs/architecture.md).

Built-ins: `ollama` (local, no auth) and `anthropic` (the Claude API's OpenAI
compatibility endpoint — tools/function-calling ARE supported there, so the full
orchestration loop works; only `strict` schema enforcement is dropped). Adding
vLLM / Together / Groq / OpenRouter is just more entries with the same shape;
config-file-driven inferencers (architecture.md's `inferencers` block) are the
natural follow-on. Secrets come from env vars, read at call time.

`claude-code/` is deliberately NOT here — it's a subprocess-delegation backend
(see claude_code.py), a different mechanism, dispatched separately in
`router.orchestrate`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import config


class InferencerError(Exception):
    """Configuration/credential problem resolving an inferencer (e.g. a missing
    API key). The router maps this to a clear client error."""


@dataclass(frozen=True)
class Inferencer:
    name: str
    base_url: str               # OpenAI-compatible base, WITHOUT /chat/completions
    api_key_env: str | None = None   # env var holding the bearer key; None = no auth
    # Provider-specific request fields merged into each ORCHESTRATION request
    # (not passthrough — there the client owns the body). Keeps Ollama's native
    # `options` and gives Anthropic a sane max_tokens / a clamped temperature.
    extra_body: dict = field(default_factory=dict)
    # Model discoverability in GET /v1/models (issue #3). woollama owns no model
    # catalog — a provider's models are surfaced only if opted in here:
    #   `models`         — explicit ids to list (no key needed to list).
    #   `discover`       — query the provider's own /v1/models live (needs key).
    #   `model_patterns` — fnmatch globs to filter a (large) discovered catalog;
    #                      empty = list all discovered. Applies to discovery only.
    models: tuple[str, ...] = ()
    discover: bool = False
    model_patterns: tuple[str, ...] = ()

    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def headers(self) -> dict[str, str]:
        """Auth headers; raises InferencerError if the configured key env is
        unset (fail fast with a clear message rather than a 401 from upstream)."""
        if not self.api_key_env:
            return {}
        key = os.environ.get(self.api_key_env)
        if not key:
            raise InferencerError(
                f"inferencer '{self.name}' requires ${self.api_key_env} to be set")
        return {"Authorization": f"Bearer {key}"}


# Well-known OpenAI-compatible cloud providers, base URLs verified from each
# vendor's docs (2026-06-02). They "just work" once the named API key env var is
# set. Anything not here — self-hosted vLLM/llama.cpp, niche clouds, or an
# override — is added via $config/inferencers.toml (config.load_inferencers),
# merged on top by name. temperature=0 for deterministic orchestration.
_BUILTIN_CLOUDS: tuple[tuple[str, str, str], ...] = (
    # (name,        base_url,                            api_key_env)
    ("openai",     "https://api.openai.com/v1",          "OPENAI_API_KEY"),
    ("groq",       "https://api.groq.com/openai/v1",     "GROQ_API_KEY"),
    ("together",   "https://api.together.ai/v1",         "TOGETHER_API_KEY"),
    ("openrouter", "https://openrouter.ai/api/v1",       "OPENROUTER_API_KEY"),
)


def _registry() -> dict[str, Inferencer]:
    """All inferencers: built-ins overlaid by user config. Rebuilt per call so
    env overrides (Ollama URL, API keys) and edits to inferencers.toml are
    picked up live — important for tests and reconfiguration."""
    ollama_url = os.environ.get("WOOLLAMA_OLLAMA_URL", "http://localhost:11434")
    reg: dict[str, Inferencer] = {
        "ollama": Inferencer(
            name="ollama",
            base_url=f"{ollama_url}/v1",
            extra_body={"options": {"temperature": 0}},  # Ollama-native; unchanged
            discover=True,    # local catalog is small + needs no key → list it (as before)
        ),
        "anthropic": Inferencer(
            name="anthropic",
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
            extra_body={"temperature": 0, "max_tokens": 4096},
        ),
    }
    for name, base_url, key_env in _BUILTIN_CLOUDS:
        reg[name] = Inferencer(name=name, base_url=base_url,
                               api_key_env=key_env, extra_body={"temperature": 0})

    # User config merges over built-ins, FIELD BY FIELD: a config entry for an
    # existing provider overlays only the keys it sets (so you can add `models` to
    # `anthropic` without restating its base_url); a new provider must supply
    # base_url. (`base_url` validation lives here, not in config.load_inferencers,
    # because only here do we know whether the name extends a built-in.)
    for name, spec in config.load_inferencers().items():
        base = reg.get(name)
        base_url = spec.get("base_url") or (base.base_url if base else None)
        if not base_url:
            raise InferencerError(
                f"inferencer '{name}' has no base_url and is not a known built-in "
                "to extend — add base_url in inferencers.toml")
        reg[name] = Inferencer(
            name=name,
            base_url=base_url,
            api_key_env=spec.get("api_key_env", base.api_key_env if base else None),
            extra_body=spec.get("extra_body") or (base.extra_body if base else {}),
            models=tuple(spec.get("models") or (base.models if base else ())),
            discover=spec.get("discover", base.discover if base else False),
            model_patterns=tuple(
                spec.get("model_patterns") or (base.model_patterns if base else ())),
        )
    return reg


def all() -> dict[str, Inferencer]:  # noqa: A001 — registry accessor
    """All inferencers (built-ins overlaid by config). Used by GET /v1/models."""
    return _registry()


def get(provider: str) -> Inferencer | None:
    """Resolve a provider name (the part before `/` in an inferencer string)."""
    return _registry().get(provider)


def names() -> list[str]:
    return list(_registry().keys())
