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


def _registry() -> dict[str, Inferencer]:
    """Built-in inferencers. Rebuilt per call so env overrides (the Ollama URL,
    API keys) are picked up live — important for tests and reconfiguration."""
    ollama_url = os.environ.get("WOOLLAMA_OLLAMA_URL", "http://localhost:11434")
    return {
        "ollama": Inferencer(
            name="ollama",
            base_url=f"{ollama_url}/v1",
            extra_body={"options": {"temperature": 0}},  # Ollama-native; unchanged
        ),
        "anthropic": Inferencer(
            name="anthropic",
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
            extra_body={"temperature": 0, "max_tokens": 4096},
        ),
    }


def get(provider: str) -> Inferencer | None:
    """Resolve a provider name (the part before `/` in an inferencer string)."""
    return _registry().get(provider)


def names() -> list[str]:
    return list(_registry().keys())
