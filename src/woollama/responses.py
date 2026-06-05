"""OpenAI *Responses* wire-shape helpers — the stateful surface's data layer.

woollama's `/v1/responses` adopts the OpenAI Responses shapes verbatim (so every
OpenAI SDK and cosmic-fabric speak it for free — docs/conversations-api-design.md
§1). This module is the PURE shaping layer: parse the polymorphic `input`, mint
ids, and build a Response dict that the real `openai` SDK parses (its computed
`.output_text` aggregates our `output_text` content parts).

It imports nothing from the router (kept acyclic); the endpoint + routing live in
router.py. Slice conv-1a covers the stateless (`store:false`) shape only — handle
routing and stateful backends are conv-1b.
"""
from __future__ import annotations

import time
import uuid


def new_id(prefix: str) -> str:
    """`resp_<hex>` / `msg_<hex>` / `conv_<hex>` — stable, opaque handles. The
    `resp_` id is designed to be a fork point (previous_response_id) that conv-1b
    keys its handle table on, so it must be unique per turn."""
    return f"{prefix}_{uuid.uuid4().hex}"


def parse_input(value) -> list[dict]:
    """Normalize the Responses `input` (a bare string OR a list of message items)
    into OpenAI chat messages. A string is one user turn; a list maps each
    {role, content} item, flattening content-part arrays ({type: input_text|
    output_text|text, text}) to plain text (a v1 simplification — multimodal
    parts are a later refinement)."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list):
        msgs: list[dict] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") in ("input_text", "output_text", "text"))
            msgs.append({"role": role,
                         "content": content if isinstance(content, str) else str(content)})
        return msgs
    raise ValueError("`input` must be a string or a list of message items")


def build_response(resp_id: str, model: str, text: str, *,
                   conversation: str | None = None,
                   status: str = "completed",
                   created_at: int | None = None) -> dict:
    """Assemble an OpenAI-Responses-shaped dict the `openai` SDK validates. The
    field set is exactly what `openai.types.responses.Response` requires plus the
    assistant message; `.output_text` is the SDK-computed join of the
    `output_text` parts (we don't emit it ourselves)."""
    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(created_at if created_at is not None else time.time()),
        "model": model,
        "status": status,
        # A Response carries a conversation OBJECT (`{id}`) or null — requests
        # still attach by bare string id (see router). None for stateless turns.
        "conversation": ({"id": conversation} if conversation else None),
        "output": [{
            "type": "message",
            "id": new_id("msg"),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        # Required by the SDK's Response model; woollama's stateless surface does
        # not expose tool config, so these are inert defaults.
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
