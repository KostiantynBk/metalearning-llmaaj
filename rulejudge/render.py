"""Shared, deterministic rendering of an item for prompts and reports (WP0, canonical).

All prompt builders (assessor, induction, baseline) must use render_item so that the
model sees identical item text everywhere. Never include labels or item IDs here.
"""
from __future__ import annotations

import json
import re

from .contracts import Item

_DELIMS = re.compile(r"(<<<|>>>)")


def _neutralize(text: str) -> str:
    # Conversation/context text is data; stop it from forging our delimiters.
    return _DELIMS.sub(lambda m: m.group(1).replace("<", "‹").replace(">", "›"), text)


def render_context(item: Item) -> str:
    return _neutralize(json.dumps(item.context, ensure_ascii=False, separators=(", ", ": ")))


def render_messages(item: Item) -> str:
    lines = []
    for i, msg in enumerate(item.conversation, start=1):
        lines.append(f"[msg {i} | {msg.get('role', '?')}] {_neutralize(str(msg.get('content', '')))}")
    return "\n".join(lines)


def render_item(item: Item, include_topic: bool = True) -> str:
    parts = []
    if include_topic:
        parts.append(f"Topic: {item.topic}")
    parts += ["Context (JSON data available to the assistant):", "<<<CONTEXT", render_context(item),
              "CONTEXT>>>", "Messages:", "<<<MESSAGES", render_messages(item), "MESSAGES>>>"]
    return "\n".join(parts)


def excerpt(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def message_text(item: Item, n: int) -> str:
    """1-based message index; '' if out of range."""
    return str(item.conversation[n - 1].get("content", "")) if 1 <= n <= len(item.conversation) else ""


_MSG = re.compile(r"^\s*msg\s*(\d+)\s*$", re.IGNORECASE)


def resolve_pointer(item: Item, pointer: str) -> bool:
    """True if pointer is 'msg <n>' with a valid n, or 'context.<path>' that exists."""
    if not isinstance(pointer, str):
        return False
    m = _MSG.match(pointer)
    if m:
        return 1 <= int(m.group(1)) <= len(item.conversation)
    if not pointer.startswith("context"):
        return False
    node = item.context
    for token in re.findall(r"[^.\[\]]+|\[\d+\]", pointer[len("context"):]):
        if token.startswith("["):
            idx = int(token[1:-1])
            if not isinstance(node, list) or idx >= len(node):
                return False
            node = node[idx]
        else:
            if not isinstance(node, dict) or token not in node:
                return False
            node = node[token]
    return True
