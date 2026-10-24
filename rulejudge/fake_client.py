"""Deterministic fake LLM client for tests and dry runs (WP0, canonical). No network."""
from __future__ import annotations

import threading
from typing import Callable, Optional, Union

from .contracts import LLMResult

Responder = Callable[[str, str, str], Union[str, LLMResult]]


class FakeLLMClient:
    """responder(role, system, user) -> str | LLMResult. Records every call in .calls."""

    def __init__(self, responder: Optional[Responder] = None, default: str = "{}"):
        self.responder = responder
        self.default = default
        self.calls: list = []
        self._lock = threading.Lock()

    def generate(self, *, role: str, system: str, user: str,
                 json_expected: bool = True, max_output_tokens: int = 2048) -> LLMResult:
        with self._lock:
            self.calls.append({"role": role, "system": system, "user": user})
        out = self.responder(role, system, user) if self.responder else self.default
        if isinstance(out, LLMResult):
            return out
        text = str(out)
        return LLMResult(ok=True, text=text, input_tokens=(len(system) + len(user)) // 4,
                         output_tokens=len(text) // 4)
