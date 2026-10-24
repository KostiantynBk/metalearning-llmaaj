"""WP1 — orchestration of `adapt` and `eval` (see IMPLEMENTATION_PLAN.md, WP1)."""
from __future__ import annotations

from typing import Optional

from .contracts import LLMClient, Settings


def main(argv: list[str]) -> int:
    """`adapt` -> run_adapt, `eval` -> run_eval, anything else -> usage error (exit 2)."""
    raise NotImplementedError


def run_adapt(settings: Settings, client: Optional[LLMClient] = None) -> int:
    """Priors-first adapt. Returns exit code (0 when a valid bundle is committed)."""
    raise NotImplementedError


def run_eval(settings: Settings, client: Optional[LLMClient] = None) -> int:
    """Priors-first eval with checkpointed atomic CSV rewrites. Returns exit code."""
    raise NotImplementedError
