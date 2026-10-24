"""WP3 — per-conversation rule assessment (one Gemma call per item, compact JSON output)."""
from __future__ import annotations

from typing import Callable, Optional

from .contracts import Assessment, Item, LLMClient, Rule, Rulebook, Settings

ASSESSOR_TEMPLATE_VERSION = "a1"


def rules_for_topic(rulebook: Rulebook, topic: str) -> list[Rule]:
    """Global rules + rules scoped to topic, ordered by rule_id."""
    raise NotImplementedError


def build_assessor_request(item: Item, rules: list[Rule]) -> tuple[str, str]:
    """Return (system, user). Uses render.render_item; never includes labels or item_id."""
    raise NotImplementedError


def parse_assessment(text: str, item: Item, expected_rule_ids: list[str]) -> Assessment:
    """Parse {"s": {rid: "na"|0..3}, "e": [{"r", "w", "y"}]} into an Assessment.

    status ok: every expected id present with a legal value. partial: >=1 legal value but some
    missing/illegal. failed: nothing usable. Evidence kept only for support >= 1 and pointers
    that resolve (render.resolve_pointer); unresolvable pointers are dropped, not fatal.
    """
    raise NotImplementedError


def assess_item(item: Item, rulebook: Rulebook, client: LLMClient, settings: Settings) -> Assessment:
    """Up to settings.max_content_attempts calls; second attempt appends a short format reminder."""
    raise NotImplementedError


def assess_many(items: list[Item], rulebook: Rulebook, client: LLMClient, settings: Settings,
                on_progress: Optional[Callable[[dict], None]] = None) -> dict[str, Assessment]:
    """Thread pool (settings.max_concurrency) main pass, then ONE slow sequential recovery pass
    for failed items (stop early on daily_quota/deadline). Calls on_progress(results_so_far)
    every settings.checkpoint_every completed items and at the end. Returns {item_id: Assessment}.
    """
    raise NotImplementedError
