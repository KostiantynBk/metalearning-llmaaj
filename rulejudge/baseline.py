"""WP7 — same-model baseline judge without learned rules (for the FAQ comparison)."""
from __future__ import annotations

from .contracts import Item, LLMClient, Settings

BASELINE_TEMPLATE_VERSION = "b1"


def build_baseline_request(item: Item) -> tuple[str, str]:
    """(system, user) using render.render_item; asks for {"label": "good"|"bad"} only."""
    raise NotImplementedError


def run_baseline(items: list[Item], client: LLMClient, settings: Settings, priors: dict) -> dict:
    """{item_id: {"label", "source"}}; parse failures after 2 attempts -> prior label, source fallback."""
    raise NotImplementedError
