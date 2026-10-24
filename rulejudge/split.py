"""WP5 — deterministic group-aware fitting/selection split."""
from __future__ import annotations

from .contracts import Item


def split_fs(items: list[Item], s_fraction: float = 0.2, seed: str = "rulejudge-v1") -> tuple[list[str], list[str]]:
    """Return (F_ids, S_ids). Stratify by (topic, label); exact-duplicate content (same
    item_text without id) stays in one partition; ordering by sha256(seed + item_id).
    Both partitions must contain all topics present in items, otherwise return (all_ids, [])."""
    raise NotImplementedError
