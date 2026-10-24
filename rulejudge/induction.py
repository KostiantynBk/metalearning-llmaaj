"""WP4 — rule induction from labelled training items (contrast pairs -> consolidation)."""
from __future__ import annotations

from .contracts import Item, LLMClient, Rulebook, Settings


def content_tokens(item: Item) -> set[str]:
    """Lower-cased word tokens from message contents and context VALUES (not keys); no digits-only."""
    raise NotImplementedError


def jaccard(a: set, b: set) -> float:
    raise NotImplementedError


def contrast_pairs(items: list[Item]) -> list[tuple[Item, Item, float]]:
    """For each labelled-bad item: most similar labelled-good item of the same topic
    (Jaccard on content_tokens; ties -> smaller item_id). Returns (bad, good, similarity)."""
    raise NotImplementedError


def validate_rulebook(rulebook: Rulebook, train_items: list[Item]) -> tuple[Rulebook, list[str]]:
    """Enforce: <= MAX_RULES, ids R01.., unique non-empty text, scope in {"global"} | TOPICS,
    no item IDs / customer first names in rule text, field length caps. Returns (clean, warnings)."""
    raise NotImplementedError


def induce_rulebook(train_items: list[Item], client: LLMClient, settings: Settings) -> tuple[Rulebook, dict]:
    """Stage 1: per-topic batches of contrast pairs -> candidate rules (proposer role).
    Stage 2: consolidation of all candidates (+ sample of unpaired good items) -> <= 16 rules.
    Fallback: if consolidation fails twice, dedupe stage-1 candidates (top 16 by confidence).
    Respects settings.max_induction_calls. Returns (rulebook, log) where log has calls, pairs,
    candidates, warnings."""
    raise NotImplementedError
