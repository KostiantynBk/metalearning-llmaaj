"""WP1 — training label priors (fallback and prior-only mode)."""
from __future__ import annotations

from .contracts import LABELS, TOPICS, Item


def fit_priors(items: list[Item]) -> dict:
    """Fit per-topic/global majorities; ties are good, absent topics use the global prior."""
    counts = {topic: {label: 0 for label in LABELS} for topic in TOPICS}
    for item in items:
        if item.label is None:
            continue
        if item.topic not in TOPICS or item.label not in LABELS:
            raise ValueError("Priors require known topics and good/bad labels")
        counts[item.topic][item.label] += 1
    good = sum(c["good"] for c in counts.values())
    bad = sum(c["bad"] for c in counts.values())
    global_label = "bad" if bad > good else "good"
    by_topic = {}
    for topic, c in counts.items():
        by_topic[topic] = ("bad" if c["bad"] > c["good"] else "good") if sum(c.values()) else global_label
    return {"by_topic": by_topic, "global": global_label, "counts": counts}


def prior_label(priors: dict, topic: str) -> str:
    """Majority label for topic, falling back to priors['global']."""
    return priors.get("by_topic", {}).get(topic, priors["global"])
