"""WP1 — training label priors (fallback and prior-only mode)."""
from __future__ import annotations

from .contracts import Item


def fit_priors(items: list[Item]) -> dict:
    """Return {"by_topic": {topic: label}, "global": label, "counts": {topic: {"good": n, "bad": n}}}.

    Majority label per topic over labelled items; ties -> "good". Topics absent from the
    training data use the global majority (ties -> "good").
    """
    raise NotImplementedError


def prior_label(priors: dict, topic: str) -> str:
    """Majority label for topic, falling back to priors["global"]."""
    raise NotImplementedError
