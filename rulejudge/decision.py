"""WP5 — Layer A: veto decision, metrics, head fitting/selection, floor (pure Python)."""
from __future__ import annotations

from typing import Optional

from .contracts import Assessment, Decision, HeadParams, Item


def decide(item: Item, assessment: Optional[Assessment], head: HeadParams, priors: dict) -> Decision:
    """head.kind == prior -> prior label (SRC_PRIOR_MODE).
    No assessment / failed -> prior label (SRC_FALLBACK).
    Any ACTIVE rule (not in head.disabled) with int support >= head.tau -> "bad"
      (SRC_MODEL if status ok, SRC_PARTIAL if partial).
    status ok and no trigger -> "good" (also when every rule is "na").
    status partial and no trigger -> prior label (SRC_FALLBACK)."""
    raise NotImplementedError


def topic_macro_accuracy(items: list[Item], predicted: dict[str, str]) -> dict:
    """{"macro": mean of per-topic accuracies over topics present, "by_topic": {t: {"n", "correct",
    "accuracy"}}, "n": int, "all_topics_present": bool}. Same metric as src/evaluate.py."""
    raise NotImplementedError


def fit_fitted_head(items: list[Item], assessments: dict[str, Assessment], priors: dict,
                    fit_on: str = "F") -> HeadParams:
    """For tau in (1, 2, 3): greedy disabling. Candidate = active rule triggering on >= 3 items
    with precision (share labelled bad among them) < 0.5; disable the candidate giving the best
    topic-macro accuracy if accuracy does not decrease; recompute; at most 3 disabled; ties ->
    lower precision, then rule_id. Choose tau by accuracy; ties -> 2, then the lower tau."""
    raise NotImplementedError


def select_head(candidates: list[HeadParams], s_items: list[Item], assessments: dict[str, Assessment],
                priors: dict) -> tuple[HeadParams, dict]:
    """candidates[0] is the incumbent (fixed). Pick the best topic-macro accuracy on S; the
    incumbent wins ties. Returns (head, report with every candidate's S score)."""
    raise NotImplementedError


def apply_floor(head: HeadParams, f_items: list[Item], s_items: list[Item],
                assessments: dict[str, Assessment], priors: dict) -> tuple[HeadParams, dict]:
    """If the prior beats `head` on BOTH F and S (strictly), return HeadParams(kind="prior")."""
    raise NotImplementedError


def refit_final(head: HeadParams, all_items: list[Item], assessments: dict[str, Assessment],
                priors: dict) -> HeadParams:
    """fitted -> fit_fitted_head(all_items, fit_on="F+S"); fixed/prior -> unchanged."""
    raise NotImplementedError


def rule_stats(items: list[Item], assessments: dict[str, Assessment], head: HeadParams) -> dict:
    """{rule_id: {"applicable", "triggered", "triggered_bad", "triggered_good", "sole_triggered",
    "precision", "disabled": bool}} over labelled items."""
    raise NotImplementedError
