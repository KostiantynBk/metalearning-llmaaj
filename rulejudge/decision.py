"""WP5 — Layer A: veto decision, metrics, head fitting/selection, floor (pure Python)."""
from __future__ import annotations

from typing import Optional

from .contracts import (LABELS, TOPICS, SRC_FALLBACK, SRC_MODEL, SRC_PARTIAL,
                        SRC_PRIOR_MODE, Assessment, Decision, HeadParams, Item)


def _prior_label(priors: dict, topic: str) -> str:
    return priors.get("by_topic", {}).get(topic, priors.get("global", "good"))


def _triggers(support: dict, head: HeadParams) -> list[str]:
    disabled = set(head.disabled)
    # bool is an int subclass, but is not an assessor support level.
    return sorted(rid for rid, value in support.items()
                  if rid not in disabled and type(value) is int and value >= head.tau)


def decide(item: Item, assessment: Optional[Assessment], head: HeadParams, priors: dict) -> Decision:
    """head.kind == prior -> prior label (SRC_PRIOR_MODE).
    No assessment / failed -> prior label (SRC_FALLBACK).
    Any ACTIVE rule (not in head.disabled) with int support >= head.tau -> "bad"
      (SRC_MODEL if status ok, SRC_PARTIAL if partial).
    status ok and no trigger -> "good" (also when every rule is "na").
    status partial and no trigger -> prior label (SRC_FALLBACK)."""
    prior = _prior_label(priors, item.topic)
    if head.kind == "prior":
        return Decision(item.item_id, prior, SRC_PRIOR_MODE)
    if assessment is None or assessment.status not in ("ok", "partial"):
        return Decision(item.item_id, prior, SRC_FALLBACK)
    triggered = _triggers(assessment.support, head)
    if triggered:
        source = SRC_MODEL if assessment.status == "ok" else SRC_PARTIAL
        return Decision(item.item_id, "bad", source, triggered)
    if assessment.status == "ok":
        return Decision(item.item_id, "good", SRC_MODEL)
    return Decision(item.item_id, prior, SRC_FALLBACK)


def topic_macro_accuracy(items: list[Item], predicted: dict[str, str]) -> dict:
    """{"macro": mean of per-topic accuracies over topics present, "by_topic": {t: {"n", "correct",
    "accuracy"}}, "n": int, "all_topics_present": bool}. Same metric as src/evaluate.py.

    Unlike the final organizer scorer, fitting may contain fewer than four topics
    or no items. An empty fit has macro 0.0; it cannot justify an improvement.
    """
    ids = {item.item_id for item in items}
    if len(ids) != len(items) or set(predicted) != ids:
        raise ValueError("Prediction IDs must match unique item IDs exactly")
    counts = {topic: {"n": 0, "correct": 0} for topic in TOPICS}
    for item in items:
        if item.topic not in counts:
            raise ValueError("Unknown or missing topic")
        guess = predicted[item.item_id]
        if item.label not in LABELS or guess not in LABELS:
            raise ValueError("Labels must be exactly good or bad")
        group = counts[item.topic]
        group["n"] += 1
        group["correct"] += int(item.label == guess)
    by_topic = {topic: {**group, "accuracy": group["correct"] / group["n"]}
                for topic, group in counts.items() if group["n"]}
    macro = sum(group["accuracy"] for group in by_topic.values()) / len(by_topic) if by_topic else 0.0
    return {"macro": macro, "by_topic": by_topic, "n": len(items),
            "all_topics_present": len(by_topic) == len(TOPICS)}


def _accuracy(items: list[Item], assessments: dict[str, Assessment], head: HeadParams,
              priors: dict) -> float:
    predictions = {item.item_id: decide(item, assessments.get(item.item_id), head, priors).label
                   for item in items}
    return topic_macro_accuracy(items, predictions)["macro"]


def _head_report(head: HeadParams) -> dict:
    return {"kind": head.kind, "tau": head.tau, "disabled": list(head.disabled), "fit_on": head.fit_on}


def fit_fitted_head(items: list[Item], assessments: dict[str, Assessment], priors: dict,
                    fit_on: str = "F") -> HeadParams:
    """For tau in (1, 2, 3): greedy disabling. Candidate = active rule triggering on >= 3 items
    with precision (share labelled bad among them) < 0.5; disable the candidate giving the best
    topic-macro accuracy if accuracy does not decrease; recompute; at most 3 disabled; ties ->
    lower precision, then rule_id. Choose tau by accuracy; ties -> 2, then the lower tau."""
    fitted = []
    for tau in (1, 2, 3):
        head = HeadParams("fitted", tau, fit_on=fit_on)
        current_score = _accuracy(items, assessments, head, priors)
        for _ in range(3):
            options = []
            for rid, stats in rule_stats(items, assessments, head).items():
                if stats["disabled"] or stats["triggered"] < 3 or stats["precision"] >= 0.5:
                    continue
                proposed = HeadParams("fitted", tau, head.disabled + [rid], fit_on)
                score = _accuracy(items, assessments, proposed, priors)
                if score >= current_score:
                    options.append((score, stats["precision"], rid, proposed))
            if not options:
                break
            best = min(options, key=lambda option: (-option[0], option[1], option[2]))
            current_score, head = best[0], best[3]
        fitted.append((current_score, head))
    # The fixed threshold 2 is the first tie-break; then prefer lower tau.
    return min(fitted, key=lambda pair: (-pair[0], pair[1].tau != 2, pair[1].tau))[1]


def select_head(candidates: list[HeadParams], s_items: list[Item], assessments: dict[str, Assessment],
                priors: dict) -> tuple[HeadParams, dict]:
    """candidates[0] is the incumbent (fixed). Pick the best topic-macro accuracy on S; the
    incumbent wins ties. Returns (head, report with every candidate's S score).
    Empty S chooses the last candidate (fitted), with reason "no_S" and null scores.
    """
    if not candidates:
        raise ValueError("At least one head candidate is required")
    reports = [{**_head_report(head),
                "S_macro": _accuracy(s_items, assessments, head, priors) if s_items else None}
               for head in candidates]
    index = max(range(len(candidates)), key=lambda n: reports[n]["S_macro"]) if s_items else len(candidates) - 1
    chosen = candidates[index]
    return chosen, {"candidates": reports, "chosen": _head_report(chosen),
                    "reason": "best_S" if s_items else "no_S"}


def apply_floor(head: HeadParams, f_items: list[Item], s_items: list[Item],
                assessments: dict[str, Assessment], priors: dict) -> tuple[HeadParams, dict]:
    """If the prior beats `head` on BOTH F and S (strictly), return HeadParams(kind="prior").
    With empty S, use only the strict F comparison.
    """
    prior = HeadParams("prior")
    f_head = _accuracy(f_items, assessments, head, priors)
    f_prior = _accuracy(f_items, assessments, prior, priors)
    s_head = _accuracy(s_items, assessments, head, priors) if s_items else None
    s_prior = _accuracy(s_items, assessments, prior, priors) if s_items else None
    applied = f_prior > f_head and (not s_items or s_prior > s_head)
    report = {"applied": applied, "F_head": f_head, "F_prior": f_prior,
              "S_head": s_head, "S_prior": s_prior,
              "reason": "prior_strictly_better" if applied else "prior_not_strictly_better",
              "comparison": "F_and_S" if s_items else "F_only_no_S"}
    return (prior if applied else head), report


def refit_final(head: HeadParams, all_items: list[Item], assessments: dict[str, Assessment],
                priors: dict) -> HeadParams:
    """fitted -> fit_fitted_head(all_items, fit_on="F+S"); fixed/prior -> unchanged."""
    if head.kind == "fitted":
        return fit_fitted_head(all_items, assessments, priors, fit_on="F+S")
    return head


def rule_stats(items: list[Item], assessments: dict[str, Assessment], head: HeadParams) -> dict:
    """{rule_id: {"applicable", "triggered", "triggered_bad", "triggered_good", "sole_triggered",
    "precision", "disabled": bool}} over labelled items.

    Only ok/partial assessments contribute. Threshold counts include disabled rules
    for noise diagnostics; sole_triggered counts only active, sole vetoes. Precision
    is None when a rule has no threshold triggers (rather than inventing precision).
    """
    stats = {}
    disabled = set(head.disabled)
    for item in items:
        assessment = assessments.get(item.item_id)
        if item.label not in LABELS or assessment is None or assessment.status not in ("ok", "partial"):
            continue
        active_triggers = _triggers(assessment.support, head)
        for rid in sorted(set(assessment.support) | set(assessment.requested_rule_ids)):
            group = stats.setdefault(rid, {"applicable": 0, "triggered": 0, "triggered_bad": 0,
                                           "triggered_good": 0, "sole_triggered": 0,
                                           "precision": None, "disabled": rid in disabled})
            value = assessment.support.get(rid)
            if type(value) is not int:
                continue
            group["applicable"] += 1
            if value >= head.tau:
                group["triggered"] += 1
                group["triggered_bad" if item.label == "bad" else "triggered_good"] += 1
                if active_triggers == [rid]:
                    group["sole_triggered"] += 1
    for group in stats.values():
        if group["triggered"]:
            group["precision"] = group["triggered_bad"] / group["triggered"]
    return {rid: stats[rid] for rid in sorted(stats)}
