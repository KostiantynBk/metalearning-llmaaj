"""WP6 — automatically generated jury explanation (Markdown) from the run's real state."""
from __future__ import annotations


def select_examples(items, assessments, head, rulebook, per_rule: int = 2) -> dict:
    """From the assessor's actual outputs on training items x gold labels:
    {rule_id: {"supporting": [...], "false_alarm": [...], "boundary_pass": [...]},
     "missed": [...], "no_rule_applied": [...]} with item_id, label, excerpt, evidence."""
    raise NotImplementedError


def render_explanation(report: dict) -> str:
    """Markdown explanation; `report` keys are defined in IMPLEMENTATION_PLAN.md (WP6)."""
    raise NotImplementedError


def render_prior_only_explanation(run_info: dict, priors: dict, reason: str) -> str:
    """Explanation for a prior-only bundle: states plainly that no rules were learned and why."""
    raise NotImplementedError
