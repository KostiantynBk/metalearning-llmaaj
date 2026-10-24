"""WP6 — deterministic jury explanations grounded in real training assessments."""
from __future__ import annotations

from .contracts import HEAD_PRIOR, ST_FAILED, ST_OK, ST_PARTIAL, to_jsonable
from .render import excerpt, render_item, resolve_pointer


def _text(value) -> str:
    """Keep untrusted training/model text on one Markdown-safe line."""
    text = " ".join(str(value).split())
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for char in ("\\", "`", "*", "_", "[", "]", "|", "#", "!"):
        text = text.replace(char, "\\" + char)
    return text


def _is_support(value) -> bool:
    return type(value) is int and 0 <= value <= 3


def _triggers(assessment, head) -> list:
    if (head.kind == HEAD_PRIOR or assessment is None
            or assessment.status not in (ST_OK, ST_PARTIAL)):
        return []
    return sorted(rid for rid, value in assessment.support.items()
                  if rid not in head.disabled and _is_support(value) and value >= head.tau)


def _valid_evidence(item, assessment, rule_id=None) -> list:
    if assessment is None or assessment.status not in (ST_OK, ST_PARTIAL):
        return []
    result = []
    for evidence in assessment.evidence:
        data = to_jsonable(evidence)
        rid = data.get("rule_id")
        support = assessment.support.get(rid)
        if rule_id is not None and rid != rule_id:
            continue
        if not _is_support(support) or support < 1 or data.get("support") != support:
            continue
        result.append({"rule_id": rid, "support": support,
                       "where": [p for p in data.get("where", []) if resolve_pointer(item, p)],
                       "why": excerpt(data.get("why", ""), 120)})
    return result


def _record(item, assessment, head, rule_id=None) -> dict:
    triggered = _triggers(assessment, head)
    return {"item_id": item.item_id, "label": item.label, "topic": item.topic,
            "rule_id": rule_id,
            "support": (assessment.support.get(rule_id) if rule_id else dict(assessment.support)),
            "triggered": rule_id in triggered if rule_id else bool(triggered),
            "triggered_rules": triggered, "assessment_status": assessment.status,
            "excerpt": excerpt(render_item(item), 600),
            "evidence": _valid_evidence(item, assessment, rule_id)}


def _category(item, assessment, head, rule_id) -> str:
    if (head.kind == HEAD_PRIOR or rule_id in head.disabled or assessment is None
            or assessment.status not in (ST_OK, ST_PARTIAL)):
        return ""
    value = assessment.support.get(rule_id)
    if not _is_support(value):
        return ""
    if value >= head.tau:
        return "supporting" if item.label == "bad" else "false_alarm" if item.label == "good" else ""
    return "boundary_pass" if item.label == "good" else ""


def select_examples(items, assessments, head, rulebook, per_rule: int = 2) -> dict:
    """Select actual training outputs, never fabricated or induction-only examples.

    Per-rule categories are capped at ``per_rule``; global categories retain all
    members so their counts remain exact. A boundary is per-rule, not necessarily
    a correct whole-pipeline prediction (another active rule may veto).
    """
    limit = max(0, int(per_rule))
    result = {rule.rule_id: {"supporting": [], "false_alarm": [], "boundary_pass": []}
              for rule in rulebook.rules}
    result.update({"missed": [], "no_rule_applied": []})
    for item in sorted(items, key=lambda i: i.item_id):
        if item.label not in ("good", "bad"):
            continue
        assessment = assessments.get(item.item_id)
        if assessment is None:
            continue
        for rule in rulebook.rules:
            if rule.scope not in ("global", item.topic):
                continue
            category = _category(item, assessment, head, rule.rule_id)
            if category:
                result[rule.rule_id][category].append(_record(item, assessment, head, rule.rule_id))
        if item.label == "bad" and not _triggers(assessment, head):
            result["missed"].append(_record(item, assessment, head))
        if assessment.status == ST_OK and all(v == "na" for v in assessment.support.values()):
            result["no_rule_applied"].append(_record(item, assessment, head))
    for rule in rulebook.rules:
        categories = result[rule.rule_id]
        for category in ("supporting", "false_alarm"):
            categories[category].sort(key=lambda row: (-row["support"], row["item_id"]))
        categories["boundary_pass"].sort(
            key=lambda row: (row["support"] != 1, row["support"], row["item_id"]))
        for category in categories:
            categories[category] = categories[category][:limit]
    return result


def _consistency_issues(report: dict) -> list[str]:
    """Audit selected rows against source labels, supports, evidence and category.

    Rendering also re-reads source items/assessments rather than trusting the
    excerpts or labels in selected rows. Reject inconsistent selection metadata.
    """
    issues = []
    items = report.get("items_by_id", {})
    assessments = report.get("assessments", {})
    head, rulebook = report["head"], report["rulebook"]
    examples = report.get("examples", {})
    rules = {r.rule_id: r for r in rulebook.rules}
    groups = []
    for rid, categories in examples.items():
        if rid in ("missed", "no_rule_applied"):
            groups.append((None, rid, categories))
        elif rid not in rules:
            issues.append(f"Unknown example rule: {rid}")
        else:
            groups.extend((rid, category, rows) for category, rows in categories.items())
    for rid, category, rows in groups:
        for row in rows:
            iid = row.get("item_id")
            item, assessment = items.get(iid), assessments.get(iid)
            if item is None or assessment is None:
                issues.append(f"{category}: {iid} has no source item/assessment")
                continue
            expected = _record(item, assessment, head, rid)
            for key in expected:
                if key in row and row[key] != expected[key]:
                    issues.append(f"{category}: {iid} disagrees on {key}")
            if item.label not in ("good", "bad"):
                issues.append(f"{category}: {iid} has no provided training label")
            if rid:
                valid = (rules[rid].scope in ("global", item.topic)
                         and _category(item, assessment, head, rid) == category)
            elif category == "missed":
                valid = item.label == "bad" and not _triggers(assessment, head)
            else:
                valid = (assessment.status == ST_OK
                         and all(v == "na" for v in assessment.support.values()))
            if not valid:
                issues.append(f"{category}: {iid} does not satisfy category for {rid or 'all rules'}")
    return issues


def _metric(value) -> str:
    if isinstance(value, dict):
        value = value.get("macro")
    return f"{value:.4f}" if isinstance(value, (int, float)) else "not recorded"


def _precision(stat) -> str:
    if "precision" not in stat:
        return "not recorded"
    return "undefined (no triggers)" if stat["precision"] is None else _metric(stat["precision"])


def _prior(report, topic):
    """Recover the exact majority label from reported training counts, if available."""
    priors = report.get("priors", {})
    if topic in priors.get("by_topic", {}):
        return priors["by_topic"][topic]
    counts = report.get("run_info", {}).get("counts", {})
    if not counts:
        return priors.get("global")
    counts = counts.get(topic) or {label: sum(c.get(label, 0) for c in counts.values())
                                 for label in ("good", "bad")}
    return "bad" if counts.get("bad", 0) > counts.get("good", 0) else "good"


def _pipeline_output(report, item, assessment) -> str:
    head = report["head"]
    triggered = _triggers(assessment, head)
    if head.kind == HEAD_PRIOR:
        return f"{_prior(report, item.topic) or 'label not recorded'} — prior-only head"
    if assessment.status in (ST_OK, ST_PARTIAL) and triggered:
        return "bad — active veto: " + ", ".join(triggered)
    if assessment.status == ST_OK:
        return "good — no active rule reaches the threshold"
    return f"{_prior(report, item.topic) or 'label not recorded'} — topic-majority fallback"


def _show_example(report, row, rule_id=None) -> list[str]:
    item = report["items_by_id"][row["item_id"]]
    assessment = report["assessments"][item.item_id]
    lines = [f"**Training item {_text(item.item_id)}** (topic: {_text(item.topic)}).",
             f"Expert label: **{item.label.upper()} (provided)**."]
    if rule_id:
        lines.append(f"Inferred rule: **{_text(rule_id)}** (model-inferred from labels, not stated by the expert).")
    for role in ("user", "assistant"):
        messages = [str(m.get("content", "")) for m in item.conversation if m.get("role") == role]
        lines.append(f"> {role.title()}: {_text(excerpt(' / '.join(messages), 300)) if messages else '(no message of this role)'}")
        lines.append("")
    lines.append("Pipeline output: " + _text(_pipeline_output(report, item, assessment)) + ".")
    if rule_id:
        value = assessment.support.get(rule_id)
        triggered = rule_id in _triggers(assessment, report["head"])
        lines.append(f"{_text(rule_id)} support: {_text(value)}; active trigger: {'yes' if triggered else 'no'}.")
    evidence = _valid_evidence(item, assessment, rule_id)
    for entry in evidence:
        where = ", ".join(entry["where"]) or "no valid source pointer recorded"
        lines.append(f"Assessor evidence ({_text(entry['rule_id'])}, support {entry['support']}): "
                     f"“{_text(entry['why'])}” — {_text(where)}.")
    if not evidence:
        lines.append("No validated evidence rationale recorded; the support value above is the actual assessor output."
                     if rule_id else "No validated evidence rationale recorded.")
    lines.append("")
    return lines


def render_explanation(report: dict) -> str:
    """Render the five jury criteria without generating new rules or examples."""
    report = dict(report)
    head, rulebook = report["head"], report["rulebook"]
    items = report.get("items_by_id", {})
    assessments = report.get("assessments", {})
    if "examples" not in report:
        report["examples"] = select_examples(items.values(), assessments, head, rulebook)
    issues = _consistency_issues(report)
    if issues:
        raise ValueError("Inconsistent training evidence: " + "; ".join(issues))
    examples = report["examples"]
    # Uncapped diagnostics: selected rows alone must never determine totals.
    diagnostics = select_examples(items.values(), assessments, head, rulebook, per_rule=len(items))
    run = report.get("run_info", {})
    selection, metrics = report.get("selection", {}), report.get("metrics", {})
    lines = ["# Learned expert judge — training explanation", "",
             f"Run: {_text(run.get('run_id', 'not recorded'))}; created UTC: {_text(run.get('created_utc', 'not recorded'))}.",
             f"Training items: {_text(run.get('n_items', len(items)))}; fingerprint: {_text(run.get('train_fingerprint', 'not recorded'))}.",
             f"Language note: {_text(run.get('language_note', 'Conversations are judged by meaning, not language.'))}", "",
             "## 1. Decision rule and priorities", "",
             f"Final head: **{_text(head.kind)}**; threshold **τ = {head.tau}**; fitted on: {_text(head.fit_on or 'not fitted')}.",
             "Disabled rules: " + (", ".join(_text(r) for r in head.disabled) or "none") + ".", ""]
    if head.kind == HEAD_PRIOR:
        lines.append("**Prior-only mode:** this head bypasses all learned rules and returns the training topic majority. No rule veto is used.")
    lines += ["For a rule-based head, **bad iff any ACTIVE rule has integer support ≥ τ** on a valid or partial assessment. "
              "No rule compensates for another; disabled rules never veto. Support is 0 (complies), 1 (weak), 2 (supported), "
              "or 3 (clear violation); `na` means not applicable.",
              "A complete assessment with no trigger returns good, including when no rule applies. A partial assessment with a trigger "
              "returns bad; a partial assessment without a trigger, a failed assessment, or a missing assessment uses the training "
              "topic majority (global majority for an unseen topic; ties go to good).", "",
              "### Head selection (before full-training refit)", "",
              "| Candidate | τ | Disabled | F macro | S macro |", "|---|---:|---|---:|---:|"]
    for candidate in selection.get("candidates", []):
        c = to_jsonable(candidate)
        lines.append(f"| {_text(c.get('kind', 'not recorded'))} | {_text(c.get('tau', '—'))} | "
                     f"{_text(', '.join(c.get('disabled', [])) or 'none')} | {_metric(c.get('F_macro'))} | {_metric(c.get('S_macro'))} |")
    lines += [f"Chosen at selection: {_text(to_jsonable(selection.get('chosen', 'not recorded')))}. "
              f"F size: {_text(selection.get('f_size', 'not recorded'))}; S size: {_text(selection.get('s_size', 'not recorded'))}.",
              "The incumbent wins selection ties. If S is empty, use the fitted candidate. The majority-prior floor switches to "
              "prior-only only when the prior is strictly better on both F and S (F only when S is empty). "
              "A selected fitted head is then refitted on all training items; its final threshold/disabled list above can differ.",
              f"Selection reason: {_text(selection.get('reason', 'not recorded'))}. "
              f"Recorded floor outcome: {_text(to_jsonable(selection.get('floor', 'not recorded')))}.", "",
              "Fitted-head search tests τ ∈ {1, 2, 3}. It may disable at most three rules, each with at least three "
              "training triggers and precision below 0.5, only when disabling does not reduce fitting accuracy. "
              "Threshold ties prefer τ = 2, then the lower threshold.", "",
              "## 2. Rules learned from this training set", "",
              "Rules are hypotheses inferred from labels, not expert-authored instructions. List order conveys induction priority; "
              "at decision time every active rule has the same non-compensating veto.", "",
              *([f"Exception: {', '.join(_text(r.rule_id) for r in rulebook.rules if r.origin == 'fixed_general_check')} "
                 "is a fixed general-quality check (correct, consistent with the context, answers the request) added "
                 "to every rulebook by design, not learned from labels. Its training triggers and precision below are "
                 "measured exactly like the learned rules, and the same disabling rule applies to it.", ""]
                if any(r.origin == "fixed_general_check" for r in rulebook.rules) else []),
              "| ID | Requirement | Applies when | Scope | Status | Triggered on training (bad/good) | Precision |",
              "|---|---|---|---|---|---:|---:|"]
    stats = report.get("rule_stats", {})
    for rule in rulebook.rules:
        stat = stats.get(rule.rule_id, {})
        lines.append(f"| {_text(rule.rule_id)} | {_text(rule.text)} | {_text(rule.applies_when or 'Always within scope')} | "
                     f"{_text(rule.scope)} | {'disabled' if rule.rule_id in head.disabled else 'active'} | "
                     f"{_text(stat.get('triggered_bad', 'not recorded'))}/{_text(stat.get('triggered_good', 'not recorded'))} | {_precision(stat)} |")
    if not rulebook.rules:
        lines.append("No learned rules in this bundle.")
    for rule in rulebook.rules:
        lines += [f"- **{_text(rule.rule_id)}** — Violation: {_text(rule.violation or 'As stated in the requirement')}. "
                  f"Not a violation: {_text(rule.boundary or 'No explicit boundary inferred')}."]
    lines += ["", "Precision is labelled-bad triggers / all labelled triggers; no triggers means precision is undefined. "
              "Counts use labelled ok/partial assessments. Disabled-rule counts show hypothetical threshold crossings "
              "for diagnosis only, not active vetoes. These are training diagnostics, not held-out performance estimates.",
              "", "## 3. Training evidence", "",
              "Examples below come only from this run's real training assessments × provided labels. "
              "A boundary pass is for the named rule only; another rule can still veto the conversation.", ""]
    for rule in rulebook.rules:
        if rule.rule_id in head.disabled or head.kind == HEAD_PRIOR:
            continue
        categories = examples.get(rule.rule_id, {})
        lines += [f"### {_text(rule.rule_id)} — {_text(rule.text)}", ""]
        supporting = categories.get("supporting", [])[:2]
        if not supporting:
            lines += ["no supporting training example found", ""]
        for row in supporting:
            lines += _show_example(report, row, rule.rule_id)
        boundary = categories.get("boundary_pass", [])[:1]
        false_alarm = categories.get("false_alarm", [])[:1]
        if boundary or false_alarm:
            lines += ["**Boundary pass**" if boundary else "**False alarm (expert accepted it)**", ""]
            lines += _show_example(report, (boundary or false_alarm)[0], rule.rule_id)
        else:
            lines += ["no boundary or false-alarm training example found", ""]
    if head.kind == HEAD_PRIOR:
        lines += ["No rule-trigger examples are presented as decision evidence: the final head is prior-only.", ""]
    lines += ["## 4. Boundaries and limitations", "", "### Disabled rules", ""]
    if not head.disabled:
        lines.append("None.")
    for rid in head.disabled:
        stat = stats.get(rid, {})
        lines.append(f"- {_text(rid)}: disabled; recorded triggers bad/good = "
                     f"{_text(stat.get('triggered_bad', 'not recorded'))}/{_text(stat.get('triggered_good', 'not recorded'))}; "
                     f"precision {_precision(stat)}. It cannot veto.")
    lines += ["", "### Bad training items with no active trigger", "",
              "These are rule-coverage misses, not necessarily wrong final predictions: a fallback prior can still return bad."]
    missed = diagnostics["missed"]
    topics = sorted({item.topic for item in items.values()})
    for topic in topics:
        lines.append(f"- {_text(topic)}: {sum(row['topic'] == topic for row in missed)}")
    if not missed:
        lines.append("no missed training example found")
    for row in missed[:2]:
        lines += _show_example(report, row)
    false_alarms = {row["item_id"] for rule in rulebook.rules
                    for row in diagnostics[rule.rule_id]["false_alarm"]}
    lines += ["", "### False alarms", "",
              f"False alarms: {len(false_alarms)} distinct expert-good items with an active veto."]
    for iid in sorted(false_alarms)[:2]:
        item = items[iid]
        rid = next(rule.rule_id for rule in rulebook.rules
                   if any(row["item_id"] == iid for row in diagnostics[rule.rule_id]["false_alarm"]))
        lines += _show_example(report, _record(item, assessments[iid], head, rid), rid)
    if not false_alarms:
        lines.append("no false-alarm training example found")
    unassessed = sum(item.label in ("good", "bad") and item.item_id not in assessments
                     for item in items.values())
    lines += ["", f"Training items without an assessment: {unassessed}; not presented as model-output examples.",
              f"No rule applied (complete all-na or empty assessments): {len(diagnostics['no_rule_applied'])} items.",
              f"Failed assessments: {_text(report.get('fallbacks', {}).get('failed', 'not recorded'))}; "
              f"partial assessments: {_text(report.get('fallbacks', {}).get('partial', 'not recorded'))}. "
              "Partial is not itself a fallback count: a partial trigger still returns bad.", "",
              f"Pre-refit S macro: {_metric(metrics.get('S'))}; S majority-prior macro: {_metric(metrics.get('S_prior'))}. "
              "S supported head selection and is not an untouched test set.",
              f"F macro: {_metric(metrics.get('F'))}; all-training final macro: {_metric(metrics.get('all_final'))} "
              "(in-sample diagnostics, not evidence of generalization).", "",
              "Rules are hypotheses inferred from labels. Missing evidence, context-dependent exceptions, multilingual ambiguity, "
              "and unseen private rules can cause errors. No test labels were used to generate these examples.", "",
              "### Induction notes"]
    notes = list(report.get("induction_notes", []))
    if not notes:
        lines.append("No induction notes recorded.")
    for note in notes:
        lines.append("- " + _text(note))
    lines += ["", "## 5. Consistency with the implementation", "",
              f"Models: {_text(to_jsonable(run.get('models', 'not recorded')))}.",
              "Settings: assessor temperature 0; one shared assessor template for adapt/eval; "
              "no rubric revision (K = 0); final head settings are shown above.", "",
              "### Recorded hashes"]
    for name in ("rulebook.json", "head.json", "assessor_template"):
        lines.append(f"- {_text(name)}: {_text(report.get('hashes', {}).get(name, 'not recorded'))}")
    lines += ["", "```text", "if head.kind == 'prior': return topic_majority",
              "if assessment is missing or failed: return topic_majority",
              "triggers = [r for r, s in support if r not in head.disabled",
              "            and type(s) is int and s >= head.tau]",
              "if triggers: return bad",
              "if assessment.status == 'ok': return good",
              "return topic_majority  # partial, with no trigger", "```", "",
              "eval loads exactly runtime/rulebook.json + head.json from this bundle "
              "(the latter is runtime/head.json), together with runtime/priors.json and the assessor configuration. "
              "Training-only provenance, labels, examples and this explanation are not used as eval inputs.", ""]
    return "\n".join(lines)


def render_prior_only_explanation(run_info: dict, priors: dict, reason: str) -> str:
    """Describe degraded/initial mode truthfully, without inventing learned rules."""
    lines = ["# Prior-only judge — no learned rules", "",
             f"Run: {_text(run_info.get('run_id', 'not recorded'))}; created UTC: {_text(run_info.get('created_utc', 'not recorded'))}.",
             f"Training fingerprint: {_text(run_info.get('train_fingerprint', 'not recorded'))}.",
             "No rules were learned in this bundle. Predictions use only the provided training labels' topic majorities.",
             f"Reason: {_text(reason)}", "", "| Topic | Prior label | Good / bad training items |", "|---|---|---:|"]
    for topic, label in sorted(priors.get("by_topic", {}).items()):
        counts = priors.get("counts", {}).get(topic, {})
        lines.append(f"| {_text(topic)} | {_text(label)} | {_text(counts.get('good', 'not recorded'))} / {_text(counts.get('bad', 'not recorded'))} |")
    lines += ["", f"Unseen-topic fallback: {_text(priors.get('global', 'not recorded'))}. Ties go to good.",
              "There are no inferred criteria, rule-backed boundaries, or model-trigger evidence to claim. "
              "This is a majority baseline/degraded output, not a successful learned expert model.", ""]
    return "\n".join(lines)
