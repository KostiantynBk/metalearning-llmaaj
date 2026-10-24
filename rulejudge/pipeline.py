"""WP1 — priors-first adapt/eval orchestration with atomic, recoverable outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import assessor, decision, explain, gemma_client, induction, split
from .artifacts import ArtifactStore
from .contracts import (SRC_FALLBACK, SRC_PRIOR_MODE, HeadParams, LLMClient,
                        Settings, canonical_json, item_text_for_hash, stable_hash, to_jsonable)
from .io_contract import load_items, write_predictions_atomic
from .priors import fit_priors, prior_label


def _strict() -> bool:
    return os.environ.get("RJ_STRICT", "").strip() == "1"


def _banner(reason: str) -> None:
    print(f"DEGRADED: {reason}", file=sys.stderr, flush=True)


def _json_text(obj) -> str:
    return canonical_json(obj) + "\n"


def _file_hash(obj) -> str:
    return hashlib.sha256(_json_text(obj).encode("utf-8")).hexdigest()


def _set_paths(settings: Settings, default_cache: str, usage_log: str) -> None:
    settings.cache_dir = os.environ.get("RJ_CACHE_DIR", "").strip() or default_cache
    settings.usage_log = usage_log


def _metric(items, assessments, head, priors) -> dict:
    predicted = {item.item_id: decision.decide(item, assessments.get(item.item_id), head, priors).label
                 for item in items}
    return decision.topic_macro_accuracy(items, predicted)


def _prior_explanation(run_info: dict, priors: dict, reason: str) -> str:
    try:
        return explain.render_prior_only_explanation(run_info, priors, reason)
    except Exception:
        # Even an explanation bug must not prevent the first valid prediction state.
        return ("# Prior-only judge\nNo rules were learned in this committed bundle.\n"
                f"Reason: {reason}\nRun: {run_info['run_id']}\n"
                f"Topic priors: {canonical_json(priors['by_topic'])}\n")


def main(argv: list[str]) -> int:
    """Dispatch adapt/eval with environment settings; argparse usage errors return 2."""
    parser = argparse.ArgumentParser(prog="rulejudge", description=__doc__)
    parser.add_argument("command", choices=("adapt", "eval"))
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        settings = Settings.from_env()
    except (ValueError, TypeError) as error:
        if _strict():
            raise
        print(f"Invalid settings: {error}", file=sys.stderr)
        return 2
    return run_adapt(settings) if args.command == "adapt" else run_eval(settings)


def run_adapt(settings: Settings, client: Optional[LLMClient] = None) -> int:
    """Commit usable priors before any model call; keep them on later failures."""
    settings.start_clock()
    try:
        items = load_items(str(Path(settings.input_dir) / "train.jsonl"), require_labels=True)
        fingerprint = stable_hash([item_text_for_hash(item) + item.label for item in items])
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + fingerprint[:6]
        priors = fit_priors(items)
        run_info = {"run_id": run_id, "created_utc": now.isoformat(),
                    "train_fingerprint": fingerprint, "n_items": len(items), "counts": priors["counts"],
                    "models": {"assessor": settings.model_assessor, "proposer": settings.model_proposer},
                    "language_note": "Original multilingual conversations; no translation or test labels used.",
                    "settings": {"s_fraction": settings.s_fraction, "max_content_attempts": settings.max_content_attempts}}
        meta = {"run_id": run_id, "created_utc": run_info["created_utc"], "train_fingerprint": fingerprint}
        store = ArtifactStore(settings.artifacts_dir)
        reason = "Priors-first checkpoint: learning has not completed."
        prior_runtime = {"rulebook.json": {"rules": [], "notes": []},
                         "head.json": to_jsonable(HeadParams("prior")), "priors.json": priors,
                         "assessor_config.json": {"template": None}}
        bid = store.write_bundle("prior_only", prior_runtime,
                                 {"explanation.md": _prior_explanation(run_info, priors, reason)}, meta)
        store.commit(bid)
    except Exception as error:
        if _strict():
            raise
        print(f"Adapt input/artifact error: {error}", file=sys.stderr)
        return 2

    try:
        # Set these before preflight as well, so an exact-cache Docker replay needs no network.
        _set_paths(settings, str(Path(settings.artifacts_dir) / "cache"),
                   str(Path(settings.artifacts_dir) / "usage.jsonl"))
        client = client or gemma_client.make_client(settings)
        ok, detail = gemma_client.preflight(client)
        if not ok:
            _banner(f"preflight failed; prior-only bundle retained: {detail}")
            return 0
        f_ids, s_ids = split.split_fs(items, settings.s_fraction)
        by_id = {item.item_id: item for item in items}
        f_items, s_items = [by_id[iid] for iid in f_ids], [by_id[iid] for iid in s_ids]
        rulebook, induction_log = induction.induce_rulebook(f_items, client, settings)
        if not rulebook.rules:
            _banner(f"no rules induced; continuing with the fixed general check {induction.GENERAL_RULE_ID} only")
        rulebook = induction.add_general_rule(rulebook)
        assessments = assessor.assess_many(items, rulebook, client, settings)
        fixed = HeadParams("fixed", 2)
        fitted = decision.fit_fitted_head(f_items, assessments, priors, fit_on="F")
        selected, selection_report = decision.select_head([fixed, fitted], s_items, assessments, priors)
        head, floor = decision.apply_floor(selected, f_items, s_items, assessments, priors)
        final = decision.refit_final(head, items, assessments, priors)
        selection = {"candidates": [dict(to_jsonable(candidate),
                         F_macro=_metric(f_items, assessments, candidate, priors).get("macro"),
                         S_macro=_metric(s_items, assessments, candidate, priors).get("macro"))
                                      for candidate in (fixed, fitted)],
                     "chosen": to_jsonable(head), "selected_before_floor": to_jsonable(selected),
                     "reason": selection_report.get("reason"),
                     "selection_report": selection_report, "floor": floor,
                     "f_size": len(f_items), "s_size": len(s_items)}
        runtime_rulebook = to_jsonable(rulebook)
        for rule in runtime_rulebook["rules"]:
            rule.pop("provenance_ids", None)
        runtime = {"rulebook.json": runtime_rulebook, "head.json": to_jsonable(final),
                   "priors.json": priors, "assessor_config.json": {
                       "template": assessor.ASSESSOR_TEMPLATE_VERSION, "model": settings.model_assessor}}
        report = {"run_info": run_info, "rulebook": rulebook, "head": final, "selection": selection,
                  "rule_stats": decision.rule_stats(items, assessments, final),
                  "examples": explain.select_examples(items, assessments, final, rulebook),
                  "metrics": {"F": _metric(f_items, assessments, head, priors),
                              "S": _metric(s_items, assessments, head, priors),
                              "S_prior": _metric(s_items, assessments, HeadParams("prior"), priors),
                              "all_final": _metric(items, assessments, final, priors)},
                  "fallbacks": {"failed": sum(assessments.get(i.item_id) is None or
                                               assessments[i.item_id].status == "failed" for i in items),
                                "partial": sum(a.status == "partial" for a in assessments.values())},
                  "induction_notes": rulebook.notes,
                  "hashes": {"rulebook.json": _file_hash(runtime["rulebook.json"]),
                             "head.json": _file_hash(runtime["head.json"]),
                             "assessor_template": stable_hash(assessor.ASSESSOR_TEMPLATE_VERSION)},
                  "items_by_id": by_id, "assessments": assessments}
        markdown = explain.render_explanation(report)
        jury = {"explanation.md": markdown,
                "training_assessments.jsonl": "".join(_json_text(assessments[iid]) for iid in sorted(assessments)),
                "rulebook_with_provenance.json": _json_text(rulebook),
                "selection.json": _json_text(selection), "induction_log.json": _json_text(induction_log)}
        bid = store.write_bundle("learned", runtime, jury, meta)
        store.commit(bid)
        print(f"Adapt committed {run_id}: {len(items)} items, {len(rulebook.rules)} rules, "
              f"head={final.kind}, tau={final.tau}, failures={report['fallbacks']['failed']}",
              file=sys.stderr, flush=True)
        return 0
    except Exception:
        if _strict():
            raise
        _banner("adapt failed after prior checkpoint; last committed bundle retained")
        traceback.print_exc(file=sys.stderr)
        return 0


def run_eval(settings: Settings, client: Optional[LLMClient] = None) -> int:
    """Write complete priors first, then atomically checkpoint model decisions."""
    settings.start_clock()
    # A reused Settings object must not report counts from an earlier eval.
    settings.eval_source_counts = {}
    try:
        rt = ArtifactStore(settings.artifacts_dir).load_runtime()
        items = load_items(str(Path(settings.input_dir) / "test.jsonl"), require_labels=False)
        ids = [item.item_id for item in items]
        path = str(Path(settings.output_dir) / "predictions.csv")
        priors, head = rt["priors"], rt["head"]
        labels = {item.item_id: prior_label(priors, item.topic) for item in items}
        write_predictions_atomic(path, labels, ids)
        # Exact sources from the decide() contract: prior head versus no assessment.
        prior_source = SRC_PRIOR_MODE if head.kind == "prior" else SRC_FALLBACK
        settings.eval_source_counts = {prior_source: len(items)}
    except Exception as error:
        if _strict():
            raise
        print(f"Eval input/artifact error: {error}", file=sys.stderr)
        return 2
    if rt["mode"] == "prior_only" or head.kind == "prior":
        _banner("prior-only evaluation; complete predictions.csv written")
        return 0

    try:
        config = rt["assessor_config"]
        if config.get("template") != assessor.ASSESSOR_TEMPLATE_VERSION:
            raise ValueError("Assessor template differs from the committed training template")
        if config.get("model"):
            settings.model_assessor = config["model"]
        tmp = tempfile.mkdtemp(prefix="rulejudge-eval-")
        _set_paths(settings, tmp, str(Path(tmp) / "usage.jsonl"))
        client = client or gemma_client.make_client(settings)
        ok, detail = gemma_client.preflight(client)
        if not ok:
            _banner(f"preflight failed; complete prior predictions retained: {detail}")
            return 0

        def checkpoint(finished: dict) -> None:
            current, checkpoint_sources = {}, Counter()
            for item in items:
                if item.item_id in finished:
                    result = decision.decide(item, finished[item.item_id], head, priors)
                    current[item.item_id] = result.label
                    checkpoint_sources[result.source] += 1
                else:
                    current[item.item_id] = prior_label(priors, item.topic)
                    checkpoint_sources[SRC_FALLBACK] += 1
            write_predictions_atomic(path, current, ids)
            # Update only after replacement succeeds: counts describe the on-disk CSV.
            settings.eval_source_counts = dict(checkpoint_sources)

        assessments = assessor.assess_many(items, rt["rulebook"], client, settings, on_progress=checkpoint)
        checkpoint(assessments)
        sources = Counter(decision.decide(item, assessments.get(item.item_id), head, priors).source for item in items)
        settings.eval_source_counts = dict(sources)
        print(f"Eval wrote {len(items)} predictions: {json.dumps(dict(sorted(sources.items())), sort_keys=True)}",
              file=sys.stderr, flush=True)
        return 0
    except Exception:
        if _strict():
            raise
        _banner("eval interrupted; last complete predictions checkpoint retained")
        traceback.print_exc(file=sys.stderr)
        return 0
