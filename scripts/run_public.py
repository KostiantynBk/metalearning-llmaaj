"""Local public comparison. Fake/subset runs are smoke tests, never benchmark results."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
# Support `python scripts/run_public.py` from any working directory.
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import load_dataset, load_labels, write_jsonl
from evaluate import evaluate
from prepare_inputs import prepare
from rulejudge import pipeline
from rulejudge.artifacts import ArtifactStore
from rulejudge.baseline import run_baseline
from rulejudge.contracts import TOPICS, Settings, stable_hash
from rulejudge.fake_client import FakeLLMClient
from rulejudge.io_contract import load_items, write_predictions_atomic
from rulejudge.priors import fit_priors, prior_label


def _load_env_file(path: Path, environ=None) -> None:
    """Tiny non-shell parser; never overwrite exported values or print secrets.

    Only GEMINI_API_KEY and RJ_* settings are accepted. No expansion, execution,
    multiline strings or escape decoding. Used only for real runs, not tests/fakes.
    """
    target = os.environ if environ is None else environ
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r"GEMINI_API_KEY|RJ_[A-Z0-9_]+", key):
            continue
        if value.startswith(("'", '"')):
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                continue
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        target.setdefault(key, value)


def _allocate(sizes: dict, count: int) -> dict:
    """Deterministic proportional quotas, reserving one per group when possible."""
    keys = sorted(sizes)
    if count >= sum(sizes.values()):
        return dict(sizes)
    quotas = {key: int(count >= len(keys)) for key in keys}
    while sum(quotas.values()) < count:
        eligible = [key for key in keys if quotas[key] < sizes[key]]
        # Largest proportional deficit, with stable alphabetical tie-breaking.
        key = min(eligible, key=lambda k: (-(count * sizes[k] / sum(sizes.values()) - quotas[k]), k))
        quotas[key] += 1
    return quotas


def _stratified_subset(rows: list[dict], limit: int | None, *, train: bool) -> list[dict]:
    """Stratify by topic (and training label only); never inspect test gold."""
    if limit is None or limit >= len(rows):
        return list(rows)
    topics = defaultdict(list)
    for row in rows:
        topics[row["topic"]].append(row)
    if limit < len(topics):
        raise ValueError("Subset limit must keep at least one item per topic (normally 4)")
    topic_quotas = _allocate({topic: len(group) for topic, group in topics.items()}, limit)
    selected = []
    for topic in sorted(topics):
        groups = defaultdict(list)
        for row in topics[topic]:
            groups[row["label"] if train else "all"].append(row)
        quotas = _allocate({key: len(group) for key, group in groups.items()}, topic_quotas[topic])
        for key in sorted(groups):
            ordered = sorted(groups[key], key=lambda row: (stable_hash("public-smoke-v1", row["item_id"]),
                                                           row["item_id"]))
            selected.extend(ordered[:quotas[key]])
    return sorted(selected, key=lambda row: row["item_id"])


def _fake_responder(role: str, system: str, user: str) -> str:
    """Synthetic WRONG-DATE marker fixture; no dataset gold is consulted."""
    if role == "baseline":
        return json.dumps({"label": "bad" if "WRONG-DATE" in user else "good"})
    if role == "proposer":
        text = "Dates and amounts must match the context."
        rule = {"text": text, "scope": "global", "applies_when": "Facts are stated.",
                "violation": "A stated fact contradicts the context.",
                "boundary": "A correct context fact is not a violation.",
                "merged_candidates": [f"C{i}" for i in range(1, 49)]}
        pairs = sorted(set(int(n) for n in re.findall(r"PAIR\s+(\d+)", user, re.I))) or [1]
        candidates = [{"pair": n, "text": text, "applies_when": rule["applies_when"],
                       "violation": rule["violation"], "boundary": rule["boundary"],
                       "topic_specific": False, "confidence": "high"} for n in pairs]
        return json.dumps({"candidates": candidates, "no_clear_rule": [], "rules": [rule], "notes": []})
    ids = sorted(set(re.findall(r"\bR\d{2,}\b", user)))
    support = 3 if "WRONG-DATE" in user else 0
    return json.dumps({"ok": True, "s": {rid: support for rid in ids},
                       "e": [{"r": rid, "w": ["msg 2"], "y": "Synthetic contradictory answer"}
                             for rid in ids] if support else []})


class _CountingClient:
    """Transparent thread-safe accounting without retaining prompts or API keys."""
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.phase = "adapt"
        self.counts = {}
        self._lock = threading.Lock()

    def generate(self, **kwargs):
        result = self.wrapped.generate(**kwargs)
        with self._lock:
            phase = self.counts.setdefault(self.phase, {})
            row = phase.setdefault(kwargs["role"], {
                "calls": 0, "failed_calls": 0, "cached_calls": 0,
                "input_tokens": 0, "output_tokens": 0,
                "uncached_input_tokens": 0, "uncached_output_tokens": 0,
            })
            row["calls"] += 1
            row["failed_calls"] += int(not result.ok)
            row["cached_calls"] += int(result.cached)
            row["input_tokens"] += result.input_tokens
            row["output_tokens"] += result.output_tokens
            if not result.cached:
                row["uncached_input_tokens"] += result.input_tokens
                row["uncached_output_tokens"] += result.output_tokens
        return result


def _paired_counts(gold: list[dict], ours: dict, baseline: dict) -> dict:
    counts = {topic: dict.fromkeys(("both_right", "only_ours", "only_baseline", "both_wrong"), 0)
              for topic in TOPICS}
    for row in gold:
        our_ok = ours[row["item_id"]] == row["label"]
        baseline_ok = baseline[row["item_id"]] == row["label"]
        key = ("both_right" if baseline_ok else "only_ours") if our_ok else (
            "only_baseline" if baseline_ok else "both_wrong")
        counts[row["topic"]][key] += 1
    return counts


def _eval_counts(settings: Settings, runtime: dict, n_items: int) -> dict:
    # Optional WP1 diagnostics; never infer a fallback merely from label equality.
    counts = getattr(settings, "eval_source_counts", None)
    if isinstance(counts, dict):
        return {"available": True, "by_source": dict(counts),
                "fallback_prior": counts.get("fallback_prior", 0)}
    if runtime["mode"] == "prior_only" or runtime["head"].kind == "prior":
        return {"available": True, "by_source": {"prior_mode": n_items},
                "fallback_prior": 0, "prior_mode": n_items}
    return {"available": False, "fallback_prior": None,
            "reason": "Pipeline did not expose eval_source_counts; see pipeline stderr."}


def _markdown(report: dict) -> str:
    lines = ["# Public comparison", "", f"**Run type: {report['run_type']}**.", ""]
    if not report["benchmark_eligible"]:
        lines += ["**SMOKE ONLY — not benchmark evidence or reportable performance.**", ""]
    lines += ["Test labels are used only after predictions, for scoring and paired counts.", "",
              "| Method | Macro topic accuracy | " + " | ".join(TOPICS) + " |",
              "|---|---:|" + "---:|" * len(TOPICS)]
    for name, result in report["scores"].items():
        lines.append(f"| {name} | {result['macro_topic_accuracy']:.4f} | " +
                     " | ".join(f"{result['by_topic'][t]['accuracy']:.4f}" for t in TOPICS) + " |")
    lines += ["", "## Paired test outcomes", "",
              "| Topic | Both right | Only ours | Only baseline | Both wrong |",
              "|---|---:|---:|---:|---:|"]
    if report["paired_by_topic"] is None:
        lines += ["Baseline skipped; paired counts unavailable."]
    else:
        for topic, row in report["paired_by_topic"].items():
            lines.append(f"| {topic} | {row['both_right']} | {row['only_ours']} | "
                         f"{row['only_baseline']} | {row['both_wrong']} |")
    lines += ["", "## Fallbacks", "", "```json", json.dumps(report["fallback_counts"], indent=2),
              "```", "", "## Wall time and token usage", "", "```json",
              json.dumps({"wall_time_s": report["wall_time_s"], "token_usage": report["token_usage"]}, indent=2),
              "```", "", "Tokens are returned client usage; uncached totals exclude replay hits. "
              "Fake usage is an estimate, not actual API usage. "
              "Transport-level retries may incur additional tokens not returned by the client.", ""]
    return "\n".join(lines)


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def _run(args) -> dict:
    started = time.perf_counter()
    if not args.fake:
        _load_env_file(ROOT / ".env")
    settings = Settings.from_env()
    settings.fake_llm = args.fake or settings.fake_llm
    if not args.skip_baseline and settings.model_baseline != settings.model_assessor:
        raise ValueError("Same-model comparison requires RJ_MODEL_BASELINE == RJ_MODEL_ASSESSOR")
    out = Path(args.out).resolve()
    # Refuse reuse, even if a partial run left files: never silently mix runs.
    out.mkdir(parents=True, exist_ok=False)
    input_dir, artifacts_dir, output_dir = (out / name for name in ("input", "artifacts", "output"))
    settings.input_dir, settings.artifacts_dir, settings.output_dir = map(
        str, (input_dir, artifacts_dir, output_dir))
    artifacts_dir.mkdir()
    output_dir.mkdir()
    prepare(str(Path(args.train).resolve()), str(Path(args.test).resolve()), str(input_dir), args.language)
    for name, limit, train in (("train", args.limit_train, True), ("test", args.limit_test, False)):
        path = input_dir / f"{name}.jsonl"
        rows = load_dataset(path, require_labels=train)
        subset = _stratified_subset(rows, limit, train=train)
        if {row["topic"] for row in subset} != set(TOPICS):
            raise ValueError("Public comparison requires all four topics in both train and test")
        write_jsonl(path, subset)
    train = load_items(str(input_dir / "train.jsonl"), require_labels=True)
    test = load_items(str(input_dir / "test.jsonl"), require_labels=False)
    priors = fit_priors(train)
    if settings.fake_llm:
        raw_client = FakeLLMClient(_fake_responder)
    else:
        from rulejudge.gemma_client import make_client
        raw_client = make_client(settings)
    client = _CountingClient(raw_client)
    times = {}
    for phase, func in (("adapt", pipeline.run_adapt), ("eval", pipeline.run_eval)):
        client.phase = phase
        settings.start_clock()
        stage_start = time.perf_counter()
        code = func(settings, client=client)
        times[phase] = time.perf_counter() - stage_start
        if code != 0:
            raise ValueError(f"Pipeline {phase} failed with exit code {code}")
    runtime = ArtifactStore(str(artifacts_dir)).load_runtime()
    eval_counts = _eval_counts(settings, runtime, len(test))
    expected = [item.item_id for item in test]
    prior_predictions = {item.item_id: prior_label(priors, item.topic) for item in test}
    prior_path = output_dir / "prior.csv"
    write_predictions_atomic(str(prior_path), prior_predictions, expected)
    baseline_predictions = None
    if not args.skip_baseline:
        client.phase = "baseline"
        settings.start_clock()
        stage_start = time.perf_counter()
        baseline_predictions = run_baseline(test, client, settings, priors)
        times["baseline"] = time.perf_counter() - stage_start
        baseline_path = output_dir / "baseline.csv"
        write_predictions_atomic(str(baseline_path),
                                 {iid: value["label"] for iid, value in baseline_predictions.items()}, expected)
    # Gold is loaded only here, after ALL methods have produced predictions.
    selected_ids = set(expected)
    gold = [row for row in load_dataset(args.test, require_labels=True) if row["item_id"] in selected_ids]
    gold_path = out / "gold_for_scoring.jsonl"
    write_jsonl(gold_path, [{"item_id": row["item_id"], "label": row["label"]} for row in gold])
    scores = {}
    for name, path in (("ours", output_dir / "predictions.csv"), ("majority_prior", prior_path)):
        scores[name] = evaluate(str(input_dir / "test.jsonl"), str(path), str(gold_path))
    paired = None
    if baseline_predictions is not None:
        scores["same_model_baseline"] = evaluate(str(input_dir / "test.jsonl"), str(baseline_path), str(gold_path))
        ours = {row["item_id"]: row["label"] for row in load_labels(output_dir / "predictions.csv")}
        paired = _paired_counts(gold, ours, {iid: value["label"] for iid, value in baseline_predictions.items()})
    baseline_counts = dict(Counter(value["source"] for value in (baseline_predictions or {}).values()))
    smoke = settings.fake_llm or args.limit_train is not None or args.limit_test is not None
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": "fake_smoke" if settings.fake_llm else "subset_smoke" if smoke else "full_public",
        "benchmark_eligible": not smoke,
        "language": args.language,
        "models": {"assessor": settings.model_assessor, "proposer": settings.model_proposer,
                   "baseline": settings.model_baseline if not args.skip_baseline else None},
        "n_train": len(train), "n_test": len(test),
        "limits": {"train": args.limit_train, "test": args.limit_test},
        "scores": scores, "paired_by_topic": paired,
        "fallback_counts": {"ours": eval_counts,
                            "baseline": None if args.skip_baseline else {
                                "by_source": baseline_counts, "fallback": baseline_counts.get("fallback", 0)}},
        "wall_time_s": {**times, "total": time.perf_counter() - started},
        "token_usage": {"by_phase_and_role": client.counts,
                        "synthetic": settings.fake_llm,
                        "note": "Client-returned token counts; uncached totals exclude replay hits; fake counts are estimates."},
        "artifacts_mode": runtime["mode"],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parser.add_argument("--out", default=str(ROOT / "runs" / f"public-{stamp}"))
    parser.add_argument("--train", default=str(ROOT / "data" / "public_train.csv"))
    parser.add_argument("--test", default=str(ROOT / "data" / "public_test.csv"))
    parser.add_argument("--fake", action="store_true", help="Scripted offline smoke run, not benchmark evidence")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--language", choices=("original", "en", "es"), default="original")
    parser.add_argument("--limit-train", type=_positive, help="Stratified smoke subset; never report as benchmark")
    parser.add_argument("--limit-test", type=_positive, help="Topic-stratified smoke subset; no gold used for selection")
    args = parser.parse_args(argv)
    old_fake = os.environ.get("RJ_FAKE_LLM")
    if args.fake:
        os.environ["RJ_FAKE_LLM"] = "1"
    try:
        report = _run(args)
    except NotImplementedError:
        print("Public runner requires the implemented WP1 pipeline and merged dependencies.", file=sys.stderr)
        return 2
    except (ValueError, OSError, KeyError):
        # Do not echo exception text: an SDK/configuration error may contain a secret.
        print("Public run failed (input, output or configuration error); no benchmark claim produced.", file=sys.stderr)
        return 2
    finally:
        if args.fake:
            if old_fake is None:
                os.environ.pop("RJ_FAKE_LLM", None)
            else:
                os.environ["RJ_FAKE_LLM"] = old_fake
    print(f"{report['run_type']}: report written to {Path(args.out).resolve() / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
