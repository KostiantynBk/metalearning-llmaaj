# rulejudge — a bank reviewer that learns from expert labels

rulejudge is a CLI for the Plata *A Judge That Learns from an Expert* challenge. It is for evaluators who need a reproducible whole-conversation `good`/`bad` judge and an inspectable explanation of its learned criteria. It uses the conversation and the account context, rather than treating individual assistant messages in isolation.

## How it works

1. Read the labeled training JSONL, fit per-topic majority priors, and immediately save a valid prior-only bundle.
2. Split training into fitting and selection sets; keep evaluation labels out of all prompts and fitting.
3. Use same-topic bad/good contrast pairs to propose and consolidate compact requirements. These are hypotheses inferred from labels, not expert-authored rules. One fixed general check, **R00** ("the reply must be correct, consistent with the context data, and actually address the customer's request"), is prepended to every rulebook by design; the total stays at most 12 rules.
4. Ask Gemma once per conversation to assess all applicable requirements, with support levels 0–3 and short evidence pointers. Parsing failures may cause retries.
5. Select a fixed or fitted non-compensating decision head using training only: any active requirement meeting the threshold makes the conversation `bad`. Disabled requirements cannot veto. Failed assessments use topic priors; a valid no-applicable-rule result means `good`.
6. Save the learned runtime and an automatic, example-backed jury explanation. In a separate `eval` process, load the frozen runtime and atomically checkpoint a complete predictions CSV, starting with prior predictions.

## Model and terms

Default core model: **Gemma 4 `gemma-4-26b-a4b-it` via the Gemini API**, used for proposing rules, assessing conversations, and the zero-shot same-model baseline. `gemma-4-31b-it` is supported through the `RJ_MODEL_*` variables, but on the free tier on 24 Oct 2026 it took 34–315 s per call (rule induction is sequential), which risks the per-stage time budget. No local model weights are bundled. See [Gemma terms](https://ai.google.dev/gemma/terms) and [Gemini API terms](https://ai.google.dev/gemini-api/terms).

Model terms and API service terms are separate from the repository's code license. The code in this repository is released under the **MIT License** (see [LICENSE](../LICENSE)). Data rights are not granted by this draft; do not republish training data or private examples without permission.

## Setup

Python 3.10+ for local runs; Docker image uses Python 3.12. Install the pinned dependencies:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

Supply `GEMINI_API_KEY` through your local environment or a git-ignored root `.env`. Never paste it into a command that will be shared, commit it, include it in an image, or put it in a report. The local public runner reads a root `.env` only for real runs and does not override existing exported values. It supports simple `KEY=VALUE` lines, optional matching quotes and `export`, but no shell evaluation or variable substitution.

Confirm quota and permission to send the task data to Gemini before a real run. This project does not enable billing, purchase credits or activate paid top-ups. Free-tier availability and quota are not guaranteed.

## Docker: prepare, adapt, then evaluate

The following commands use a POSIX shell, with the key already exported locally. Run them from the repository root. Choose a fresh run directory each time.

```sh
python src/prepare_inputs.py --train data/public_train.csv --test data/public_test.csv --out-dir runs/demo/input
mkdir -p runs/demo/artifacts runs/demo/output
docker build -t rulejudge .
docker run --rm -e GEMINI_API_KEY \
  -v "$PWD/runs/demo/input:/input:ro" \
  -v "$PWD/runs/demo/artifacts:/artifacts" \
  rulejudge adapt
docker run --rm -e GEMINI_API_KEY \
  -v "$PWD/runs/demo/input:/input:ro" \
  -v "$PWD/runs/demo/artifacts:/artifacts:ro" \
  -v "$PWD/runs/demo/output:/output" \
  rulejudge eval
python src/evaluate.py --test data/public_test.csv --predictions runs/demo/output/predictions.csv --out runs/demo/score.json
```

If using a git-ignored root `.env` instead of an exported key, replace `-e GEMINI_API_KEY` with `--env-file .env` in **both** container commands. Docker does not automatically load the file. Do not print its contents. The artifacts volume must be writable for `adapt` and can be read-only for `eval`.

Expected organizer input files are `/input/train.jsonl` (labeled) and `/input/test.jsonl` (unlabeled). Output is `/output/predictions.csv`, header exactly `item_id,label`, every test ID exactly once, labels exactly `good` or `bad`. Adapt and eval are separate containers; eval never reads training examples or jury-only files.

## Local public comparison and dry run

```sh
# Offline plumbing smoke with scripted outputs — NOT evidence of model performance:
python scripts/run_public.py --fake --limit-train 20 --limit-test 8
# Real quota-limited subset smoke — also NOT a benchmark:
python scripts/run_public.py --limit-train 20 --limit-test 8
# Full public comparison (actual model/API calls):
python scripts/run_public.py
# Optional baseline omission:
python scripts/run_public.py --skip-baseline
```

Optional flags: `--out runs/<fresh-name>`, `--language original|en|es`, `--train <csv-or-jsonl>`, `--test <csv-or-jsonl>`. The default language is `original`; translated variants are alternative versions, not additional independent examples. Output directory reuse is rejected. Both datasets must retain all four topics. Subsets are deterministic: train is stratified by topic and label, test by topic **without using test labels**.

The runner writes `report.json` and `report.md` with official `src/evaluate.py` scores, per-topic accuracy, macro topic accuracy, paired per-topic counts (both right / only ours / only baseline / both wrong), fallback diagnostics, wall time, and client-returned token usage. Cached usage is separated from uncached totals; fake usage is synthetic. If a pipeline version does not expose exact eval-source counts, the report says unavailable instead of guessing from labels. Gold is available only to preparation/validation and the final scorer, never to model requests or head fitting. Smoke reports are prominently labeled and must not be cited as benchmark evidence.

## Configuration

| Variable | Default / purpose |
|---|---|
| `GEMINI_API_KEY` | Required for real API calls; never stored in artifacts |
| `RJ_INPUT_DIR` | `/input`; local runner sets a run-specific directory |
| `RJ_ARTIFACTS_DIR` | `/artifacts`; local runner sets a run-specific directory |
| `RJ_OUTPUT_DIR` | `/output`; local runner sets a run-specific directory |
| `RJ_MODEL_ASSESSOR` | `gemma-4-26b-a4b-it` |
| `RJ_MODEL_PROPOSER` | `gemma-4-26b-a4b-it` |
| `RJ_MODEL_BASELINE` | `gemma-4-26b-a4b-it`; must equal assessor for the same-model comparison |
| `RJ_RPM` | 25 client-side requests/minute |
| `RJ_TPM` | 15000 estimated input tokens/minute |
| `RJ_MAX_CONCURRENCY` | 8 workers; the rate limiter still applies |
| `RJ_TIME_BUDGET_S` | 3000 seconds per stage |
| `RJ_MAX_INDUCTION_CALLS` | 40 maximum proposer calls |
| `RJ_S_FRACTION` | 0.2 training selection fraction |
| `RJ_CHECKPOINT_EVERY` | 10 completed eval items between checkpoints |
| `RJ_CACHE_DIR` | Optional explicit cache directory; default chosen by pipeline |
| `RJ_FAKE_LLM` | Offline fake mode; use `--fake` on the public runner for its scripted responder |
| `RJ_STRICT` | `1` re-raises development errors instead of graceful degradation |

Quota is project-wide; multiple keys do not create independent quota. Supplied event measurements were approximately 30 requests/minute and 16K input tokens/minute per model. These are not guaranteed API entitlements. A full adapt/eval/baseline public run can take roughly 35 minutes, and retries can extend it. Record actual usage; do not infer savings or latency from the fake run.

## Where the expertise and model calls live

- `/artifacts/manifest.json` atomically selects `/artifacts/bundles/<id>/` and records file hashes.
- `runtime/rulebook.json`: frozen learned requirements, with provenance IDs removed.
- `runtime/head.json`: threshold, disabled requirements and selected head kind.
- `runtime/priors.json`: training-only fallback majority labels.
- `runtime/assessor_config.json`: assessor template/model metadata.
- `jury/explanation.md`: automatically generated criteria, priorities, real training evidence, boundaries and implementation-consistency details.
- Other `jury/` files contain training assessments and provenance for local audit, not runtime inference. They may contain sensitive training text; do not publish them indiscriminately.
- `rulejudge/gemma_client.py`: actual Gemini API SDK calls, feature fallbacks, rate limiting, caching and usage records.
- `rulejudge/assessor.py`: learned-rule prompt and per-item assessment.
- `rulejudge/induction.py`: training-only rule proposer/consolidation, plus the fixed general check `R00` (`general_rule`, `add_general_rule`).
- `rulejudge/baseline.py`: same-model zero-shot comparison without learned requirements.

For a short demo: run the fake plumbing smoke first, then show a **clearly labeled** real completed run if available; open its manifest, runtime rulebook and generated explanation; finish with the real comparison report and limitations. Cached or recorded outputs must be described as cached or recorded, not live inference.

## Results — full real public run (24 Oct 2026)

Run: code commit `e52cb62` (the final HEAD changes only docs, LICENSE and Docker pinning); `python scripts/run_public.py`; original-language public data, **150 train / 80 test**; `gemma-4-26b-a4b-it` for all roles (free tier). Scores come from the official `src/evaluate.py`.

| Method | Macro topic accuracy | Accuracy | Credit | Financial position | Savings | Spending |
|---|---:|---:|---:|---:|---:|---:|
| **rulejudge (R00 + 11 learned rules)** | **0.833** | 0.812 (65/80) | 1.000 | 0.818 | 0.950 | 0.565 |
| Same-model zero-shot reviewer | 0.826 | 0.812 (65/80) | 0.933 | 0.818 | 0.900 | 0.652 |
| Training topic-majority prior | 0.626 | 0.625 (50/80) | 0.600 | 0.682 | 0.700 | 0.522 |

Paired outcomes against the same-model baseline:

| Topic | Both right | Only ours | Only baseline | Both wrong |
|---|---:|---:|---:|---:|
| credit | 14 | 1 | 0 | 0 |
| financial_position | 18 | 0 | 0 | 4 |
| savings | 17 | 2 | 1 | 0 |
| spending | 11 | 2 | 4 | 6 |
| **all** | 60 | 5 | 5 | 10 |

**Interpretation:**
- **Accuracy is a statistical tie with plain Gemma.** Each method is right on 5 conversations the other gets wrong, so this does not demonstrate an accuracy improvement.
- **Better where specific rules were learned** (credit, savings); **weaker on spending**, where no spending-specific rule was learned and only R00 applies.
- rulejudge catches more bad conversations (22/30 vs 18/30) at the cost of more false alarms (7/50 vs 3/50). Cohen's κ: 0.597 vs 0.574.
- What it adds is an inspectable, training-derived rulebook, per-rule evidence, and an automatic jury explanation backed by real training examples.

**Training diagnostics:**
- 12 rules: R00 plus 11 learned (credit 4, financial_position 4, savings 2, global 1).
- Selection split F/S = 121/29. Selected head: fixed, τ = 2. Training-S macro 0.804 vs prior 0.627, so the floor was not applied.
- R00 triggered on 39 bad / 8 good training conversations (precision 0.83).

**Cost and reliability (this run):**
- Wall time: adapt 38.5 min, eval 8.1 min, baseline 7.2 min.
- Calls: 22 proposer (2 server-side timeouts, skipped), 152 + 81 assessor (0 failed), 80 baseline.
- About 406K input / 29K output tokens in total.
- 0 prior fallbacks for both methods.

**Docker verification:**
- Image pinned by base digest and `constraints.txt`; `pip-audit` found no known vulnerabilities.
- Replaying this run inside separate containers with `RJ_CACHE_DIR` produced an **identical rulebook and head** (adapt) and **identical predictions** (eval; official score 0.8333).

**Disclosure — the public test was not untouched:**
- A 16-item public-test smoke subset was scored before the final design. With learned rules only, it gave 0.483 vs the baseline's 0.838.
- That result led to two changes before the full run:
  - adding the fixed general check **R00**;
  - switching from `gemma-4-31b-it` (34–315 s per call on the free tier) to `gemma-4-26b-a4b-it`.
- Nothing was changed after the full run above. These public-test numbers therefore informed one design change and are not an untouched held-out estimate.

The organizer's task score is `10 × macro_topic_accuracy + 0.4 × jury points`; the local runner reports accuracy only.

## Limitations and failure behavior

- A small labeled training set cannot identify every hidden rule. Inferred requirements can be wrong, incomplete, overly broad, or inconsistent.
- Textual rationales are not proof of faithful model reasoning. Evidence pointers and gold labels support auditing, not guarantees.
- Context/conversation text is untrusted input. Delimiter neutralization and reviewer instructions reduce prompt-injection risk but cannot eliminate it; the zero-shot comparison uses the prescribed simpler baseline prompt.
- Multi-language meaning, numerical reasoning and context interpretation may fail. The rule cap and prompt budget can omit important details.
- Rate limits, daily quota exhaustion, authentication problems and malformed replies can force majority fallback. A complete CSV can still be mostly prior predictions; inspect source counts and degraded-mode banners.
- The learned rubric does not compensate across criteria. A single active trigger can classify an otherwise good conversation as `bad`.
- Training/selection metrics are not held-out test performance; final refitting reuses selection data. Small public test differences are not evidence of a statistically reliable improvement.
- **Stage time budget.** `RJ_TIME_BUDGET_S` defaults to 3000 s per stage. On the free tier, adapt on 150 items took 38.5 min, and about 190 private training items may approach the budget. When it is reached, the remaining calls return `deadline`. Those training items count as failed assessments, and adapt still commits the learned bundle. Eval (about 6 min per 80 items here) keeps a complete, checkpointed CSV. Raise the budget if organizer limits allow.
- Gemini API processing permission and quota at the organizer run were assumed available as instructed; free-tier latency varies (two induction calls timed out server-side in the reported run).

## AI-assistance disclosure

Craft Agent and parallel AI coding agents assisted with architecture review, implementation, tests and this README draft. Gemma proposes requirements from expert-provided labels and assesses conversations at runtime. The project author reviews the final implementation, dependencies, real results, license and submission. No generated smoke output is presented as expert annotation or measured Gemma performance.
