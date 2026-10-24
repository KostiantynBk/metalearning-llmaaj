# rulejudge — a bank reviewer that learns from expert labels

**Integration draft: merge into the root README after verifying the final run, license and commands.**

rulejudge is a CLI for the Plata *A Judge That Learns from an Expert* challenge. It is for evaluators who need a reproducible whole-conversation `good`/`bad` judge and an inspectable explanation of its learned criteria. It uses the conversation and the account context, rather than treating individual assistant messages in isolation.

## How it works

1. Read the labeled training JSONL, fit per-topic majority priors, and immediately save a valid prior-only bundle.
2. Split training into fitting and selection sets; keep evaluation labels out of all prompts and fitting.
3. Use same-topic bad/good contrast pairs to propose and consolidate at most 12 compact requirements. These are hypotheses inferred from labels, not expert-authored rules.
4. Ask Gemma once per conversation to assess all applicable requirements, with support levels 0–3 and short evidence pointers. Parsing failures may cause retries.
5. Select a fixed or fitted non-compensating decision head using training only: any active requirement meeting the threshold makes the conversation `bad`. Disabled requirements cannot veto. Failed assessments use topic priors; a valid no-applicable-rule result means `good`.
6. Save the learned runtime and an automatic, example-backed jury explanation. In a separate `eval` process, load the frozen runtime and atomically checkpoint a complete predictions CSV, starting with prior predictions.

## Model and terms

Default core model: **Gemma 4 `gemma-4-31b-it` via the Gemini API**, used for proposing rules, assessing conversations, and the zero-shot same-model baseline. No local model weights are bundled. See [Gemma terms](https://ai.google.dev/gemma/terms) and [Gemini API terms](https://ai.google.dev/gemini-api/terms).

Model terms and API service terms are separate from the repository's code license. **Root LICENSE: TBD by integrator / Constantin.** Data rights are not granted by this draft; do not republish training data or private examples without permission.

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
| `RJ_MODEL_ASSESSOR` | `gemma-4-31b-it` |
| `RJ_MODEL_PROPOSER` | `gemma-4-31b-it` |
| `RJ_MODEL_BASELINE` | `gemma-4-31b-it`; must equal assessor for the same-model comparison |
| `RJ_RPM` | 25 client-side requests/minute |
| `RJ_TPM` | 15000 estimated input tokens/minute |
| `RJ_MAX_CONCURRENCY` | 4 workers; the rate limiter still applies |
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
- `rulejudge/induction.py`: training-only rule proposer/consolidation.
- `rulejudge/baseline.py`: same-model zero-shot comparison without learned requirements.

For a short demo: run the fake plumbing smoke first, then show a **clearly labeled** real completed run if available; open its manifest, runtime rulebook and generated explanation; finish with the real comparison report and limitations. Cached or recorded outputs must be described as cached or recorded, not live inference.

## Results — fill only after a verified full real public run

| Method | Macro topic accuracy | Credit | Financial position | Savings | Spending |
|---|---|---|---|---|---|
| Learned rulejudge | TBD by integrator | TBD | TBD | TBD | TBD |
| Same-model zero-shot reviewer | TBD by integrator | TBD | TBD | TBD | TBD |
| Training topic-majority prior | TBD by integrator | TBD | TBD | TBD | TBD |

Also record the exact SHA, dataset language, model names, fallback/source counts, runtime, token usage, and whether the public test was previously inspected. No performance gain is claimed here. Public test results are reporting-only; changes motivated by those results must be disclosed. The organizer's task score is `10 × macro_topic_accuracy + 0.4 × jury points`; the local runner reports accuracy, not unawarded jury points.

## Limitations and failure behavior

- A small labeled training set cannot identify every hidden rule. Inferred requirements can be wrong, incomplete, overly broad, or inconsistent.
- Textual rationales are not proof of faithful model reasoning. Evidence pointers and gold labels support auditing, not guarantees.
- Context/conversation text is untrusted input. Delimiter neutralization and reviewer instructions reduce prompt-injection risk but cannot eliminate it; the zero-shot comparison uses the prescribed simpler baseline prompt.
- Multi-language meaning, numerical reasoning and context interpretation may fail. The rule cap and prompt budget can omit important details.
- Rate limits, daily quota exhaustion, authentication problems and malformed replies can force majority fallback. A complete CSV can still be mostly prior predictions; inspect source counts and degraded-mode banners.
- The learned rubric does not compensate across criteria. A single active trigger can classify an otherwise good conversation as `bad`.
- Training/selection metrics are not held-out test performance; final refitting reuses selection data. Small public test differences are not evidence of a statistically reliable improvement.
- API processing permission, available quota, organizer runtime/network access and the final code license must be checked independently before submission.

## AI-assistance disclosure

Craft Agent and parallel AI coding agents assisted with architecture review, implementation, tests and this README draft. Gemma proposes requirements from expert-provided labels and assesses conversations at runtime. The project author reviews the final implementation, dependencies, real results, license and submission. No generated smoke output is presented as expert annotation or measured Gemma performance.
