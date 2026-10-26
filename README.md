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

Model terms and API service terms are separate from the repository's code license. The code in this repository is released under the **MIT License** (see [LICENSE](LICENSE)). Data rights are not granted by this draft; do not republish training data or private examples without permission.

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

### Organizer run on private data

- **Key.** Pass your own key to both containers as `GEMINI_API_KEY` (`-e GEMINI_API_KEY` or `--env-file`). The google-genai SDK also accepts `GOOGLE_API_KEY` and uses it if both are set. The key needs Gemini API access to `gemma-4-26b-a4b-it`. Nothing is baked into the image.
- **Network.** Both containers need outbound HTTPS to the Gemini API (`generativelanguage.googleapis.com`). Without it every call fails. Eval still writes a valid CSV, but it then holds training topic-majority predictions only.
- **Time.** The defaults throttle to free-tier-safe rates (`RJ_RPM=25`, `RJ_TPM=15000`). At those rates, expect about 45–50 min for adapt on 190 training items and about 30 min for eval on 296 test items. If your key has higher limits, raise `RJ_RPM`, `RJ_TPM` and `RJ_MAX_CONCURRENCY` below them for a faster run. If a container has a hard time limit, set `RJ_TIME_BUDGET_S` a few minutes below it. At the budget, adapt still commits a learned bundle and eval keeps its checkpointed CSV complete.
- **Jury artifact.** `/artifacts/manifest.json` points to the selected bundle; the explanation is `bundles/<id>/jury/explanation.md`.

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
- **No accuracy improvement over plain Gemma was demonstrated.** Each method is right on 5 conversations the other gets wrong. This is not a formal demonstration of statistical equivalence.
- **Higher observed accuracy on credit and savings; lower on spending**, where no spending-specific rule was learned and only R00 applies. This suggests a coverage gap, but does not isolate the cause of the performance differences.
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

## Research basis

The design draws on seven sources. Each row names the part of this code it shaped. All results cited there are the **authors' results on their own tasks**; none reports results on Plata data, and nothing below is a claim that their gains transfer.

| Source | What we took from it | Where it lives here |
|---|---|---|
| Kola et al., **SAJA: A Simple Approach to Judge Alignment for LLM-as-a-Judge** (ACL 2026 Industry, `2026.acl-industry.45`) | One structured LLM call per item extracts features; a small, separate decision layer is fitted on labels. Fixed and fitted heads are compared rather than assuming calibration helps: their full head lowered binary accuracy (0.65 → 0.61). | `assessor.py` (one call per conversation); `decision.py` (fixed vs fitted head, selection on training data) |
| Kwako et al., **LLM-Based Rubric Refinement in Multi-Agent Automated Scoring** (AIME-Con 2026, `2026.aimecon-sessions.2`) | Rubric text derived from labelled examples, with evidence-linked proposals and an explicit "no clear rule" option. The result is checked on a separate full split, not only on the cases that motivated it. | `induction.py` (`no_clear_rule`, provenance kept for audit); head selection and the prior floor on the held-out training split S |
| Lu et al., **SPEAR: Code-Augmented Agentic Prompt Optimization** (arXiv `2605.26275`) | Keep analysis and selection separate, select with the official metric, budget all calls, and expect small dev sets to rank noisily. Plus a recovery pass for failed calls. | Topic-macro selection metric; `RJ_MAX_INDUCTION_CALLS`; incumbent wins ties; assessor recovery pass |
| Sun et al., **Support Vector Rubrics: Closing the Gap Between Self-Generated and Human Rubrics** (arXiv `2606.08077`) | Contrastive good-vs-bad induction, and evidence that per-criterion treatment matters (their uniform-weight ablation fell from 82.8 to 64.3). We did not copy compensating weights into an any-violation task. | Same-topic bad/good contrast pairs; precision-gated rule disabling; non-compensating veto |
| Kawakami et al., **"I want to be pushed, I want to grow": Enabling social workers to design evaluations of LLM augmentation in their work** (arXiv `2608.22459`) | Criteria need explicit applicability, boundaries ("what is *not* a violation") and contrasting examples; "not applicable" is a normal outcome. | `Rule.applies_when` / `boundary`; explanation sections on boundaries and false alarms; "no applicable rule ⇒ good" |
| Robsky, Lavitas and Wang, **Reliable Evaluations for LLMs and AI Agents** (book) | Hard gates on critical dimensions, separate data roles, paired-difference uncertainty when comparing systems, and failure handling where one failed call never invalidates a run. | Veto decision layer; train-only `adapt` / test-only `eval`; paired comparison table above; priors-first bundles and checkpointed CSV |
| Mehrabi et al., **Prompting Strategy Effects Depend on the Judge: A Multilingual Study of LLM-as-a-Judge Robustness** (arXiv `2610.31172`) | Prompt effects depend on the judge model and language, so tune and test on the deployed model rather than transferring a "best" prompt. Their Gemma evidence is Gemma 3, not Gemma 4. | All measurements on `gemma-4-26b-a4b-it` itself; same-model zero-shot baseline |

Two choices came from our own measurements rather than the literature:
- the fixed general check **R00**, added after a public-test smoke subset showed a learned-rules-only judge missing most bad answers (see the disclosure above);
- the switch to 26B-A4B, driven by measured free-tier latency.

## Limitations and failure behavior

- **Single-pass assessment.** One substantive Gemma request assesses all applicable rules for each conversation. Content retries and failed-call recovery repair format or transport failures, not plausible but incorrect judgments. Independent semantic verification, per-rule calls and multi-pass consensus have not been evaluated; more calls are not assumed to improve accuracy.
- **Incomplete coverage and bounded induction.** A small labeled training set cannot identify every hidden rule. Inferred requirements can be wrong, incomplete, overly broad, or inconsistent. The public run learned no spending-specific rule. Lexical contrast pairing and the compact rule budget can miss requirements; the current pipeline does not revise the rubric in response to residual training errors.
- **Inspectable evidence is not independent fact-checking.** Textual rationales are not proof of faithful model reasoning. The parser checks whether evidence pointers resolve to existing messages or context fields, not whether they substantiate a violation. Vetoes depend on model-produced support levels and do not require independently verified evidence.
- Context/conversation text is untrusted input. Delimiter neutralization and reviewer instructions reduce prompt-injection risk but cannot eliminate it; the zero-shot comparison uses the prescribed simpler baseline prompt.
- Multi-language meaning, numerical reasoning and context interpretation may fail. The rule cap and prompt budget can omit important details.
- Rate limits, daily quota exhaustion, authentication problems and malformed replies can force majority fallback. A complete CSV can still be mostly prior predictions; inspect source counts and degraded-mode banners.
- The learned rubric does not compensate across criteria. A single active trigger can classify an otherwise good conversation as `bad`.
- Training/selection metrics are not held-out test performance; final refitting reuses selection data. Small public test differences are not evidence of a statistically reliable improvement.
- **Stage time budget.** `RJ_TIME_BUDGET_S` defaults to 3000 s per stage. On the free tier, adapt on 150 items took 38.5 min, and about 190 private training items may approach the budget. When it is reached, the remaining calls return `deadline`. Those training items count as failed assessments, and adapt still commits the learned bundle. Eval (about 6 min per 80 items here) keeps a complete, checkpointed CSV. Raise the budget if organizer limits allow.
- Gemini API processing permission and quota at the organizer run were assumed available as instructed; free-tier latency varies (two induction calls timed out server-side in the reported run).

## Next experiments — not implemented or validated

1. **Verify proposed violations.** Compare the single-pass assessor with a separately prompted evidence-verification pass and a higher-thinking assessor. Test multi-assessment aggregation rather than assuming consensus helps; the task's any-violation rule is not a reason to reject whenever any reviewer says `bad`.
2. **Improve coverage using training only.** Diagnose uncovered topics and residual errors, then test one bounded rubric-refinement round and semantic contrast pairing on separate validation data.
3. **Isolate what helps.** Compare zero-shot, R00-only, learned-only and R00-plus-learned variants under matched conditions. The current comparison does not isolate the incremental benefit of learned rules from R00 or prompt structure.
4. **Strengthen evaluation.** Freeze variants before untouched evaluation; report paired topic-macro uncertainty, bad-case recall, false alarms, fallback rate, tokens and latency. Repeat fresh inference/adaptation runs to measure variability. Exact-request cache replay is a reproducibility check for saved responses, not an independent model trial.

These are research directions, not claimed improvements. The public-test influence disclosed above remains a limitation of the reported results.

## AI-assistance disclosure

Craft Agent and parallel AI coding agents assisted with architecture review, implementation, tests and this README draft. Gemma proposes requirements from expert-provided labels and assesses conversations at runtime. The project author reviews the final implementation, dependencies, real results, license and submission. No generated smoke output is presented as expert annotation or measured Gemma performance.

---

# Original challenge description: A Judge That Learns from an Expert

Infer an expert's evaluation rules from labeled conversations, then assign one of two labels to new conversations: `good` or `bad`. Evaluate the entire conversation. Any violation of an applicable rule makes the conversation `bad`.

**Dataset sizes.** Public: 150 training examples and 80 test examples. Final evaluation: 190 private training examples and 296 private test examples.

The public and private datasets cover the same four topics. The private dataset introduces additional expert requirements whose violations are not represented in the public data, so your solution must adapt again using the training set provided for the final evaluation. The hidden rules are not shared with participants.

## Files

```text
data/
  public_train.csv           # 150 labeled examples for adaptation
  public_test.csv            # 80 labeled examples for local evaluation
src/
  common.py                  # data loading and format validation
  prepare_inputs.py          # CSV -> JSONL; removes labels from prediction inputs
  evaluate.py                # prediction validation and metric calculation
  test_evaluation.py         # evaluation contract tests
README.md
```

The public training set contains 90 `good` and 60 `bad` examples; the public test set contains 50 `good` and 30 `bad` examples. Both sets cover all four topics.

Public test labels are provided for local evaluation. During the final evaluation, your solution receives a labeled private training set and an unlabeled private test set. Adaptation starts with a clean artifacts directory for each public or private run.

## Data format

CSV files use UTF-8 encoding and include a header row. Use a CSV parser: cells may contain quotes, commas, and JSON.

| Column | Contents |
|---|---|
| `item_id` | An opaque, unique example identifier. |
| `topic` | `financial_position`, `savings`, `spending`, or `credit`. |
| `context` | A JSON object containing the source data needed to evaluate the assistant's response. |
| `conversation` | A JSON array of messages with `role` and `content` fields. |
| `label` | The ground-truth label: `good` or `bad`. Omitted from the final prediction input. |
| `input_es` | A Spanish version of the complete input: JSON containing `context` and `conversation`. |
| `input_en` | An English version of the complete input in the same format. |

Translations are alternative representations of the same example, not additional rows or separate tasks. They share the same `item_id`, `topic`, and label. By default, inputs use the original `context` and `conversation`; you can select a translation when preparing inputs. Record the selected language with your results to make experiments comparable.

## Quick start

The scripts in `src/` require Python 3.10 or later. No external libraries or API keys are needed to run these scripts. Run the commands below from the repository root.

Prepare inputs for the two stages of your solution:

```sh
python3 src/prepare_inputs.py --train data/public_train.csv --test data/public_test.csv --out-dir out/public_input
```

The new directory will contain labeled `train.jsonl` and unlabeled `test.jsonl`. Both files contain only the required fields; alternative translations are not duplicated in the inputs. Add `--language es` or `--language en` to select a translation.

During adaptation, use only the training set. During prediction, use the unlabeled test set and the artifacts produced during adaptation. The labeled public test set is intended for scoring and development; the final private test set will be hidden.

Your solution must output **exactly one label for every test ID**. Probabilities and confidence scores are not required. For local evaluation, write predictions as CSV:

```csv
item_id,label
FIRST_EXAMPLE_ID,good
SECOND_EXAMPLE_ID,bad
```

The local scorer also accepts JSONL with exactly the same two fields:

```json
{"item_id":"EXAMPLE_ID","label":"good"}
```

Column names and class labels are case-sensitive. Row order does not matter. Missing, duplicate, or unknown IDs, extra fields, and labels other than `good` or `bad` are rejected with a nonzero exit code.

Measure prediction quality for a complete prediction file:

```sh
python3 src/evaluate.py --test data/public_test.csv --predictions predictions.csv --out out/score.json
```

## Submitting your solution

**Fork this repository and push your complete solution to your team's fork.** Click **Fork** on GitHub, then clone your fork. Replace `YOUR_ACCOUNT` with the account or organization that owns your fork:

```sh
git clone https://github.com/YOUR_ACCOUNT/metalearning-llmaaj.git
cd metalearning-llmaaj
```

Commit the source code, a Dockerfile, dependency specifications, and instructions for building and running both `adapt` and `eval`. Include any files needed to reproduce the solution, and document configuration through environment variables. Keep credentials out of the repository.

After committing your solution, push it to your fork and record the exact commit:

```sh
git push origin HEAD
git rev-parse HEAD
```

Send the organizers **the URL of your fork and the commit SHA**. The organizers will evaluate that submitted version. Make sure they can access your fork and fetch the submitted commit.

Your Docker image must support two commands, `adapt` and `eval`, with this interface:

| Stage | Inputs | Outputs |
|---|---|---|
| `adapt` | Labeled training examples at `/input/train.jsonl`. | Adaptation artifacts in `/artifacts`, including the automatically generated explanation for the jury. |
| `eval` | Unlabeled test examples at `/input/test.jsonl` and the saved `/artifacts` directory. | `/output/predictions.csv` with exactly the columns `item_id,label`, as shown above. |

`eval` runs in a separate container, so persist everything it needs from `adapt` in `/artifacts`.

## Final evaluation

**You submit a runnable solution; the organizers run it themselves and calculate the official result.** Local scores help you develop your pipeline.

1. The organizers clone your fork, check out the submitted commit, and build your Docker image using the supplied instructions.
2. They prepare a new, labeled private training set and an unlabeled private test set. Private test labels remain with the organizers and are never passed to your solution.
3. They run `adapt` on the private training set with a fresh, empty artifacts directory. This stage learns the expert's criteria and automatically produces both prediction artifacts and an explanation for the jury.
4. They run `eval` in a separate container using the private test inputs and those saved artifacts. Your solution writes one `good` or `bad` prediction for every test ID.
5. The organizers validate the prediction file and calculate `acc = macro_topic_accuracy` against the hidden labels. The score is printed and saved to a file.
6. The jury evaluates the explanation generated during that same adaptation run using the five criteria below. The organizers combine `acc` and the jury points using **Final score = 10 × acc + 0.4 × I**.

## Automated metric

The automated metric is **`macro_topic_accuracy`** (`acc`): calculate accuracy separately for each of the four topics, then average those four values. Every topic has equal weight, regardless of its number of examples.

```text
acc = (accuracy_credit + accuracy_financial_position + accuracy_savings + accuracy_spending) / 4
score = macro_topic_accuracy = acc
```

Scores range from 0 to 1; higher is better. For example, topic accuracies of 0.8, 0.7, 0.9, and 0.6 give `acc = 0.75`. Model/API cost does not affect the score, and no cost reports are required for evaluation.

The bundled `src/evaluate.py` calculates this metric directly: its `macro_topic_accuracy` and `score` fields contain the same value. The organizer evaluator uses the same metric. Jury points are added separately as described below.

The scorer also reports overall accuracy, Cohen's kappa, and mean kappa across topics as diagnostics. In confusion matrices, rows represent ground-truth labels and columns represent predictions; the class order is `good`, `bad`. If kappa has a zero denominator, it is reported as 0.

Brier score and log loss are not calculated because solutions output labels only. The scorer evaluates one saved set of predictions without rerunning the model. Identical predictions against the same test set produce identical scores. This does not imply that the solution itself is deterministic.

A difference of a few thousandths in the score is not, by itself, evidence of a statistically significant advantage.

## Project defense and final score

**Final score = 10 × acc + 0.4 × I**, where **acc = `macro_topic_accuracy`** and **I ∈ [0, 5]** is the sum of points in the table below.

The automated component contributes up to 10 points and the jury component up to 2 points. Final scores range from 0 to 12. For example, `acc = 0.80` and `I = 4` give a final score of `10 × 0.80 + 0.4 × 4 = 9.6`.

The jury evaluates an artifact automatically generated by `adapt` on a new training set. Include this explanation alongside the artifacts used for prediction. The examples below are hypothetical illustrations for a text-to-`good`/`bad` task.

The interpretability score **`I` ranges from 0 to 5** and is the sum of five binary criteria:

| Criterion | +1 point | 0 points |
|---|---|---|
| **Clear criteria** | "Unsupported claims indicate a bad response; asking for clarification when essential data are missing indicates a good response." A decision tree with understandable conditions also qualifies. | "The model assesses text quality," or a list of unexplained technical features. |
| **Clear priorities** | "A factual error outweighs completeness." For a decision tree, the order of conditions and thresholds is visible. | Criteria are listed, but it is unclear how conflicting criteria produce the final `good`/`bad` label. |
| **Links to training examples** | Specific texts, their expert labels, and an explanation of which criterion each example illustrates are provided. | Only invented examples are given, or a generic claim such as "the training data support this." |
| **Clear boundaries** | "Labels for short responses are inconsistent: similar texts received different labels," supported by examples from the training set. | "The model sometimes makes mistakes," without specific conditions or examples. |
| **Consistency with the implementation** | The described tree is actually used for prediction; the displayed rules appear in the final prompt; a table of weights matches the model. | A polished explanation is attached separately, but its connection to the working solution is not demonstrated. |

**Rules for the jury:** each row receives exactly 0 or 1 point. The format is unrestricted: text, rules, a tree, a table, or a visualization. Length and literary quality earn no additional points; a classical model can receive all five points.

"Common sense" here means checking whether the explanation is understandable. An unusual expert preference is acceptable when it is supported by the labels.

## Check the evaluation scripts

```sh
python3 -m unittest discover -s src -p 'test_*.py'
```
