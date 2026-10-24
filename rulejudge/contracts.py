"""Shared data contracts for the rulejudge pipeline.

CANONICAL FILE (owner: integrator / WP0). Every work package imports from here.
Do not change field names or semantics on a work-package branch; ask the
integrator instead. Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Optional, Protocol, Union

LABELS = ("good", "bad")
TOPICS = ("credit", "financial_position", "savings", "spending")
NA = "na"                       # rule does not apply to this conversation
SUPPORT_LEVELS = (0, 1, 2, 3)   # 0 none, 1 ambiguous, 2 supported, 3 clear violation
MAX_RULES = 12                  # input-token budget (Gemma free tier ~16K input tokens/min)
ROLES = ("assessor", "proposer", "baseline")
SupportValue = Union[int, str]  # 0..3 or "na"

# Head kinds / decision sources / statuses / bundle modes (compare by value)
HEAD_FIXED, HEAD_FITTED, HEAD_PRIOR = "fixed", "fitted", "prior"
SRC_MODEL, SRC_PARTIAL, SRC_FALLBACK, SRC_PRIOR_MODE = (
    "model", "partial_model", "fallback_prior", "prior_mode")
ST_OK, ST_PARTIAL, ST_FAILED = "ok", "partial", "failed"
MODE_PRIOR_ONLY, MODE_LEARNED = "prior_only", "learned"


@dataclass(frozen=True)
class Item:
    item_id: str
    topic: str                      # one of TOPICS
    context: dict                   # source data the assistant had
    conversation: list              # [{"role": "user"|"assistant"|"system"|"tool", "content": str}]
    label: Optional[str] = None     # "good"/"bad" for training items, None for test items


@dataclass
class Rule:
    rule_id: str                    # "R01".."R16", stable within a run
    text: str                       # one-sentence verifiable requirement; no item IDs, no customer names
    scope: str = "global"           # "global" or one of TOPICS (default global)
    applies_when: str = ""          # condition under which the rule applies ("" = always within scope)
    violation: str = ""             # what counts as a violation
    boundary: str = ""              # what does NOT count as a violation (near miss); may be ""
    provenance_ids: list = field(default_factory=list)  # training IDs behind the proposal (audit only)
    origin: str = "inferred_from_labels"                # or "explicit_in_context"


@dataclass
class Rulebook:
    rules: list                     # list[Rule], at most MAX_RULES
    notes: list = field(default_factory=list)  # contradictions / uncovered cases from induction


@dataclass
class Evidence:
    rule_id: str
    support: int                    # 1..3
    where: list                     # e.g. ["msg 4", "context.credit_statement.next_cut_off_date"]
    why: str                        # <= ~15 words


@dataclass
class Assessment:
    item_id: str
    status: str                     # ST_OK | ST_PARTIAL | ST_FAILED
    support: dict                   # rule_id -> "na" | 0..3 (only rules that were in the request)
    evidence: list = field(default_factory=list)          # list[Evidence], only for support >= 1
    requested_rule_ids: list = field(default_factory=list)
    request_hash: str = ""
    attempts: int = 0
    error: str = ""


@dataclass
class HeadParams:
    kind: str                       # HEAD_FIXED | HEAD_FITTED | HEAD_PRIOR
    tau: int = 2                    # violation-support threshold, 1..3
    disabled: list = field(default_factory=list)  # rule IDs ignored by the decision (still prompted)
    fit_on: str = ""                # "F", "F+S" or "" (fixed/prior)


@dataclass
class Decision:
    item_id: str
    label: str                      # "good" | "bad"
    source: str                     # SRC_MODEL | SRC_PARTIAL | SRC_FALLBACK | SRC_PRIOR_MODE
    triggered: list = field(default_factory=list)  # active rule IDs with support >= tau


@dataclass
class LLMResult:
    ok: bool
    text: str = ""
    # "" | "rate_limit" | "daily_quota" | "unavailable" | "auth" | "bad_request" | "timeout" | "deadline" | "other"
    error_kind: str = ""
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


class LLMClient(Protocol):
    def generate(self, *, role: str, system: str, user: str,
                 json_expected: bool = True, max_output_tokens: int = 2048) -> LLMResult:
        """role is one of ROLES. Never raises for API errors: returns ok=False with error_kind."""
        ...


@dataclass
class Settings:
    input_dir: str = "/input"
    artifacts_dir: str = "/artifacts"
    output_dir: str = "/output"
    model_assessor: str = "gemma-4-31b-it"
    model_proposer: str = "gemma-4-31b-it"
    model_baseline: str = "gemma-4-31b-it"
    max_concurrency: int = 4
    rpm_limit: int = 25                 # client-side requests/minute cap (API limit 30)
    tpm_limit: int = 15000              # client-side input tokens/minute cap (API limit ~16K)
    time_budget_s: float = 3000.0       # per stage (adapt or eval)
    deadline_epoch: float = 0.0         # absolute time.time() deadline; 0 = none
    max_content_attempts: int = 2       # parse/validation attempts per assessment (main pass)
    max_induction_calls: int = 40
    s_fraction: float = 0.2
    checkpoint_every: int = 10          # eval: rewrite predictions.csv every N finished items
    fake_llm: bool = False
    cache_dir: str = ""                 # set by pipeline (adapt: <artifacts>/cache, eval: temp dir)
    usage_log: str = ""                 # JSONL path for per-call usage records

    @staticmethod
    def from_env(env: Optional[dict] = None) -> "Settings":
        env = dict(os.environ if env is None else env)
        s = Settings()
        mapping = {
            "RJ_INPUT_DIR": ("input_dir", str), "RJ_ARTIFACTS_DIR": ("artifacts_dir", str),
            "RJ_OUTPUT_DIR": ("output_dir", str), "RJ_MODEL_ASSESSOR": ("model_assessor", str),
            "RJ_MODEL_PROPOSER": ("model_proposer", str), "RJ_MODEL_BASELINE": ("model_baseline", str),
            "RJ_MAX_CONCURRENCY": ("max_concurrency", int), "RJ_RPM": ("rpm_limit", int), "RJ_TPM": ("tpm_limit", int),
            "RJ_TIME_BUDGET_S": ("time_budget_s", float),
            "RJ_MAX_INDUCTION_CALLS": ("max_induction_calls", int),
            "RJ_S_FRACTION": ("s_fraction", float), "RJ_CHECKPOINT_EVERY": ("checkpoint_every", int),
        }
        for key, (attr, cast) in mapping.items():
            if str(env.get(key, "")).strip():
                setattr(s, attr, cast(env[key]))
        s.fake_llm = str(env.get("RJ_FAKE_LLM", "")).strip().lower() in {"1", "true", "yes"}
        return s

    def start_clock(self) -> None:
        self.deadline_epoch = time.time() + self.time_budget_s

    def seconds_left(self) -> float:
        return float("inf") if not self.deadline_epoch else self.deadline_epoch - time.time()


# ----------------------------------------------------------------------------- JSON helpers

def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).hexdigest()


def _from_dict(cls, data: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def rule_from_dict(d: dict) -> Rule:
    return _from_dict(Rule, d)


def rulebook_from_dict(d: dict) -> Rulebook:
    return Rulebook(rules=[rule_from_dict(r) for r in d.get("rules", [])], notes=list(d.get("notes", [])))


def evidence_from_dict(d: dict) -> Evidence:
    return _from_dict(Evidence, d)


def assessment_from_dict(d: dict) -> Assessment:
    a = _from_dict(Assessment, d)
    a.evidence = [evidence_from_dict(e) if isinstance(e, dict) else e for e in a.evidence]
    return a


def head_from_dict(d: dict) -> HeadParams:
    return _from_dict(HeadParams, d)


def item_from_dict(d: dict) -> Item:
    return Item(item_id=d["item_id"], topic=d["topic"], context=d["context"],
                conversation=d["conversation"], label=d.get("label"))


def item_text_for_hash(item: Item) -> str:
    """Label-free canonical text of an item (used for fingerprints and cache keys)."""
    return canonical_json({"item_id": item.item_id, "topic": item.topic,
                           "context": item.context, "conversation": item.conversation})
