"""Same-model, zero-shot bank reviewer without any learned rules or gold labels."""
from __future__ import annotations

from .contracts import LABELS, Item, LLMClient, Settings
from .jsonutil import extract_json
from .render import render_item

BASELINE_TEMPLATE_VERSION = "b1"
_SYSTEM = (
    'You are a senior quality reviewer at a bank. Decide if the assistant\'s handling '
    'of this conversation is acceptable ("good") or not ("bad"), using the context '
    'data as the source of truth. A conversation is "bad" if the assistant breaks '
    'any rule a careful bank expert would apply. Return ONLY JSON '
    '{"label": "good"|"bad"}.'
)
_TERMINAL_ERRORS = {"auth", "daily_quota", "deadline"}


def build_baseline_request(item: Item) -> tuple[str, str]:
    """(system, user) using render.render_item; no labels, ids or learned rules."""
    return _SYSTEM, render_item(item)


def _prior(priors: dict, topic: str) -> str:
    # Keep this independent of pipeline so the baseline works before WP1 is merged.
    label = priors.get("by_topic", {}).get(topic, priors.get("global", "good"))
    if label not in LABELS:
        raise ValueError("Baseline prior must be exactly good or bad")
    return label


def run_baseline(items: list[Item], client: LLMClient, settings: Settings, priors: dict) -> dict:
    """Return {item_id: {label, source}}; after 2 failed attempts use the prior.

    source is ``model`` or ``fallback``. Terminal API errors short-circuit the
    remaining items to their priors; the client owns transport retries/limiting.
    """
    predictions = {}
    stopped = False
    for item in items:
        prediction = {"label": _prior(priors, item.topic), "source": "fallback"}
        if not stopped and settings.seconds_left() > 0:
            system, user = build_baseline_request(item)
            for attempt in range(2):
                if settings.seconds_left() <= 0:
                    stopped = True
                    break
                prompt = user if attempt == 0 else (
                    user + '\nYour previous reply was invalid. Return ONLY JSON '
                    '{"label": "good"|"bad"}.'
                )
                result = client.generate(role="baseline", system=system, user=prompt,
                                         json_expected=True, max_output_tokens=128)
                if result.ok:
                    value = extract_json(result.text)
                    label = value.get("label") if isinstance(value, dict) else None
                    if isinstance(label, str) and label in LABELS:
                        prediction = {"label": label, "source": "model"}
                        break
                elif result.error_kind in _TERMINAL_ERRORS:
                    stopped = True
                    break
        predictions[item.item_id] = prediction
    return predictions
