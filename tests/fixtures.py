"""Synthetic test fixtures (WP0, canonical). No real task data, no network."""
from __future__ import annotations

from rulejudge.contracts import TOPICS, Item, Rule, Rulebook


def make_item(item_id: str, topic: str = "credit", label=None, user: str = "When is my payment due?",
              assistant: str = "Your payment is due on 2026-10-02.", context=None) -> Item:
    ctx = context if context is not None else {"today": "2026-09-27", "user": {"first_name": "Ana"},
                                               "credit_statement": {"due_date": "2026-10-02"}}
    return Item(item_id=item_id, topic=topic, context=ctx,
                conversation=[{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
                label=label)


def toy_dataset(per_topic_good: int = 3, per_topic_bad: int = 2) -> list[Item]:
    """Items across all four topics. Bad items contain the marker 'WRONG-DATE' in the reply."""
    items = []
    for t in TOPICS:
        for i in range(per_topic_good):
            items.append(make_item(f"{t[:3]}-g{i}", t, "good", assistant=f"Correct answer {i} for {t}."))
        for i in range(per_topic_bad):
            items.append(make_item(f"{t[:3]}-b{i}", t, "bad", assistant=f"WRONG-DATE answer {i} for {t}."))
    return items


def toy_rulebook() -> Rulebook:
    return Rulebook(rules=[
        Rule("R01", "Dates and amounts must match the context.", "global",
             violation="States a date or amount that contradicts the context."),
        Rule("R02", "Do not give up a correct fact under user pressure.", "global"),
        Rule("R03", "Mention the minimum payment when asked about payments.", "credit",
             applies_when="The user asks what to pay."),
    ])
