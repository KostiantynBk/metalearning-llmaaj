"""WP5 — deterministic group-aware fitting/selection split."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from .contracts import Item, canonical_json


def split_fs(items: list[Item], s_fraction: float = 0.2, seed: str = "rulejudge-v1") -> tuple[list[str], list[str]]:
    """Return (F_ids, S_ids). Stratify by (topic, label); exact-duplicate content (same
    item_text without id) stays in one partition; ordering by sha256(seed + item_id).
    Both partitions must contain all topics present in items, otherwise return (all_ids, []).

    S targets round(s_fraction * n) per stratum, with a minimum of one for n >= 3.
    Duplicate groups are indivisible: choose the closest group boundary to the
    target (ties favour S), forcing a nonempty S for a positive target. Groups
    spanning conflicting labels share one assignment; a later stratum may promote
    a group to S to meet its minimum. Group constraints make targets approximate.
    """
    if not math.isfinite(s_fraction) or not 0 <= s_fraction <= 1:
        raise ValueError("s_fraction must be finite and between zero and one")
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Item IDs must be unique")

    def order(item_id):
        return hashlib.sha256((seed + item_id).encode("utf-8")).hexdigest(), item_id

    ordered_ids = sorted(ids, key=order)
    # Labels are deliberately excluded, so contradictory duplicates cannot leak.
    content_key = {}
    strata = defaultdict(list)
    for item in items:
        key = canonical_json({"topic": item.topic, "context": item.context,
                              "conversation": item.conversation})
        content_key[item.item_id] = key
        strata[(item.topic, item.label)].append(item.item_id)

    assignment = {}  # canonical content -> True for S, False for F
    for stratum in sorted(strata, key=lambda key: (key[0], str(key[1]))):
        members = sorted(strata[stratum], key=order)
        target = max(1 if len(members) >= 3 else 0, round(s_fraction * len(members)))
        # Groups assigned in earlier strata are already committed (even if their
        # member appears later in this stratum's hash order).
        selected = sum(assignment.get(content_key[iid], False) for iid in members)
        group_counts = defaultdict(int)
        for iid in members:
            group_counts[content_key[iid]] += 1
        for iid in members:
            key = content_key[iid]
            if key in assignment:
                continue
            size = group_counts[key]
            use_s = selected < target and abs(selected + size - target) <= abs(selected - target)
            assignment[key] = use_s
            if use_s:
                selected += size
        if target > 0 and selected == 0:
            # A large group must not make us abandon a feasible split merely
            # because every nonempty boundary overshoots the fractional target.
            # Dict insertion order follows the hash ordering, resolving ties.
            closest = min(group_counts, key=lambda key: abs(group_counts[key] - target))
            assignment[closest] = True

    s_ids = [iid for iid in ordered_ids if assignment[content_key[iid]]]
    f_ids = [iid for iid in ordered_ids if not assignment[content_key[iid]]]
    topics = {item.topic for item in items}
    topics_by_id = {item.item_id: item.topic for item in items}
    if {topics_by_id[iid] for iid in f_ids} != topics or {topics_by_id[iid] for iid in s_ids} != topics:
        return ordered_ids, []
    return f_ids, s_ids
