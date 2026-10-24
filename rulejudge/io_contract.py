"""WP1 — organizer I/O contract. Standard library only."""
from __future__ import annotations

from .contracts import Item


def load_items(path: str, require_labels: bool) -> list[Item]:
    """Read organizer JSONL (/input/train.jsonl or /input/test.jsonl) into Items.

    Validate like src/common.py: unique non-empty item_id, topic in TOPICS, context is an
    object, conversation a non-empty list of {role, content}; labels good/bad when
    require_labels. Raise ValueError with a clear message on invalid input.
    """
    raise NotImplementedError


def write_predictions_atomic(path: str, labels: dict[str, str], expected_ids: list[str]) -> None:
    """Write CSV with header exactly 'item_id,label' and one row per expected id.

    Validate: set(labels) == set(expected_ids), every label in LABELS. Write to a temp file
    in the same directory, then os.replace (atomic). Never leave a partial file.
    """
    raise NotImplementedError
