"""WP1 — organizer I/O contract. Standard library only."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from .contracts import LABELS, TOPICS, Item


def _check_content(context, conversation) -> None:
    if not isinstance(context, dict):
        raise ValueError("context must be a JSON object")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("conversation must be a nonempty JSON array")
    for message in conversation:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Each message must contain exactly role and content")
        if message["role"] not in ("system", "user", "assistant", "tool"):
            raise ValueError("Invalid message role")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("Message content must be a nonempty string")


def load_items(path: str, require_labels: bool) -> list[Item]:
    """Read and validate organizer JSONL; reject malformed/empty data and duplicate IDs."""
    items, seen = [], set()
    required = {"item_id", "topic", "context", "conversation"}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not required <= set(row):
                    raise ValueError("Missing required dataset fields or row is not an object")
                if set(row) - required - {"label", "input_es", "input_en"}:
                    raise ValueError("Unexpected dataset columns")
                iid = row["item_id"]
                if not isinstance(iid, str) or not iid.strip():
                    raise ValueError("item_id must be a nonempty string")
                if iid in seen:
                    raise ValueError(f"Duplicate item_id: {iid}")
                if row["topic"] not in TOPICS:
                    raise ValueError("Unknown topic")
                for field in ("context", "conversation"):
                    if isinstance(row[field], str):
                        row[field] = json.loads(row[field])
                _check_content(row["context"], row["conversation"])
                if require_labels and "label" not in row:
                    raise ValueError("Gold labels are required")
                if "label" in row and row["label"] not in LABELS:
                    raise ValueError("Labels must be exactly good or bad")
                for field in ("input_es", "input_en"):
                    if field in row:
                        value = row[field]
                        value = json.loads(value) if isinstance(value, str) else value
                        if not isinstance(value, dict) or set(value) != {"context", "conversation"}:
                            raise ValueError(f"Invalid {field} structure")
                        _check_content(value["context"], value["conversation"])
                items.append(Item(iid, row["topic"], row["context"], row["conversation"], row.get("label")))
                seen.add(iid)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}: line {line_number}: {error}") from error
    if not items:
        raise ValueError(f"{path}: empty input")
    return items


def write_predictions_atomic(path: str, labels: dict[str, str], expected_ids: list[str]) -> None:
    """Validate and atomically replace a complete item_id,label CSV in input-ID order."""
    if any(not isinstance(iid, str) or not iid.strip() for iid in expected_ids):
        raise ValueError("Expected IDs must be nonempty strings")
    if len(expected_ids) != len(set(expected_ids)) or set(labels) != set(expected_ids):
        raise ValueError("Prediction IDs must match unique expected IDs exactly")
    if any(label not in LABELS for label in labels.values()):
        raise ValueError("Labels must be exactly good or bad")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp", delete=False) as stream:
            tmp = stream.name
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["item_id", "label"])
            writer.writerows((iid, labels[iid]) for iid in expected_ids)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
