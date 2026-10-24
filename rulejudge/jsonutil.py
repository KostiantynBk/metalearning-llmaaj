"""Shared JSON extraction for model outputs (WP0, canonical). Standard library only."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Optional[Any]:
    """Return the first JSON object/array found in text, or None.

    Tolerates markdown code fences, leading prose and trailing text.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for chunk in candidates:
        chunk = chunk.strip()
        try:
            return json.loads(chunk)
        except ValueError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = chunk.find(opener)
            while start != -1:
                depth, in_str, esc = 0, False, False
                for pos in range(start, len(chunk)):
                    ch = chunk[pos]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    elif ch == '"':
                        in_str = True
                    elif ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(chunk[start:pos + 1])
                            except ValueError:
                                break
                start = chunk.find(opener, start + 1)
    return None
