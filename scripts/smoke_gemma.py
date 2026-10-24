"""Explicit real-API synthetic probe. Run manually; never called by unit tests."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulejudge.contracts import Settings
from rulejudge.gemma_client import GemmaClient


def _load_api_key(root: Path):
    """Environment wins; only read the API-key assignment from a local .env.

    No dependency, interpolation, execution, or printing of file contents.
    """
    if os.environ.get("GEMINI_API_KEY"):
        return
    try:
        lines = (root / ".env").read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, sep, value = line.partition("=")
        if sep and name.strip() == "GEMINI_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].rstrip()
            if value:
                os.environ["GEMINI_API_KEY"] = value
            return


def main():
    _load_api_key(Path(__file__).resolve().parents[1])
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is missing; configure the environment or repo-root .env.", file=sys.stderr)
        return 2
    settings = Settings.from_env()
    settings.time_budget_s = min(settings.time_budget_s, 300)
    settings.cache_dir = ""  # Smoke must exercise actual capabilities, not a replay.
    settings.start_clock()
    client = GemmaClient(settings)
    for model in ("gemma-4-31b-it", "gemma-4-26b-a4b-it"):
        settings.model_assessor = settings.model_proposer = model
        probes = (
            ("json_assessor", "assessor", "Return only JSON.", 'Return {"s":{"R01":0},"e":[]}.'),
            ("high_proposer", "proposer", "Return only JSON.", 'Propose one rule about checking dates. Return {"rules":["..."]}.'),
            ("no_system", "assessor", "", 'Return only JSON: {"ok":true}.'),
        )
        for name, role, system, user in probes:
            result = client.generate(role=role, system=system, user=user, max_output_tokens=256)
            print(json.dumps({"model": model, "probe": name, "ok": result.ok,
                              "error_kind": result.error_kind, "latency_s": round(result.latency_s, 3),
                              "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                              "capability_flags": client.capability_flags.get(model, {})}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
