"""Explicit real-API synthetic probe. Run manually; never called by unit tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulejudge.contracts import Settings
from rulejudge.gemma_client import GemmaClient


def main():
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
