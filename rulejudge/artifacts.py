"""WP1 — artifact bundles under /artifacts with an atomically switched manifest.

Layout:
  /artifacts/manifest.json                 -> {"bundle": "bundles/<id>", "mode": ..., "run_id": ...,
                                              "train_fingerprint": ..., "files": {name: sha256}}
  /artifacts/bundles/<id>/runtime/*.json   -> RUNTIME_FILES only (eval reads ONLY these)
  /artifacts/bundles/<id>/jury/*           -> explanation.md and audit files (eval never reads)
  /artifacts/cache/                        -> adapt-time LLM cache (not used by eval)
"""
from __future__ import annotations

RUNTIME_FILES = ("rulebook.json", "head.json", "priors.json", "assessor_config.json")


class ArtifactStore:
    def __init__(self, root: str):
        self.root = root

    def write_bundle(self, mode: str, runtime: dict, jury: dict, meta: dict) -> str:
        """Write a complete bundle and return its id. Does not switch the manifest.

        runtime: {filename in RUNTIME_FILES: JSON-serialisable object}
        jury:    {relative filename: str content}  (explanation.md is required)
        meta:    {"run_id", "train_fingerprint", ...} copied into the manifest on commit.
        """
        raise NotImplementedError

    def commit(self, bundle_id: str) -> None:
        """Atomically (temp file + os.replace) point manifest.json at bundle_id."""
        raise NotImplementedError

    def load_runtime(self) -> dict:
        """Read manifest + RUNTIME_FILES of the committed bundle; verify sha256 of each file.

        Returns {"mode", "manifest", "rulebook": Rulebook|None, "head": HeadParams,
                 "priors": dict, "assessor_config": dict}. Raise ValueError if missing/corrupt.
        """
        raise NotImplementedError
