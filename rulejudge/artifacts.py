"""WP1 — immutable artifact bundles with an atomically switched manifest.

Only runtime/*.json is opened by eval. Jury evidence can remain private or be removed.
File hashes are SHA-256 of the exact UTF-8 bytes on disk (including the final newline).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path, PurePosixPath

from .contracts import (LABELS, MAX_RULES, TOPICS, canonical_json, head_from_dict,
                        rulebook_from_dict)

RUNTIME_FILES = ("rulebook.json", "head.json", "priors.json", "assessor_config.json")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Invalid bundle path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or any(p in (".", "..") for p in relative.parts):
        raise ValueError("Bundle path must be relative and cannot traverse parents")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Bundle path escapes artifact root")
    return path


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            tmp = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def _json_bytes(value) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


class ArtifactStore:
    def __init__(self, root: str):
        self.root = root

    def write_bundle(self, mode: str, runtime: dict, jury: dict, meta: dict) -> str:
        """Write a full, uncommitted bundle; only commit() changes the active manifest."""
        if mode not in ("prior_only", "learned"):
            raise ValueError("Invalid bundle mode")
        if set(runtime) != set(RUNTIME_FILES):
            raise ValueError("Bundle must contain exactly the runtime files")
        if "explanation.md" not in jury or any(not isinstance(v, str) for v in jury.values()):
            raise ValueError("Jury files must be text and include explanation.md")
        if not meta.get("run_id") or not meta.get("train_fingerprint"):
            raise ValueError("Bundle metadata requires run_id and train_fingerprint")
        root = Path(self.root)
        bundle_id = uuid.uuid4().hex
        bundle = _safe_path(root, f"bundles/{bundle_id}")
        bundle.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for name, value in runtime.items():
            relative = f"runtime/{name}"
            content = _json_bytes(value)
            _atomic_bytes(_safe_path(bundle, relative), content)
            hashes[relative] = _sha(content)
        for name, text in jury.items():
            relative = f"jury/{name}"
            content = text.encode("utf-8")
            _atomic_bytes(_safe_path(bundle, relative), content)
            hashes[relative] = _sha(content)
        manifest = dict(meta, bundle=f"bundles/{bundle_id}", mode=mode, files=hashes)
        _atomic_bytes(bundle / "manifest.json", _json_bytes(manifest))
        return bundle_id

    def commit(self, bundle_id: str) -> None:
        """Verify the completed bundle, then atomically switch the root manifest."""
        if not isinstance(bundle_id, str) or len(bundle_id) != 32 or any(c not in "0123456789abcdef" for c in bundle_id):
            raise ValueError("Invalid bundle id")
        root = Path(self.root)
        bundle = _safe_path(root, f"bundles/{bundle_id}")
        try:
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            if manifest["bundle"] != f"bundles/{bundle_id}":
                raise ValueError("Bundle manifest points elsewhere")
            for name, digest in manifest["files"].items():
                if _sha(_safe_path(bundle, name).read_bytes()) != digest:
                    raise ValueError(f"SHA-256 mismatch: {name}")
            self._load_manifest(manifest)
            _atomic_bytes(root / "manifest.json", _json_bytes(manifest))
        except (OSError, KeyError, TypeError) as error:
            raise ValueError(f"Cannot commit artifact bundle: {error}") from error

    def _load_manifest(self, manifest: dict) -> dict:
        try:
            if manifest["mode"] not in ("learned", "prior_only"):
                raise ValueError("Invalid bundle mode")
            bundle_name = manifest["bundle"]
            if not bundle_name.startswith("bundles/") or len(PurePosixPath(bundle_name).parts) != 2:
                raise ValueError("Invalid bundle location")
            bundle = _safe_path(Path(self.root), bundle_name)
            runtime = {}
            for name in RUNTIME_FILES:
                relative = f"runtime/{name}"
                content = _safe_path(bundle, relative).read_bytes()
                if _sha(content) != manifest["files"][relative]:
                    raise ValueError(f"SHA-256 mismatch: {relative}")
                runtime[name] = json.loads(content)
                if not isinstance(runtime[name], dict):
                    raise ValueError(f"Expected an object: {name}")
            head = head_from_dict(runtime["head.json"])
            if head.kind not in ("prior", "fixed", "fitted") or type(head.tau) is not int or head.tau not in (1, 2, 3):
                raise ValueError("Invalid head parameters")
            if not isinstance(head.disabled, list) or any(not isinstance(r, str) for r in head.disabled):
                raise ValueError("Invalid disabled rules")
            priors = runtime["priors.json"]
            if priors["global"] not in LABELS or not isinstance(priors["by_topic"], dict):
                raise ValueError("Invalid priors")
            if any(t not in TOPICS or label not in LABELS for t, label in priors["by_topic"].items()):
                raise ValueError("Invalid topic priors")
            rulebook = rulebook_from_dict(runtime["rulebook.json"])
            ids = [r.rule_id for r in rulebook.rules]
            if len(ids) > MAX_RULES or len(ids) != len(set(ids)):
                raise ValueError("Invalid rulebook IDs or count")
            if any(not r.rule_id or not isinstance(r.text, str) or not r.text.strip()
                   or r.scope not in ("global", *TOPICS) for r in rulebook.rules):
                raise ValueError("Invalid rulebook rule")
            if manifest["mode"] == "prior_only" and (rulebook.rules or head.kind != "prior"):
                raise ValueError("Inconsistent prior-only bundle")
            return {"mode": manifest["mode"], "manifest": manifest, "rulebook": rulebook,
                    "head": head, "priors": priors, "assessor_config": runtime["assessor_config.json"]}
        except (OSError, KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"Missing or corrupt artifact bundle: {error}") from error

    def load_runtime(self) -> dict:
        """Verify committed runtime hashes and deserialize; never open jury files."""
        try:
            manifest = json.loads((Path(self.root) / "manifest.json").read_text(encoding="utf-8"))
            return self._load_manifest(manifest)
        except (OSError, TypeError) as error:
            raise ValueError(f"Missing or corrupt artifact manifest: {error}") from error
