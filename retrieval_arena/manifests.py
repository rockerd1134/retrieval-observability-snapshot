from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import ValidationError
from .hashing import sha256_json


SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_REQUIRED_FIELDS = ("schema_version", "created_at", "manifest_hash")


def manifest_to_dict(manifest: Any) -> dict[str, Any]:
    if is_dataclass(manifest) and not isinstance(manifest, type):
        manifest = asdict(manifest)
    if not isinstance(manifest, dict):
        raise ValidationError("Manifest must be a JSON object.")
    return dict(manifest)


def canonical_manifest_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def manifest_payload_for_hash(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return payload


def compute_manifest_hash(manifest: dict[str, Any]) -> str:
    return sha256_json(manifest_payload_for_hash(manifest))


def finalize_manifest(manifest: Any) -> dict[str, Any]:
    finalized = manifest_to_dict(manifest)
    finalized["manifest_hash"] = compute_manifest_hash(finalized)
    return finalized


def validate_manifest(
    manifest: Any,
    *,
    required_fields: Iterable[str] = DEFAULT_REQUIRED_FIELDS,
    verify_hash: bool = False,
) -> dict[str, Any]:
    value = manifest_to_dict(manifest)
    for field in required_fields:
        if field not in value:
            raise ValidationError(f"Manifest missing required field: {field}")
        if value[field] in (None, ""):
            raise ValidationError(f"Manifest field must be non-empty: {field}")
    manifest_hash = value.get("manifest_hash")
    if not isinstance(manifest_hash, str) or not SHA256_HEX_RE.fullmatch(manifest_hash):
        raise ValidationError("Manifest manifest_hash must be a 64-character lowercase SHA-256 hex digest.")
    if verify_hash and manifest_hash != compute_manifest_hash(value):
        raise ValidationError("Manifest manifest_hash does not match manifest payload.")
    return value


def write_manifest(path: Path, manifest: Any, *, finalize: bool = True) -> dict[str, Any]:
    value = finalize_manifest(manifest) if finalize else manifest_to_dict(manifest)
    validate_manifest(value, verify_hash=finalize)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_manifest_json(value), encoding="utf-8")
    return value


def read_manifest(path: Path, *, verify_hash: bool = True) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid manifest JSON: {exc}") from exc
    return validate_manifest(value, verify_hash=verify_hash)
