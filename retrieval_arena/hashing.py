from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def directory_inventory(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sorted(files, key=lambda item: item["path"])


def sha256_directory(root: Path) -> str:
    return sha256_json({"files": directory_inventory(root)})


def sha256_jsonl(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            digest.update(canonical_json_bytes(value))
            digest.update(b"\n")
    return digest.hexdigest()
