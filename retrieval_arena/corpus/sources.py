from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ValidationError


@dataclass(frozen=True)
class CorpusSourceDescriptor:
    corpus_id: str
    snapshot_id: str
    source_type: str
    destination_workspace: Path
    source_url: str | None = None
    source_path: Path | None = None
    requested_ref: str | None = None
    docs_root: str | None = None
    include: tuple[str, ...] = field(default_factory=tuple)
    exclude: tuple[str, ...] = field(default_factory=tuple)


def descriptor_from_dict(raw: dict[str, Any], *, base_path: Path | None = None) -> CorpusSourceDescriptor:
    base = (base_path or Path.cwd()).resolve()
    if not isinstance(raw, dict):
        raise ValidationError("Corpus source descriptor must be a mapping.")
    corpus_id = _required_str(raw, "corpus_id")
    snapshot_id = _required_str(raw, "snapshot_id")
    source_type = _required_str(raw, "source_type")
    if source_type not in {"git", "local"}:
        raise ValidationError("source_type must be 'git' or 'local'.")
    destination_workspace = _path(_required_str(raw, "destination_workspace"), base)
    source_url = raw.get("source_url")
    source_path_raw = raw.get("source_path")
    if source_url is not None and not isinstance(source_url, str):
        raise ValidationError("source_url must be a string when present.")
    source_path = _path(source_path_raw, base) if source_path_raw else None
    if source_type == "local" and source_path is None:
        raise ValidationError("local corpus sources require source_path.")
    if source_type == "git" and source_path is None and not source_url:
        raise ValidationError("git corpus sources require source_path or source_url.")
    requested_ref = raw.get("requested_ref", raw.get("ref"))
    if requested_ref is not None and not isinstance(requested_ref, str):
        raise ValidationError("requested_ref must be a string when present.")
    docs_root = raw.get("docs_root")
    if docs_root is not None and not isinstance(docs_root, str):
        raise ValidationError("docs_root must be a string when present.")
    return CorpusSourceDescriptor(
        corpus_id=corpus_id,
        snapshot_id=snapshot_id,
        source_type=source_type,
        source_url=source_url,
        source_path=source_path,
        requested_ref=requested_ref,
        docs_root=_normalize_docs_root(docs_root),
        include=_string_tuple(raw.get("include", ["**/*"])),
        exclude=_string_tuple(raw.get("exclude", [])),
        destination_workspace=destination_workspace,
    )


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{key} is required.")
    return value


def _path(value: str, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValidationError("include and exclude must be strings or lists of non-empty strings.")
    return tuple(value)


def _normalize_docs_root(value: str | None) -> str | None:
    if not value:
        return None
    normalized = Path(value).as_posix().strip("/")
    return normalized or None
