from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Any

from .errors import ValidationError
from .hashing import directory_inventory, sha256_file, sha256_json, sha256_jsonl
from .manifests import write_manifest
from .schemas import read_jsonl, validate_dataset


SNAPSHOT_SCHEMA_VERSION = "retrieval_arena.snapshot_manifest.v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_id(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{name} is required.")
    return value


def optional_metadata(**metadata: str | None) -> dict[str, str]:
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def corpus_doc_id(relative_path: str) -> str:
    path = Path(relative_path)
    if path.suffix:
        return path.with_suffix("").as_posix()
    return path.as_posix()


def corpus_inventory(corpus_dir: Path) -> list[dict[str, Any]]:
    inventory = []
    for item in directory_inventory(corpus_dir):
        inventory.append(
            {
                "path": item["path"],
                "doc_id": corpus_doc_id(item["path"]),
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
        )
    return inventory


def build_corpus_snapshot_manifest(
    dataset_path: Path,
    *,
    corpus_id: str,
    snapshot_id: str,
    extraction_version: str,
    parser_version: str,
    require_dataset_contract: bool = True,
    created_at: str | None = None,
    source_name: str | None = None,
    source_url: str | None = None,
    source_commit: str | None = None,
    source_release: str | None = None,
    source_timestamp: str | None = None,
) -> dict[str, Any]:
    require_id("corpus_id", corpus_id)
    require_id("snapshot_id", snapshot_id)
    require_id("extraction_version", extraction_version)
    require_id("parser_version", parser_version)
    if require_dataset_contract:
        validate_dataset(dataset_path)

    corpus_dir = dataset_path / "corpus"
    if not corpus_dir.is_dir():
        raise ValidationError(f"Dataset corpus must be a directory: {corpus_dir}")
    files = corpus_inventory(corpus_dir)
    manifest: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": created_at or utc_now_iso(),
        "manifest_type": "corpus_snapshot",
        "corpus_id": corpus_id,
        "snapshot_id": snapshot_id,
        "extraction_version": extraction_version,
        "parser_version": parser_version,
        "page_count": len(files),
        "file_count": len(files),
        "corpus_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "file_inventory": files,
        "content_hash": sha256_json({"files": files}),
    }
    manifest.update(
        optional_metadata(
            source_name=source_name,
            source_url=source_url,
            source_commit=source_commit,
            source_release=source_release,
            source_timestamp=source_timestamp,
        )
    )
    return manifest


def read_graph_edges(edge_path: Path) -> list[tuple[str, str]]:
    if not edge_path.exists():
        raise ValidationError(f"Missing graph edge file: {edge_path}")
    edges: list[tuple[str, str]] = []
    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source", "target"]:
            raise ValidationError("graph_edges.csv header must be source,target")
        for line_number, row in enumerate(reader, start=2):
            source = row.get("source")
            target = row.get("target")
            if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
                raise ValidationError(f"graph_edges.csv line {line_number} requires non-empty source and target.")
            edges.append((source, target))
    return sorted(edges)


def build_graph_snapshot_manifest(
    dataset_path: Path,
    *,
    corpus_id: str,
    snapshot_id: str,
    corpus_snapshot_id: str,
    graph_extraction_version: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    require_id("corpus_id", corpus_id)
    require_id("snapshot_id", snapshot_id)
    require_id("corpus_snapshot_id", corpus_snapshot_id)
    require_id("graph_extraction_version", graph_extraction_version)

    edge_path = dataset_path / "graph_edges.csv"
    edges = read_graph_edges(edge_path)
    nodes = sorted({node for edge in edges for node in edge})
    edge_inventory = [{"source": source, "target": target} for source, target in edges]
    manifest: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": created_at or utc_now_iso(),
        "manifest_type": "graph_snapshot",
        "corpus_id": corpus_id,
        "snapshot_id": snapshot_id,
        "corpus_snapshot_id": corpus_snapshot_id,
        "graph_extraction_version": graph_extraction_version,
        "source_dataset_identity_hash": _dataset_identity_hash(dataset_path),
        "graph_extraction_config": _graph_transformation_config(dataset_path),
        "edge_file": "graph_edges.csv",
        "edge_file_hash": sha256_file(edge_path),
        "edge_inventory": edge_inventory,
        "graph_hash": sha256_json({"edges": edge_inventory}),
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
    metrics_path = dataset_path / "graph_metrics.json"
    if metrics_path.exists():
        try:
            graph_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid graph_metrics.json: {exc}") from exc
        if not isinstance(graph_metrics, dict):
            raise ValidationError("graph_metrics.json must be a JSON object.")
        manifest["graph_metrics_file"] = "graph_metrics.json"
        manifest["graph_metrics_file_hash"] = sha256_file(metrics_path)
        manifest["graph_metrics"] = graph_metrics
        if isinstance(graph_metrics.get("node_count"), int):
            manifest["node_count"] = graph_metrics["node_count"]
        if isinstance(graph_metrics.get("edge_count"), int):
            manifest["edge_count"] = graph_metrics["edge_count"]
    return manifest


def query_set_hash(dataset_path: Path) -> str:
    questions = dataset_path / "questions.jsonl"
    if not questions.exists():
        raise ValidationError(f"Dataset missing required path: {questions}")
    return sha256_jsonl(questions)


def support_doc_ids_from_row(row: dict[str, Any], *, line_number: int) -> list[str]:
    docs = row.get("top_docs", row.get("support_target_doc_ids", row.get("support_doc_ids", [])))
    if not isinstance(docs, list):
        raise ValidationError(f"faq_support_audit.jsonl line {line_number} support docs must be a list.")
    doc_ids: list[str] = []
    for item in docs:
        if isinstance(item, str):
            doc_id = item
        elif isinstance(item, dict):
            doc_id = item.get("doc_id")
        else:
            raise ValidationError(f"faq_support_audit.jsonl line {line_number} support doc entries must be strings or objects.")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValidationError(f"faq_support_audit.jsonl line {line_number} support doc entries require non-empty doc_id.")
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def read_support_targets(path: Path) -> tuple[list[str], dict[str, list[str]], list[str]]:
    if not path.exists():
        raise ValidationError(f"Missing support audit file: {path}")
    rows = read_jsonl(path)
    supported_question_ids: list[str] = []
    support_targets_by_question: dict[str, list[str]] = {}
    all_targets: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValidationError(f"faq_support_audit.jsonl line {line_number} requires non-empty question_id.")
        if question_id in support_targets_by_question:
            raise ValidationError(f"Duplicate question_id in faq_support_audit.jsonl: {question_id}")
        doc_ids = support_doc_ids_from_row(row, line_number=line_number)
        if doc_ids:
            supported_question_ids.append(question_id)
        support_targets_by_question[question_id] = doc_ids
        all_targets.update(doc_ids)
    return sorted(supported_question_ids), dict(sorted(support_targets_by_question.items())), sorted(all_targets)


def support_label_counts(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    counts = Counter(str(row.get("support_label", "unknown")) for row in rows)
    return dict(sorted(counts.items()))


def build_support_surface_manifest(
    dataset_path: Path,
    *,
    corpus_id: str,
    snapshot_id: str,
    corpus_snapshot_id: str,
    query_set_id: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    require_id("corpus_id", corpus_id)
    require_id("snapshot_id", snapshot_id)
    require_id("corpus_snapshot_id", corpus_snapshot_id)
    require_id("query_set_id", query_set_id)

    audit_path = dataset_path / "faq_support_audit.jsonl"
    supported_question_ids, targets_by_question, target_doc_ids = read_support_targets(audit_path)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": created_at or utc_now_iso(),
        "manifest_type": "support_surface",
        "corpus_id": corpus_id,
        "snapshot_id": snapshot_id,
        "corpus_snapshot_id": corpus_snapshot_id,
        "query_set_id": query_set_id,
        "source_dataset_identity_hash": _dataset_identity_hash(dataset_path),
        "support_construction_config": _support_construction_config(dataset_path),
        "query_set_file": "questions.jsonl",
        "query_set_hash": query_set_hash(dataset_path),
        "support_audit_file": "faq_support_audit.jsonl",
        "support_audit_file_hash": sha256_jsonl(audit_path),
        "supported_question_ids": supported_question_ids,
        "support_targets_by_question": targets_by_question,
        "support_target_doc_ids": target_doc_ids,
        "support_target_count": len(target_doc_ids),
        "support_label_counts": support_label_counts(audit_path),
    }


def write_corpus_snapshot_manifest(output_path: Path, dataset_path: Path, **kwargs: Any) -> dict[str, Any]:
    return write_manifest(output_path, build_corpus_snapshot_manifest(dataset_path, **kwargs))


def write_graph_snapshot_manifest(output_path: Path, dataset_path: Path, **kwargs: Any) -> dict[str, Any]:
    return write_manifest(output_path, build_graph_snapshot_manifest(dataset_path, **kwargs))


def write_support_surface_manifest(output_path: Path, dataset_path: Path, **kwargs: Any) -> dict[str, Any]:
    return write_manifest(output_path, build_support_surface_manifest(dataset_path, **kwargs))


def _dataset_identity_hash(dataset_path: Path) -> str | None:
    manifest_path = dataset_path / "dataset_preparation_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid dataset_preparation_manifest.json: {exc}") from exc
    return value.get("output_dataset_identity_hash") if isinstance(value, dict) else None


def _graph_transformation_config(dataset_path: Path) -> dict[str, Any] | None:
    manifest_path = dataset_path / "graph_transformation_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid graph_transformation_manifest.json: {exc}") from exc
    if not isinstance(value, dict):
        return None
    return {
        "graph_extraction_version": value.get("graph_extraction_version"),
        "link_extraction_configuration": value.get("link_extraction_configuration"),
        "link_resolver_configuration": value.get("link_resolver_configuration"),
    }


def _support_construction_config(dataset_path: Path) -> dict[str, Any] | None:
    manifest_path = dataset_path / "support_construction_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid support_construction_manifest.json: {exc}") from exc
    if not isinstance(value, dict):
        return None
    return {
        "support_construction_method": value.get("support_construction_method"),
        "support_audit_configuration": value.get("support_audit_configuration"),
        "query_set_id": value.get("query_set_id"),
    }
