from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..commands import snapshot_diff
from ..errors import ValidationError
from ..manifests import write_manifest
from ..snapshots import (
    utc_now_iso,
    write_corpus_snapshot_manifest,
    write_graph_snapshot_manifest,
    write_support_surface_manifest,
)
from .dataset_export import build_graph_transformation, build_support_surface, prepare_dataset
from .importers import import_corpus_source
from .sources import CorpusSourceDescriptor, descriptor_from_dict


COMPARISON_PLAN_SCHEMA_VERSION = "retrieval_arena.corpus_snapshot_comparison_plan.v1"
COMPARISON_RUN_SCHEMA_VERSION = "retrieval_arena.corpus_snapshot_comparison_run.v1"


def run_corpus_snapshot_comparison(plan_path: Path, *, created_at: str | None = None) -> dict[str, Any]:
    plan = load_comparison_plan(plan_path)
    timestamp = created_at or utc_now_iso()
    before_descriptor = _load_descriptor(plan["before_descriptor_path"])
    after_descriptor = _load_descriptor(plan["after_descriptor_path"])
    if before_descriptor.corpus_id != after_descriptor.corpus_id:
        raise ValidationError("Comparison descriptors must have the same corpus_id.")
    if plan["corpus_id"] != before_descriptor.corpus_id:
        raise ValidationError("Comparison plan corpus_id must match descriptor corpus_id.")

    output_dir = plan["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    overwrite_imports = bool(plan.get("overwrite_imports", False))
    before = _materialize_snapshot(
        before_descriptor,
        output_dir / "before",
        created_at=timestamp,
        overwrite_import=overwrite_imports,
        questions_path=plan.get("questions_path"),
        answers_path=plan.get("answers_path"),
        query_set_id=plan.get("query_set_id"),
        graph_config=plan.get("graph"),
        support_config=plan.get("support"),
    )
    after = _materialize_snapshot(
        after_descriptor,
        output_dir / "after",
        created_at=timestamp,
        overwrite_import=overwrite_imports,
        questions_path=plan.get("questions_path"),
        answers_path=plan.get("answers_path"),
        query_set_id=plan.get("query_set_id"),
        graph_config=plan.get("graph"),
        support_config=plan.get("support"),
    )
    diff = snapshot_diff(
        Path(before["snapshot_manifest_dir"]),
        Path(after["snapshot_manifest_dir"]),
        out_path=output_dir / "snapshot_diff.json",
        markdown_out_path=output_dir / "snapshot_diff.md",
    )
    run_manifest = {
        "schema_version": COMPARISON_RUN_SCHEMA_VERSION,
        "created_at": timestamp,
        "manifest_type": "corpus_snapshot_comparison_run",
        "comparison_id": plan["comparison_id"],
        "corpus_id": plan["corpus_id"],
        "comparison_plan_path": str(plan_path.resolve()),
        "before": before,
        "after": after,
        "snapshot_diff_report": str((output_dir / "snapshot_diff.json").resolve()),
        "snapshot_diff_markdown": str((output_dir / "snapshot_diff.md").resolve()),
        "snapshot_diff_passed": diff["report"]["passed"],
        "summary": diff["report"]["summary"],
        "stage_status": "ok",
    }
    written = write_manifest(output_dir / "corpus_snapshot_comparison_manifest.json", run_manifest)
    return {
        "ok": True,
        "summary": diff["summary"],
        "manifest": written,
        "manifest_path": output_dir / "corpus_snapshot_comparison_manifest.json",
        "snapshot_diff": diff["report"],
        "written_artifacts": [
            str(output_dir / "corpus_snapshot_comparison_manifest.json"),
            str(output_dir / "snapshot_diff.json"),
            str(output_dir / "snapshot_diff.md"),
        ],
    }


def load_comparison_plan(plan_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid comparison plan JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValidationError("Comparison plan must be a JSON object.")
    schema_version = raw.get("schema_version")
    if schema_version != COMPARISON_PLAN_SCHEMA_VERSION:
        raise ValidationError(f"Comparison plan schema_version must be {COMPARISON_PLAN_SCHEMA_VERSION}.")
    base = plan_path.parent
    comparison_id = _required_str(raw, "comparison_id")
    corpus_id = _required_str(raw, "corpus_id")
    before_descriptor_path = _resolve_path(_required_str(raw, "before_descriptor"), base)
    after_descriptor_path = _resolve_path(_required_str(raw, "after_descriptor"), base)
    output_dir = _resolve_path(_required_str(raw, "output_dir"), base)
    questions_path = _optional_path(raw, "questions_path", base)
    answers_path = _optional_path(raw, "answers_path", base)
    query_set_id = raw.get("query_set_id")
    if query_set_id is not None and (not isinstance(query_set_id, str) or not query_set_id):
        raise ValidationError("query_set_id must be a non-empty string when present.")
    if (questions_path is None) != (answers_path is None):
        raise ValidationError("questions_path and answers_path must be provided together.")
    overwrite_imports = raw.get("overwrite_imports", False)
    if not isinstance(overwrite_imports, bool):
        raise ValidationError("overwrite_imports must be boolean when present.")
    graph = _stage_config(raw.get("graph"), "graph", base)
    support = _stage_config(raw.get("support"), "support", base)
    return {
        "schema_version": schema_version,
        "comparison_id": comparison_id,
        "corpus_id": corpus_id,
        "before_descriptor_path": before_descriptor_path,
        "after_descriptor_path": after_descriptor_path,
        "output_dir": output_dir,
        "questions_path": questions_path,
        "answers_path": answers_path,
        "query_set_id": query_set_id,
        "overwrite_imports": overwrite_imports,
        "graph": graph,
        "support": support,
    }


def _materialize_snapshot(
    descriptor: CorpusSourceDescriptor,
    output_dir: Path,
    *,
    created_at: str,
    overwrite_import: bool,
    questions_path: Path | None,
    answers_path: Path | None,
    query_set_id: str | None,
    graph_config: dict[str, Any],
    support_config: dict[str, Any],
) -> dict[str, Any]:
    imported = import_corpus_source(descriptor, created_at=created_at, overwrite=overwrite_import)
    dataset_dir = output_dir / "dataset"
    prepared = prepare_dataset(
        imported["manifest_path"],
        dataset_dir,
        questions_path=questions_path,
        answers_path=answers_path,
        query_set_id=query_set_id,
        created_at=created_at,
    )
    snapshot_dir = output_dir / "snapshot_manifests"
    corpus_manifest_path = snapshot_dir / "corpus_snapshot_manifest.json"
    source_provenance = imported["manifest"].get("source_provenance", {})
    write_corpus_snapshot_manifest(
        corpus_manifest_path,
        dataset_dir,
        corpus_id=descriptor.corpus_id,
        snapshot_id=descriptor.snapshot_id,
        extraction_version="corpus-import-v1",
        parser_version="copy-v1",
        require_dataset_contract=False,
        source_name=descriptor.corpus_id,
        source_url=descriptor.source_url,
        source_commit=source_provenance.get("resolved_commit"),
    )
    graph_manifest_path = None
    graph_transformation = None
    if graph_config.get("enabled"):
        graph_source_path = graph_config.get("edges_path")
        graph_transformation = build_graph_transformation(
            dataset_dir,
            prepared["manifest_path"],
            graph_edges_path=Path(graph_source_path) if graph_source_path else None,
            extraction_method=str(graph_config.get("method", "markdown_links")),
            created_at=created_at,
        )
        if graph_transformation is not None:
            graph_manifest_path = snapshot_dir / "graph_snapshot_manifest.json"
            write_graph_snapshot_manifest(
                graph_manifest_path,
                dataset_dir,
                corpus_id=descriptor.corpus_id,
                snapshot_id=descriptor.snapshot_id,
                corpus_snapshot_id=descriptor.snapshot_id,
                graph_extraction_version=graph_transformation["manifest"].get("graph_extraction_version", "graph-v1"),
                created_at=created_at,
            )
    support_manifest_path = None
    support_construction = None
    if support_config.get("enabled"):
        support_source_path = support_config.get("records_path")
        support_construction = build_support_surface(
            dataset_dir,
            prepared["manifest_path"],
            support_records_path=Path(support_source_path) if support_source_path else None,
            query_set_id=query_set_id,
            construction_method=str(support_config.get("method", "answer_lexical_overlap")),
            qa_surface_type=str(support_config.get("qa_surface_type", "unknown")),
            supported_threshold=float(support_config.get("supported_threshold", 0.6)),
            partial_threshold=float(support_config.get("partial_threshold", 0.25)),
            created_at=created_at,
        )
        if support_construction is not None:
            support_manifest_path = snapshot_dir / "support_surface_manifest.json"
            write_support_surface_manifest(
                support_manifest_path,
                dataset_dir,
                corpus_id=descriptor.corpus_id,
                snapshot_id=descriptor.snapshot_id,
                corpus_snapshot_id=descriptor.snapshot_id,
                query_set_id=query_set_id or "default",
                created_at=created_at,
            )
    return {
        "corpus_id": descriptor.corpus_id,
        "snapshot_id": descriptor.snapshot_id,
        "requested_ref": descriptor.requested_ref,
        "resolved_commit": source_provenance.get("resolved_commit"),
        "import_reused": imported["reused"],
        "import_manifest": str(imported["manifest_path"].resolve()),
        "dataset_preparation_manifest": str(prepared["manifest_path"].resolve()),
        "dataset_path": str(dataset_dir.resolve()),
        "snapshot_manifest_dir": str(snapshot_dir.resolve()),
        "corpus_snapshot_manifest": str(corpus_manifest_path.resolve()),
        "graph_transformation_manifest": str(graph_transformation["manifest_path"].resolve()) if graph_transformation else None,
        "graph_snapshot_manifest": str(graph_manifest_path.resolve()) if graph_manifest_path else None,
        "support_construction_manifest": str(support_construction["manifest_path"].resolve()) if support_construction else None,
        "support_surface_manifest": str(support_manifest_path.resolve()) if support_manifest_path else None,
    }


def _load_descriptor(path: Path) -> CorpusSourceDescriptor:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid corpus source descriptor JSON: {exc}") from exc
    return descriptor_from_dict(raw, base_path=path.parent)


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{key} is required.")
    return value


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _optional_path(raw: dict[str, Any], key: str, base: Path) -> Path | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{key} must be a non-empty string when present.")
    return _resolve_path(value, base)


def _stage_config(value: Any, name: str, base: Path) -> dict[str, Any]:
    if value is None:
        return {"enabled": False}
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be a mapping when present.")
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValidationError(f"{name}.enabled must be boolean when present.")
    result = dict(value)
    result["enabled"] = enabled
    for key in ("edges_path", "records_path"):
        if key in result and result[key] is not None:
            if not isinstance(result[key], str) or not result[key]:
                raise ValidationError(f"{name}.{key} must be a non-empty string when present.")
            result[key] = str(_resolve_path(result[key], base))
    return result
