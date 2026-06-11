from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval_arena.commands import snapshot_diff, snapshot_manifest
from retrieval_arena.corpus.dataset_export import build_graph_transformation, build_support_surface, prepare_dataset
from retrieval_arena.corpus.importers import import_corpus_source
from retrieval_arena.corpus.sources import CorpusSourceDescriptor
from retrieval_arena.manifests import write_manifest
from retrieval_arena.snapshots import utc_now_iso


STUDY_SCHEMA_VERSION = "retrieval_arena.review2026_toy_study.v1"


def run_toy_study(
    before: CorpusSourceDescriptor,
    after: CorpusSourceDescriptor,
    output_dir: Path,
    *,
    before_questions: Path,
    before_answers: Path,
    after_questions: Path | None = None,
    after_answers: Path | None = None,
    before_graph_edges: Path | None = None,
    after_graph_edges: Path | None = None,
    before_support_records: Path | None = None,
    after_support_records: Path | None = None,
    query_set_id: str = "toy-queries",
    created_at: str | None = None,
) -> dict[str, Any]:
    timestamp = created_at or utc_now_iso()
    output_dir.mkdir(parents=True, exist_ok=True)
    before_result = _prepare_snapshot(
        before,
        output_dir / "before",
        questions=before_questions,
        answers=before_answers,
        graph_edges=before_graph_edges,
        support_records=before_support_records,
        query_set_id=query_set_id,
        created_at=timestamp,
    )
    after_result = _prepare_snapshot(
        after,
        output_dir / "after",
        questions=after_questions or before_questions,
        answers=after_answers or before_answers,
        graph_edges=after_graph_edges,
        support_records=after_support_records,
        query_set_id=query_set_id,
        created_at=timestamp,
    )
    diff_result = snapshot_diff(
        Path(before_result["snapshot_manifest_dir"]),
        Path(after_result["snapshot_manifest_dir"]),
        out_path=output_dir / "snapshot_diff.json",
        markdown_out_path=output_dir / "snapshot_diff.md",
    )
    manifest = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "created_at": timestamp,
        "manifest_type": "review2026_toy_study",
        "study_id": "review2026-toy-source-to-snapshot",
        "stages": ["corpus_import", "dataset_preparation", "snapshot_manifest", "snapshot_diff"],
        "before": before_result,
        "after": after_result,
        "snapshot_diff_report": str((output_dir / "snapshot_diff.json").resolve()),
        "snapshot_diff_passed": diff_result["report"]["passed"],
        "stage_status": "ok",
    }
    written = write_manifest(output_dir / "study_manifest.json", manifest)
    return {"manifest": written, "manifest_path": output_dir / "study_manifest.json", "snapshot_diff": diff_result["report"]}


def _prepare_snapshot(
    descriptor: CorpusSourceDescriptor,
    snapshot_output_dir: Path,
    *,
    questions: Path,
    answers: Path,
    graph_edges: Path | None,
    support_records: Path | None,
    query_set_id: str,
    created_at: str,
) -> dict[str, Any]:
    imported = import_corpus_source(descriptor, created_at=created_at)
    dataset_dir = snapshot_output_dir / "dataset"
    prepared = prepare_dataset(
        imported["manifest_path"],
        dataset_dir,
        questions_path=questions,
        answers_path=answers,
        query_set_id=query_set_id,
        created_at=created_at,
    )
    graph = build_graph_transformation(dataset_dir, prepared["manifest_path"], graph_edges_path=graph_edges, created_at=created_at)
    support = build_support_surface(
        dataset_dir,
        prepared["manifest_path"],
        support_records_path=support_records,
        query_set_id=query_set_id,
        created_at=created_at,
    )
    manifest_types = {"corpus"}
    if graph is not None:
        manifest_types.add("graph")
    if support is not None:
        manifest_types.add("support")
    snapshot_dir = snapshot_output_dir / "snapshot_manifests"
    snapshot_result = snapshot_manifest(
        dataset_dir,
        snapshot_dir,
        corpus_id=descriptor.corpus_id,
        snapshot_id=descriptor.snapshot_id,
        query_set_id=query_set_id if "support" in manifest_types else None,
        extraction_version="corpus-import-v1",
        parser_version="copy-v1",
        graph_extraction_version="copy-v1",
        source_name=descriptor.corpus_id,
        source_url=descriptor.source_url,
        source_commit=imported["manifest"]["source_provenance"].get("resolved_commit"),
        manifest_types=manifest_types,
    )
    return {
        "corpus_id": descriptor.corpus_id,
        "snapshot_id": descriptor.snapshot_id,
        "import_manifest": str(imported["manifest_path"].resolve()),
        "dataset_preparation_manifest": str(prepared["manifest_path"].resolve()),
        "graph_transformation_manifest": str(graph["manifest_path"].resolve()) if graph else None,
        "support_construction_manifest": str(support["manifest_path"].resolve()) if support else None,
        "dataset_path": str(dataset_dir.resolve()),
        "snapshot_manifest_dir": str(snapshot_dir.resolve()),
        "snapshot_manifest_paths": snapshot_result["written_artifacts"],
    }
