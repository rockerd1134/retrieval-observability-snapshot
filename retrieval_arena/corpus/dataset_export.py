from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from ..errors import ValidationError
from ..hashing import sha256_directory, sha256_file, sha256_json, sha256_jsonl
from ..manifests import read_manifest, write_manifest
from ..schemas import validate_dataset
from ..snapshots import utc_now_iso
from ..graph.links import GRAPH_EXTRACTION_VERSION, write_markdown_link_graph
from ..support.surface import SUPPORT_CONSTRUCTION_METHOD, write_answer_overlap_support


DATASET_PREPARATION_SCHEMA_VERSION = "retrieval_arena.dataset_preparation_manifest.v1"
GRAPH_TRANSFORMATION_SCHEMA_VERSION = "retrieval_arena.graph_transformation_manifest.v1"
SUPPORT_CONSTRUCTION_SCHEMA_VERSION = "retrieval_arena.support_construction_manifest.v1"
DATASET_PREPARATION_VERSION = "retrieval_arena.dataset_preparation.copy-v1"


def prepare_dataset(
    import_manifest_path: Path,
    output_path: Path,
    *,
    questions_path: Path | None = None,
    answers_path: Path | None = None,
    query_set_id: str | None = None,
    graph_edges_path: Path | None = None,
    support_records_path: Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    import_manifest = read_manifest(import_manifest_path)
    documents_path = Path(str(import_manifest["documents_path"]))
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(documents_path, output_path / "corpus")
    if questions_path is not None:
        shutil.copy2(questions_path, output_path / "questions.jsonl")
    if answers_path is not None:
        shutil.copy2(answers_path, output_path / "answers.jsonl")
    graph_hash = None
    support_hash = None
    if graph_edges_path is not None:
        shutil.copy2(graph_edges_path, output_path / "graph_edges.csv")
        graph_hash = sha256_file(output_path / "graph_edges.csv")
    if support_records_path is not None:
        shutil.copy2(support_records_path, output_path / "faq_support_audit.jsonl")
        support_hash = sha256_jsonl(output_path / "faq_support_audit.jsonl")
    corpus_inventory = _corpus_inventory(output_path / "corpus")
    normalization_config = {"method": "copy_imported_documents", "version": DATASET_PREPARATION_VERSION}
    chunking_config = {"method": "one_file_per_document", "chunking": "none"}
    preparation_config = {
        "normalization_config": normalization_config,
        "chunking_config": chunking_config,
        "query_set_id": query_set_id,
        "questions_file": "questions.jsonl" if questions_path is not None else None,
        "answers_file": "answers.jsonl" if answers_path is not None else None,
    }
    dataset_identity = {
        "import_manifest_hash": import_manifest["manifest_hash"],
        "corpus_directory_hash": sha256_directory(output_path / "corpus"),
        "questions_file_hash": sha256_jsonl(output_path / "questions.jsonl") if questions_path is not None else None,
        "answers_file_hash": sha256_jsonl(output_path / "answers.jsonl") if answers_path is not None else None,
        "preparation_config_hash": sha256_json(preparation_config),
    }

    validation_status = "not_requested"
    validation_error = None
    if questions_path is not None or answers_path is not None:
        try:
            validate_dataset(output_path)
            validation_status = "ok"
        except ValidationError as exc:
            validation_status = "failed"
            validation_error = str(exc)

    manifest = {
        "schema_version": DATASET_PREPARATION_SCHEMA_VERSION,
        "created_at": created_at or utc_now_iso(),
        "manifest_type": "dataset_preparation",
        "corpus_id": import_manifest["corpus_id"],
        "snapshot_id": import_manifest["snapshot_id"],
        "import_manifest_path": str(import_manifest_path.resolve()),
        "import_manifest_hash": import_manifest["manifest_hash"],
        "input_snapshot_identity_hash": import_manifest.get("snapshot_identity_hash"),
        "dataset_output_path": str(output_path.resolve()),
        "corpus_directory_hash": sha256_directory(output_path / "corpus"),
        "normalization_config": normalization_config,
        "chunking_config": chunking_config,
        "preparation_config": preparation_config,
        "preparation_config_hash": sha256_json(preparation_config),
        "generated_document_ids": [item["doc_id"] for item in corpus_inventory],
        "document_inventory": corpus_inventory,
        "excluded_artifacts": [
            {"path": "graph_edges.csv", "reason": "optional_graph_stage_not_requested"} if graph_edges_path is None else None,
            {"path": "faq_support_audit.jsonl", "reason": "optional_support_stage_not_requested"} if support_records_path is None else None,
        ],
        "output_dataset_identity_hash": sha256_json(dataset_identity),
        "software_lineage": {
            "stage": "dataset_preparation",
            "version": DATASET_PREPARATION_VERSION,
            "import_schema_version": import_manifest.get("schema_version"),
        },
        "query_set_id": query_set_id,
        "questions_file": "questions.jsonl" if questions_path is not None else None,
        "questions_file_hash": sha256_jsonl(output_path / "questions.jsonl") if questions_path is not None else None,
        "answers_file": "answers.jsonl" if answers_path is not None else None,
        "answers_file_hash": sha256_jsonl(output_path / "answers.jsonl") if answers_path is not None else None,
        "graph_edge_file_hash": graph_hash,
        "support_input_hash": support_hash,
        "validation_status": validation_status,
        "validation_error": validation_error,
        "stage_status": "ok" if validation_status != "failed" else "failed",
    }
    manifest["excluded_artifacts"] = [item for item in manifest["excluded_artifacts"] if item is not None]
    written = write_manifest(output_path / "dataset_preparation_manifest.json", manifest)
    if validation_error:
        raise ValidationError(validation_error)
    return {"manifest": written, "manifest_path": output_path / "dataset_preparation_manifest.json", "dataset_path": output_path}


def build_graph_transformation(
    dataset_path: Path,
    dataset_preparation_manifest_path: Path,
    *,
    graph_edges_path: Path | None = None,
    extraction_method: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any] | None:
    preparation = read_manifest(dataset_preparation_manifest_path)
    if graph_edges_path is not None:
        destination = dataset_path / "graph_edges.csv"
        shutil.copy2(graph_edges_path, destination)
        edges = _read_edges(destination)
        nodes = sorted({node for edge in edges for node in edge})
        metrics = {"node_count": len(nodes), "edge_count": len(edges)}
        (dataset_path / "graph_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        graph_artifacts = {
            "edge_file": "graph_edges.csv",
            "edge_file_hash": sha256_file(destination),
            "graph_metrics_file": "graph_metrics.json",
            "graph_metrics_file_hash": sha256_file(dataset_path / "graph_metrics.json"),
            "graph_hash": sha256_json({"edges": [{"source": source, "target": target} for source, target in edges]}),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "graph_metrics": metrics,
        }
        version = "copy-v1"
        resolver = {"source": "provided_graph_edges_csv"}
    elif extraction_method == "markdown_links":
        graph_artifacts = write_markdown_link_graph(dataset_path)
        version = GRAPH_EXTRACTION_VERSION
        resolver = {
            "source": "dataset_corpus_markdown",
            "internal_link_policy": "resolve_relative_absolute_html_and_markdown_links_to_dataset_doc_ids",
        }
    else:
        return None
    manifest = {
        "schema_version": GRAPH_TRANSFORMATION_SCHEMA_VERSION,
        "created_at": created_at or utc_now_iso(),
        "manifest_type": "graph_transformation",
        "corpus_id": preparation["corpus_id"],
        "snapshot_id": preparation["snapshot_id"],
        "dataset_preparation_manifest_path": str(dataset_preparation_manifest_path.resolve()),
        "dataset_preparation_manifest_hash": preparation["manifest_hash"],
        "source_dataset_identity_hash": preparation.get("output_dataset_identity_hash"),
        "graph_extraction_version": version,
        "link_extraction_configuration": {"method": extraction_method},
        "link_resolver_configuration": resolver,
        **graph_artifacts,
        "stage_status": "ok",
    }
    written = write_manifest(dataset_path / "graph_transformation_manifest.json", manifest)
    return {"manifest": written, "manifest_path": dataset_path / "graph_transformation_manifest.json"}


def build_support_surface(
    dataset_path: Path,
    dataset_preparation_manifest_path: Path,
    *,
    support_records_path: Path | None = None,
    query_set_id: str | None = None,
    construction_method: str | None = None,
    qa_surface_type: str = "unknown",
    supported_threshold: float = 0.6,
    partial_threshold: float = 0.25,
    created_at: str | None = None,
) -> dict[str, Any] | None:
    preparation = read_manifest(dataset_preparation_manifest_path)
    if support_records_path is not None:
        destination = dataset_path / "faq_support_audit.jsonl"
        shutil.copy2(support_records_path, destination)
        rows = _read_jsonl(destination)
        targets = sorted({doc_id for row in rows for doc_id in _row_doc_ids(row)})
        support_artifacts = {
            "support_construction_method": "copy-v1",
            "support_audit_configuration": {"source": "provided_support_records_jsonl"},
            "supported_question_ids": sorted(row["question_id"] for row in rows if _row_doc_ids(row)),
            "support_target_doc_ids": targets,
            "support_target_count": len(targets),
        }
    elif construction_method == "answer_lexical_overlap":
        support_artifacts = write_answer_overlap_support(
            dataset_path,
            qa_surface_type=qa_surface_type,
            supported_threshold=supported_threshold,
            partial_threshold=partial_threshold,
        )
        destination = dataset_path / "faq_support_audit.jsonl"
        rows = _read_jsonl(destination)
        support_artifacts["support_audit_configuration"] = {
            "source": "answers_jsonl_and_dataset_corpus",
            "method": SUPPORT_CONSTRUCTION_METHOD,
            "top_k": support_artifacts["top_k"],
            "qa_surface_type": support_artifacts["qa_surface_type"],
            "supported_threshold": support_artifacts["supported_threshold"],
            "partial_threshold": support_artifacts["partial_threshold"],
        }
    else:
        return None
    manifest = {
        "schema_version": SUPPORT_CONSTRUCTION_SCHEMA_VERSION,
        "created_at": created_at or utc_now_iso(),
        "manifest_type": "support_construction",
        "corpus_id": preparation["corpus_id"],
        "snapshot_id": preparation["snapshot_id"],
        "dataset_preparation_manifest_path": str(dataset_preparation_manifest_path.resolve()),
        "dataset_preparation_manifest_hash": preparation["manifest_hash"],
        "source_dataset_identity_hash": preparation.get("output_dataset_identity_hash"),
        "query_set_id": query_set_id or preparation.get("query_set_id"),
        "support_audit_file": "faq_support_audit.jsonl",
        "support_audit_file_hash": sha256_jsonl(destination),
        **support_artifacts,
        "stage_status": "ok",
    }
    written = write_manifest(dataset_path / "support_construction_manifest.json", manifest)
    return {"manifest": written, "manifest_path": dataset_path / "support_construction_manifest.json"}


def _read_edges(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source", "target"]:
            raise ValidationError("graph_edges.csv header must be source,target")
        return sorted((row["source"], row["target"]) for row in reader if row.get("source") and row.get("target"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_doc_ids(row: dict[str, Any]) -> list[str]:
    docs = row.get("top_docs", row.get("support_target_doc_ids", row.get("support_doc_ids", [])))
    result = []
    if isinstance(docs, list):
        for item in docs:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict) and isinstance(item.get("doc_id"), str):
                result.append(item["doc_id"])
    return result


def _corpus_inventory(corpus_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in corpus_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(corpus_dir).as_posix()
        rows.append(
            {
                "path": relative,
                "doc_id": Path(relative).with_suffix("").as_posix() if Path(relative).suffix else relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sorted(rows, key=lambda item: item["path"])
