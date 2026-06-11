from __future__ import annotations

import json
import os
import platform
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import __version__
from .git_provenance import describe_git_provenance
from .hashing import directory_inventory, sha256_file
from .manifests import canonical_manifest_json, read_manifest
from .schemas import read_jsonl


MEASUREMENTS_SCHEMA_VERSION = "retrieval_arena.systems_measurements.v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def timed_stage(stage_name: str, *, artifact_references: list[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
    started = utc_now_iso()
    start = time.perf_counter()
    metric = {
        "stage_name": stage_name,
        "started_at": started,
        "completed_at": None,
        "duration_seconds": None,
        "status": "running",
        "artifact_references": artifact_references or [],
    }
    try:
        yield metric
    except Exception:
        metric["completed_at"] = utc_now_iso()
        metric["duration_seconds"] = time.perf_counter() - start
        metric["status"] = "failed"
        raise
    else:
        metric["completed_at"] = utc_now_iso()
        metric["duration_seconds"] = time.perf_counter() - start
        metric["status"] = "completed"


def collect_systems_measurements(
    out_dir: Path,
    *,
    snapshot_dir: Path | None = None,
    run_dir: Path | None = None,
    snapshot_diff_json: Path | None = None,
    drift_jsonl: Path | None = None,
    drift_summary_json: Path | None = None,
    audit_jsonl: Path | None = None,
    audit_summary_json: Path | None = None,
    created_at: str | None = None,
    stage_runtime_metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    timestamp = created_at or utc_now_iso()
    unavailable: list[dict[str, str]] = []
    workload = _empty_workload_metrics()
    artifact_metrics = _empty_artifact_metrics()

    _measure_snapshot(snapshot_dir, workload, artifact_metrics, unavailable)
    _measure_run(run_dir, workload, artifact_metrics, unavailable)
    _measure_snapshot_diff(snapshot_diff_json, workload, artifact_metrics, unavailable)
    _measure_drift(drift_jsonl, drift_summary_json, workload, artifact_metrics, unavailable)
    _measure_audit(audit_jsonl, audit_summary_json, workload, artifact_metrics, unavailable)

    _add_ratios(workload, artifact_metrics)
    report = {
        "schema_version": MEASUREMENTS_SCHEMA_VERSION,
        "created_at": timestamp,
        "workload_metrics": workload,
        "artifact_metrics": artifact_metrics,
        "stage_runtime_metrics": _normalize_stage_runtime_metrics(stage_runtime_metrics or []),
        "environment": environment_metadata(),
        "unavailable_metrics": sorted(unavailable, key=lambda item: (item["artifact_family"], item["metric"])),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "systems_measurements.json"
    md_path = out_dir / "systems_measurements.md"
    json_path.write_text(canonical_manifest_json(report), encoding="utf-8")
    md_path.write_text(render_measurements_markdown(report), encoding="utf-8")
    report["written_artifacts"] = [str(json_path), str(md_path)]
    return report


def environment_metadata() -> dict[str, Any]:
    memory = _available_memory()
    root = Path(__file__).resolve().parent.parent
    return {
        "os": os.name,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "available_memory_bytes": memory.get("available_memory_bytes"),
        "available_memory_unavailable_reason": memory.get("reason"),
        "retrieval_arena_version": __version__,
        "retrieval_arena_git_provenance": describe_git_provenance(root),
    }


def render_measurements_markdown(report: dict[str, Any]) -> str:
    workload = report["workload_metrics"]
    artifacts = report["artifact_metrics"]
    ratios = artifacts["ratios"]
    lines = [
        "# Systems Measurements",
        "",
        "Portable workload and artifact measurements are primary evidence. Runtime values are environment-qualified secondary evidence.",
        "",
        "## Workload",
        "",
        f"- Pages: {_value(workload['page_count'])}",
        f"- Corpus bytes: {_value(workload['corpus_byte_size'])}",
        f"- Graph nodes / edges: {_value(workload['graph_node_count'])} / {_value(workload['graph_edge_count'])}",
        f"- Queries: {_value(workload['query_count'])}",
        f"- Predictions / item scores: {_value(workload['prediction_row_count'])} / {_value(workload['item_score_row_count'])}",
        f"- Drift rows / audit rows: {_value(workload['drift_row_count'])} / {_value(workload['audit_row_count'])}",
        "",
        "## Artifacts",
        "",
        f"- Input artifacts: {artifacts['input_artifacts']['artifact_count']} files, {artifacts['input_artifacts']['byte_size']} bytes",
        f"- Output artifacts: {artifacts['output_artifacts']['artifact_count']} files, {artifacts['output_artifacts']['byte_size']} bytes",
        f"- Manifests: {artifacts['manifest_artifacts']['artifact_count']} files, {artifacts['manifest_artifacts']['byte_size']} bytes",
        f"- Reports: {artifacts['report_artifacts']['artifact_count']} files, {artifacts['report_artifacts']['byte_size']} bytes",
        "",
        "## Ratios",
        "",
    ]
    for name in sorted(ratios):
        item = ratios[name]
        if item.get("available"):
            lines.append(f"- {name}: {item['value']} ({item['numerator']} / {item['denominator']})")
        else:
            lines.append(f"- {name}: unavailable ({item['reason']})")
    lines.extend(["", "## Runtime", ""])
    if report["stage_runtime_metrics"]:
        for item in report["stage_runtime_metrics"]:
            lines.append(f"- {item['stage_name']}: {item['status']}, {_value(item['duration_seconds'])} seconds")
    else:
        lines.append("- No stage runtime metrics supplied.")
    lines.extend(["", "## Missing Optional Evidence", ""])
    if report["unavailable_metrics"]:
        for item in report["unavailable_metrics"]:
            lines.append(f"- {item['artifact_family']}.{item['metric']}: {item['reason']}")
    else:
        lines.append("- None")
    lines.extend(["", "No efficiency score, overall scale score, or composite systems metric is computed."])
    return "\n".join(lines).rstrip() + "\n"


def _empty_workload_metrics() -> dict[str, Any]:
    return {
        "selected_file_count": None,
        "normalized_document_count": None,
        "page_count": None,
        "corpus_byte_size": None,
        "graph_node_count": None,
        "graph_edge_count": None,
        "weak_component_count": None,
        "strong_component_count": None,
        "largest_component_size": None,
        "support_target_count": None,
        "supported_query_count": None,
        "query_count": None,
        "retrieved_context_count_per_query": {},
        "prediction_row_count": None,
        "item_score_row_count": None,
        "action_trace_count": None,
        "action_trace_step_count": None,
        "snapshot_diff_counts": {
            "added_pages": None,
            "removed_pages": None,
            "changed_pages": None,
            "added_graph_edges": None,
            "removed_graph_edges": None,
            "added_support_targets": None,
            "removed_support_targets": None,
            "changed_support_targets": None,
        },
        "drift_row_count": None,
        "audit_row_count": None,
        "audit_cause_label_counts": {},
        "evidence_availability_counts": {},
    }


def _empty_artifact_metrics() -> dict[str, Any]:
    return {
        "input_artifacts": _artifact_group(),
        "output_artifacts": _artifact_group(),
        "manifest_artifacts": _artifact_group(),
        "report_artifacts": _artifact_group(),
        "jsonl_row_counts": {},
        "manifest_size_bytes": {},
        "report_size_bytes": {},
        "ratios": {},
    }


def _artifact_group() -> dict[str, Any]:
    return {"artifact_count": 0, "byte_size": 0, "artifacts": []}


def _measure_snapshot(
    snapshot_dir: Path | None,
    workload: dict[str, Any],
    artifact_metrics: dict[str, Any],
    unavailable: list[dict[str, str]],
) -> None:
    if snapshot_dir is None:
        _unavailable(unavailable, "snapshot", "snapshot_metrics", "snapshot_dir_not_provided")
        return
    if not snapshot_dir.exists():
        _unavailable(unavailable, "snapshot", "snapshot_metrics", "snapshot_dir_missing")
        return
    _add_directory_artifacts(snapshot_dir, artifact_metrics, default_group="input_artifacts")
    corpus = _read_optional_manifest(snapshot_dir / "corpus_snapshot_manifest.json")
    graph = _read_optional_manifest(snapshot_dir / "graph_snapshot_manifest.json")
    support = _read_optional_manifest(snapshot_dir / "support_surface_manifest.json")
    if corpus:
        workload["selected_file_count"] = _number(corpus.get("file_count"))
        workload["normalized_document_count"] = _number(corpus.get("file_count"))
        workload["page_count"] = _number(corpus.get("page_count"))
        workload["corpus_byte_size"] = _number(corpus.get("corpus_size_bytes"))
    else:
        _unavailable(unavailable, "snapshot", "corpus_manifest", "corpus_snapshot_manifest_missing")
    if graph:
        workload["graph_node_count"] = _number(graph.get("node_count"))
        workload["graph_edge_count"] = _number(graph.get("edge_count"))
        metrics = graph.get("graph_metrics") if isinstance(graph.get("graph_metrics"), dict) else {}
        workload["weak_component_count"] = _component_count(metrics, "weak_components")
        workload["strong_component_count"] = _component_count(metrics, "strong_components")
        workload["largest_component_size"] = _number(metrics.get("largest_component_size"))
    else:
        _unavailable(unavailable, "snapshot", "graph_manifest", "graph_snapshot_manifest_missing")
    if support:
        workload["support_target_count"] = _number(support.get("support_target_count"))
        supported = support.get("supported_question_ids")
        workload["supported_query_count"] = len(supported) if isinstance(supported, list) else None
    else:
        _unavailable(unavailable, "snapshot", "support_manifest", "support_surface_manifest_missing")


def _measure_run(
    run_dir: Path | None,
    workload: dict[str, Any],
    artifact_metrics: dict[str, Any],
    unavailable: list[dict[str, str]],
) -> None:
    if run_dir is None:
        _unavailable(unavailable, "run", "run_metrics", "run_dir_not_provided")
        return
    if not run_dir.exists():
        _unavailable(unavailable, "run", "run_metrics", "run_dir_missing")
        return
    input_dir = run_dir / "input"
    if input_dir.exists():
        _add_directory_artifacts(input_dir, artifact_metrics, default_group="input_artifacts")
    for path in sorted(run_dir.iterdir(), key=lambda item: item.name):
        if path.is_file():
            _add_artifact(path, artifact_metrics, "output_artifacts", root=run_dir)
    predictions = _read_optional_jsonl(run_dir / "predictions.jsonl")
    item_scores = _read_optional_jsonl(run_dir / "item_scores.jsonl")
    traces = _read_optional_jsonl(run_dir / "action_traces.jsonl")
    if predictions is None:
        _unavailable(unavailable, "run", "prediction_row_count", "predictions_jsonl_missing")
    else:
        workload["prediction_row_count"] = len(predictions)
        workload["query_count"] = len(predictions)
        workload["retrieved_context_count_per_query"] = {
            str(row.get("question_id")): len(row.get("retrieved_context", [])) if isinstance(row.get("retrieved_context"), list) else 0
            for row in sorted(predictions, key=lambda item: str(item.get("question_id") or ""))
        }
    if item_scores is None:
        _unavailable(unavailable, "run", "item_score_row_count", "item_scores_jsonl_missing")
    else:
        workload["item_score_row_count"] = len(item_scores)
    if traces is None:
        _unavailable(unavailable, "run", "action_trace_metrics", "action_traces_jsonl_missing")
    else:
        workload["action_trace_count"] = len(traces)
        workload["action_trace_step_count"] = sum(_action_step_count(row) for row in traces)


def _measure_snapshot_diff(
    path: Path | None,
    workload: dict[str, Any],
    artifact_metrics: dict[str, Any],
    unavailable: list[dict[str, str]],
) -> None:
    if path is None:
        _unavailable(unavailable, "snapshot_diff", "snapshot_diff_counts", "snapshot_diff_json_not_provided")
        return
    report = _read_optional_json(path)
    if report is None:
        _unavailable(unavailable, "snapshot_diff", "snapshot_diff_counts", "snapshot_diff_json_missing")
        return
    _add_artifact(path, artifact_metrics, "report_artifacts")
    corpus = report.get("corpus_result") if isinstance(report.get("corpus_result"), dict) else {}
    graph = report.get("graph_result") if isinstance(report.get("graph_result"), dict) else {}
    support = report.get("support_surface_result") if isinstance(report.get("support_surface_result"), dict) else {}
    counts = workload["snapshot_diff_counts"]
    counts["added_pages"] = _number(corpus.get("added_file_count"), len(corpus.get("added_files", [])) if isinstance(corpus.get("added_files"), list) else 0)
    counts["removed_pages"] = _number(corpus.get("removed_file_count"), len(corpus.get("removed_files", [])) if isinstance(corpus.get("removed_files"), list) else 0)
    counts["changed_pages"] = _number(corpus.get("changed_file_count"), len(corpus.get("changed_files", [])) if isinstance(corpus.get("changed_files"), list) else 0)
    counts["added_graph_edges"] = _number(graph.get("added_edge_count"), len(graph.get("added_edges", [])) if isinstance(graph.get("added_edges"), list) else 0)
    counts["removed_graph_edges"] = _number(graph.get("removed_edge_count"), len(graph.get("removed_edges", [])) if isinstance(graph.get("removed_edges"), list) else 0)
    counts["added_support_targets"] = _number(support.get("added_target_reference_count"))
    counts["removed_support_targets"] = _number(support.get("removed_target_reference_count"))
    counts["changed_support_targets"] = _number(support.get("changed_question_count"), len(support.get("changed_questions", [])) if isinstance(support.get("changed_questions"), list) else 0)


def _measure_drift(
    drift_jsonl: Path | None,
    summary_json: Path | None,
    workload: dict[str, Any],
    artifact_metrics: dict[str, Any],
    unavailable: list[dict[str, str]],
) -> None:
    if drift_jsonl is None:
        _unavailable(unavailable, "drift", "drift_row_count", "drift_jsonl_not_provided")
    else:
        rows = _read_optional_jsonl(drift_jsonl)
        if rows is None:
            _unavailable(unavailable, "drift", "drift_row_count", "drift_jsonl_missing")
        else:
            workload["drift_row_count"] = len(rows)
            _add_artifact(drift_jsonl, artifact_metrics, "report_artifacts")
    if summary_json is None:
        _unavailable(unavailable, "drift", "drift_summary", "drift_summary_json_not_provided")
    elif summary_json.exists():
        _add_artifact(summary_json, artifact_metrics, "report_artifacts")
    else:
        _unavailable(unavailable, "drift", "drift_summary", "drift_summary_json_missing")


def _measure_audit(
    audit_jsonl: Path | None,
    summary_json: Path | None,
    workload: dict[str, Any],
    artifact_metrics: dict[str, Any],
    unavailable: list[dict[str, str]],
) -> None:
    if audit_jsonl is None:
        _unavailable(unavailable, "audit", "audit_row_count", "audit_jsonl_not_provided")
    else:
        rows = _read_optional_jsonl(audit_jsonl)
        if rows is None:
            _unavailable(unavailable, "audit", "audit_row_count", "audit_jsonl_missing")
        else:
            workload["audit_row_count"] = len(rows)
            counts: dict[str, int] = {}
            availability: dict[str, dict[str, int]] = {}
            for row in rows:
                for label in row.get("cause_labels", []) if isinstance(row.get("cause_labels"), list) else []:
                    counts[str(label)] = counts.get(str(label), 0) + 1
                evidence = row.get("evidence_availability")
                if isinstance(evidence, dict):
                    for name, item in evidence.items():
                        bucket = "available" if isinstance(item, dict) and item.get("available") else "unavailable"
                        availability.setdefault(str(name), {"available": 0, "unavailable": 0})[bucket] += 1
            workload["audit_cause_label_counts"] = dict(sorted(counts.items()))
            workload["evidence_availability_counts"] = dict(sorted(availability.items()))
            _add_artifact(audit_jsonl, artifact_metrics, "report_artifacts")
    if summary_json is None:
        _unavailable(unavailable, "audit", "audit_summary", "audit_summary_json_not_provided")
    else:
        summary = _read_optional_json(summary_json)
        if summary is None:
            _unavailable(unavailable, "audit", "audit_summary", "audit_summary_json_missing")
        else:
            workload["audit_cause_label_counts"] = dict(sorted(summary.get("cause_label_counts", workload["audit_cause_label_counts"]).items()))
            workload["evidence_availability_counts"] = dict(sorted(summary.get("evidence_availability_counts", workload["evidence_availability_counts"]).items()))
            _add_artifact(summary_json, artifact_metrics, "report_artifacts")


def _add_directory_artifacts(root: Path, artifact_metrics: dict[str, Any], *, default_group: str) -> None:
    for item in directory_inventory(root):
        _add_artifact(root / item["path"], artifact_metrics, _group_for_path(root / item["path"], default_group), root=root, precomputed=item)


def _add_artifact(
    path: Path,
    artifact_metrics: dict[str, Any],
    group_name: str,
    *,
    root: Path | None = None,
    precomputed: dict[str, Any] | None = None,
) -> None:
    if not path.exists() or not path.is_file():
        return
    item = precomputed or {"path": path.name if root is None else path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    reference = {"path": item["path"] if root is not None else str(path), "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
    group = artifact_metrics[group_name]
    if any(existing["path"] == reference["path"] and existing["sha256"] == reference["sha256"] for existing in group["artifacts"]):
        return
    group["artifacts"].append(reference)
    group["artifacts"].sort(key=lambda value: value["path"])
    group["artifact_count"] = len(group["artifacts"])
    group["byte_size"] = sum(int(value["size_bytes"]) for value in group["artifacts"])
    suffix = path.suffix.lower()
    name = reference["path"]
    if path.name.endswith(".jsonl"):
        rows = _read_optional_jsonl(path)
        if rows is not None:
            artifact_metrics["jsonl_row_counts"][name] = len(rows)
    if suffix == ".json" and ("manifest" in path.name or path.name.endswith("_summary.json")):
        artifact_metrics["manifest_size_bytes"][name] = path.stat().st_size
    if suffix in {".json", ".jsonl", ".md", ".csv"} and ("report" in path.name or "drift" in path.name or "audit" in path.name or "diff" in path.name):
        artifact_metrics["report_size_bytes"][name] = path.stat().st_size


def _group_for_path(path: Path, default_group: str) -> str:
    name = path.name
    if "manifest" in name:
        return "manifest_artifacts"
    if name.endswith((".md", ".jsonl")) or "summary" in name or "diff" in name or "audit" in name or "drift" in name:
        return "report_artifacts"
    return default_group


def _add_ratios(workload: dict[str, Any], artifact_metrics: dict[str, Any]) -> None:
    page_count = workload.get("page_count")
    query_count = workload.get("query_count")
    edge_count = workload.get("graph_edge_count")
    audit_rows = workload.get("audit_row_count")
    artifact_bytes = sum(artifact_metrics[group]["byte_size"] for group in ("input_artifacts", "output_artifacts", "manifest_artifacts", "report_artifacts"))
    report_bytes = artifact_metrics["report_artifacts"]["byte_size"]
    ratios = artifact_metrics["ratios"]
    ratios["bytes_per_page"] = _ratio(workload.get("corpus_byte_size"), page_count)
    ratios["edges_per_page"] = _ratio(edge_count, page_count)
    ratios["report_bytes_per_query"] = _ratio(report_bytes, query_count)
    ratios["artifact_bytes_per_page"] = _ratio(artifact_bytes, page_count)
    ratios["audit_rows_per_query"] = _ratio(audit_rows, query_count)


def _ratio(numerator: Any, denominator: Any) -> dict[str, Any]:
    if not isinstance(numerator, int | float) or not isinstance(denominator, int | float) or denominator == 0:
        return {"available": False, "reason": "missing_or_zero_denominator", "numerator": numerator, "denominator": denominator}
    return {"available": True, "value": numerator / denominator, "numerator": numerator, "denominator": denominator}


def _normalize_stage_runtime_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "stage_name": str(row.get("stage_name", row.get("name", "unknown"))),
                "started_at": row.get("started_at"),
                "completed_at": row.get("completed_at"),
                "duration_seconds": row.get("duration_seconds"),
                "status": str(row.get("status", "unknown")),
                "artifact_references": row.get("artifact_references", []),
            }
        )
    return sorted(normalized, key=lambda item: item["stage_name"])


def _component_count(metrics: dict[str, Any], key: str) -> int | None:
    value = metrics.get(key)
    if isinstance(value, list):
        return len(value)
    return _number(value)


def _read_optional_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_manifest(path, verify_hash=False)


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    return read_jsonl(path)


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _action_step_count(row: dict[str, Any]) -> int:
    actions = row.get("actions")
    if isinstance(actions, list):
        return len(actions)
    steps = row.get("steps")
    if isinstance(steps, list):
        return len(steps)
    return 0


def _available_memory() -> dict[str, Any]:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int):
                return {"available_memory_bytes": pages * page_size}
        except (OSError, ValueError):
            pass
    return {"available_memory_bytes": None, "reason": "standard_library_available_memory_unavailable"}


def _number(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _unavailable(unavailable: list[dict[str, str]], artifact_family: str, metric: str, reason: str) -> None:
    unavailable.append({"artifact_family": artifact_family, "metric": metric, "reason": reason})


def _value(value: Any) -> str:
    return "n/a" if value is None else str(value)
