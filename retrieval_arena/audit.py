from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .drift import compare_retrieval_runs
from .errors import ValidationError
from .hashing import sha256_file
from .manifests import read_manifest, write_manifest
from .schemas import read_jsonl, write_jsonl


AUDIT_ROW_SCHEMA_VERSION = "retrieval_arena.regression_audit_row.v1"
AUDIT_SUMMARY_SCHEMA_VERSION = "retrieval_arena.regression_audit_summary.v1"
EPSILON = 1e-12
CAUSE_LABEL_SCHEMA_VERSION = "retrieval_arena.audit_cause_labels.v1"
CAUSE_LABEL_SCHEMA = {
    "action_trace_changed": "Deterministic or iterative retrieval trace changed.",
    "answer_lexical_score_regressed": "Answer lexical coverage decreased.",
    "corpus_page_added": "A retrieved or support document was added in the corpus snapshot.",
    "corpus_page_changed": "A retrieved or support document changed in the corpus snapshot.",
    "corpus_page_removed": "A retrieved or support document was removed from the corpus snapshot.",
    "distance_to_support_increased": "Minimum observed graph distance to support increased.",
    "evidence_coverage_regressed": "Evidence coverage decreased.",
    "evidence_unavailable": "One or more optional evidence families were unavailable.",
    "graph_edge_added": "A graph edge incident to retrieved or support documents was added.",
    "graph_edge_removed": "A graph edge incident to retrieved or support documents was removed.",
    "rank_changed": "Retained retrieved candidates moved in rank.",
    "replay_manifest_changed": "Replay manifest lineage or references changed.",
    "retrieval_config_changed": "Retrieval configuration identity changed.",
    "retrieval_score_changed": "Retained retrieved candidate scores changed.",
    "retrieved_candidate_changed": "Retrieved candidate set changed.",
    "scoring_config_changed": "Scoring configuration identity changed.",
    "support_exposure_regressed": "Retrieved support exposure decreased.",
    "support_recall_regressed": "Support recall decreased.",
    "support_target_added": "Support target was added for the query.",
    "support_target_changed": "Support target set changed for the query.",
    "support_target_removed": "Support target was removed for the query.",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_regression_audit(
    before_run_dir: Path,
    after_run_dir: Path,
    out_dir: Path,
    *,
    drift_jsonl: Path | None = None,
    drift_summary_json: Path | None = None,
    snapshot_diff_json: Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    timestamp = created_at or utc_now_iso()
    before_manifest = _read_optional_manifest(before_run_dir / "retrieval_replay_manifest.json")
    after_manifest = _read_optional_manifest(after_run_dir / "retrieval_replay_manifest.json")
    _validate_run_manifests_joinable(before_manifest, after_manifest)

    drift_rows, drift_summary = _load_or_build_drift(
        before_run_dir,
        after_run_dir,
        drift_jsonl=drift_jsonl,
        drift_summary_json=drift_summary_json,
        created_at=timestamp,
    )
    snapshot_diff = _read_optional_json(snapshot_diff_json)
    _validate_drift_rows(drift_rows, before_manifest, after_manifest)

    snapshot_index = _snapshot_index(snapshot_diff)
    rows = [
        _audit_row(
            row,
            before_manifest,
            after_manifest,
            snapshot_index,
            created_at=timestamp,
        )
        for row in sorted(drift_rows, key=_drift_sort_key)
    ]
    summary = _audit_summary(
        before_run_dir,
        after_run_dir,
        before_manifest,
        after_manifest,
        drift_summary,
        snapshot_diff,
        rows,
        created_at=timestamp,
        drift_jsonl=drift_jsonl,
        drift_summary_json=drift_summary_json,
        snapshot_diff_json=snapshot_diff_json,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "regression_audit.jsonl"
    summary_path = out_dir / "regression_audit_summary.json"
    markdown_path = out_dir / "regression_audit.md"
    write_jsonl(jsonl_path, rows)
    write_manifest(summary_path, summary)
    markdown_path.write_text(render_audit_markdown(summary, rows), encoding="utf-8")
    return {
        "summary": summary,
        "rows": rows,
        "written_artifacts": [str(jsonl_path), str(summary_path), str(markdown_path)],
    }


def render_audit_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Regression Audit",
        "",
        f"- Queries audited: {summary['query_count']}",
        f"- Queries with labels: {summary['labeled_query_count']}",
        "- Overall drift score: not computed",
        "",
        "## Cause Labels",
        "",
    ]
    if summary["cause_label_counts"]:
        for label, count in sorted(summary["cause_label_counts"].items()):
            lines.append(f"- `{label}`: {count}")
    else:
        lines.append("- None")
    lines.extend(["", "## Evidence Availability", ""])
    for name, counts in sorted(summary["evidence_availability_counts"].items()):
        lines.append(f"- `{name}`: {counts.get('available', 0)} available / {counts.get('unavailable', 0)} unavailable")
    labeled_rows = [row for row in rows if row["cause_labels"]]
    if labeled_rows:
        lines.extend(["", "## Query Candidates", ""])
        for row in labeled_rows[:20]:
            labels = ", ".join(f"`{label}`" for label in row["cause_labels"])
            lines.append(f"- `{row['question_id']}`: {labels}")
    return "\n".join(lines).rstrip() + "\n"


def _load_or_build_drift(
    before_run_dir: Path,
    after_run_dir: Path,
    *,
    drift_jsonl: Path | None,
    drift_summary_json: Path | None,
    created_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if drift_jsonl is None and drift_summary_json is None:
        report = compare_retrieval_runs(before_run_dir, after_run_dir, created_at=created_at)
        return report["rows"], report["summary"]
    if drift_jsonl is None or drift_summary_json is None:
        raise ValidationError("Both drift_jsonl and drift_summary_json are required when loading existing drift reports.")
    if not drift_jsonl.exists():
        raise ValidationError(f"Missing retrieval drift JSONL: {drift_jsonl}")
    if not drift_summary_json.exists():
        raise ValidationError(f"Missing retrieval drift summary JSON: {drift_summary_json}")
    return read_jsonl(drift_jsonl), _read_json(drift_summary_json)


def _audit_row(
    drift_row: dict[str, Any],
    before_manifest: dict[str, Any] | None,
    after_manifest: dict[str, Any] | None,
    snapshot_index: dict[str, Any],
    *,
    created_at: str,
) -> dict[str, Any]:
    question_id = _required_string(drift_row, "question_id")
    before_docs = _string_list((drift_row.get("before") or {}).get("doc_ids"))
    after_docs = _string_list((drift_row.get("after") or {}).get("doc_ids"))
    metrics = drift_row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValidationError(f"Drift row {question_id} requires metrics.")

    labels: set[str] = set()
    evidence: dict[str, dict[str, Any]] = {}
    notes: list[str] = []

    _metric_labels(metrics, labels, evidence)
    _manifest_labels(before_manifest, after_manifest, labels, evidence)
    _snapshot_labels(question_id, before_docs, after_docs, snapshot_index, labels, evidence)

    for name, item in sorted(evidence.items()):
        if not item.get("available"):
            notes.append(f"{name}:{item.get('reason', 'unavailable')}")
    if notes:
        labels.add("evidence_unavailable")

    return {
        "schema_version": AUDIT_ROW_SCHEMA_VERSION,
        "created_at": created_at,
        "dataset": drift_row.get("dataset"),
        "test": drift_row.get("test"),
        "query_set_id": drift_row.get("query_set_id"),
        "question_id": question_id,
        "before": {"doc_ids": before_docs},
        "after": {"doc_ids": after_docs},
        "cause_labels": sorted(labels),
        "cause_label_schema_version": CAUSE_LABEL_SCHEMA_VERSION,
        "evidence_availability": evidence,
        "missing_evidence": notes,
        "associated_evidence": _associated_evidence(question_id, before_docs, after_docs, snapshot_index),
        "drift_metrics": metrics,
    }


def _metric_labels(metrics: dict[str, Any], labels: set[str], evidence: dict[str, dict[str, Any]]) -> None:
    top_k = _metric(metrics, "top_k_jaccard")
    evidence["retrieval_candidates"] = _availability(top_k)
    if top_k.get("available") and _number(top_k.get("value"), 1.0) < 1.0 - EPSILON:
        labels.add("retrieved_candidate_changed")

    rank = _metric(metrics, "rank_displacement")
    evidence["rank"] = _availability(rank)
    if rank.get("available") and _number(rank.get("mean_absolute_delta"), 0.0) > EPSILON:
        labels.add("rank_changed")

    score = _metric(metrics, "retained_score_delta")
    evidence["retrieval_score"] = _availability(score)
    if score.get("available") and any(abs(_number(item.get("score_delta"), 0.0)) > EPSILON for item in score.get("documents", []) if isinstance(item, dict)):
        labels.add("retrieval_score_changed")

    lexical = _metric(metrics, "lexical_score_delta")
    evidence["lexical_answer_score"] = _availability(lexical)
    if lexical.get("available") and any(key.endswith("_delta") and _number(value, 0.0) < -EPSILON for key, value in lexical.items()):
        labels.add("answer_lexical_score_regressed")

    support_exposure = _metric(metrics, "support_exposure")
    evidence["support_exposure"] = _availability(support_exposure)
    if support_exposure.get("available") and _number(support_exposure.get("exposed_count_delta"), 0.0) < -EPSILON:
        labels.add("support_exposure_regressed")

    support_recall = _metric(metrics, "support_recall")
    evidence["support_recall"] = _availability(support_recall)
    if support_recall.get("available") and _number(support_recall.get("recall_delta"), 0.0) < -EPSILON:
        labels.add("support_recall_regressed")

    coverage = _metric(metrics, "evidence_coverage")
    evidence["evidence_coverage"] = _availability(coverage)
    if coverage.get("available") and _number(coverage.get("coverage_delta"), 0.0) < -EPSILON:
        labels.add("evidence_coverage_regressed")

    distance = _metric(metrics, "distance_to_support")
    evidence["distance_to_support"] = _availability(distance)
    if distance.get("available") and _number(distance.get("min_distance_delta"), 0.0) > EPSILON:
        labels.add("distance_to_support_increased")

    trace = _metric(metrics, "action_trace")
    evidence["action_trace"] = _availability(trace)
    if trace.get("available"):
        action_delta = trace.get("action_count_delta")
        final_jaccard = trace.get("final_context_jaccard")
        if (isinstance(action_delta, int | float) and action_delta != 0) or (isinstance(final_jaccard, int | float) and final_jaccard < 1.0 - EPSILON):
            labels.add("action_trace_changed")


def _manifest_labels(
    before_manifest: dict[str, Any] | None,
    after_manifest: dict[str, Any] | None,
    labels: set[str],
    evidence: dict[str, dict[str, Any]],
) -> None:
    if before_manifest is None or after_manifest is None:
        evidence["replay_manifest"] = {"available": False, "reason": "replay_manifest_unavailable"}
        evidence["retrieval_config"] = {"available": False, "reason": "replay_manifest_unavailable"}
        evidence["scoring_config"] = {"available": False, "reason": "replay_manifest_unavailable"}
        return
    evidence["replay_manifest"] = {"available": True}
    evidence["retrieval_config"] = {"available": True}
    if before_manifest.get("retrieval_config_hash") != after_manifest.get("retrieval_config_hash") or before_manifest.get("retrieval_config_id") != after_manifest.get("retrieval_config_id"):
        labels.add("retrieval_config_changed")
        labels.add("replay_manifest_changed")
    evidence["scoring_config"] = {"available": True}
    if before_manifest.get("scoring_hash") != after_manifest.get("scoring_hash") or before_manifest.get("scoring_method") != after_manifest.get("scoring_method"):
        labels.add("scoring_config_changed")
        labels.add("replay_manifest_changed")
    if before_manifest.get("snapshot_manifest_references") != after_manifest.get("snapshot_manifest_references"):
        labels.add("replay_manifest_changed")


def _snapshot_labels(
    question_id: str,
    before_docs: list[str],
    after_docs: list[str],
    snapshot_index: dict[str, Any],
    labels: set[str],
    evidence: dict[str, dict[str, Any]],
) -> None:
    if not snapshot_index:
        evidence["snapshot_diff"] = {"available": False, "reason": "snapshot_diff_unavailable"}
        evidence["graph"] = {"available": False, "reason": "snapshot_diff_unavailable"}
        evidence["support_surface"] = {"available": False, "reason": "snapshot_diff_unavailable"}
        return
    evidence["snapshot_diff"] = {"available": True}
    docs = set(before_docs) | set(after_docs)

    support = snapshot_index.get("support", {})
    evidence["support_surface"] = support.get("availability", {"available": False, "reason": "support_surface_diff_unavailable"})
    change = support.get("by_question", {}).get(question_id, {})
    if change.get("removed_targets"):
        labels.add("support_target_removed")
    if change.get("added_targets"):
        labels.add("support_target_added")
    if change.get("changed"):
        labels.add("support_target_changed")
    docs |= set(change.get("before_targets", [])) | set(change.get("after_targets", []))

    corpus = snapshot_index.get("corpus", {})
    evidence["corpus"] = corpus.get("availability", {"available": False, "reason": "corpus_diff_unavailable"})
    if docs & corpus.get("removed_doc_ids", set()):
        labels.add("corpus_page_removed")
    if docs & corpus.get("added_doc_ids", set()):
        labels.add("corpus_page_added")
    if docs & corpus.get("changed_doc_ids", set()):
        labels.add("corpus_page_changed")

    graph = snapshot_index.get("graph", {})
    evidence["graph"] = graph.get("availability", {"available": False, "reason": "graph_diff_unavailable"})
    if docs & graph.get("removed_nodes", set()):
        labels.add("graph_edge_removed")
    if docs & graph.get("added_nodes", set()):
        labels.add("graph_edge_added")


def _associated_evidence(question_id: str, before_docs: list[str], after_docs: list[str], snapshot_index: dict[str, Any]) -> dict[str, Any]:
    if not snapshot_index:
        return {}
    docs = set(before_docs) | set(after_docs)
    support_change = snapshot_index.get("support", {}).get("by_question", {}).get(question_id, {})
    docs |= set(support_change.get("before_targets", [])) | set(support_change.get("after_targets", []))
    corpus = snapshot_index.get("corpus", {})
    graph = snapshot_index.get("graph", {})
    return {
        "support_surface": support_change,
        "corpus": {
            "added_doc_ids": sorted(docs & corpus.get("added_doc_ids", set())),
            "removed_doc_ids": sorted(docs & corpus.get("removed_doc_ids", set())),
            "changed_doc_ids": sorted(docs & corpus.get("changed_doc_ids", set())),
        },
        "graph": {
            "added_edges": [edge for edge in graph.get("added_edges", []) if edge["source"] in docs or edge["target"] in docs],
            "removed_edges": [edge for edge in graph.get("removed_edges", []) if edge["source"] in docs or edge["target"] in docs],
        },
    }


def _audit_summary(
    before_run_dir: Path,
    after_run_dir: Path,
    before_manifest: dict[str, Any] | None,
    after_manifest: dict[str, Any] | None,
    drift_summary: dict[str, Any],
    snapshot_diff: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    *,
    created_at: str,
    drift_jsonl: Path | None,
    drift_summary_json: Path | None,
    snapshot_diff_json: Path | None,
) -> dict[str, Any]:
    label_counts = Counter(label for row in rows for label in row["cause_labels"])
    availability: dict[str, Counter[str]] = {}
    for row in rows:
        for name, item in row["evidence_availability"].items():
            availability.setdefault(name, Counter())["available" if item.get("available") else "unavailable"] += 1
    return {
        "schema_version": AUDIT_SUMMARY_SCHEMA_VERSION,
        "created_at": created_at,
        "manifest_type": "regression_audit_summary",
        "comparison_type": "regression_audit",
        "before": _run_reference(before_run_dir, before_manifest),
        "after": _run_reference(after_run_dir, after_manifest),
        "query_count": len(rows),
        "labeled_query_count": sum(1 for row in rows if row["cause_labels"] and row["cause_labels"] != ["evidence_unavailable"]),
        "cause_label_counts": dict(sorted(label_counts.items())),
        "cause_label_schema_version": CAUSE_LABEL_SCHEMA_VERSION,
        "cause_label_schema": {key: CAUSE_LABEL_SCHEMA[key] for key in sorted(CAUSE_LABEL_SCHEMA)},
        "evidence_availability_counts": {name: dict(counter) for name, counter in sorted(availability.items())},
        "input_artifacts": _input_artifacts(
            before_run_dir,
            after_run_dir,
            drift_jsonl=drift_jsonl,
            drift_summary_json=drift_summary_json,
            snapshot_diff_json=snapshot_diff_json,
        ),
        "source_report_references": {
            "drift_summary_schema_version": drift_summary.get("schema_version"),
            "snapshot_diff_schema_version": snapshot_diff.get("schema_version") if snapshot_diff else None,
        },
        "overall_drift_score": None,
        "overall_drift_score_note": "No overall drift score or composite summary metric is computed.",
    }


def _snapshot_index(snapshot_diff: dict[str, Any] | None) -> dict[str, Any]:
    if snapshot_diff is None:
        return {}
    corpus_result = snapshot_diff.get("corpus_result") if isinstance(snapshot_diff.get("corpus_result"), dict) else {}
    graph_result = snapshot_diff.get("graph_result") if isinstance(snapshot_diff.get("graph_result"), dict) else {}
    support_result = snapshot_diff.get("support_surface_result") if isinstance(snapshot_diff.get("support_surface_result"), dict) else {}
    return {
        "corpus": {
            "availability": {"available": bool(corpus_result.get("available", True))},
            "added_doc_ids": _doc_ids_from_files(corpus_result.get("added_files", [])),
            "removed_doc_ids": _doc_ids_from_files(corpus_result.get("removed_files", [])),
            "changed_doc_ids": _doc_ids_from_files(corpus_result.get("changed_files", [])),
        },
        "graph": {
            "availability": _optional_availability(graph_result, "graph_diff_unavailable"),
            "added_edges": _edge_list(graph_result.get("added_edges", [])),
            "removed_edges": _edge_list(graph_result.get("removed_edges", [])),
            "added_nodes": _edge_nodes(graph_result.get("added_edges", [])),
            "removed_nodes": _edge_nodes(graph_result.get("removed_edges", [])),
        },
        "support": {
            "availability": _optional_availability(support_result, "support_surface_diff_unavailable"),
            "by_question": _support_changes_by_question(support_result),
        },
    }


def _support_changes_by_question(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}
    for item in result.get("added_questions", []) if isinstance(result.get("added_questions"), list) else []:
        qid = item.get("question_id")
        if isinstance(qid, str):
            changes[qid] = {"added_targets": _string_list(item.get("targets")), "removed_targets": [], "before_targets": [], "after_targets": _string_list(item.get("targets")), "changed": True}
    for item in result.get("removed_questions", []) if isinstance(result.get("removed_questions"), list) else []:
        qid = item.get("question_id")
        if isinstance(qid, str):
            changes[qid] = {"added_targets": [], "removed_targets": _string_list(item.get("targets")), "before_targets": _string_list(item.get("targets")), "after_targets": [], "changed": True}
    for item in result.get("changed_questions", []) if isinstance(result.get("changed_questions"), list) else []:
        qid = item.get("question_id")
        if isinstance(qid, str):
            changes[qid] = {
                "added_targets": _string_list(item.get("added_targets")),
                "removed_targets": _string_list(item.get("removed_targets")),
                "before_targets": _string_list(item.get("before_targets")),
                "after_targets": _string_list(item.get("after_targets")),
                "changed": True,
            }
    return changes


def _validate_run_manifests_joinable(before: dict[str, Any] | None, after: dict[str, Any] | None) -> None:
    if before is None or after is None:
        return
    for field in ("dataset", "test", "query_set_id"):
        before_value = before.get(field)
        after_value = after.get(field)
        if before_value and after_value and before_value != after_value:
            raise ValidationError(f"Run directories are not joinable for audit: replay manifest {field} differs.")


def _validate_drift_rows(rows: list[dict[str, Any]], before_manifest: dict[str, Any] | None, after_manifest: dict[str, Any] | None) -> None:
    seen: set[str] = set()
    for row in rows:
        qid = _required_string(row, "question_id")
        if qid in seen:
            raise ValidationError(f"Duplicate question_id in retrieval drift rows: {qid}")
        seen.add(qid)
        for field in ("dataset", "test", "query_set_id"):
            row_value = row.get(field)
            manifest_value = (before_manifest or {}).get(field) or (after_manifest or {}).get(field)
            if row_value and manifest_value and row_value != manifest_value:
                raise ValidationError(f"Drift row {qid} is not joinable with replay manifests: {field} differs.")


def _read_optional_manifest(path: Path) -> dict[str, Any] | None:
    return read_manifest(path, verify_hash=False) if path.exists() else None


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise ValidationError(f"Missing snapshot diff JSON: {path}")
    return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON artifact must be an object: {path}")
    return value


def _input_artifacts(before_run_dir: Path, after_run_dir: Path, **paths: Path | None) -> dict[str, Any]:
    artifacts = {
        "before_run_dir": str(before_run_dir),
        "after_run_dir": str(after_run_dir),
    }
    for name, path in paths.items():
        if path is not None:
            artifacts[name] = _artifact_reference(path)
    for side, run_dir in (("before", before_run_dir), ("after", after_run_dir)):
        manifest = run_dir / "retrieval_replay_manifest.json"
        if manifest.exists():
            artifacts[f"{side}_replay_manifest"] = _artifact_reference(manifest)
    return artifacts


def _artifact_reference(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _run_reference(run_dir: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    manifest = manifest or {}
    return {
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "dataset": manifest.get("dataset"),
        "test": manifest.get("test"),
        "query_set_id": manifest.get("query_set_id"),
        "corpus_snapshot_id": manifest.get("corpus_snapshot_id"),
        "graph_snapshot_id": manifest.get("graph_snapshot_id"),
        "support_surface_id": manifest.get("support_surface_id"),
        "retrieval_config_id": manifest.get("retrieval_config_id"),
        "retrieval_config_hash": manifest.get("retrieval_config_hash"),
        "scoring_hash": manifest.get("scoring_hash"),
        "manifest_hash": manifest.get("manifest_hash"),
    }


def _metric(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    value = metrics.get(name)
    return value if isinstance(value, dict) else {"available": False, "reason": f"{name}_unavailable"}


def _availability(metric: dict[str, Any]) -> dict[str, Any]:
    if metric.get("available"):
        return {"available": True}
    return {"available": False, "reason": str(metric.get("reason", "unavailable"))}


def _optional_availability(result: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"available": True} if result.get("available") else {"available": False, "reason": reason}


def _doc_ids_from_files(rows: Any) -> set[str]:
    docs: set[str] = set()
    if not isinstance(rows, list):
        return docs
    for item in rows:
        if not isinstance(item, dict):
            continue
        for field in ("doc_id", "before_doc_id", "after_doc_id", "path"):
            value = item.get(field)
            if isinstance(value, str) and value:
                docs.add(value)
    return docs


def _edge_list(rows: Any) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    if not isinstance(rows, list):
        return edges
    for item in rows:
        if isinstance(item, dict) and isinstance(item.get("source"), str) and isinstance(item.get("target"), str):
            edges.append({"source": item["source"], "target": item["target"], "edge_id": item.get("edge_id", f"{item['source']}->{item['target']}")})
    return edges


def _edge_nodes(rows: Any) -> set[str]:
    return {node for edge in _edge_list(rows) for node in (edge["source"], edge["target"])}


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _number(value: Any, default: float) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return default


def _required_string(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Audit input row requires non-empty {field}.")
    return value


def _drift_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (str(row.get("dataset") or ""), str(row.get("test") or ""), str(row.get("query_set_id") or ""), str(row.get("question_id") or ""))
