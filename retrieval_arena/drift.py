from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .hashing import sha256_file
from .manifests import canonical_manifest_json, read_manifest, write_manifest
from .schemas import read_jsonl, write_jsonl


DRIFT_SCHEMA_VERSION = "retrieval_arena.retrieval_drift.v1"
SUMMARY_SCHEMA_VERSION = "retrieval_arena.retrieval_drift_summary.v1"
REPORT_FILENAMES = ("retrieval_drift.jsonl", "retrieval_drift_summary.json", "retrieval_drift_summary.md")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compare_retrieval_runs(
    before_run_dir: Path,
    after_run_dir: Path,
    *,
    out_dir: Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    before = _load_run(before_run_dir, "before")
    after = _load_run(after_run_dir, "after")
    timestamp = created_at or utc_now_iso()
    _validate_comparable_runs(before, after)

    rows = [
        _query_drift_row(
            question_id,
            before,
            after,
            created_at=timestamp,
        )
        for question_id in sorted(before["predictions"])
    ]
    summary = build_drift_summary(before, after, rows, created_at=timestamp)

    written: list[str] = []
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "retrieval_drift.jsonl"
        summary_path = out_dir / "retrieval_drift_summary.json"
        markdown_path = out_dir / "retrieval_drift_summary.md"
        write_jsonl(jsonl_path, rows)
        write_manifest(summary_path, summary)
        markdown_path.write_text(render_drift_markdown(summary), encoding="utf-8")
        written = [str(jsonl_path), str(summary_path), str(markdown_path)]
    return {"summary": summary, "rows": rows, "written_artifacts": written}


def build_drift_summary(
    before: dict[str, Any],
    after: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    created_at: str,
) -> dict[str, Any]:
    optional_names = [
        "retained_score_delta",
        "support_exposure",
        "evidence_coverage",
        "distance_to_support",
        "support_recall",
        "lexical_score_delta",
        "action_trace",
    ]
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "created_at": created_at,
        "manifest_type": "retrieval_drift_summary",
        "comparison_type": "retrieval_drift",
        "before": _run_reference(before),
        "after": _run_reference(after),
        "query_count": len(rows),
        "mean_top_k_jaccard": _mean(_metric_values(rows, "top_k_jaccard", "value")),
        "mean_ordered_top_k_overlap": _mean(_metric_values(rows, "ordered_top_k_overlap", "value")),
        "mean_rank_displacement": _mean(_metric_values(rows, "rank_displacement", "mean_absolute_delta")),
        "optional_signal_availability": {
            name: {
                "available_count": sum(1 for row in rows if row["metrics"][name]["available"]),
                "unavailable_count": sum(1 for row in rows if not row["metrics"][name]["available"]),
            }
            for name in optional_names
        },
        "support_exposure_regression_count": sum(
            1 for row in rows if _metric_delta(row, "support_exposure", "exposed_count_delta") < 0
        ),
        "evidence_coverage_regression_count": sum(
            1 for row in rows if _metric_delta(row, "evidence_coverage", "coverage_delta") < 0
        ),
        "support_recall_regression_count": sum(
            1 for row in rows if _metric_delta(row, "support_recall", "recall_delta") < 0
        ),
        "mean_support_recall_delta": _mean(_metric_values(rows, "support_recall", "recall_delta")),
        "mean_distance_to_support_delta": _mean(_metric_values(rows, "distance_to_support", "min_distance_delta")),
        "mean_lexical_f1_delta": _mean(_metric_values(rows, "lexical_score_delta", "f1_delta")),
        "artifact_hashes": _artifact_hashes(before["run_dir"], after["run_dir"]),
    }
    return summary


def render_drift_markdown(summary: dict[str, Any]) -> str:
    availability = summary["optional_signal_availability"]
    lines = [
        "# Retrieval Drift",
        "",
        f"- Queries compared: {summary['query_count']}",
        f"- Mean top-k Jaccard: {_format_number(summary['mean_top_k_jaccard'])}",
        f"- Mean ordered top-k overlap: {_format_number(summary['mean_ordered_top_k_overlap'])}",
        f"- Mean retained rank displacement: {_format_number(summary['mean_rank_displacement'])}",
        f"- Support exposure regressions: {summary['support_exposure_regression_count']}",
        f"- Evidence coverage regressions: {summary['evidence_coverage_regression_count']}",
        f"- Support recall regressions: {summary['support_recall_regression_count']}",
        "",
        "## Optional Signals",
        "",
    ]
    for name in sorted(availability):
        item = availability[name]
        lines.append(f"- {name}: {item['available_count']} available / {item['unavailable_count']} unavailable")
    return "\n".join(lines).rstrip() + "\n"


def _load_run(run_dir: Path, side: str) -> dict[str, Any]:
    resolved = run_dir.resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValidationError(f"{side} run directory does not exist: {resolved}")
    predictions = _read_indexed_jsonl(resolved / "predictions.jsonl", "predictions.jsonl")
    item_scores = _read_indexed_jsonl(resolved / "item_scores.jsonl", "item_scores.jsonl")
    manifest = _read_optional_manifest(resolved / "retrieval_replay_manifest.json")
    traces = _read_optional_indexed_jsonl(resolved / "action_traces.jsonl", "action_traces.jsonl")
    return {
        "run_dir": resolved,
        "predictions": predictions,
        "item_scores": item_scores,
        "manifest": manifest,
        "support_targets": _support_targets_from_run_manifest(manifest, resolved),
        "action_traces": traces,
    }


def _read_indexed_jsonl(path: Path, filename: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise ValidationError(f"Missing required drift artifact: {path}")
    return _index_by_question_id(read_jsonl(path), filename)


def _read_optional_indexed_jsonl(path: Path, filename: str) -> dict[str, dict[str, Any]] | None:
    if not path.exists():
        return None
    return _index_by_question_id(read_jsonl(path), filename)


def _read_optional_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_manifest(path, verify_hash=False)


def _index_by_question_id(rows: list[dict[str, Any]], filename: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ValidationError(f"{filename} rows require non-empty question_id for drift comparison.")
        if qid in indexed:
            raise ValidationError(f"Duplicate question_id in {filename}: {qid}")
        indexed[qid] = row
    return indexed


def _validate_comparable_runs(before: dict[str, Any], after: dict[str, Any]) -> None:
    before_ids = set(before["predictions"])
    after_ids = set(after["predictions"])
    if before_ids != after_ids:
        raise ValidationError(
            "Run directories are not comparable by query IDs: "
            f"before_only={sorted(before_ids - after_ids)} after_only={sorted(after_ids - before_ids)}"
        )
    if set(before["item_scores"]) != before_ids or set(after["item_scores"]) != after_ids:
        raise ValidationError("item_scores.jsonl query IDs must match predictions.jsonl for drift comparison.")
    before_manifest = before.get("manifest") or {}
    after_manifest = after.get("manifest") or {}
    for field in ("dataset", "test", "query_set_id"):
        before_value = before_manifest.get(field)
        after_value = after_manifest.get(field)
        if before_value and after_value and before_value != after_value:
            raise ValidationError(f"Run directories are not comparable: replay manifest {field} differs.")


def _query_drift_row(question_id: str, before: dict[str, Any], after: dict[str, Any], *, created_at: str) -> dict[str, Any]:
    before_prediction = before["predictions"][question_id]
    after_prediction = after["predictions"][question_id]
    before_context = _context(before_prediction)
    after_context = _context(after_prediction)
    before_docs = _doc_ids(before_context)
    after_docs = _doc_ids(after_context)
    support_targets = _support_targets(question_id, before, after, before_prediction, after_prediction, before_context, after_context)
    evidence_targets = _evidence_targets(before_prediction, after_prediction, before_context, after_context)
    metrics = {
        "top_k_jaccard": _top_k_jaccard(before_docs, after_docs),
        "ordered_top_k_overlap": _ordered_overlap(before_docs, after_docs),
        "rank_displacement": _rank_displacement(before_docs, after_docs),
        "retained_score_delta": _score_delta(before_context, after_context),
        "support_exposure": _support_exposure(before_docs, after_docs, support_targets),
        "evidence_coverage": _evidence_coverage(before_docs, after_docs, evidence_targets),
        "distance_to_support": _distance_to_support(before_context, after_context),
        "support_recall": _support_recall(before_docs, after_docs, support_targets),
        "lexical_score_delta": _lexical_score_delta(before["item_scores"][question_id], after["item_scores"][question_id]),
        "action_trace": _action_trace_drift(question_id, before.get("action_traces"), after.get("action_traces")),
    }
    return {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "created_at": created_at,
        "dataset": _manifest_field(before, after, "dataset"),
        "test": _manifest_field(before, after, "test"),
        "query_set_id": _manifest_field(before, after, "query_set_id"),
        "question_id": question_id,
        "before": {"doc_ids": before_docs, "context_count": len(before_context)},
        "after": {"doc_ids": after_docs, "context_count": len(after_context)},
        "metrics": metrics,
    }


def _context(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    value = prediction.get("retrieved_context", [])
    return value if isinstance(value, list) else []


def _doc_ids(context: list[dict[str, Any]]) -> list[str]:
    docs: list[str] = []
    for item in context:
        if isinstance(item, dict) and isinstance(item.get("doc_id"), str) and item["doc_id"]:
            docs.append(item["doc_id"])
    return docs


def _top_k_jaccard(before_docs: list[str], after_docs: list[str]) -> dict[str, Any]:
    before_set = set(before_docs)
    after_set = set(after_docs)
    union = before_set | after_set
    value = 1.0 if not union else len(before_set & after_set) / len(union)
    return {
        "available": True,
        "value": value,
        "intersection_count": len(before_set & after_set),
        "union_count": len(union),
    }


def _ordered_overlap(before_docs: list[str], after_docs: list[str]) -> dict[str, Any]:
    denominator = max(len(before_docs), len(after_docs))
    matches = sum(1 for before_doc, after_doc in zip(before_docs, after_docs) if before_doc == after_doc)
    return {"available": True, "value": 1.0 if denominator == 0 else matches / denominator, "matching_rank_count": matches, "rank_count": denominator}


def _rank_displacement(before_docs: list[str], after_docs: list[str]) -> dict[str, Any]:
    before_ranks = {doc_id: index + 1 for index, doc_id in enumerate(before_docs)}
    after_ranks = {doc_id: index + 1 for index, doc_id in enumerate(after_docs)}
    retained = sorted(set(before_ranks) & set(after_ranks))
    docs = [
        {
            "doc_id": doc_id,
            "before_rank": before_ranks[doc_id],
            "after_rank": after_ranks[doc_id],
            "rank_delta": after_ranks[doc_id] - before_ranks[doc_id],
        }
        for doc_id in retained
    ]
    return {
        "available": True,
        "retained_count": len(docs),
        "mean_absolute_delta": _mean([abs(item["rank_delta"]) for item in docs]),
        "documents": docs,
    }


def _score_delta(before_context: list[dict[str, Any]], after_context: list[dict[str, Any]]) -> dict[str, Any]:
    before_scores = _scores_by_doc(before_context)
    after_scores = _scores_by_doc(after_context)
    retained = sorted(set(before_scores) & set(after_scores))
    docs = [
        {
            "doc_id": doc_id,
            "before_score": before_scores[doc_id],
            "after_score": after_scores[doc_id],
            "score_delta": after_scores[doc_id] - before_scores[doc_id],
        }
        for doc_id in retained
    ]
    return _available_or_false(docs, {"retained_count": len(docs), "mean_delta": _mean([item["score_delta"] for item in docs]), "documents": docs})


def _scores_by_doc(context: list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in context:
        doc_id = item.get("doc_id") if isinstance(item, dict) else None
        score = item.get("score") if isinstance(item, dict) else None
        if isinstance(doc_id, str) and isinstance(score, int | float) and not isinstance(score, bool):
            scores[doc_id] = float(score)
    return scores


def _support_exposure(before_docs: list[str], after_docs: list[str], targets: set[str]) -> dict[str, Any]:
    if not targets:
        return {"available": False, "reason": "support_targets_unavailable"}
    before_exposed = sorted(set(before_docs) & targets)
    after_exposed = sorted(set(after_docs) & targets)
    return {
        "available": True,
        "target_count": len(targets),
        "before_exposed_doc_ids": before_exposed,
        "after_exposed_doc_ids": after_exposed,
        "before_exposed_count": len(before_exposed),
        "after_exposed_count": len(after_exposed),
        "exposed_count_delta": len(after_exposed) - len(before_exposed),
    }


def _support_recall(before_docs: list[str], after_docs: list[str], targets: set[str]) -> dict[str, Any]:
    if not targets:
        return {"available": False, "reason": "support_targets_unavailable"}
    before_recall = len(set(before_docs) & targets) / len(targets)
    after_recall = len(set(after_docs) & targets) / len(targets)
    return {"available": True, "before_recall": before_recall, "after_recall": after_recall, "recall_delta": after_recall - before_recall}


def _evidence_coverage(before_docs: list[str], after_docs: list[str], targets: set[str]) -> dict[str, Any]:
    if not targets:
        return {"available": False, "reason": "evidence_targets_unavailable"}
    before_coverage = len(set(before_docs) & targets) / len(targets)
    after_coverage = len(set(after_docs) & targets) / len(targets)
    return {"available": True, "target_count": len(targets), "before_coverage": before_coverage, "after_coverage": after_coverage, "coverage_delta": after_coverage - before_coverage}


def _distance_to_support(before_context: list[dict[str, Any]], after_context: list[dict[str, Any]]) -> dict[str, Any]:
    before_distance = _min_distance(before_context)
    after_distance = _min_distance(after_context)
    if before_distance is None and after_distance is None:
        return {"available": False, "reason": "distance_fields_unavailable"}
    return {
        "available": True,
        "before_min_distance": before_distance,
        "after_min_distance": after_distance,
        "min_distance_delta": None if before_distance is None or after_distance is None else after_distance - before_distance,
    }


def _min_distance(context: list[dict[str, Any]]) -> float | None:
    fields = ("distance_to_support", "graph_distance_to_support", "support_distance", "graph_distance")
    distances: list[float] = []
    for item in context:
        for field in fields:
            value = item.get(field) if isinstance(item, dict) else None
            if isinstance(value, int | float) and not isinstance(value, bool):
                distances.append(float(value))
                break
    return min(distances) if distances else None


def _lexical_score_delta(before_score: dict[str, Any], after_score: dict[str, Any]) -> dict[str, Any]:
    fields = sorted(
        field
        for field in set(before_score) & set(after_score)
        if field != "question_id" and isinstance(before_score.get(field), int | float) and isinstance(after_score.get(field), int | float)
    )
    if not fields:
        return {"available": False, "reason": "numeric_item_score_fields_unavailable"}
    deltas = {f"{field}_delta": float(after_score[field]) - float(before_score[field]) for field in fields}
    values = {f"before_{field}": float(before_score[field]) for field in fields} | {f"after_{field}": float(after_score[field]) for field in fields}
    return {"available": True, "fields": fields, **values, **deltas}


def _action_trace_drift(question_id: str, before_traces: dict[str, dict[str, Any]] | None, after_traces: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    if before_traces is None or after_traces is None:
        return {"available": False, "reason": "action_traces_unavailable"}
    before_trace = before_traces.get(question_id)
    after_trace = after_traces.get(question_id)
    if before_trace is None or after_trace is None:
        return {"available": False, "reason": "action_trace_missing_for_query"}
    before_actions = before_trace.get("actions", [])
    after_actions = after_trace.get("actions", [])
    before_final = _string_set(before_trace.get("final_context_doc_ids", []))
    after_final = _string_set(after_trace.get("final_context_doc_ids", []))
    union = before_final | after_final
    return {
        "available": True,
        "before_action_count": len(before_actions) if isinstance(before_actions, list) else None,
        "after_action_count": len(after_actions) if isinstance(after_actions, list) else None,
        "action_count_delta": None if not isinstance(before_actions, list) or not isinstance(after_actions, list) else len(after_actions) - len(before_actions),
        "before_final_context_doc_ids": sorted(before_final),
        "after_final_context_doc_ids": sorted(after_final),
        "final_context_jaccard": 1.0 if not union else len(before_final & after_final) / len(union),
    }


def _support_targets(
    question_id: str,
    before: dict[str, Any],
    after: dict[str, Any],
    before_prediction: dict[str, Any],
    after_prediction: dict[str, Any],
    before_context: list[dict[str, Any]],
    after_context: list[dict[str, Any]],
) -> set[str]:
    targets = set(before["support_targets"].get(question_id, [])) | set(after["support_targets"].get(question_id, []))
    targets |= _row_doc_targets(before_prediction, ("support_target_doc_ids", "support_doc_ids", "top_docs"))
    targets |= _row_doc_targets(after_prediction, ("support_target_doc_ids", "support_doc_ids", "top_docs"))
    for item in [*before_context, *after_context]:
        targets |= _context_doc_targets(item, ("support_target_doc_ids", "support_doc_ids"))
    return targets


def _evidence_targets(
    before_prediction: dict[str, Any],
    after_prediction: dict[str, Any],
    before_context: list[dict[str, Any]],
    after_context: list[dict[str, Any]],
) -> set[str]:
    targets = _row_doc_targets(before_prediction, ("evidence_doc_ids", "evidence_target_doc_ids")) | _row_doc_targets(after_prediction, ("evidence_doc_ids", "evidence_target_doc_ids"))
    for item in [*before_context, *after_context]:
        if _truthy_any(item, ("is_evidence", "evidence", "evidence_label", "support", "is_support")) and isinstance(item.get("doc_id"), str):
            targets.add(item["doc_id"])
        targets |= _context_doc_targets(item, ("evidence_doc_ids", "evidence_target_doc_ids"))
    return targets


def _row_doc_targets(row: dict[str, Any], fields: tuple[str, ...]) -> set[str]:
    targets: set[str] = set()
    for field in fields:
        targets |= _doc_id_set(row.get(field))
    return targets


def _context_doc_targets(item: dict[str, Any], fields: tuple[str, ...]) -> set[str]:
    if not isinstance(item, dict):
        return set()
    return _row_doc_targets(item, fields)


def _doc_id_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    docs: set[str] = set()
    for item in value:
        if isinstance(item, str) and item:
            docs.add(item)
        elif isinstance(item, dict) and isinstance(item.get("doc_id"), str) and item["doc_id"]:
            docs.add(item["doc_id"])
    return docs


def _truthy_any(item: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return isinstance(item, dict) and any(bool(item.get(field)) for field in fields)


def _support_targets_from_run_manifest(manifest: dict[str, Any] | None, run_dir: Path) -> dict[str, list[str]]:
    if not manifest:
        return {}
    refs = manifest.get("snapshot_manifest_references")
    support_ref = refs.get("support_surface") if isinstance(refs, dict) else None
    path_value = support_ref.get("path") if isinstance(support_ref, dict) else None
    if not isinstance(path_value, str) or not path_value:
        return {}
    path = Path(path_value)
    if not path.is_absolute():
        path = run_dir / path
    if not path.exists():
        return {}
    support_manifest = read_manifest(path, verify_hash=False)
    targets = support_manifest.get("support_targets_by_question", {})
    if not isinstance(targets, dict):
        return {}
    return {str(qid): sorted(str(doc_id) for doc_id in doc_ids) for qid, doc_ids in targets.items() if isinstance(doc_ids, list)}


def _run_reference(run: dict[str, Any]) -> dict[str, Any]:
    manifest = run.get("manifest") or {}
    return {
        "run_dir": str(run["run_dir"]),
        "run_id": manifest.get("run_id"),
        "dataset": manifest.get("dataset"),
        "test": manifest.get("test"),
        "query_set_id": manifest.get("query_set_id"),
        "corpus_snapshot_id": manifest.get("corpus_snapshot_id"),
        "graph_snapshot_id": manifest.get("graph_snapshot_id"),
        "support_surface_id": manifest.get("support_surface_id"),
        "retrieval_config_id": manifest.get("retrieval_config_id"),
        "retrieval_config_hash": manifest.get("retrieval_config_hash"),
        "manifest_hash": manifest.get("manifest_hash"),
    }


def _manifest_field(before: dict[str, Any], after: dict[str, Any], field: str) -> Any:
    before_value = (before.get("manifest") or {}).get(field)
    after_value = (after.get("manifest") or {}).get(field)
    return before_value if before_value == after_value else before_value or after_value


def _artifact_hashes(before_run_dir: Path, after_run_dir: Path) -> dict[str, dict[str, Any]]:
    hashes: dict[str, dict[str, Any]] = {}
    for side, run_dir in (("before", before_run_dir), ("after", after_run_dir)):
        hashes[side] = {}
        for filename in ("retrieval_replay_manifest.json", "predictions.jsonl", "item_scores.jsonl", "scores.json", "metadata.json", "action_traces.jsonl"):
            path = run_dir / filename
            if path.exists() and path.is_file():
                hashes[side][filename] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return hashes


def _metric_values(rows: list[dict[str, Any]], metric: str, field: str) -> list[float]:
    values = []
    for row in rows:
        item = row["metrics"][metric]
        value = item.get(field)
        if item.get("available") and isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _metric_delta(row: dict[str, Any], metric: str, field: str) -> float:
    item = row["metrics"][metric]
    value = item.get(field)
    if item.get("available") and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _mean(values: list[int | float]) -> float | None:
    return sum(float(value) for value in values) / len(values) if values else None


def _string_set(value: Any) -> set[str]:
    return {item for item in value if isinstance(item, str) and item} if isinstance(value, list) else set()


def _available_or_false(rows: list[Any], payload: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"available": False, "reason": "required_fields_unavailable"}
    return {"available": True, **payload}


def _format_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
