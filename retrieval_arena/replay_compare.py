from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .hashing import sha256_file, sha256_json
from .manifests import canonical_manifest_json, read_manifest
from .schemas import read_jsonl


REPLAY_COMPARISON_SCHEMA_VERSION = "retrieval_arena.replay_fidelity_report.v1"
REQUIRED_ARTIFACTS = ("retrieval_replay_manifest.json", "predictions.jsonl", "metadata.json", "item_scores.jsonl", "scores.json")
OPTIONAL_ARTIFACTS = ("action_traces.jsonl",)
BEHAVIOR_ARTIFACTS = ("predictions.jsonl", "metadata.json", "item_scores.jsonl", "action_traces.jsonl")
VOLATILE_MANIFEST_FIELDS = ("created_at", "manifest_hash", "run_started_at", "run_completed_at")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_replay_fidelity_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_manifest_json(report), encoding="utf-8")


def compare_run_dirs(
    expected_run_dir: Path,
    actual_run_dir: Path,
    *,
    created_at: str | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    expected = expected_run_dir.resolve()
    actual = actual_run_dir.resolve()
    artifact_results = _compare_artifacts(expected, actual)
    manifest_result = _compare_manifests(expected, actual)
    prediction_results = _compare_predictions(expected, actual)
    score_results = _compare_scores(expected, actual)
    metadata_result = _compare_metadata(expected, actual)
    action_trace_results = _compare_action_traces(expected, actual)

    required_artifacts_passed = all(
        item["expected_exists"] and item["actual_exists"]
        for item in artifact_results
        if item["path"] in REQUIRED_ARTIFACTS
    )
    optional_artifacts_passed = all(item["passed"] for item in artifact_results if item["path"] in OPTIONAL_ARTIFACTS)
    behavior_artifacts_passed = all(item["passed"] for item in artifact_results if item["path"] in BEHAVIOR_ARTIFACTS)
    behavior_passed = _behavior_passed(prediction_results, score_results, metadata_result, action_trace_results)
    manifest_passed = manifest_result["passed"] or manifest_result["only_volatile_differences"]
    replay_matched = required_artifacts_passed and optional_artifacts_passed and behavior_passed

    report = {
        "schema_version": REPLAY_COMPARISON_SCHEMA_VERSION,
        "created_at": created_at or utc_now_iso(),
        "comparison_type": "replay_fidelity",
        "expected_run_dir": str(expected),
        "actual_run_dir": str(actual),
        "passed": replay_matched,
        "replay_matched": replay_matched,
        "summary": _summary(
            artifact_results=artifact_results,
            manifest_result=manifest_result,
            prediction_results=prediction_results,
            score_results=score_results,
            metadata_result=metadata_result,
            action_trace_results=action_trace_results,
            required_artifacts_passed=required_artifacts_passed,
            optional_artifacts_passed=optional_artifacts_passed,
            behavior_artifacts_passed=behavior_artifacts_passed,
            behavior_passed=behavior_passed,
            manifest_passed=manifest_passed,
        ),
        "artifact_results": artifact_results,
        "manifest_result": manifest_result,
        "prediction_results": prediction_results,
        "score_results": score_results,
        "metadata_result": metadata_result,
        "action_trace_results": action_trace_results,
    }
    if out_path is not None:
        write_replay_fidelity_report(out_path, report)
    return report


def _behavior_passed(
    prediction_results: dict[str, Any],
    score_results: dict[str, Any],
    metadata_result: dict[str, Any],
    action_trace_results: dict[str, Any] | None,
) -> bool:
    results = [
        prediction_results.get("passed"),
        score_results.get("item_scores", {}).get("passed"),
        metadata_result.get("passed"),
    ]
    if action_trace_results is not None:
        results.append(action_trace_results.get("passed"))
    return all(bool(item) for item in results)


def _compare_artifacts(expected: Path, actual: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for filename in sorted([*REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS]):
        expected_path = expected / filename
        actual_path = actual / filename
        expected_exists = expected_path.exists()
        actual_exists = actual_path.exists()
        result: dict[str, Any] = {
            "path": filename,
            "required": filename in REQUIRED_ARTIFACTS,
            "expected_exists": expected_exists,
            "actual_exists": actual_exists,
            "passed": expected_exists == actual_exists if filename in OPTIONAL_ARTIFACTS else expected_exists and actual_exists,
        }
        if expected_exists and expected_path.is_file():
            result["expected_sha256"] = sha256_file(expected_path)
            result["expected_size_bytes"] = expected_path.stat().st_size
        if actual_exists and actual_path.is_file():
            result["actual_sha256"] = sha256_file(actual_path)
            result["actual_size_bytes"] = actual_path.stat().st_size
        if expected_exists and actual_exists:
            result["passed"] = result["expected_sha256"] == result["actual_sha256"]
            result["status"] = "equal" if result["passed"] else "different"
        elif result["passed"]:
            result["status"] = "absent"
        else:
            result["status"] = "missing"
        results.append(result)
    return results


def _compare_manifests(expected: Path, actual: Path) -> dict[str, Any]:
    filename = "retrieval_replay_manifest.json"
    expected_path = expected / filename
    actual_path = actual / filename
    if not expected_path.exists() or not actual_path.exists():
        return {
            "passed": False,
            "only_volatile_differences": False,
            "missing": _missing_sides(expected_path, actual_path),
            "field_differences": [],
            "volatile_differences": [],
        }
    expected_manifest = read_manifest(expected_path, verify_hash=False)
    actual_manifest = read_manifest(actual_path, verify_hash=False)
    field_differences = _field_differences(expected_manifest, actual_manifest)
    volatile_differences = [diff for diff in field_differences if diff["field"] in VOLATILE_MANIFEST_FIELDS]
    stable_differences = [diff for diff in field_differences if diff["field"] not in VOLATILE_MANIFEST_FIELDS]
    return {
        "passed": not field_differences,
        "only_volatile_differences": bool(volatile_differences) and not stable_differences,
        "volatile_fields": list(VOLATILE_MANIFEST_FIELDS),
        "volatile_differences": volatile_differences,
        "field_differences": stable_differences,
        "expected_manifest_hash": expected_manifest.get("manifest_hash"),
        "actual_manifest_hash": actual_manifest.get("manifest_hash"),
    }


def _compare_predictions(expected: Path, actual: Path) -> dict[str, Any]:
    loaded = _load_jsonl_pair(expected, actual, "predictions.jsonl")
    if loaded["missing"]:
        return {"passed": False, "missing": loaded["missing"], "changed_question_ids": [], "differences": []}
    expected_by_id = _index_by_question_id(loaded["expected"], "predictions.jsonl")
    actual_by_id = _index_by_question_id(loaded["actual"], "predictions.jsonl")
    differences: list[dict[str, Any]] = []
    for qid in sorted(set(expected_by_id) | set(actual_by_id)):
        if qid not in expected_by_id or qid not in actual_by_id:
            differences.append({"question_id": qid, "type": "missing_prediction", "missing": "expected" if qid not in expected_by_id else "actual"})
            continue
        expected_row = expected_by_id[qid]
        actual_row = actual_by_id[qid]
        if expected_row.get("question") != actual_row.get("question"):
            differences.append(_value_diff(qid, "question", expected_row.get("question"), actual_row.get("question")))
        if expected_row.get("generated_answer") != actual_row.get("generated_answer"):
            differences.append(_hashed_value_diff(qid, "generated_answer", expected_row.get("generated_answer"), actual_row.get("generated_answer")))
        context_diffs = _context_differences(qid, expected_row.get("retrieved_context", []), actual_row.get("retrieved_context", []))
        differences.extend(context_diffs)
        row_diffs = _field_differences(expected_row, actual_row, prefix="", ignore={"question", "generated_answer", "retrieved_context"})
        for diff in row_diffs:
            diff["question_id"] = qid
            diff["type"] = "prediction_field_changed"
            differences.append(diff)
    return {
        "passed": not differences,
        "question_count": len(set(expected_by_id) | set(actual_by_id)),
        "changed_question_ids": sorted({diff["question_id"] for diff in differences if "question_id" in diff}),
        "differences": differences,
    }


def _context_differences(question_id: str, expected_context: Any, actual_context: Any) -> list[dict[str, Any]]:
    if not isinstance(expected_context, list) or not isinstance(actual_context, list):
        return [_value_diff(question_id, "retrieved_context", expected_context, actual_context, diff_type="retrieved_context_changed")]
    differences: list[dict[str, Any]] = []
    max_len = max(len(expected_context), len(actual_context))
    for index in range(max_len):
        rank = index + 1
        if index >= len(expected_context) or index >= len(actual_context):
            differences.append(
                {
                    "question_id": question_id,
                    "type": "retrieved_context_missing_rank",
                    "rank": rank,
                    "expected_present": index < len(expected_context),
                    "actual_present": index < len(actual_context),
                }
            )
            continue
        expected_item = expected_context[index]
        actual_item = actual_context[index]
        if not isinstance(expected_item, dict) or not isinstance(actual_item, dict):
            if expected_item != actual_item:
                differences.append(_value_diff(question_id, f"retrieved_context[{rank}]", expected_item, actual_item, diff_type="retrieved_context_changed"))
            continue
        for field in sorted(set(expected_item) | set(actual_item)):
            if expected_item.get(field) == actual_item.get(field):
                continue
            differences.append(
                {
                    "question_id": question_id,
                    "type": "retrieved_context_changed",
                    "rank": rank,
                    "field": field,
                    "expected": _compact_value(expected_item.get(field)),
                    "actual": _compact_value(actual_item.get(field)),
                    "expected_context_doc_id": expected_item.get("doc_id"),
                    "actual_context_doc_id": actual_item.get("doc_id"),
                }
            )
    return differences


def _compare_scores(expected: Path, actual: Path) -> dict[str, Any]:
    item_result = _compare_jsonl_by_question_id(expected, actual, "item_scores.jsonl", result_name="item_score")
    aggregate_result = _compare_json_file(expected, actual, "scores.json")
    passed = item_result["passed"] and aggregate_result["passed"]
    return {
        "passed": passed,
        "item_scores": item_result,
        "aggregate_scores": aggregate_result,
    }


def _compare_metadata(expected: Path, actual: Path) -> dict[str, Any]:
    return _compare_json_file(expected, actual, "metadata.json")


def _compare_action_traces(expected: Path, actual: Path) -> dict[str, Any] | None:
    expected_path = expected / "action_traces.jsonl"
    actual_path = actual / "action_traces.jsonl"
    if not expected_path.exists() and not actual_path.exists():
        return None
    return _compare_jsonl_by_question_id(expected, actual, "action_traces.jsonl", result_name="action_trace")


def _compare_json_file(expected: Path, actual: Path, filename: str) -> dict[str, Any]:
    expected_path = expected / filename
    actual_path = actual / filename
    if not expected_path.exists() or not actual_path.exists():
        return {"passed": False, "path": filename, "missing": _missing_sides(expected_path, actual_path), "field_differences": []}
    expected_value = _load_json(expected_path)
    actual_value = _load_json(actual_path)
    differences = _field_differences(expected_value, actual_value)
    return {"passed": not differences, "path": filename, "field_differences": differences}


def _compare_jsonl_by_question_id(expected: Path, actual: Path, filename: str, *, result_name: str) -> dict[str, Any]:
    loaded = _load_jsonl_pair(expected, actual, filename)
    if loaded["missing"]:
        return {"passed": False, "path": filename, "missing": loaded["missing"], "changed_question_ids": [], "field_differences": []}
    expected_by_id = _index_by_question_id(loaded["expected"], filename)
    actual_by_id = _index_by_question_id(loaded["actual"], filename)
    differences: list[dict[str, Any]] = []
    for qid in sorted(set(expected_by_id) | set(actual_by_id)):
        if qid not in expected_by_id or qid not in actual_by_id:
            differences.append({"question_id": qid, "type": f"missing_{result_name}", "missing": "expected" if qid not in expected_by_id else "actual"})
            continue
        for diff in _field_differences(expected_by_id[qid], actual_by_id[qid]):
            diff["question_id"] = qid
            diff["type"] = f"{result_name}_field_changed"
            differences.append(diff)
    return {
        "passed": not differences,
        "path": filename,
        "changed_question_ids": sorted({diff["question_id"] for diff in differences if "question_id" in diff}),
        "field_differences": differences,
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {path}: {exc}") from exc


def _load_jsonl_pair(expected: Path, actual: Path, filename: str) -> dict[str, Any]:
    expected_path = expected / filename
    actual_path = actual / filename
    missing = _missing_sides(expected_path, actual_path)
    if missing:
        return {"missing": missing, "expected": [], "actual": []}
    return {"missing": [], "expected": read_jsonl(expected_path), "actual": read_jsonl(actual_path)}


def _index_by_question_id(rows: list[dict[str, Any]], filename: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ValidationError(f"{filename} rows require non-empty question_id for replay comparison.")
        if qid in indexed:
            raise ValidationError(f"Duplicate question_id in {filename}: {qid}")
        indexed[qid] = row
    return indexed


def _field_differences(expected: Any, actual: Any, *, prefix: str = "", ignore: set[str] | None = None) -> list[dict[str, Any]]:
    ignore = ignore or set()
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            if not prefix and key in ignore:
                continue
            field = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                differences.append(
                    {
                        "field": field,
                        "expected_present": key in expected,
                        "actual_present": key in actual,
                        "expected": _compact_value(expected.get(key)),
                        "actual": _compact_value(actual.get(key)),
                    }
                )
                continue
            differences.extend(_field_differences(expected[key], actual[key], prefix=field))
        return differences
    if expected != actual:
        return [{"field": prefix, "expected": _compact_value(expected), "actual": _compact_value(actual)}]
    return []


def _missing_sides(expected_path: Path, actual_path: Path) -> list[str]:
    missing: list[str] = []
    if not expected_path.exists():
        missing.append("expected")
    if not actual_path.exists():
        missing.append("actual")
    return missing


def _value_diff(question_id: str, field: str, expected: Any, actual: Any, *, diff_type: str = "prediction_field_changed") -> dict[str, Any]:
    return {
        "question_id": question_id,
        "type": diff_type,
        "field": field,
        "expected": _compact_value(expected),
        "actual": _compact_value(actual),
    }


def _hashed_value_diff(question_id: str, field: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "type": "generated_answer_changed",
        "field": field,
        "expected_sha256": sha256_json(expected),
        "actual_sha256": sha256_json(actual),
        "expected_length": len(expected) if isinstance(expected, str) else None,
        "actual_length": len(actual) if isinstance(actual, str) else None,
    }


def _compact_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 160:
            return {"sha256": sha256_json(value), "length": len(value)}
        return value
    return {"sha256": sha256_json(value), "type": type(value).__name__}


def _summary(
    *,
    artifact_results: list[dict[str, Any]],
    manifest_result: dict[str, Any],
    prediction_results: dict[str, Any],
    score_results: dict[str, Any],
    metadata_result: dict[str, Any],
    action_trace_results: dict[str, Any] | None,
    required_artifacts_passed: bool,
    optional_artifacts_passed: bool,
    behavior_artifacts_passed: bool,
    behavior_passed: bool,
    manifest_passed: bool,
) -> dict[str, Any]:
    changed_artifacts = sorted(item["path"] for item in artifact_results if not item["passed"])
    changed_question_ids = sorted(
        set(prediction_results.get("changed_question_ids", []))
        | set(score_results.get("item_scores", {}).get("changed_question_ids", []))
        | set((action_trace_results or {}).get("changed_question_ids", []))
    )
    true_behavior_difference_count = (
        len(prediction_results.get("differences", []))
        + len(score_results.get("item_scores", {}).get("field_differences", []))
        + len(metadata_result.get("field_differences", []))
        + len((action_trace_results or {}).get("field_differences", []))
    )
    aggregate_score_difference_count = len(score_results.get("aggregate_scores", {}).get("field_differences", []))
    artifact_byte_equality_passed = not changed_artifacts
    provenance_sensitive = bool(manifest_result.get("volatile_differences")) or aggregate_score_difference_count > 0
    required_or_optional_missing = not required_artifacts_passed or not optional_artifacts_passed
    replay_matched = not required_or_optional_missing and true_behavior_difference_count == 0
    if true_behavior_difference_count:
        outcome = "behavior_mismatch"
        operator_status = "MISMATCHED"
    elif required_or_optional_missing:
        outcome = "incomplete_artifacts"
        operator_status = "INCOMPLETE"
    elif not manifest_passed or provenance_sensitive or not artifact_byte_equality_passed:
        outcome = "matched_with_provenance_or_artifact_differences"
        operator_status = "MATCHED_WITH_DIFFERENCES"
    else:
        outcome = "exact_match"
        operator_status = "MATCHED_EXACTLY"
    return {
        "replay_outcome": outcome,
        "operator_status": operator_status,
        "replay_matched": replay_matched,
        "behavior_equality_passed": replay_matched,
        "behavior_artifact_byte_equality_passed": behavior_artifacts_passed,
        "provenance_equality_passed": bool(manifest_result.get("passed")),
        "provenance_sensitive_difference_count": len(manifest_result.get("field_differences", [])) + len(manifest_result.get("volatile_differences", [])) + aggregate_score_difference_count,
        "artifact_byte_equality_passed": artifact_byte_equality_passed,
        "aggregate_score_equality_passed": aggregate_score_difference_count == 0,
        "changed_artifact_count": len(changed_artifacts),
        "changed_artifacts": changed_artifacts,
        "changed_question_count": len(changed_question_ids),
        "changed_question_ids": changed_question_ids,
        "true_behavior_difference_count": true_behavior_difference_count,
        "manifest_field_difference_count": len(manifest_result.get("field_differences", [])),
        "manifest_volatile_difference_count": len(manifest_result.get("volatile_differences", [])),
        "prediction_difference_count": len(prediction_results.get("differences", [])),
        "item_score_difference_count": len(score_results.get("item_scores", {}).get("field_differences", [])),
        "aggregate_score_difference_count": len(score_results.get("aggregate_scores", {}).get("field_differences", [])),
        "metadata_difference_count": len(metadata_result.get("field_differences", [])),
        "action_trace_difference_count": len((action_trace_results or {}).get("field_differences", [])),
    }
