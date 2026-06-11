from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .hashing import sha256_json
from .manifests import canonical_manifest_json, read_manifest


SNAPSHOT_DIFF_SCHEMA_VERSION = "retrieval_arena.snapshot_diff.v1"
VOLATILE_MANIFEST_FIELDS = ("created_at", "manifest_hash")
MANIFEST_FILENAMES = {
    "corpus": "corpus_snapshot_manifest.json",
    "graph": "graph_snapshot_manifest.json",
    "support_surface": "support_surface_manifest.json",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_snapshot_diff_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_manifest_json(report), encoding="utf-8")


def write_snapshot_diff_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_snapshot_diff_markdown(report), encoding="utf-8")


def compare_snapshot_bundles(
    before: Path,
    after: Path,
    *,
    created_at: str | None = None,
    out_path: Path | None = None,
    markdown_out_path: Path | None = None,
) -> dict[str, Any]:
    before_bundle = load_snapshot_bundle(before)
    after_bundle = load_snapshot_bundle(after)

    corpus_result = compare_corpus_manifests(before_bundle["corpus"]["manifest"], after_bundle["corpus"]["manifest"])
    graph_result = compare_optional_graph(before_bundle.get("graph"), after_bundle.get("graph"))
    support_surface_result = compare_optional_support_surface(before_bundle.get("support_surface"), after_bundle.get("support_surface"))
    manifest_results = compare_manifest_bundles(before_bundle, after_bundle)

    summary = _summary(corpus_result, graph_result, support_surface_result, manifest_results)
    report = {
        "schema_version": SNAPSHOT_DIFF_SCHEMA_VERSION,
        "created_at": created_at or utc_now_iso(),
        "comparison_type": "snapshot_diff",
        "before": _bundle_reference(before_bundle),
        "after": _bundle_reference(after_bundle),
        "passed": summary["total_delta_count"] == 0,
        "summary": summary,
        "corpus_result": corpus_result,
        "graph_result": graph_result,
        "support_surface_result": support_surface_result,
        "manifest_results": manifest_results,
    }
    if out_path is not None:
        write_snapshot_diff_report(out_path, report)
    if markdown_out_path is not None:
        write_snapshot_diff_markdown(markdown_out_path, report)
    return report


def load_snapshot_bundle(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_dir():
        base_dir = resolved
        corpus_path = base_dir / MANIFEST_FILENAMES["corpus"]
    else:
        base_dir = resolved.parent
        corpus_path = resolved
    if not corpus_path.exists():
        raise ValidationError(f"Missing required corpus snapshot manifest: {corpus_path}")

    corpus_manifest = read_manifest(corpus_path, verify_hash=False)
    if corpus_manifest.get("manifest_type") != "corpus_snapshot":
        raise ValidationError(f"Required corpus manifest has wrong manifest_type: {corpus_path}")

    bundle: dict[str, Any] = {
        "input_path": str(resolved),
        "base_dir": str(base_dir),
        "corpus": {"path": str(corpus_path), "manifest": corpus_manifest},
    }
    for bundle_key, filename in (("graph", MANIFEST_FILENAMES["graph"]), ("support_surface", MANIFEST_FILENAMES["support_surface"])):
        optional_path = base_dir / filename
        if optional_path.exists():
            bundle[bundle_key] = {"path": str(optional_path), "manifest": read_manifest(optional_path, verify_hash=False)}
    return bundle


def compare_corpus_manifests(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_files = _file_inventory_by_path(before)
    after_files = _file_inventory_by_path(after)
    added = [_file_added(after_files[path]) for path in sorted(set(after_files) - set(before_files))]
    removed = [_file_removed(before_files[path]) for path in sorted(set(before_files) - set(after_files))]
    changed = []
    for path in sorted(set(before_files) & set(after_files)):
        before_item = before_files[path]
        after_item = after_files[path]
        if _file_identity(before_item) == _file_identity(after_item):
            continue
        changed.append(
            {
                "path": path,
                "doc_id": after_item.get("doc_id", before_item.get("doc_id")),
                "before_doc_id": before_item.get("doc_id"),
                "after_doc_id": after_item.get("doc_id"),
                "before_sha256": before_item.get("sha256"),
                "after_sha256": after_item.get("sha256"),
                "before_size_bytes": before_item.get("size_bytes"),
                "after_size_bytes": after_item.get("size_bytes"),
                "size_delta_bytes": _number(after_item.get("size_bytes")) - _number(before_item.get("size_bytes")),
            }
        )
    before_page_count = _number(before.get("page_count", len(before_files)))
    after_page_count = _number(after.get("page_count", len(after_files)))
    before_bytes = _number(before.get("corpus_size_bytes"))
    after_bytes = _number(after.get("corpus_size_bytes"))
    return {
        "available": True,
        "passed": not added and not removed and not changed and before_bytes == after_bytes and before_page_count == after_page_count,
        "before_snapshot_id": before.get("snapshot_id"),
        "after_snapshot_id": after.get("snapshot_id"),
        "before_content_hash": before.get("content_hash"),
        "after_content_hash": after.get("content_hash"),
        "before_page_count": before_page_count,
        "after_page_count": after_page_count,
        "page_count_delta": after_page_count - before_page_count,
        "before_corpus_size_bytes": before_bytes,
        "after_corpus_size_bytes": after_bytes,
        "corpus_size_delta_bytes": after_bytes - before_bytes,
        "added_files": added,
        "removed_files": removed,
        "changed_files": changed,
        "added_file_count": len(added),
        "removed_file_count": len(removed),
        "changed_file_count": len(changed),
    }


def compare_optional_graph(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    if before is None or after is None:
        return _unavailable_optional("graph", before, after)
    before_manifest = before["manifest"]
    after_manifest = after["manifest"]
    before_edges = _load_graph_edges(before)
    after_edges = _load_graph_edges(after)
    added_edges = [_edge_result(edge) for edge in sorted(after_edges - before_edges)]
    removed_edges = [_edge_result(edge) for edge in sorted(before_edges - after_edges)]
    descriptor_deltas = _graph_descriptor_deltas(before_manifest, after_manifest)
    return {
        "available": True,
        "passed": not added_edges and not removed_edges and not descriptor_deltas,
        "before_snapshot_id": before_manifest.get("snapshot_id"),
        "after_snapshot_id": after_manifest.get("snapshot_id"),
        "before_graph_hash": before_manifest.get("graph_hash"),
        "after_graph_hash": after_manifest.get("graph_hash"),
        "before_node_count": before_manifest.get("node_count"),
        "after_node_count": after_manifest.get("node_count"),
        "node_count_delta": _number(after_manifest.get("node_count")) - _number(before_manifest.get("node_count")),
        "before_edge_count": before_manifest.get("edge_count", len(before_edges)),
        "after_edge_count": after_manifest.get("edge_count", len(after_edges)),
        "edge_count_delta": _number(after_manifest.get("edge_count", len(after_edges))) - _number(before_manifest.get("edge_count", len(before_edges))),
        "edge_source": _edge_source(before, after),
        "added_edges": added_edges,
        "removed_edges": removed_edges,
        "added_edge_count": len(added_edges),
        "removed_edge_count": len(removed_edges),
        "descriptor_deltas": descriptor_deltas,
        "descriptor_delta_count": len(descriptor_deltas),
    }


def compare_optional_support_surface(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    if before is None or after is None:
        return _unavailable_optional("support_surface", before, after)
    before_manifest = before["manifest"]
    after_manifest = after["manifest"]
    before_targets = _support_targets(before_manifest)
    after_targets = _support_targets(after_manifest)
    added_questions = sorted(set(after_targets) - set(before_targets))
    removed_questions = sorted(set(before_targets) - set(after_targets))
    changed_questions: list[dict[str, Any]] = []
    added_targets_total = 0
    removed_targets_total = 0
    for question_id in sorted(set(before_targets) & set(after_targets)):
        before_set = set(before_targets[question_id])
        after_set = set(after_targets[question_id])
        added_targets = sorted(after_set - before_set)
        removed_targets = sorted(before_set - after_set)
        if not added_targets and not removed_targets:
            continue
        added_targets_total += len(added_targets)
        removed_targets_total += len(removed_targets)
        changed_questions.append(
            {
                "question_id": question_id,
                "added_targets": added_targets,
                "removed_targets": removed_targets,
                "before_targets": before_targets[question_id],
                "after_targets": after_targets[question_id],
            }
        )
    added_question_results = [{"question_id": qid, "targets": after_targets[qid]} for qid in added_questions]
    removed_question_results = [{"question_id": qid, "targets": before_targets[qid]} for qid in removed_questions]
    added_targets_total += sum(len(item["targets"]) for item in added_question_results)
    removed_targets_total += sum(len(item["targets"]) for item in removed_question_results)
    before_all_targets = set(before_manifest.get("support_target_doc_ids", []))
    after_all_targets = set(after_manifest.get("support_target_doc_ids", []))
    return {
        "available": True,
        "passed": not added_question_results and not removed_question_results and not changed_questions,
        "before_snapshot_id": before_manifest.get("snapshot_id"),
        "after_snapshot_id": after_manifest.get("snapshot_id"),
        "before_query_set_id": before_manifest.get("query_set_id"),
        "after_query_set_id": after_manifest.get("query_set_id"),
        "before_support_target_count": before_manifest.get("support_target_count", len(before_all_targets)),
        "after_support_target_count": after_manifest.get("support_target_count", len(after_all_targets)),
        "support_target_count_delta": len(after_all_targets) - len(before_all_targets),
        "added_support_target_doc_ids": sorted(after_all_targets - before_all_targets),
        "removed_support_target_doc_ids": sorted(before_all_targets - after_all_targets),
        "added_questions": added_question_results,
        "removed_questions": removed_question_results,
        "changed_questions": changed_questions,
        "added_question_count": len(added_question_results),
        "removed_question_count": len(removed_question_results),
        "changed_question_count": len(changed_questions),
        "added_target_reference_count": added_targets_total,
        "removed_target_reference_count": removed_targets_total,
    }


def compare_manifest_bundles(before_bundle: dict[str, Any], after_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for key in ("corpus", "graph", "support_surface"):
        before = before_bundle.get(key)
        after = after_bundle.get(key)
        if before is None or after is None:
            results.append(
                {
                    "manifest": key,
                    "available": False,
                    "missing": _missing_bundle_sides(before, after),
                    "passed": False,
                    "only_volatile_differences": False,
                    "field_differences": [],
                    "volatile_differences": [],
                }
            )
            continue
        field_differences = _field_differences(before["manifest"], after["manifest"])
        volatile_differences = [diff for diff in field_differences if diff["field"] in VOLATILE_MANIFEST_FIELDS]
        stable_differences = [diff for diff in field_differences if diff["field"] not in VOLATILE_MANIFEST_FIELDS]
        results.append(
            {
                "manifest": key,
                "available": True,
                "before_path": before["path"],
                "after_path": after["path"],
                "passed": not field_differences,
                "only_volatile_differences": bool(volatile_differences) and not stable_differences,
                "volatile_fields": list(VOLATILE_MANIFEST_FIELDS),
                "volatile_differences": volatile_differences,
                "field_differences": stable_differences,
                "before_manifest_hash": before["manifest"].get("manifest_hash"),
                "after_manifest_hash": after["manifest"].get("manifest_hash"),
            }
        )
    return results


def render_snapshot_diff_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Snapshot Diff",
        "",
        f"- Status: {'PASSED' if report['passed'] else 'FAILED'}",
        f"- Corpus files: +{summary['added_file_count']} / -{summary['removed_file_count']} / changed {summary['changed_file_count']}",
        f"- Graph edges: +{summary['added_edge_count']} / -{summary['removed_edge_count']}",
        f"- Support questions changed: {summary['changed_support_question_count']}",
        "",
        "## Corpus",
        "",
    ]
    for label, key in (("Added", "added_files"), ("Removed", "removed_files"), ("Changed", "changed_files")):
        lines.append(f"### {label} Files")
        items = report["corpus_result"][key]
        if not items:
            lines.append("- None")
        else:
            for item in items:
                lines.append(f"- `{item['path']}`")
        lines.append("")
    if report["graph_result"]["available"]:
        lines.extend(["## Graph", ""])
        for label, key in (("Added", "added_edges"), ("Removed", "removed_edges")):
            lines.append(f"### {label} Edges")
            items = report["graph_result"][key]
            if not items:
                lines.append("- None")
            else:
                for item in items:
                    lines.append(f"- `{item['source']}` -> `{item['target']}`")
            lines.append("")
    if report["support_surface_result"]["available"]:
        lines.extend(["## Support Surface", ""])
        changed = report["support_surface_result"]["changed_questions"]
        if not changed and not report["support_surface_result"]["added_questions"] and not report["support_surface_result"]["removed_questions"]:
            lines.append("- None")
        for item in report["support_surface_result"]["added_questions"]:
            lines.append(f"- Added question `{item['question_id']}`")
        for item in report["support_surface_result"]["removed_questions"]:
            lines.append(f"- Removed question `{item['question_id']}`")
        for item in changed:
            lines.append(f"- Changed question `{item['question_id']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _summary(
    corpus_result: dict[str, Any],
    graph_result: dict[str, Any],
    support_surface_result: dict[str, Any],
    manifest_results: list[dict[str, Any]],
) -> dict[str, Any]:
    stable_manifest_difference_count = sum(len(result.get("field_differences", [])) for result in manifest_results if result.get("available"))
    total_delta_count = (
        corpus_result["added_file_count"]
        + corpus_result["removed_file_count"]
        + corpus_result["changed_file_count"]
        + graph_result.get("added_edge_count", 0)
        + graph_result.get("removed_edge_count", 0)
        + graph_result.get("descriptor_delta_count", 0)
        + support_surface_result.get("added_question_count", 0)
        + support_surface_result.get("removed_question_count", 0)
        + support_surface_result.get("changed_question_count", 0)
    )
    return {
        "added_file_count": corpus_result["added_file_count"],
        "removed_file_count": corpus_result["removed_file_count"],
        "changed_file_count": corpus_result["changed_file_count"],
        "page_count_delta": corpus_result["page_count_delta"],
        "corpus_size_delta_bytes": corpus_result["corpus_size_delta_bytes"],
        "graph_available": graph_result["available"],
        "added_edge_count": graph_result.get("added_edge_count", 0),
        "removed_edge_count": graph_result.get("removed_edge_count", 0),
        "graph_descriptor_delta_count": graph_result.get("descriptor_delta_count", 0),
        "support_surface_available": support_surface_result["available"],
        "added_support_question_count": support_surface_result.get("added_question_count", 0),
        "removed_support_question_count": support_surface_result.get("removed_question_count", 0),
        "changed_support_question_count": support_surface_result.get("changed_question_count", 0),
        "stable_manifest_difference_count": stable_manifest_difference_count,
        "volatile_manifest_difference_count": sum(len(result.get("volatile_differences", [])) for result in manifest_results if result.get("available")),
        "unavailable_optional_manifests": [
            result["manifest"] for result in manifest_results if not result["available"] and result["manifest"] != "corpus"
        ],
        "total_delta_count": total_delta_count,
    }


def _bundle_reference(bundle: dict[str, Any]) -> dict[str, Any]:
    corpus = bundle["corpus"]["manifest"]
    return {
        "input_path": bundle["input_path"],
        "base_dir": bundle["base_dir"],
        "corpus_id": corpus.get("corpus_id"),
        "snapshot_id": corpus.get("snapshot_id"),
        "corpus_manifest_path": bundle["corpus"]["path"],
        "corpus_manifest_hash": corpus.get("manifest_hash"),
    }


def _file_inventory_by_path(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = manifest.get("file_inventory")
    if not isinstance(files, list):
        raise ValidationError("Corpus snapshot manifest requires file_inventory.")
    by_path: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
            raise ValidationError("Corpus file_inventory entries require non-empty path.")
        if item["path"] in by_path:
            raise ValidationError(f"Duplicate corpus file_inventory path: {item['path']}")
        by_path[item["path"]] = item
    return by_path


def _file_added(item: dict[str, Any]) -> dict[str, Any]:
    return {"path": item["path"], "doc_id": item.get("doc_id"), "sha256": item.get("sha256"), "size_bytes": item.get("size_bytes")}


def _file_removed(item: dict[str, Any]) -> dict[str, Any]:
    return _file_added(item)


def _file_identity(item: dict[str, Any]) -> tuple[Any, Any, Any]:
    return item.get("sha256"), item.get("size_bytes"), item.get("doc_id")


def _load_graph_edges(bundle_item: dict[str, Any]) -> set[tuple[str, str]]:
    manifest = bundle_item["manifest"]
    for key in ("edge_inventory", "graph_edges", "edges"):
        if isinstance(manifest.get(key), list):
            return _edges_from_list(manifest[key], key)
    edge_file = manifest.get("edge_file")
    if isinstance(edge_file, str) and edge_file:
        path = Path(bundle_item["path"]).parent / edge_file
        if path.exists():
            return _edges_from_csv(path)
    return set()


def _edges_from_list(rows: list[Any], key: str) -> set[tuple[str, str]]:
    edges = set()
    for row in rows:
        if isinstance(row, dict):
            source = row.get("source")
            target = row.get("target")
        elif isinstance(row, list | tuple) and len(row) == 2:
            source, target = row
        else:
            raise ValidationError(f"Graph {key} entries require source and target.")
        if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
            raise ValidationError(f"Graph {key} entries require non-empty source and target.")
        edges.add((source, target))
    return edges


def _edges_from_csv(path: Path) -> set[tuple[str, str]]:
    edges = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source", "target"]:
            raise ValidationError(f"Graph edge file must have source,target header: {path}")
        for row in reader:
            source = row.get("source")
            target = row.get("target")
            if isinstance(source, str) and source and isinstance(target, str) and target:
                edges.add((source, target))
    return edges


def _edge_result(edge: tuple[str, str]) -> dict[str, str]:
    return {"source": edge[0], "target": edge[1], "edge_id": f"{edge[0]}->{edge[1]}"}


def _edge_source(before: dict[str, Any], after: dict[str, Any]) -> str:
    if _load_graph_edges(before) or _load_graph_edges(after):
        return "manifest_or_referenced_file"
    return "unavailable"


def _graph_descriptor_deltas(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    compared_fields = {
        "node_count",
        "edge_count",
        "graph_metrics",
        "component_descriptors",
        "components",
        "weak_components",
        "strong_components",
        "largest_component_size",
    }
    before_descriptors = {key: before.get(key) for key in compared_fields if key in before}
    after_descriptors = {key: after.get(key) for key in compared_fields if key in after}
    return _field_differences(before_descriptors, after_descriptors)


def _support_targets(manifest: dict[str, Any]) -> dict[str, list[str]]:
    value = manifest.get("support_targets_by_question")
    if not isinstance(value, dict):
        raise ValidationError("Support-surface manifest requires support_targets_by_question.")
    targets: dict[str, list[str]] = {}
    for question_id in sorted(value):
        docs = value[question_id]
        if not isinstance(question_id, str) or not isinstance(docs, list):
            raise ValidationError("support_targets_by_question must map question IDs to lists.")
        targets[question_id] = sorted(str(doc_id) for doc_id in docs)
    return targets


def _unavailable_optional(name: str, before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    return {"available": False, "passed": False, "manifest_type": name, "missing": _missing_bundle_sides(before, after)}


def _missing_bundle_sides(before: dict[str, Any] | None, after: dict[str, Any] | None) -> list[str]:
    missing = []
    if before is None:
        missing.append("before")
    if after is None:
        missing.append("after")
    return missing


def _field_differences(before: Any, after: Any, *, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        differences = []
        for key in sorted(set(before) | set(after)):
            field = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                differences.append(
                    {
                        "field": field,
                        "before_present": key in before,
                        "after_present": key in after,
                        "before": _compact_value(before.get(key)),
                        "after": _compact_value(after.get(key)),
                    }
                )
                continue
            differences.extend(_field_differences(before[key], after[key], prefix=field))
        return differences
    if before != after:
        return [{"field": prefix, "before": _compact_value(before), "after": _compact_value(after)}]
    return []


def _compact_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 160:
            return {"sha256": sha256_json(value), "length": len(value)}
        return value
    return {"sha256": sha256_json(value), "type": type(value).__name__}


def _number(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
