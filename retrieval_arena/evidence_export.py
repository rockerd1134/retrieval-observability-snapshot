from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .html_report import assemble_report_data


EVIDENCE_EXPORT_SCHEMA_VERSION = "retrieval_arena.paper_evidence_export.v1"


def export_paper_evidence(bundle_root: Path, out_dir: Path) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = assemble_report_data(bundle_root)
    artifacts = [
        _write_csv(out_dir / "corpus_graph_support_summary.csv", _corpus_graph_support_rows(report)),
        _write_csv(out_dir / "family_drift_matrix.csv", _family_drift_rows(report)),
        _write_csv(out_dir / "audit_cause_labels.csv", _audit_label_rows(report)),
        _write_csv(out_dir / "systems_storage_measurements.csv", _systems_rows(report)),
        _write_csv(out_dir / "selected_case_studies.csv", _case_study_rows(report)),
        _write_text(out_dir / "provenance_replay_summary.md", _provenance_summary(report)),
    ]
    return {
        "ok": True,
        "schema_version": EVIDENCE_EXPORT_SCHEMA_VERSION,
        "summary": f"Paper evidence exported {len(artifacts)} artifacts to {out_dir}",
        "bundle_root": str(bundle_root),
        "out_dir": str(out_dir),
        "written_artifacts": [str(path) for path in artifacts],
    }


def _corpus_graph_support_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    manifests = report.get("snapshot_manifests", {}) if isinstance(report.get("snapshot_manifests"), dict) else {}
    summary = report.get("snapshot_diff", {}).get("summary", {}) if isinstance(report.get("snapshot_diff"), dict) else {}
    before_corpus = manifests.get("before_corpus", {})
    after_corpus = manifests.get("after_corpus", {})
    before_graph = manifests.get("before_graph", {})
    after_graph = manifests.get("after_graph", {})
    before_support = manifests.get("before_support", {})
    after_support = manifests.get("after_support", {})
    return [
        {
            "schema_version": EVIDENCE_EXPORT_SCHEMA_VERSION,
            "pilot_id": report.get("pilot_id"),
            "corpus_id": report.get("corpus_id"),
            "query_set_id": report.get("query_set_id"),
            "before_snapshot_id": before_corpus.get("snapshot_id"),
            "after_snapshot_id": after_corpus.get("snapshot_id"),
            "before_pages": before_corpus.get("page_count"),
            "after_pages": after_corpus.get("page_count"),
            "page_count_delta": summary.get("page_count_delta"),
            "changed_file_count": summary.get("changed_file_count"),
            "added_file_count": summary.get("added_file_count"),
            "removed_file_count": summary.get("removed_file_count"),
            "before_graph_edges": before_graph.get("edge_count"),
            "after_graph_edges": after_graph.get("edge_count"),
            "added_edge_count": summary.get("added_edge_count"),
            "removed_edge_count": summary.get("removed_edge_count"),
            "before_support_targets": before_support.get("support_target_count"),
            "after_support_targets": after_support.get("support_target_count"),
            "changed_support_question_count": summary.get("changed_support_question_count"),
            "source_bundle": report.get("bundle_root"),
        }
    ]


def _family_drift_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comparison in report.get("comparisons", []):
        drift = comparison.get("drift", {}) if isinstance(comparison.get("drift"), dict) else {}
        rows.append(
            {
                "schema_version": EVIDENCE_EXPORT_SCHEMA_VERSION,
                "pilot_id": report.get("pilot_id"),
                "corpus_id": report.get("corpus_id"),
                "comparison_id": comparison.get("comparison_id"),
                "query_count": drift.get("query_count"),
                "mean_top_k_jaccard": drift.get("mean_top_k_jaccard"),
                "mean_ordered_top_k_overlap": drift.get("mean_ordered_top_k_overlap"),
                "mean_rank_displacement": drift.get("mean_rank_displacement"),
                "support_exposure_regression_count": drift.get("support_exposure_regression_count"),
                "evidence_coverage_regression_count": drift.get("evidence_coverage_regression_count"),
                "support_recall_regression_count": drift.get("support_recall_regression_count"),
                "drift_summary_path": comparison.get("paths", {}).get("drift_summary"),
                "drift_rows_path": comparison.get("paths", {}).get("drift_rows"),
            }
        )
    return rows


def _audit_label_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comparison in report.get("comparisons", []):
        audit = comparison.get("audit", {}) if isinstance(comparison.get("audit"), dict) else {}
        counts = audit.get("cause_label_counts", {}) if isinstance(audit.get("cause_label_counts"), dict) else {}
        for label in sorted(counts):
            rows.append(
                {
                    "schema_version": EVIDENCE_EXPORT_SCHEMA_VERSION,
                    "pilot_id": report.get("pilot_id"),
                    "corpus_id": report.get("corpus_id"),
                    "comparison_id": comparison.get("comparison_id"),
                    "cause_label": label,
                    "count": counts[label],
                    "labeled_query_count": audit.get("labeled_query_count"),
                    "audit_summary_path": comparison.get("paths", {}).get("audit_summary"),
                    "audit_rows_path": comparison.get("paths", {}).get("audit_rows"),
                }
            )
    return rows


def _systems_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comparison in report.get("comparisons", []):
        systems = comparison.get("systems", {}) if isinstance(comparison.get("systems"), dict) else {}
        workload = systems.get("workload_metrics", {}) if isinstance(systems.get("workload_metrics"), dict) else {}
        artifacts = systems.get("artifact_metrics", {}) if isinstance(systems.get("artifact_metrics"), dict) else {}
        ratios = artifacts.get("ratios", {}) if isinstance(artifacts.get("ratios"), dict) else {}
        rows.append(
            {
                "schema_version": EVIDENCE_EXPORT_SCHEMA_VERSION,
                "pilot_id": report.get("pilot_id"),
                "corpus_id": report.get("corpus_id"),
                "comparison_id": comparison.get("comparison_id"),
                "page_count": workload.get("page_count"),
                "query_count": workload.get("query_count"),
                "graph_edge_count": workload.get("graph_edge_count"),
                "support_target_count": workload.get("support_target_count"),
                "drift_row_count": workload.get("drift_row_count"),
                "audit_row_count": workload.get("audit_row_count"),
                "report_bytes_per_query": _ratio_value(ratios, "report_bytes_per_query"),
                "artifact_bytes_per_page": _ratio_value(ratios, "artifact_bytes_per_page"),
                "systems_measurements_path": comparison.get("paths", {}).get("systems_measurements"),
            }
        )
    return rows


def _case_study_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comparison in report.get("comparisons", []):
        for case in comparison.get("case_studies", [])[:4]:
            metrics = case.get("drift_metrics", {}) if isinstance(case.get("drift_metrics"), dict) else {}
            rows.append(
                {
                    "schema_version": EVIDENCE_EXPORT_SCHEMA_VERSION,
                    "pilot_id": report.get("pilot_id"),
                    "corpus_id": report.get("corpus_id"),
                    "comparison_id": comparison.get("comparison_id"),
                    "question_id": case.get("question_id"),
                    "cause_labels": ";".join(case.get("cause_labels", [])) if isinstance(case.get("cause_labels"), list) else "",
                    "before_doc_ids": ";".join(case.get("before", {}).get("doc_ids", [])),
                    "after_doc_ids": ";".join(case.get("after", {}).get("doc_ids", [])),
                    "top_k_jaccard": _nested(metrics, "top_k_jaccard", "value"),
                    "ordered_top_k_overlap": _nested(metrics, "ordered_top_k_overlap", "value"),
                    "support_exposure_delta": _nested(metrics, "support_exposure", "exposed_count_delta"),
                    "evidence_coverage_delta": _nested(metrics, "evidence_coverage", "coverage_delta"),
                    "support_recall_delta": _nested(metrics, "support_recall", "recall_delta"),
                    "distance_to_support_delta": _nested(metrics, "distance_to_support", "min_distance_delta"),
                    "action_trace_delta": _nested(metrics, "action_trace", "action_count_delta"),
                    "audit_rows_path": comparison.get("paths", {}).get("audit_rows"),
                }
            )
    return rows


def _provenance_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Provenance And Replay Summary",
        "",
        f"- Schema version: `{EVIDENCE_EXPORT_SCHEMA_VERSION}`",
        f"- Pilot: `{report.get('pilot_id')}`",
        f"- Corpus: `{report.get('corpus_id')}`",
        f"- Query set: `{report.get('query_set_id')}`",
        f"- Source bundle: `{report.get('bundle_root')}`",
        "- Source artifacts: `pilot_manifest.json`, `plan_resolved.json`, `snapshot_comparison/snapshot_diff.json`, per-family drift, audit, systems, and replay manifests.",
        "",
        "## Retrieval Families",
        "",
    ]
    for comparison in report.get("comparisons", []):
        paths = comparison.get("paths", {}) if isinstance(comparison.get("paths"), dict) else {}
        lines.extend(
            [
                f"### {comparison.get('comparison_id')}",
                "",
                f"- Drift summary: `{paths.get('drift_summary')}`",
                f"- Audit rows: `{paths.get('audit_rows')}`",
                f"- Systems measurements: `{paths.get('systems_measurements')}`",
                f"- Before replay manifest: `{paths.get('before_replay')}`",
                f"- After replay manifest: `{paths.get('after_replay')}`",
                "",
            ]
        )
    return "\n".join(lines)


def _ratio_value(ratios: dict[str, Any], key: str) -> Any:
    raw = ratios.get(key, {}) if isinstance(ratios.get(key), dict) else {}
    return raw.get("value") if raw.get("available") else ""


def _nested(raw: dict[str, Any], first: str, second: str) -> Any:
    child = raw.get(first, {}) if isinstance(raw.get(first), dict) else {}
    return child.get(second)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path
