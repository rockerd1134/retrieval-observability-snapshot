from __future__ import annotations

from pathlib import Path
from typing import Any

from .longitudinal_pilot import longitudinal_pilot as run_longitudinal_pilot
from .git_provenance import describe_git_provenance, plan_git_comparison
from .harness import run_config, validate_phase1
from .audit import build_regression_audit
from .drift import compare_retrieval_runs
from .evidence_export import export_paper_evidence
from .html_report import build_html_observability_report
from .measurements import collect_systems_measurements
from .replay_compare import compare_run_dirs
from .snapshot_diff import compare_snapshot_bundles
from .snapshots import write_corpus_snapshot_manifest, write_graph_snapshot_manifest, write_support_surface_manifest


def run_experiment(config_path: Path | str) -> dict[str, Any]:
    rows = run_config(config_path)
    return {"ok": True, "summary": f"retrieval audit run complete: {len(rows)} dataset/test runs", "run_count": len(rows)}


def validate_experiment(config_path: Path | str) -> dict[str, Any]:
    validate_phase1(config_path)
    return {"ok": True, "summary": "Retrieval Audit Framework Phase 1 validation passed"}


def longitudinal_pilot(
    plan_path: Path,
    *,
    dry_run: bool = False,
    stage: str | None = None,
    from_stage: str | None = None,
    force: bool = False,
    no_baseline_bundle: bool = False,
    refresh_baseline: bool = False,
) -> dict[str, Any]:
    return run_longitudinal_pilot(
        plan_path,
        dry_run=dry_run,
        stage=stage,
        from_stage=from_stage,
        force=force,
        no_baseline_bundle=no_baseline_bundle,
        refresh_baseline=refresh_baseline,
    )


def git_provenance(path: Path, *, ref: str | None = None) -> dict[str, Any]:
    provenance = describe_git_provenance(path, ref=ref)
    return {"ok": True, "summary": "Git provenance captured", "provenance": provenance}


def git_comparison_plan(repo_path: Path, *, left_ref: str, right_ref: str, output_dir: Path | None = None) -> dict[str, Any]:
    plan = plan_git_comparison(repo_path, left_ref=left_ref, right_ref=right_ref, output_dir=output_dir)
    return {"ok": True, "summary": "Git comparison plan created", "plan": plan}


def replay_compare(expected_run_dir: Path, actual_run_dir: Path, *, out_path: Path | None = None) -> dict[str, Any]:
    report = compare_run_dirs(expected_run_dir, actual_run_dir, out_path=out_path)
    summary = report["summary"]
    status = summary["operator_status"]
    detail = (
        f"{summary['changed_question_count']} changed questions, "
        f"{summary['changed_artifact_count']} changed artifacts"
    )
    if summary["changed_artifacts"]:
        detail += f" ({', '.join(summary['changed_artifacts'])})"
    return {
        "ok": bool(summary["replay_matched"]),
        "summary": f"Replay {status}: {detail}",
        "report": report,
        "written_artifacts": [str(out_path)] if out_path is not None else [],
    }


def retrieval_drift(before_run_dir: Path, after_run_dir: Path, *, out_dir: Path | None = None) -> dict[str, Any]:
    report = compare_retrieval_runs(before_run_dir, after_run_dir, out_dir=out_dir)
    summary = report["summary"]
    return {
        "ok": True,
        "summary": (
            "Retrieval drift compared "
            f"{summary['query_count']} queries: "
            f"mean top-k Jaccard {summary['mean_top_k_jaccard']}, "
            f"mean ordered overlap {summary['mean_ordered_top_k_overlap']}"
        ),
        "report": report,
        "written_artifacts": report["written_artifacts"],
    }


def regression_audit(
    before_run_dir: Path,
    after_run_dir: Path,
    out_dir: Path,
    *,
    drift_jsonl: Path | None = None,
    drift_summary_json: Path | None = None,
    snapshot_diff_json: Path | None = None,
) -> dict[str, Any]:
    report = build_regression_audit(
        before_run_dir,
        after_run_dir,
        out_dir,
        drift_jsonl=drift_jsonl,
        drift_summary_json=drift_summary_json,
        snapshot_diff_json=snapshot_diff_json,
    )
    summary = report["summary"]
    return {
        "ok": True,
        "summary": (
            "Regression audit wrote "
            f"{summary['query_count']} query rows with "
            f"{summary['labeled_query_count']} labeled behavior-change candidates"
        ),
        "report": report,
        "written_artifacts": report["written_artifacts"],
    }


def systems_measurements(
    out_dir: Path,
    *,
    snapshot_dir: Path | None = None,
    run_dir: Path | None = None,
    snapshot_diff_json: Path | None = None,
    drift_jsonl: Path | None = None,
    drift_summary_json: Path | None = None,
    audit_jsonl: Path | None = None,
    audit_summary_json: Path | None = None,
) -> dict[str, Any]:
    report = collect_systems_measurements(
        out_dir,
        snapshot_dir=snapshot_dir,
        run_dir=run_dir,
        snapshot_diff_json=snapshot_diff_json,
        drift_jsonl=drift_jsonl,
        drift_summary_json=drift_summary_json,
        audit_jsonl=audit_jsonl,
        audit_summary_json=audit_summary_json,
    )
    workload = report["workload_metrics"]
    artifacts = report["artifact_metrics"]
    artifact_count = sum(artifacts[group]["artifact_count"] for group in ("input_artifacts", "output_artifacts", "manifest_artifacts", "report_artifacts"))
    return {
        "ok": True,
        "summary": (
            "Systems measurements wrote "
            f"{len(report['written_artifacts'])} reports for "
            f"{workload.get('page_count') or 0} pages, "
            f"{workload.get('query_count') or 0} queries, "
            f"and {artifact_count} measured artifacts"
        ),
        "report": report,
        "written_artifacts": report["written_artifacts"],
    }


def html_observability_report(bundle_root: Path, *, out_path: Path | None = None) -> dict[str, Any]:
    result = build_html_observability_report(bundle_root, out_path=out_path)
    return {
        "ok": True,
        "summary": result["summary"],
        "report": result["report"],
        "written_artifacts": result["written_artifacts"],
        "report_path": result["report_path"],
    }


def paper_evidence_export(bundle_root: Path, out_dir: Path) -> dict[str, Any]:
    return export_paper_evidence(bundle_root, out_dir)


def snapshot_manifest(
    dataset_path: Path,
    out_dir: Path,
    *,
    corpus_id: str,
    snapshot_id: str,
    corpus_snapshot_id: str | None = None,
    query_set_id: str | None = None,
    extraction_version: str = "unknown",
    parser_version: str = "unknown",
    graph_extraction_version: str = "unknown",
    source_name: str | None = None,
    source_url: str | None = None,
    source_commit: str | None = None,
    source_release: str | None = None,
    source_timestamp: str | None = None,
    manifest_types: set[str] | None = None,
) -> dict[str, Any]:
    requested_types = manifest_types or {"corpus", "graph", "support"}
    valid_types = {"corpus", "graph", "support"}
    unknown_types = requested_types - valid_types
    if unknown_types:
        from .errors import RetrievalAuditError

        raise RetrievalAuditError(f"Unknown snapshot manifest types: {sorted(unknown_types)}")

    written: list[Path] = []
    manifests: dict[str, dict[str, Any]] = {}
    if "corpus" in requested_types:
        path = out_dir / "corpus_snapshot_manifest.json"
        manifests["corpus"] = write_corpus_snapshot_manifest(
            path,
            dataset_path,
            corpus_id=corpus_id,
            snapshot_id=snapshot_id,
            extraction_version=extraction_version,
            parser_version=parser_version,
            source_name=source_name,
            source_url=source_url,
            source_commit=source_commit,
            source_release=source_release,
            source_timestamp=source_timestamp,
        )
        written.append(path)
    resolved_corpus_snapshot_id = corpus_snapshot_id or snapshot_id
    if "graph" in requested_types:
        path = out_dir / "graph_snapshot_manifest.json"
        manifests["graph"] = write_graph_snapshot_manifest(
            path,
            dataset_path,
            corpus_id=corpus_id,
            snapshot_id=snapshot_id,
            corpus_snapshot_id=resolved_corpus_snapshot_id,
            graph_extraction_version=graph_extraction_version,
        )
        written.append(path)
    if "support" in requested_types:
        if not query_set_id:
            from .errors import RetrievalAuditError

            raise RetrievalAuditError("--query-set-id is required when writing support manifests.")
        path = out_dir / "support_surface_manifest.json"
        manifests["support"] = write_support_surface_manifest(
            path,
            dataset_path,
            corpus_id=corpus_id,
            snapshot_id=snapshot_id,
            corpus_snapshot_id=resolved_corpus_snapshot_id,
            query_set_id=query_set_id,
        )
        written.append(path)
    return {
        "ok": True,
        "summary": f"retrieval audit snapshot manifests written: {len(written)}",
        "written_artifacts": [str(path) for path in written],
        "manifests": manifests,
    }


def snapshot_diff(before: Path, after: Path, *, out_path: Path | None = None, markdown_out_path: Path | None = None) -> dict[str, Any]:
    report = compare_snapshot_bundles(before, after, out_path=out_path, markdown_out_path=markdown_out_path)
    status = "PASSED" if report["passed"] else "FAILED"
    summary = report["summary"]
    return {
        "ok": bool(report["passed"]),
        "summary": (
            "Snapshot diff "
            f"{status}: files +{summary['added_file_count']} "
            f"-{summary['removed_file_count']} "
            f"changed {summary['changed_file_count']}; "
            f"edges +{summary['added_edge_count']} "
            f"-{summary['removed_edge_count']}; "
            f"support questions changed {summary['changed_support_question_count']}"
        ),
        "report": report,
        "written_artifacts": [str(path) for path in (out_path, markdown_out_path) if path is not None],
    }
