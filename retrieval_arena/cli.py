from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .commands import (
    longitudinal_pilot,
    git_comparison_plan,
    git_provenance,
    html_observability_report,
    paper_evidence_export,
    regression_audit,
    replay_compare,
    retrieval_drift,
    run_experiment,
    snapshot_diff,
    snapshot_manifest,
    systems_measurements,
    validate_experiment,
)
from .corpus.comparison import run_corpus_snapshot_comparison
from .errors import RetrievalAuditError
from .replay_manifests import manifest_json_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval-arena")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run a retrieval audit experiment.")
    run.add_argument("--config", help="Path to experiment YAML config.")
    run_subparsers = run.add_subparsers(dest="run_command")
    compare_git = run_subparsers.add_parser("compare-git", help="Plan a branch-aware comparison without checkout.")
    compare_git.add_argument("--config", required=True, help="Path to experiment YAML config.")
    compare_git.add_argument("--left-ref", required=True, help="Left Git ref.")
    compare_git.add_argument("--right-ref", required=True, help="Right Git ref.")
    compare_git.add_argument("--repo", default=".", help="Git repository path to resolve refs.")
    compare_git.add_argument("--out-dir", help="Base output directory for planned isolated runs.")
    validate = subparsers.add_parser("validate", help="Run Phase 1 automated validation gate.")
    validate.add_argument("--config", required=True, help="Path to experiment YAML config.")
    provenance = subparsers.add_parser("provenance", help="Inspect provenance sources.")
    provenance_subparsers = provenance.add_subparsers(dest="provenance_command", required=True)
    git = provenance_subparsers.add_parser("git", help="Print Git provenance for a path.")
    git.add_argument("--path", required=True, help="Path to inspect.")
    git.add_argument("--ref", help="Optional ref to validate without checkout.")
    replay = subparsers.add_parser("replay", help="Compare replay outputs.")
    replay_subparsers = replay.add_subparsers(dest="replay_command", required=True)
    replay_compare = replay_subparsers.add_parser("compare", help="Compare two completed run directories.")
    replay_compare.add_argument("--expected", required=True, help="Expected run directory.")
    replay_compare.add_argument("--actual", required=True, help="Actual run directory.")
    replay_compare.add_argument("--out", help="Optional path for replay_fidelity_report.json.")
    drift = subparsers.add_parser("drift", help="Compare retrieval behavior across completed runs.")
    drift_subparsers = drift.add_subparsers(dest="drift_command", required=True)
    drift_compare = drift_subparsers.add_parser("compare", help="Write retrieval drift reports for two run directories.")
    drift_compare.add_argument("--before-run", required=True, help="Before run directory.")
    drift_compare.add_argument("--after-run", required=True, help="After run directory.")
    drift_compare.add_argument("--out-dir", help="Directory for retrieval_drift reports.")
    audit = subparsers.add_parser("audit", help="Assemble regression audit reports.")
    audit_subparsers = audit.add_subparsers(dest="audit_command", required=True)
    audit_report = audit_subparsers.add_parser("report", help="Write regression audit reports for two completed run directories.")
    audit_report.add_argument("--before-run", required=True, help="Before run directory.")
    audit_report.add_argument("--after-run", required=True, help="After run directory.")
    audit_report.add_argument("--drift-jsonl", help="Optional existing retrieval_drift.jsonl path.")
    audit_report.add_argument("--drift-summary", help="Optional existing retrieval_drift_summary.json path.")
    audit_report.add_argument("--snapshot-diff", help="Optional snapshot_diff.json path.")
    audit_report.add_argument("--out-dir", required=True, help="Directory for regression audit reports.")
    measurements = subparsers.add_parser("measurements", help="Collect systems workload and artifact measurements.")
    measurements_subparsers = measurements.add_subparsers(dest="measurements_command", required=True)
    measurements_collect = measurements_subparsers.add_parser("collect", help="Write systems measurement reports.")
    measurements_collect.add_argument("--snapshot-dir", help="Optional snapshot manifest directory.")
    measurements_collect.add_argument("--run-dir", help="Optional completed run directory.")
    measurements_collect.add_argument("--snapshot-diff", help="Optional snapshot_diff.json path.")
    measurements_collect.add_argument("--drift-jsonl", help="Optional retrieval_drift.jsonl path.")
    measurements_collect.add_argument("--drift-summary", help="Optional retrieval_drift_summary.json path.")
    measurements_collect.add_argument("--audit-jsonl", help="Optional regression_audit.jsonl path.")
    measurements_collect.add_argument("--audit-summary", help="Optional regression_audit_summary.json path.")
    measurements_collect.add_argument("--out-dir", required=True, help="Directory for systems measurement reports.")
    report = subparsers.add_parser("report", help="Generate user-facing report artifacts.")
    report_subparsers = report.add_subparsers(dest="report_command", required=True)
    html_report = report_subparsers.add_parser("html", help="Write a self-contained HTML observability report.")
    html_report.add_argument("--bundle", required=True, help="Completed longitudinal audit pilot output or calibration bundle root.")
    html_report.add_argument("--out", help="Optional HTML output path. Defaults to observability_report.html in the bundle root.")
    evidence_report = report_subparsers.add_parser("evidence", help="Export paper-facing evidence tables from a completed report bundle.")
    evidence_report.add_argument("--bundle", required=True, help="Completed longitudinal audit pilot output or calibration bundle root.")
    evidence_report.add_argument("--out-dir", required=True, help="Directory for exported CSV and Markdown evidence artifacts.")
    corpus = subparsers.add_parser("corpus", help="Import and compare corpus snapshots.")
    corpus_subparsers = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_compare = corpus_subparsers.add_parser("compare-snapshots", help="Run a corpus source comparison plan.")
    corpus_compare.add_argument("--plan", required=True, help="Path to corpus snapshot comparison plan JSON.")
    snapshot = subparsers.add_parser("snapshot", help="Generate snapshot observability artifacts.")
    snapshot_subparsers = snapshot.add_subparsers(dest="snapshot_command", required=True)
    manifest = snapshot_subparsers.add_parser("manifest", help="Write snapshot manifests for a prepared dataset.")
    manifest.add_argument("--dataset", required=True, help="Prepared dataset directory.")
    manifest.add_argument("--out-dir", required=True, help="Directory for written manifests.")
    manifest.add_argument("--corpus-id", required=True, help="Stable corpus identifier.")
    manifest.add_argument("--snapshot-id", required=True, help="Stable snapshot identifier.")
    manifest.add_argument("--corpus-snapshot-id", help="Corpus snapshot identifier referenced by graph/support manifests.")
    manifest.add_argument("--query-set-id", help="Query set identifier for support-surface manifests.")
    manifest.add_argument("--extraction-version", default="unknown", help="Corpus extraction version.")
    manifest.add_argument("--parser-version", default="unknown", help="Corpus parser version.")
    manifest.add_argument("--graph-extraction-version", default="unknown", help="Graph extraction version.")
    manifest.add_argument("--source-name")
    manifest.add_argument("--source-url")
    manifest.add_argument("--source-commit")
    manifest.add_argument("--source-release")
    manifest.add_argument("--source-timestamp")
    manifest.add_argument("--types", default="corpus,graph,support", help="Comma-separated manifest types: corpus,graph,support.")
    snapshot_diff = snapshot_subparsers.add_parser("diff", help="Compare two completed snapshot manifest bundles.")
    snapshot_diff.add_argument("--before", required=True, help="Before snapshot directory or corpus manifest path.")
    snapshot_diff.add_argument("--after", required=True, help="After snapshot directory or corpus manifest path.")
    snapshot_diff.add_argument("--out", help="Optional path for snapshot_diff.json.")
    snapshot_diff.add_argument("--markdown-out", help="Optional path for snapshot_diff.md.")
    study = subparsers.add_parser("study", help="Run paper study orchestration commands.")
    study_subparsers = study.add_subparsers(dest="study_command", required=True)
    review = study_subparsers.add_parser("review", help="Run anonymous review study commands.")
    review_subparsers = review.add_subparsers(dest="review_command", required=True)
    pilot = review_subparsers.add_parser("pilot", help="Run the anonymous review pilot orchestrator.")
    pilot.add_argument("--plan", required=True, help="Path to the top-level longitudinal audit pilot plan JSON.")
    pilot.add_argument("--dry-run", action="store_true", help="Validate and resolve the pilot without writing generated artifacts.")
    pilot.add_argument("--stage", help="Run one stage and required predecessors.")
    pilot.add_argument("--from-stage", help="Run from this stage through the end of the plan.")
    pilot.add_argument("--force", action="store_true", help="Override reuse policy for generated reports and selected runs.")
    pilot.add_argument("--refresh-baseline", action="store_true", help="Reuse valid baseline artifacts and run only missing or invalidated selected tests.")
    pilot.add_argument("--no-baseline-bundle", action="store_true", help="Run observability stages without refreshing the calibration store.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if args.run_command == "compare-git":
                result = git_comparison_plan(
                    Path(args.repo),
                    left_ref=args.left_ref,
                    right_ref=args.right_ref,
                    output_dir=Path(args.out_dir) if args.out_dir else None,
                )
                print(manifest_json_text(result["plan"]), end="")
            else:
                if not args.config:
                    raise RetrievalAuditError("run requires --config.")
                print(run_experiment(args.config)["summary"])
        elif args.command == "validate":
            print(validate_experiment(args.config)["summary"])
        elif args.command == "provenance" and args.provenance_command == "git":
            print(manifest_json_text(git_provenance(Path(args.path), ref=args.ref)["provenance"]), end="")
        elif args.command == "replay" and args.replay_command == "compare":
            result = replay_compare(
                Path(args.expected),
                Path(args.actual),
                out_path=Path(args.out) if args.out else None,
            )
            print(result["summary"])
        elif args.command == "drift" and args.drift_command == "compare":
            result = retrieval_drift(
                Path(args.before_run),
                Path(args.after_run),
                out_dir=Path(args.out_dir) if args.out_dir else None,
            )
            print(result["summary"])
        elif args.command == "audit" and args.audit_command == "report":
            result = regression_audit(
                Path(args.before_run),
                Path(args.after_run),
                Path(args.out_dir),
                drift_jsonl=Path(args.drift_jsonl) if args.drift_jsonl else None,
                drift_summary_json=Path(args.drift_summary) if args.drift_summary else None,
                snapshot_diff_json=Path(args.snapshot_diff) if args.snapshot_diff else None,
            )
            print(result["summary"])
        elif args.command == "measurements" and args.measurements_command == "collect":
            result = systems_measurements(
                Path(args.out_dir),
                snapshot_dir=Path(args.snapshot_dir) if args.snapshot_dir else None,
                run_dir=Path(args.run_dir) if args.run_dir else None,
                snapshot_diff_json=Path(args.snapshot_diff) if args.snapshot_diff else None,
                drift_jsonl=Path(args.drift_jsonl) if args.drift_jsonl else None,
                drift_summary_json=Path(args.drift_summary) if args.drift_summary else None,
                audit_jsonl=Path(args.audit_jsonl) if args.audit_jsonl else None,
                audit_summary_json=Path(args.audit_summary) if args.audit_summary else None,
            )
            print(result["summary"])
        elif args.command == "report" and args.report_command == "html":
            result = html_observability_report(
                Path(args.bundle),
                out_path=Path(args.out) if args.out else None,
            )
            print(result["summary"])
        elif args.command == "report" and args.report_command == "evidence":
            result = paper_evidence_export(
                Path(args.bundle),
                Path(args.out_dir),
            )
            print(result["summary"])
        elif args.command == "corpus" and args.corpus_command == "compare-snapshots":
            result = run_corpus_snapshot_comparison(Path(args.plan))
            print(result["summary"])
        elif args.command == "snapshot" and args.snapshot_command == "manifest":
            dataset_path = Path(args.dataset)
            out_dir = Path(args.out_dir)
            requested_types = {item.strip() for item in args.types.split(",") if item.strip()}
            result = snapshot_manifest(
                dataset_path,
                out_dir,
                corpus_id=args.corpus_id,
                snapshot_id=args.snapshot_id,
                corpus_snapshot_id=args.corpus_snapshot_id,
                query_set_id=args.query_set_id,
                extraction_version=args.extraction_version,
                parser_version=args.parser_version,
                graph_extraction_version=args.graph_extraction_version,
                source_name=args.source_name,
                source_url=args.source_url,
                source_commit=args.source_commit,
                source_release=args.source_release,
                source_timestamp=args.source_timestamp,
                manifest_types=requested_types,
            )
            print(result["summary"])
        elif args.command == "snapshot" and args.snapshot_command == "diff":
            result = snapshot_diff(
                Path(args.before),
                Path(args.after),
                out_path=Path(args.out) if args.out else None,
                markdown_out_path=Path(args.markdown_out) if args.markdown_out else None,
            )
            print(result["summary"])
        elif args.command == "study" and args.study_command == "review" and args.review_command == "pilot":
            result = longitudinal_pilot(
                Path(args.plan),
                dry_run=args.dry_run,
                stage=args.stage,
                from_stage=args.from_stage,
                force=args.force,
                no_baseline_bundle=args.no_baseline_bundle,
                refresh_baseline=args.refresh_baseline,
            )
            print(result["summary"])
        return 0
    except RetrievalAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
