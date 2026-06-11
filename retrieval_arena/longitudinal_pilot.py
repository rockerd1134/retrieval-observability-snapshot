from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import build_regression_audit
from .config import load_config
from .drift import compare_retrieval_runs
from .errors import ValidationError
from .harness import run_config, run_experiment
from .hashing import sha256_file, sha256_jsonl
from .html_report import build_html_observability_report
from .manifests import canonical_manifest_json, read_manifest, write_manifest
from .measurements import collect_systems_measurements, environment_metadata
from .replay_manifests import resolved_run_identity_hash, scoring_hash, write_experiment_manifest


PILOT_PLAN_SCHEMA_VERSION = "retrieval_arena.review2026_pilot_plan.v1"
PILOT_RUN_SCHEMA_VERSION = "retrieval_arena.review2026_pilot_run.v1"
VALID_STAGES = (
    "prerequisites",
    "snapshot_compare",
    "retrieval_before",
    "retrieval_after",
    "drift",
    "audit",
    "measurements",
    "html_report",
    "baseline_bundle",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_pilot_plan(plan_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid longitudinal audit pilot plan JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValidationError("longitudinal audit pilot plan must be a JSON object.")
    return resolve_pilot_plan(raw, plan_path)


def resolve_pilot_plan(raw: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    if raw.get("schema_version") != PILOT_PLAN_SCHEMA_VERSION:
        raise ValidationError(f"schema_version must be {PILOT_PLAN_SCHEMA_VERSION}.")
    base = plan_path.resolve().parent
    plan = dict(raw)
    plan["plan_path"] = str(plan_path.resolve())
    for field in ("pilot_id", "corpus_id", "query_set_id"):
        _required_str(plan, field)

    snapshot_pair = _required_mapping(plan, "snapshot_pair")
    for side in ("before", "after"):
        item = _required_mapping(snapshot_pair, side)
        _required_str(item, "snapshot_id")
        item["source_descriptor"] = str(_resolve_path(_required_str(item, "source_descriptor"), base))
    snapshot_pair["comparison_plan"] = str(_resolve_path(_required_str(snapshot_pair, "comparison_plan"), base))

    retrieval = _required_mapping(plan, "retrieval")
    retrieval["before_config"] = str(_resolve_path(_required_str(retrieval, "before_config"), base))
    retrieval["after_config"] = str(_resolve_path(_required_str(retrieval, "after_config"), base))
    selected_tests = retrieval.get("selected_tests", [])
    if not isinstance(selected_tests, list) or not all(isinstance(item, str) and item for item in selected_tests):
        raise ValidationError("retrieval.selected_tests must be a list of non-empty strings.")
    _required_str(retrieval, "selected_dataset")

    comparisons = plan.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValidationError("comparisons must be a non-empty list.")
    seen_comparisons: set[str] = set()
    for item in comparisons:
        if not isinstance(item, dict):
            raise ValidationError("Each comparison must be a mapping.")
        comparison_id = _required_str(item, "comparison_id")
        if comparison_id in seen_comparisons:
            raise ValidationError(f"Duplicate comparison_id: {comparison_id}")
        seen_comparisons.add(comparison_id)
        _validate_selector(_required_mapping(item, "before_run_selector"), "before_run_selector")
        _validate_selector(_required_mapping(item, "after_run_selector"), "after_run_selector")

    plan["output_root"] = str(_resolve_path(_required_str(plan, "output_root"), base))
    plan["calibration_store"] = str(_resolve_path(_required_str(plan, "calibration_store"), base))
    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValidationError("stages must be a non-empty list.")
    unknown = [stage for stage in stages if stage not in VALID_STAGES]
    if unknown:
        raise ValidationError(f"Unknown longitudinal audit pilot stages: {unknown}")
    plan["stages"] = stages

    reuse = plan.get("reuse_policy", {})
    if not isinstance(reuse, dict):
        raise ValidationError("reuse_policy must be a mapping when present.")
    for key in ("reuse_imports", "reuse_prepared_snapshots", "reuse_retrieval_runs", "overwrite_reports"):
        if key in reuse and not isinstance(reuse[key], bool):
            raise ValidationError(f"reuse_policy.{key} must be boolean when present.")
    plan["reuse_policy"] = reuse

    baseline = _required_mapping(plan, "baseline_bundle")
    if baseline.get("include_raw") is True:
        raise ValidationError("baseline_bundle.include_raw must not be true; raw artifacts are excluded.")
    baseline["include_raw"] = False
    plan["baseline_bundle"] = baseline
    return plan


def resolve_stage_list(plan_stages: list[str], *, stage: str | None = None, from_stage: str | None = None) -> list[str]:
    if stage and from_stage:
        raise ValidationError("--stage and --from-stage cannot be used together.")
    for value, label in ((stage, "--stage"), (from_stage, "--from-stage")):
        if value and value not in plan_stages:
            raise ValidationError(f"{label} value is not present in the pilot plan stages: {value}")
    if stage:
        index = plan_stages.index(stage)
        required = [item for item in VALID_STAGES[: VALID_STAGES.index(stage) + 1] if item in plan_stages[: index + 1]]
        return required
    if from_stage:
        return plan_stages[plan_stages.index(from_stage) :]
    return list(plan_stages)


def select_run_from_experiment_manifest(manifest_path: Path, selector: dict[str, Any]) -> Path:
    manifest = read_manifest(manifest_path, verify_hash=False)
    experiment_dir = manifest_path.resolve().parent
    matches = []
    for item in manifest.get("run_manifests", []):
        if not isinstance(item, dict):
            continue
        if item.get("dataset") != selector.get("dataset") or item.get("test") != selector.get("test"):
            continue
        if selector.get("run_id") and item.get("run_id") != selector.get("run_id"):
            continue
        matches.append(item)
    label = f"dataset={selector.get('dataset')} test={selector.get('test')}"
    if selector.get("run_id"):
        label += f" run_id={selector.get('run_id')}"
    if not matches:
        raise ValidationError(f"No run matched selector in {manifest_path}: {label}")
    if len(matches) > 1:
        raise ValidationError(f"Ambiguous run selector in {manifest_path}: {label}")
    item = matches[0]
    path_value = item.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValidationError(f"Experiment manifest run entry missing path for selector: {label}")
    run_manifest = Path(path_value)
    if not run_manifest.is_absolute():
        run_manifest = experiment_dir / run_manifest
    return run_manifest.resolve().parent


def longitudinal_pilot(
    plan_path: Path,
    *,
    dry_run: bool = False,
    stage: str | None = None,
    from_stage: str | None = None,
    force: bool = False,
    no_baseline_bundle: bool = False,
    refresh_baseline: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    plan = load_pilot_plan(plan_path)
    stages_to_run = resolve_stage_list(plan["stages"], stage=stage, from_stage=from_stage)
    if no_baseline_bundle:
        stages_to_run = [item for item in stages_to_run if item != "baseline_bundle"]
    timestamp = created_at or utc_now_iso()
    output_root = Path(plan["output_root"])
    stage_records: list[dict[str, Any]] = []
    runtime_metrics: list[dict[str, Any]] = []
    context = _initial_context(plan, plan_path)
    if refresh_baseline:
        context["refresh_baseline"] = True

    if dry_run:
        manifest = _build_pilot_manifest(
            plan,
            plan_path,
            created_at=timestamp,
            stage_records=[_dry_stage_record(name) for name in stages_to_run],
            context=context,
            baseline_ref=None,
        )
        return {
            "ok": True,
            "summary": f"longitudinal audit pilot {plan['pilot_id']} dry-run: {len(stages_to_run)} stages resolved, {len(plan['comparisons'])} run comparison(s)",
            "dry_run": True,
            "plan": plan,
            "stages": stages_to_run,
            "manifest": manifest,
            "written_artifacts": [],
        }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "plan_resolved.json").write_text(canonical_manifest_json(plan), encoding="utf-8")
    baseline_ref: dict[str, Any] | None = None
    failed = False
    for stage_name in stages_to_run:
        if failed:
            stage_records.append(_skipped_stage_record(stage_name))
            continue
        started_at = utc_now_iso()
        start = datetime.now(timezone.utc)
        try:
            record = _run_stage(stage_name, plan, context, force=force, runtime_metrics=runtime_metrics)
            record.update({"stage": stage_name, "status": "completed", "started_at": started_at, "completed_at": utc_now_iso()})
            if stage_name == "baseline_bundle":
                baseline_ref = record.get("baseline_bundle")
        except Exception as exc:
            record = {
                "stage": stage_name,
                "status": "failed",
                "started_at": started_at,
                "completed_at": utc_now_iso(),
                "input_artifacts": [],
                "output_artifacts": [],
                "manifest_references": [],
                "unavailable_optional_inputs": [],
                "error": str(exc),
            }
            failed = True
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        runtime_metrics.append({"stage_name": stage_name, "started_at": started_at, "completed_at": record["completed_at"], "duration_seconds": duration, "status": record["status"], "artifact_references": record.get("output_artifacts", [])})
        stage_records.append(record)
        manifest = _build_pilot_manifest(plan, plan_path, created_at=timestamp, stage_records=stage_records, context=context, baseline_ref=baseline_ref)
        write_manifest(output_root / "pilot_manifest.json", manifest)
        if failed:
            raise ValidationError(f"longitudinal audit pilot stage failed: {stage_name}: {record['error']}")

    manifest = _build_pilot_manifest(plan, plan_path, created_at=timestamp, stage_records=stage_records, context=context, baseline_ref=baseline_ref)
    written_manifest = write_manifest(output_root / "pilot_manifest.json", manifest)
    if baseline_ref is not None:
        shutil.copy2(output_root / "pilot_manifest.json", Path(plan["calibration_store"]) / "pilot_manifest.json")
    return {
        "ok": True,
        "summary": f"longitudinal audit pilot {plan['pilot_id']} completed: {len(stages_to_run)} stages, {len(plan['comparisons'])} run comparison(s), baseline {'refreshed' if baseline_ref else 'not refreshed'}",
        "dry_run": False,
        "manifest": written_manifest,
        "manifest_path": output_root / "pilot_manifest.json",
        "written_artifacts": [str(output_root / "pilot_manifest.json"), str(output_root / "plan_resolved.json")],
    }


def assemble_baseline_bundle(
    output_root: Path,
    calibration_store: Path,
    *,
    pilot_id: str,
    comparisons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if calibration_store.exists():
        shutil.rmtree(calibration_store)
    calibration_store.mkdir(parents=True)
    copied: list[str] = []
    for item in sorted(output_root.iterdir(), key=lambda path: path.name):
        if item.name == "raw":
            continue
        target = calibration_store / item.name
        if item.is_dir():
            _copytree_excluding_raw(item, target)
        elif item.is_file():
            shutil.copy2(item, target)
        if target.exists():
            copied.append(str(target))
    copied.extend(_copy_selected_retrieval_artifacts(calibration_store, comparisons or []))
    baseline_md = calibration_store / "BASELINE.md"
    baseline_md.write_text(
        "\n".join(
            [
                f"# longitudinal audit pilot Baseline: {pilot_id}",
                "",
                "This calibration store contains the current selected longitudinal audit pilot baseline.",
                "Raw source trees, copied corpus content, and copied query/answer text are excluded by policy.",
                "No efficiency score, overall scale score, or composite systems metric is included.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    copied.append(str(baseline_md))
    return {"path": str(calibration_store.resolve()), "copied_artifacts": copied, "include_raw": False}


def _run_stage(stage_name: str, plan: dict[str, Any], context: dict[str, Any], *, force: bool, runtime_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    output_root = Path(plan["output_root"])
    if stage_name == "prerequisites":
        return _run_prerequisites(plan)
    if stage_name == "snapshot_compare":
        from .corpus.comparison import load_comparison_plan, run_corpus_snapshot_comparison

        comparison_plan = Path(plan["snapshot_pair"]["comparison_plan"])
        comparison_output = load_comparison_plan(comparison_plan)["output_dir"]
        existing_manifest = comparison_output / "corpus_snapshot_comparison_manifest.json"
        if context.get("refresh_baseline") and not force and _snapshot_comparison_reusable(existing_manifest, comparison_plan):
            manifest = read_manifest(existing_manifest, verify_hash=False)
            context["snapshot_comparison_manifest"] = existing_manifest
            context["snapshot_diff_json"] = Path(manifest["snapshot_diff_report"])
            context["snapshot_diff_md"] = Path(manifest["snapshot_diff_markdown"])
            context["before_snapshot_dir"] = Path(manifest["before"]["snapshot_manifest_dir"])
            context["after_snapshot_dir"] = Path(manifest["after"]["snapshot_manifest_dir"])
            decision = _stage_decision("snapshot_compare", "reused", "existing snapshot comparison manifest and diff reports matched the active plan", artifact=str(existing_manifest))
            context.setdefault("reuse_decisions", []).append(decision)
            return {**_record([str(comparison_plan)], [str(existing_manifest), str(context["snapshot_diff_json"]), str(context["snapshot_diff_md"])], [str(existing_manifest)]), "reuse_decisions": [decision]}
        result = run_corpus_snapshot_comparison(comparison_plan)
        context["snapshot_comparison_manifest"] = Path(result["manifest_path"])
        context["snapshot_diff_json"] = Path(result["manifest_path"]).parent / "snapshot_diff.json"
        context["snapshot_diff_md"] = Path(result["manifest_path"]).parent / "snapshot_diff.md"
        manifest = result["manifest"]
        context["before_snapshot_dir"] = Path(manifest["before"]["snapshot_manifest_dir"])
        context["after_snapshot_dir"] = Path(manifest["after"]["snapshot_manifest_dir"])
        decision = _stage_decision("snapshot_compare", "generated", "snapshot comparison generated", artifact=str(result["manifest_path"]))
        context.setdefault("reuse_decisions", []).append(decision)
        return {**_record([plan["snapshot_pair"]["comparison_plan"]], result["written_artifacts"], [str(result["manifest_path"])]), "reuse_decisions": [decision]}
    if stage_name in {"retrieval_before", "retrieval_after"}:
        side = "before" if stage_name.endswith("before") else "after"
        config_path = Path(plan["retrieval"][f"{side}_config"])
        if context.get("refresh_baseline") and not force:
            refresh = _refresh_selected_retrieval_config(config_path, plan, side=side)
            manifest_path = refresh["experiment_manifest"]
            context[f"{side}_experiment_manifest"] = manifest_path
            context.setdefault("reuse_decisions", []).extend(refresh["decisions"])
            context.setdefault("retrieval_refresh", {})[side] = refresh
            return {
                **_record([str(config_path)], [str(manifest_path), *refresh["written_artifacts"]], [str(manifest_path)]),
                "reuse_decisions": refresh["decisions"],
            }
        if not force and _experiment_manifest_path(config_path).exists() and plan["reuse_policy"].get("reuse_retrieval_runs"):
            manifest_path = _experiment_manifest_path(config_path)
            decision = _stage_decision("retrieval_runs", "reused", f"{side} experiment manifest reuse_policy allowed reuse", artifact=str(manifest_path))
            context.setdefault("reuse_decisions", []).append(decision)
        else:
            _run_selected_retrieval_config(config_path, plan["retrieval"].get("selected_tests", []))
            manifest_path = _experiment_manifest_path(config_path)
            decision = _stage_decision("retrieval_runs", "generated", f"{side} selected retrieval tests executed", artifact=str(manifest_path))
            context.setdefault("reuse_decisions", []).append(decision)
        context[f"{side}_experiment_manifest"] = manifest_path
        return {**_record([str(config_path)], [str(manifest_path)], [str(manifest_path)]), "reuse_decisions": [decision]}
    if stage_name == "drift":
        _ensure_runs_selected(plan, context)
        outputs = []
        decisions = []
        for comparison in context["selected_comparisons"]:
            out_dir = output_root / "comparisons" / comparison["comparison_id"]
            if context.get("refresh_baseline") and not _comparison_invalidated(comparison, context) and _comparison_reports_valid(out_dir, ["retrieval_drift.jsonl", "retrieval_drift_summary.json"], comparison, "retrieval_drift_summary.json"):
                report = {"written_artifacts": []}
                decisions.append(_comparison_decision(comparison["comparison_id"], "drift", "reused", "existing drift reports are present", artifact=str(out_dir)))
            else:
                report = compare_retrieval_runs(comparison["before_run_dir"], comparison["after_run_dir"], out_dir=out_dir)
                decisions.append(_comparison_decision(comparison["comparison_id"], "drift", "generated", "drift reports refreshed", artifact=str(out_dir)))
            comparison["drift_jsonl"] = out_dir / "retrieval_drift.jsonl"
            comparison["drift_summary_json"] = out_dir / "retrieval_drift_summary.json"
            outputs.extend(report["written_artifacts"])
        context.setdefault("reuse_decisions", []).extend(decisions)
        return {**_record([], outputs, [path for item in context["selected_comparisons"] for path in (str(item["drift_summary_json"]),)]), "reuse_decisions": decisions}
    if stage_name == "audit":
        _ensure_runs_selected(plan, context)
        outputs = []
        decisions = []
        for comparison in context["selected_comparisons"]:
            out_dir = output_root / "comparisons" / comparison["comparison_id"]
            if context.get("refresh_baseline") and not _comparison_invalidated(comparison, context) and _comparison_reports_valid(out_dir, ["regression_audit.jsonl", "regression_audit_summary.json"], comparison, "regression_audit_summary.json"):
                report = {"written_artifacts": []}
                decisions.append(_comparison_decision(comparison["comparison_id"], "audit", "reused", "existing audit reports are present", artifact=str(out_dir)))
            else:
                report = build_regression_audit(
                    comparison["before_run_dir"],
                    comparison["after_run_dir"],
                    out_dir,
                    drift_jsonl=comparison.get("drift_jsonl"),
                    drift_summary_json=comparison.get("drift_summary_json"),
                    snapshot_diff_json=context.get("snapshot_diff_json"),
                )
                decisions.append(_comparison_decision(comparison["comparison_id"], "audit", "generated", "audit reports refreshed", artifact=str(out_dir)))
            comparison["audit_jsonl"] = out_dir / "regression_audit.jsonl"
            comparison["audit_summary_json"] = out_dir / "regression_audit_summary.json"
            outputs.extend(report["written_artifacts"])
        context.setdefault("reuse_decisions", []).extend(decisions)
        return {**_record([], outputs, [path for item in context["selected_comparisons"] for path in (str(item["audit_summary_json"]),)]), "reuse_decisions": decisions}
    if stage_name == "measurements":
        _ensure_runs_selected(plan, context)
        outputs = []
        decisions = []
        for comparison in context["selected_comparisons"]:
            out_dir = output_root / "comparisons" / comparison["comparison_id"]
            if context.get("refresh_baseline") and not _comparison_invalidated(comparison, context) and _comparison_reports_valid(out_dir, ["systems_measurements.json"]):
                report = {"written_artifacts": []}
                decisions.append(_comparison_decision(comparison["comparison_id"], "measurements", "reused", "existing systems measurements are present", artifact=str(out_dir)))
            else:
                report = collect_systems_measurements(
                    out_dir,
                    snapshot_dir=context.get("after_snapshot_dir"),
                    run_dir=comparison["after_run_dir"],
                    snapshot_diff_json=context.get("snapshot_diff_json"),
                    drift_jsonl=comparison.get("drift_jsonl"),
                    drift_summary_json=comparison.get("drift_summary_json"),
                    audit_jsonl=comparison.get("audit_jsonl"),
                    audit_summary_json=comparison.get("audit_summary_json"),
                    stage_runtime_metrics=runtime_metrics,
                )
                decisions.append(_comparison_decision(comparison["comparison_id"], "measurements", "generated", "systems measurements refreshed", artifact=str(out_dir)))
            comparison["systems_measurements_json"] = out_dir / "systems_measurements.json"
            outputs.extend(report["written_artifacts"])
        context.setdefault("reuse_decisions", []).extend(decisions)
        return {**_record([], outputs, [path for item in context["selected_comparisons"] for path in (str(item["systems_measurements_json"]),)]), "reuse_decisions": decisions}
    if stage_name == "html_report":
        report = build_html_observability_report(output_root, out_path=output_root / "observability_report.html")
        context["html_report"] = Path(report["report_path"])
        return _record([str(output_root)], report["written_artifacts"], [])
    if stage_name == "baseline_bundle":
        bundle = assemble_baseline_bundle(
            output_root,
            Path(plan["calibration_store"]),
            pilot_id=plan["pilot_id"],
            comparisons=context.get("selected_comparisons", []),
        )
        return {**_record([str(output_root)], bundle["copied_artifacts"], []), "baseline_bundle": bundle}
    raise ValidationError(f"Unsupported longitudinal audit pilot stage: {stage_name}")


def _run_prerequisites(plan: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    _check_existing_file("snapshot_comparison_plan", Path(plan["snapshot_pair"]["comparison_plan"]), checks)
    for side in ("before", "after"):
        descriptor_path = Path(plan["snapshot_pair"][side]["source_descriptor"])
        descriptor = _load_source_descriptor(descriptor_path)
        _check_source_descriptor(side, descriptor_path, descriptor, checks)
    before_config = load_config(Path(plan["retrieval"]["before_config"]))
    after_config = load_config(Path(plan["retrieval"]["after_config"]))
    checks.append({"name": "retrieval_before_config", "status": "ok", "path": str(before_config.config_path)})
    checks.append({"name": "retrieval_after_config", "status": "ok", "path": str(after_config.config_path)})

    selected_dataset = plan["retrieval"]["selected_dataset"]
    selected_tests = set(plan["retrieval"].get("selected_tests", []))
    for label, config in (("before", before_config), ("after", after_config)):
        dataset_names = {dataset.name for dataset in config.datasets}
        test_names = {test.name for test in config.tests}
        if selected_dataset not in dataset_names:
            raise ValidationError(f"Prerequisite failed: retrieval.{label}_config missing selected dataset {selected_dataset}.")
        missing_tests = sorted(selected_tests - test_names)
        if missing_tests:
            raise ValidationError(f"Prerequisite failed: retrieval.{label}_config missing selected tests {missing_tests}.")
        checks.append({"name": f"retrieval_{label}_selected_dataset", "status": "ok", "dataset": selected_dataset})
        checks.append({"name": f"retrieval_{label}_selected_tests", "status": "ok", "tests": sorted(selected_tests)})
        for test in config.tests:
            if test.name not in selected_tests:
                continue
            if test.build_context is not None:
                if not test.build_context.is_dir():
                    raise ValidationError(f"Prerequisite failed: Docker build_context not found for {test.name}: {test.build_context}")
                checks.append({"name": f"docker_build_context_{label}_{test.name}", "status": "ok", "path": str(test.build_context)})
            for volume in test.volumes:
                if not volume.host_path.exists():
                    raise ValidationError(f"Prerequisite failed: Docker volume host_path not found for {test.name}: {volume.host_path}")
                checks.append({"name": f"docker_volume_{label}_{test.name}", "status": "ok", "path": str(volume.host_path)})

    server_version = _docker_server_version()
    checks.append({"name": "docker_engine", "status": "ok", "server_version": server_version})
    return {
        "input_artifacts": [
            {"path": plan["snapshot_pair"]["comparison_plan"]},
            {"path": plan["retrieval"]["before_config"]},
            {"path": plan["retrieval"]["after_config"]},
        ],
        "output_artifacts": [],
        "manifest_references": [],
        "unavailable_optional_inputs": [],
        "checks": checks,
    }


def _run_selected_retrieval_config(config_path: Path, selected_tests: list[str]) -> list[dict[str, Any]]:
    config = load_config(config_path)
    if not selected_tests:
        return run_config(config_path)
    requested = set(selected_tests)
    selected = [test for test in config.tests if test.name in requested]
    missing = sorted(requested - {test.name for test in selected})
    if missing:
        raise ValidationError(f"Retrieval config {config_path} missing selected tests {missing}.")
    return run_experiment(replace(config, tests=selected))


def _refresh_selected_retrieval_config(config_path: Path, plan: dict[str, Any], *, side: str) -> dict[str, Any]:
    config = load_config(config_path)
    selected_tests = plan["retrieval"].get("selected_tests", [])
    selected_dataset = plan["retrieval"]["selected_dataset"]
    dataset = next((item for item in config.datasets if item.name == selected_dataset), None)
    if dataset is None:
        raise ValidationError(f"Retrieval config {config_path} missing selected dataset {selected_dataset}.")
    requested = set(selected_tests)
    selected = [test for test in config.tests if test.name in requested]
    missing = sorted(requested - {test.name for test in selected})
    if missing:
        raise ValidationError(f"Retrieval config {config_path} missing selected tests {missing}.")

    experiment_manifest = _experiment_manifest_path(config_path)
    existing_by_test = _existing_run_manifests_by_test(experiment_manifest, selected_dataset)
    valid_manifests: list[Path] = []
    invalid_tests: list[str] = []
    decisions: list[dict[str, Any]] = []
    for test in selected:
        manifest_path = existing_by_test.get(test.name)
        if manifest_path is None:
            invalid_tests.append(test.name)
            decisions.append(_stage_decision("retrieval_runs", "new", f"{side} {test.name} has no existing run manifest", dataset=selected_dataset, test=test.name))
            continue
        decision = _validate_reusable_run_manifest(manifest_path, config, dataset, test, plan)
        decision.update({"stage": "retrieval_runs", "side": side, "dataset": selected_dataset, "test": test.name, "artifact": str(manifest_path)})
        decisions.append(decision)
        if decision["decision"] == "reused":
            valid_manifests.append(manifest_path)
        else:
            invalid_tests.append(test.name)

    written_artifacts: list[str] = []
    if invalid_tests:
        run_experiment(replace(config, datasets=[dataset], tests=[test for test in selected if test.name in set(invalid_tests)]))
        written_artifacts.append(str(experiment_manifest))
    for test in selected:
        run_dir = _run_dir_for(config, dataset.name, test.name)
        manifest_path = run_dir / "retrieval_replay_manifest.json"
        if not manifest_path.is_file():
            raise ValidationError(f"Refresh failed to produce run manifest for {dataset.name}/{test.name}: {manifest_path}")
        if manifest_path not in valid_manifests:
            valid_manifests.append(manifest_path)

    experiment_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_experiment_manifest(
        experiment_manifest,
        config=replace(config, datasets=[dataset], tests=selected),
        experiment_dir=experiment_manifest.parent,
        run_manifests=valid_manifests,
    )
    return {
        "experiment_manifest": experiment_manifest,
        "decisions": decisions,
        "written_artifacts": written_artifacts,
        "reused_count": sum(1 for item in decisions if item["decision"] == "reused"),
        "generated_count": len(invalid_tests),
        "generated_tests": sorted(invalid_tests),
    }


def _existing_run_manifests_by_test(experiment_manifest: Path, dataset_name: str) -> dict[str, Path]:
    if not experiment_manifest.is_file():
        return {}
    manifest = read_manifest(experiment_manifest, verify_hash=False)
    base = experiment_manifest.parent
    result: dict[str, Path] = {}
    for item in manifest.get("run_manifests", []):
        if not isinstance(item, dict) or item.get("dataset") != dataset_name:
            continue
        test = item.get("test")
        path_value = item.get("path")
        if not isinstance(test, str) or not isinstance(path_value, str):
            continue
        path = Path(path_value)
        result[test] = (path if path.is_absolute() else base / path).resolve()
    return result


def _validate_reusable_run_manifest(
    manifest_path: Path,
    config: Any,
    dataset: Any,
    test: Any,
    plan: dict[str, Any],
) -> dict[str, Any]:
    required_files = ["retrieval_replay_manifest.json", "predictions.jsonl", "metadata.json", "scores.json", "item_scores.jsonl"]
    missing = [name for name in required_files if not (manifest_path.parent / name).is_file()]
    if missing:
        return {"decision": "invalidated", "reason": f"missing required run outputs: {missing}"}
    manifest = read_manifest(manifest_path, verify_hash=False)
    checks = {
        "dataset": manifest.get("dataset") == dataset.name,
        "test": manifest.get("test") == test.name,
        "query_set_id": manifest.get("query_set_id") == (dataset.query_set_id or plan["query_set_id"]),
        "query_set_hash": manifest.get("query_set_hash") == sha256_jsonl(dataset.path / "questions.jsonl"),
        "scoring_hash": manifest.get("scoring_hash") == scoring_hash(config.scoring),
        "resolved_run_identity_hash": manifest.get("resolved_run_identity_hash") == resolved_run_identity_hash(dataset=dataset, test=test, scoring=config.scoring),
    }
    snapshot_refs = manifest.get("snapshot_manifest_references", {})
    if isinstance(snapshot_refs, dict):
        checks["corpus_manifest_hash"] = _manifest_hash_matches(snapshot_refs.get("corpus"), dataset.corpus_snapshot_manifest)
        checks["graph_manifest_hash"] = _manifest_hash_matches(snapshot_refs.get("graph"), dataset.graph_snapshot_manifest)
        checks["support_manifest_hash"] = _manifest_hash_matches(snapshot_refs.get("support_surface"), dataset.support_surface_manifest)
    if manifest.get("resolved_run_identity_hash") is None:
        checks["retrieval_config_hash"] = manifest.get("retrieval_config_hash") == sha256_file(config.config_path)
        checks.pop("resolved_run_identity_hash", None)
    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        return {"decision": "invalidated", "reason": f"reuse validation failed: {failed}", "hash_checks": checks}
    return {"decision": "reused", "reason": "run manifest, query, snapshot, scoring, and resolved run identity matched", "hash_checks": checks}


def _manifest_hash_matches(reference: Any, path: Path | None) -> bool:
    if path is None:
        return reference is None
    if not isinstance(reference, dict):
        return False
    if not path.is_file():
        return False
    expected = read_manifest(path, verify_hash=False).get("manifest_hash")
    return reference.get("manifest_hash") == expected


def _run_dir_for(config: Any, dataset_name: str, test_name: str) -> Path:
    from .harness import stable_run_id

    return (config.output_dir / config.experiment_name / "runs" / f"{dataset_name}__{test_name}__{stable_run_id(dataset_name, test_name)}").resolve()


def _stage_decision(stage: str, decision: str, reason: str, **details: Any) -> dict[str, Any]:
    return {"stage": stage, "decision": decision, "reason": reason, **details}


def _comparison_decision(comparison_id: str, stage: str, decision: str, reason: str, **details: Any) -> dict[str, Any]:
    return {"comparison_id": comparison_id, "stage": stage, "decision": decision, "reason": reason, **details}


def _comparison_reports_valid(out_dir: Path, filenames: list[str], comparison: dict[str, Any] | None = None, summary_filename: str | None = None) -> bool:
    if not all((out_dir / filename).is_file() for filename in filenames):
        return False
    if comparison is None or summary_filename is None:
        return True
    summary_path = out_dir / summary_filename
    if not summary_path.is_file():
        return False
    summary = read_manifest(summary_path, verify_hash=False)
    return _summary_run_hash_matches(summary.get("before"), comparison["before_run_dir"]) and _summary_run_hash_matches(summary.get("after"), comparison["after_run_dir"])


def _summary_run_hash_matches(reference: Any, run_dir: Path) -> bool:
    if not isinstance(reference, dict):
        return False
    manifest_path = run_dir / "retrieval_replay_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = read_manifest(manifest_path, verify_hash=False)
    return reference.get("manifest_hash") == manifest.get("manifest_hash")


def _snapshot_comparison_reusable(manifest_path: Path, comparison_plan: Path) -> bool:
    if not manifest_path.is_file():
        return False
    manifest = read_manifest(manifest_path, verify_hash=False)
    if Path(str(manifest.get("comparison_plan_path", ""))).resolve() != comparison_plan.resolve():
        return False
    required = [
        manifest.get("snapshot_diff_report"),
        manifest.get("snapshot_diff_markdown"),
        manifest.get("before", {}).get("corpus_snapshot_manifest") if isinstance(manifest.get("before"), dict) else None,
        manifest.get("after", {}).get("corpus_snapshot_manifest") if isinstance(manifest.get("after"), dict) else None,
    ]
    return all(isinstance(path, str) and Path(path).is_file() for path in required)


def _comparison_invalidated(comparison: dict[str, Any], context: dict[str, Any]) -> bool:
    refresh = context.get("retrieval_refresh", {})
    before_tests = set(refresh.get("before", {}).get("generated_tests", []))
    after_tests = set(refresh.get("after", {}).get("generated_tests", []))
    before_name = Path(comparison["before_run_dir"]).name
    after_name = Path(comparison["after_run_dir"]).name
    return any(f"__{test}__" in before_name for test in before_tests) or any(f"__{test}__" in after_name for test in after_tests)


def _load_source_descriptor(path: Path) -> Any:
    from .corpus.sources import descriptor_from_dict

    _check_existing_file("source_descriptor", path, [])
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Prerequisite failed: invalid source descriptor JSON {path}: {exc}") from exc
    return descriptor_from_dict(raw, base_path=path.parent)


def _check_source_descriptor(side: str, descriptor_path: Path, descriptor: Any, checks: list[dict[str, Any]]) -> None:
    _check_existing_file(f"source_descriptor_{side}", descriptor_path, checks)
    if descriptor.source_type == "local":
        if descriptor.source_path is None or not descriptor.source_path.is_dir():
            raise ValidationError(f"Prerequisite failed: local source path not found for {side}: {descriptor.source_path}")
        _check_docs_root(side, descriptor.source_path, descriptor.docs_root, checks)
        checks.append({"name": f"local_source_{side}", "status": "ok", "path": str(descriptor.source_path)})
        return
    if descriptor.source_path is None:
        raise ValidationError(f"Prerequisite failed: remote Git clone is not implemented for {side}; provide source_path.")
    if not descriptor.source_path.is_dir():
        raise ValidationError(f"Prerequisite failed: Git source path not found for {side}: {descriptor.source_path}")
    commit = _git_output(descriptor.source_path, ["rev-parse", "--verify", f"{descriptor.requested_ref or 'HEAD'}^{{commit}}"])
    clean = _git_output(descriptor.source_path, ["status", "--porcelain", "--untracked-files=all"]) == ""
    if not clean:
        raise ValidationError(f"Prerequisite failed: Git source worktree must be clean for {side}: {descriptor.source_path}")
    if descriptor.docs_root:
        _git_output(descriptor.source_path, ["cat-file", "-e", f"{commit}:{descriptor.docs_root}"])
    checks.append(
        {
            "name": f"git_source_{side}",
            "status": "ok",
            "path": str(descriptor.source_path),
            "requested_ref": descriptor.requested_ref,
            "resolved_commit": commit,
            "docs_root": descriptor.docs_root,
            "source_worktree_clean": clean,
        }
    )


def _check_docs_root(side: str, source_path: Path, docs_root: str | None, checks: list[dict[str, Any]]) -> None:
    root = source_path / docs_root if docs_root else source_path
    if not root.is_dir():
        raise ValidationError(f"Prerequisite failed: docs_root not found for {side}: {root}")
    checks.append({"name": f"docs_root_{side}", "status": "ok", "path": str(root)})


def _check_existing_file(name: str, path: Path, checks: list[dict[str, Any]]) -> None:
    if not path.is_file():
        raise ValidationError(f"Prerequisite failed: {name} not found: {path}")
    checks.append({"name": name, "status": "ok", "path": str(path)})


def _docker_server_version() -> str:
    try:
        completed = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        raise ValidationError("Prerequisite failed: Docker executable is not available.") from exc
    output = completed.stdout.strip()
    if completed.returncode != 0:
        raise ValidationError(f"Prerequisite failed: Docker engine is not available:\n{output}")
    return output or "unknown"


def _git_output(repo: Path, args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ValidationError("Prerequisite failed: Git executable is not available.") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValidationError(f"Prerequisite failed: Git command failed in {repo}: {detail}")
    return completed.stdout.strip()


def _ensure_runs_selected(plan: dict[str, Any], context: dict[str, Any]) -> None:
    if "selected_comparisons" in context:
        return
    before_manifest = context.get("before_experiment_manifest") or _experiment_manifest_path(Path(plan["retrieval"]["before_config"]))
    after_manifest = context.get("after_experiment_manifest") or _experiment_manifest_path(Path(plan["retrieval"]["after_config"]))
    selected = []
    for comparison in plan["comparisons"]:
        selected.append(
            {
                "comparison_id": comparison["comparison_id"],
                "before_run_dir": select_run_from_experiment_manifest(before_manifest, comparison["before_run_selector"]),
                "after_run_dir": select_run_from_experiment_manifest(after_manifest, comparison["after_run_selector"]),
            }
        )
    context["selected_comparisons"] = selected


def _experiment_manifest_path(config_path: Path) -> Path:
    config = load_config(config_path)
    return (config.output_dir / config.experiment_name / "experiment_manifest.json").resolve()


def _build_pilot_manifest(
    plan: dict[str, Any],
    plan_path: Path,
    *,
    created_at: str,
    stage_records: list[dict[str, Any]],
    context: dict[str, Any],
    baseline_ref: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": PILOT_RUN_SCHEMA_VERSION,
        "created_at": created_at,
        "manifest_type": "review2026_pilot_run",
        "pilot_id": plan["pilot_id"],
        "plan_path": str(plan_path.resolve()),
        "plan_hash": sha256_file(plan_path),
        "corpus_id": plan["corpus_id"],
        "query_set_id": plan["query_set_id"],
        "stage_status": _overall_status(stage_records),
        "stages": stage_records,
        "snapshot_pair": plan["snapshot_pair"],
        "retrieval_runs": _retrieval_run_refs(context),
        "comparisons": _comparison_refs(context),
        "reuse_decisions": context.get("reuse_decisions", []),
        "baseline_bundle": baseline_ref or {"include_raw": False, "status": "not_refreshed"},
        "environment": environment_metadata(),
    }


def _retrieval_run_refs(context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value)
        for key, value in sorted(context.items())
        if key.endswith("_experiment_manifest") and isinstance(value, Path)
    }


def _comparison_refs(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in context.get("selected_comparisons", []):
        rows.append({key: str(value) if isinstance(value, Path) else value for key, value in sorted(item.items())})
    return rows


def _overall_status(stage_records: list[dict[str, Any]]) -> str:
    if any(item["status"] == "failed" for item in stage_records):
        return "failed"
    if stage_records and all(item["status"] == "dry_run" for item in stage_records):
        return "dry_run"
    return "completed" if stage_records else "not_started"


def _dry_stage_record(stage_name: str) -> dict[str, Any]:
    return {
        "stage": stage_name,
        "status": "dry_run",
        "started_at": None,
        "completed_at": None,
        "input_artifacts": [],
        "output_artifacts": [],
        "manifest_references": [],
        "unavailable_optional_inputs": [],
    }


def _skipped_stage_record(stage_name: str) -> dict[str, Any]:
    record = _dry_stage_record(stage_name)
    record["status"] = "skipped"
    return record


def _record(inputs: list[str], outputs: list[str], manifests: list[str]) -> dict[str, Any]:
    return {
        "input_artifacts": [{"path": str(item)} for item in inputs],
        "output_artifacts": [{"path": str(item)} for item in outputs],
        "manifest_references": [{"path": str(item)} for item in manifests],
        "unavailable_optional_inputs": [],
    }


def _initial_context(plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    return {"plan_path": plan_path.resolve(), "pilot_id": plan["pilot_id"]}


def _copytree_excluding_raw(source: Path, target: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in {"raw", "corpus", "questions.jsonl", "answers.jsonl"}}

    shutil.copytree(source, target, ignore=ignore)


def _copy_selected_retrieval_artifacts(calibration_store: Path, comparisons: list[dict[str, Any]]) -> list[str]:
    copied: list[str] = []
    safe_files = ("retrieval_replay_manifest.json", "scores.json", "metadata.json")
    for comparison in comparisons:
        comparison_id = str(comparison.get("comparison_id", "comparison"))
        for side in ("before", "after"):
            run_dir = comparison.get(f"{side}_run_dir")
            if not isinstance(run_dir, Path):
                continue
            target_dir = calibration_store / "retrieval" / comparison_id / side
            target_dir.mkdir(parents=True, exist_ok=True)
            for filename in safe_files:
                source_file = run_dir / filename
                if source_file.exists():
                    target_file = target_dir / filename
                    shutil.copy2(source_file, target_file)
                    copied.append(str(target_file))
    return copied


def _validate_selector(selector: dict[str, Any], field: str) -> None:
    _required_str(selector, "dataset")
    _required_str(selector, "test")
    if "run_id" in selector and not isinstance(selector["run_id"], str):
        raise ValidationError(f"{field}.run_id must be a string when present.")


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValidationError(f"{key} is required and must be a mapping.")
    return value


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{key} is required.")
    return value


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()
