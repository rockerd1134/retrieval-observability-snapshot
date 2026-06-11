from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import ExperimentConfig, TestConfig, load_config
from .diagnostics import enrich_run_diagnostics
from .docker import build_image, run_container
from .errors import RetrievalAuditError, ValidationError
from .reports import write_run_scores, write_summary
from .replay_manifests import write_experiment_manifest, write_run_replay_manifest
from .schemas import validate_action_traces, validate_dataset, validate_metadata, validate_predictions, write_jsonl
from .scoring import score_predictions


def git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def stable_run_id(dataset_name: str, test_name: str) -> str:
    return hashlib.sha1(f"{dataset_name}\0{test_name}".encode("utf-8")).hexdigest()[:10]


def prepare_input(dataset_path: Path, test: TestConfig, input_dir: Path) -> None:
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)
    shutil.copytree(dataset_path / "corpus", input_dir / "corpus")
    for filename in ["questions.jsonl", "answers.jsonl", "graph_edges.csv", "graph_metrics.json", "faq_support_audit.jsonl"]:
        source = dataset_path / filename
        if source.exists():
            shutil.copy2(source, input_dir / filename)
    (input_dir / "config.yaml").write_text(json.dumps(test.config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(config: ExperimentConfig, *, output_dir: Path | None = None) -> list[dict[str, Any]]:
    root_output = (output_dir or config.output_dir).resolve()
    experiment_dir = root_output / config.experiment_name
    runs_dir = experiment_dir / "runs"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    repo_root = config.config_path.parent.parent

    for dataset in config.datasets:
        validate_dataset(dataset.path)
    for test in config.tests:
        build_image(test)

    run_manifest_paths: list[Path] = []
    for dataset in config.datasets:
        questions, answers = validate_dataset(dataset.path)
        for test in config.tests:
            run_id = stable_run_id(dataset.name, test.name)
            run_dir = runs_dir / f"{dataset.name}__{test.name}__{run_id}"
            input_dir = run_dir / "input"
            output_subdir = run_dir / "container_output"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)
            prepare_input(dataset.path, test, input_dir)
            started_at = datetime.now(timezone.utc).isoformat()
            run_container(test, input_dir, output_subdir)
            metadata = validate_metadata(output_subdir / "metadata.json")
            predictions = validate_predictions(output_subdir / "predictions.jsonl", questions)
            shutil.copy2(output_subdir / "predictions.jsonl", run_dir / "predictions.jsonl")
            shutil.copy2(output_subdir / "metadata.json", run_dir / "metadata.json")
            action_trace_path = output_subdir / "action_traces.jsonl"
            if action_trace_path.exists():
                validate_action_traces(action_trace_path, questions)
                shutil.copy2(action_trace_path, run_dir / "action_traces.jsonl")
            diagnostic_overlay = enrich_run_diagnostics(run_dir, dataset.path)
            predictions = validate_predictions(run_dir / "predictions.jsonl", questions)
            if (run_dir / "action_traces.jsonl").exists():
                validate_action_traces(run_dir / "action_traces.jsonl", questions)
            item_scores, aggregate = score_predictions(predictions, answers, config.scoring.match_threshold)
            completed_at = datetime.now(timezone.utc).isoformat()
            aggregate.update({
                "dataset": dataset.name,
                "test": test.name,
                "image": test.image,
                "build_context": str(test.build_context) if test.build_context else "",
                "run_id": run_id,
                "started_at": started_at,
                "completed_at": completed_at,
                "config_path": str(config.config_path),
                "retrieval_arena_version": __version__,
                "git_commit": git_commit(repo_root),
                "container_metadata": metadata,
                "diagnostic_overlay": diagnostic_overlay,
            })
            write_run_scores(run_dir, item_scores, aggregate)
            manifest_path = run_dir / "retrieval_replay_manifest.json"
            write_run_replay_manifest(
                manifest_path,
                config=config,
                dataset=dataset,
                test=test,
                run_id=run_id,
                run_dir=run_dir,
                metadata=metadata,
                run_started_at=started_at,
                run_completed_at=completed_at,
            )
            run_manifest_paths.append(manifest_path)
            summary_rows.append({**aggregate, "run_dir": str(run_dir)})
    write_summary(experiment_dir, summary_rows)
    write_experiment_manifest(
        experiment_dir / "experiment_manifest.json",
        config=config,
        experiment_dir=experiment_dir,
        run_manifests=run_manifest_paths,
    )
    return summary_rows


def run_config(config_path: str | Path) -> list[dict[str, Any]]:
    return run_experiment(load_config(config_path))


def assert_expected_oracle_scores(rows: list[dict[str, Any]]) -> None:
    by_test = {row["test"]: row for row in rows}
    required = {"oracle_perfect", "oracle_empty", "oracle_partial"}
    missing = required - set(by_test)
    if missing:
        raise ValidationError(f"Phase 1 validation config must include oracle tests: {sorted(missing)}")
    if by_test["oracle_perfect"]["match_percent"] != 1.0 or by_test["oracle_perfect"]["mean_f1"] != 1.0:
        raise ValidationError("oracle_perfect must score exactly 1.0 match_percent and mean_f1.")
    if by_test["oracle_empty"]["match_percent"] != 0.0 or by_test["oracle_empty"]["mean_f1"] != 0.0:
        raise ValidationError("oracle_empty must score exactly 0.0 match_percent and mean_f1.")
    partial = by_test["oracle_partial"]
    if not (0.0 < partial["mean_f1"] < 1.0):
        raise ValidationError("oracle_partial must produce intermediate mean_f1.")


def comparable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["dataset", "test", "match_percent", "mean_f1", "mean_precision", "mean_recall", "mean_lexical_overlap", "num_questions"]
    return [{key: row[key] for key in keys} for row in sorted(rows, key=lambda r: (r["dataset"], r["test"]))]


def assert_contract_negative_checks(root: Path, questions: list[dict[str, str]]) -> None:
    rows = [
        {"question_id": q["question_id"], "question": q["question"], "generated_answer": "", "retrieved_context": []}
        for q in questions
    ]
    if not rows:
        raise ValidationError("Phase 1 validation requires at least one question.")
    cases: list[tuple[str, list[dict[str, Any]], str]] = [
        ("duplicate_prediction", [*rows, dict(rows[0])], "Duplicate prediction"),
        ("unknown_question_id", [{**rows[0], "question_id": "__unknown__"}, *rows[1:]], "unknown question_id"),
        ("missing_generated_answer", [{key: value for key, value in rows[0].items() if key != "generated_answer"}, *rows[1:]], "generated_answer"),
        ("wrong_question_text", [{**rows[0], "question": rows[-1]["question"] + " wrong"} if len(rows) == 1 else {**rows[0], "question": rows[1]["question"]}, *rows[1:]], "question text"),
        ("missing_prediction", rows[:-1], "Missing predictions"),
    ]
    for name, case_rows, expected in cases:
        path = root / name / "predictions.jsonl"
        write_jsonl(path, case_rows)
        try:
            validate_predictions(path, questions)
        except ValidationError as exc:
            if expected not in str(exc):
                raise RetrievalAuditError(f"Contract negative check {name} failed with unexpected error: {exc}") from exc
        else:
            raise RetrievalAuditError(f"Contract negative check failed: {name} was accepted.")

    invalid_jsonl = root / "invalid_jsonl" / "predictions.jsonl"
    invalid_jsonl.parent.mkdir(parents=True, exist_ok=True)
    invalid_jsonl.write_text('{"question_id": "q001"\n', encoding="utf-8")
    try:
        validate_predictions(invalid_jsonl, questions)
    except ValidationError as exc:
        if "Invalid JSON" not in str(exc):
            raise RetrievalAuditError(f"Contract negative check invalid_jsonl failed with unexpected error: {exc}") from exc
    else:
        raise RetrievalAuditError("Contract negative check failed: invalid JSONL was accepted.")

    bad_metadata = root / "missing_metadata_field" / "metadata.json"
    bad_metadata.parent.mkdir(parents=True, exist_ok=True)
    bad_metadata.write_text('{"name":"bad"}\n', encoding="utf-8")
    try:
        validate_metadata(bad_metadata)
    except ValidationError as exc:
        if "deterministic" not in str(exc):
            raise RetrievalAuditError(f"Contract negative check missing_metadata_field failed with unexpected error: {exc}") from exc
    else:
        raise RetrievalAuditError("Contract negative check failed: incomplete metadata.json was accepted.")


def validate_phase1(config_path: str | Path) -> None:
    config = load_config(config_path)
    for dataset in config.datasets:
        validate_dataset(dataset.path)
    validation_root = config.output_dir / "_validation_phase1"
    first = run_experiment(config, output_dir=validation_root / "run1")
    assert_expected_oracle_scores(first)
    second = run_experiment(config, output_dir=validation_root / "run2")
    assert_expected_oracle_scores(second)
    if comparable(first) != comparable(second):
        raise RetrievalAuditError("Reproducibility check failed: two deterministic validation runs produced different scores.")

    questions, _ = validate_dataset(config.datasets[0].path)
    assert_contract_negative_checks(validation_root / "contract_negative", questions)
