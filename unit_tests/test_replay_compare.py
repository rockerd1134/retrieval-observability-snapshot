from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from retrieval_arena.manifests import write_manifest
from retrieval_arena.replay_compare import compare_run_dirs


CREATED_AT = "2026-05-25T00:00:00+00:00"


class ReplayCompareTests(unittest.TestCase):
    @contextmanager
    def workspace_tempdir(self) -> Iterator[Path]:
        root = Path(".tmp_unit_tests")
        root.mkdir(exist_ok=True)
        path = root / f"tmp_{uuid.uuid4().hex}"
        path.mkdir()
        try:
            yield path.resolve()
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def write_run(self, run_dir: Path, *, answer: str = "Use pip.", context: list[dict[str, Any]] | None = None) -> None:
        run_dir.mkdir(parents=True)
        context = context if context is not None else [{"doc_id": "install", "score": 1.0}]
        prediction = {
            "question_id": "q1",
            "question": "How install?",
            "generated_answer": answer,
            "retrieved_context": context,
        }
        (run_dir / "predictions.jsonl").write_text(json.dumps(prediction, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "metadata.json").write_text(
            json.dumps({"deterministic": True, "name": "toy", "version": "1"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_dir / "item_scores.jsonl").write_text(
            json.dumps({"question_id": "q1", "f1": 1.0, "match": True}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_dir / "scores.json").write_text(
            json.dumps({"mean_f1": 1.0, "num_questions": 1}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_manifest(run_dir / "retrieval_replay_manifest.json", self.manifest_payload(created_at=CREATED_AT))

    def manifest_payload(self, *, created_at: str) -> dict[str, Any]:
        return {
            "schema_version": "retrieval_arena.replay_manifest.v1",
            "created_at": created_at,
            "manifest_type": "retrieval_replay",
            "run_id": "run123",
            "experiment_name": "toy",
            "dataset": "toy",
            "test": "oracle",
            "query_set_id": "toy-queries",
            "query_set_hash": "a" * 64,
            "corpus_snapshot_id": "corpus-v1",
            "graph_snapshot_id": "graph-v1",
            "support_surface_id": "support-v1",
            "retrieval_config_id": "oracle",
            "retrieval_config_hash": "b" * 64,
            "container_image": "toy-image",
            "container_metadata_hash": "c" * 64,
            "build_context_hash": "d" * 64,
            "scoring_method": "lexical_baseline",
            "scoring_hash": "e" * 64,
            "retrieval_arena_version": "0.1.0",
            "retrieval_arena_git_provenance": {"is_git_worktree": False},
            "run_started_at": created_at,
            "run_completed_at": created_at,
            "input_artifact_hashes": {},
            "output_artifact_hashes": {},
        }

    def rewrite_manifest(self, run_dir: Path, **updates: Any) -> None:
        payload = self.manifest_payload(created_at=updates.pop("created_at", CREATED_AT))
        payload.update(updates)
        write_manifest(run_dir / "retrieval_replay_manifest.json", payload)

    def test_identical_run_directories_pass(self):
        with self.workspace_tempdir() as tmp:
            expected = tmp / "expected"
            actual = tmp / "actual"
            self.write_run(expected)
            shutil.copytree(expected, actual)

            report = compare_run_dirs(expected, actual, created_at=CREATED_AT)

            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["changed_artifact_count"], 0)
            self.assertEqual(report["summary"]["operator_status"], "MATCHED_EXACTLY")
            self.assertEqual(report["summary"]["replay_outcome"], "exact_match")
            self.assertTrue(report["summary"]["replay_matched"])
            self.assertTrue(report["summary"]["artifact_byte_equality_passed"])
            self.assertEqual(report["prediction_results"]["changed_question_ids"], [])

    def test_changed_prediction_answer_fails_with_question_id(self):
        with self.workspace_tempdir() as tmp:
            expected = tmp / "expected"
            actual = tmp / "actual"
            self.write_run(expected)
            self.write_run(actual, answer="Use uv.")

            report = compare_run_dirs(expected, actual, created_at=CREATED_AT)

            self.assertFalse(report["passed"])
            self.assertEqual(report["summary"]["operator_status"], "MISMATCHED")
            self.assertEqual(report["summary"]["replay_outcome"], "behavior_mismatch")
            self.assertFalse(report["summary"]["behavior_equality_passed"])
            self.assertEqual(report["prediction_results"]["changed_question_ids"], ["q1"])
            self.assertEqual(report["prediction_results"]["differences"][0]["type"], "generated_answer_changed")

    def test_changed_retrieved_context_rank_fails_with_rank_detail(self):
        with self.workspace_tempdir() as tmp:
            expected = tmp / "expected"
            actual = tmp / "actual"
            self.write_run(expected, context=[{"doc_id": "install", "score": 1.0}, {"doc_id": "usage", "score": 0.5}])
            self.write_run(actual, context=[{"doc_id": "usage", "score": 0.5}, {"doc_id": "install", "score": 1.0}])

            report = compare_run_dirs(expected, actual, created_at=CREATED_AT)

            self.assertFalse(report["passed"])
            self.assertEqual(report["summary"]["operator_status"], "MISMATCHED")
            context_diffs = [diff for diff in report["prediction_results"]["differences"] if diff["type"] == "retrieved_context_changed"]
            self.assertEqual(context_diffs[0]["rank"], 1)
            self.assertEqual(context_diffs[0]["field"], "doc_id")

    def test_changed_scores_fail_with_score_diagnostics(self):
        with self.workspace_tempdir() as tmp:
            expected = tmp / "expected"
            actual = tmp / "actual"
            self.write_run(expected)
            self.write_run(actual)
            (actual / "scores.json").write_text(
                json.dumps({"mean_f1": 0.5, "num_questions": 1}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (actual / "item_scores.jsonl").write_text(
                json.dumps({"question_id": "q1", "f1": 0.5, "match": False}, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = compare_run_dirs(expected, actual, created_at=CREATED_AT)

            self.assertFalse(report["passed"])
            self.assertEqual(report["summary"]["operator_status"], "MISMATCHED")
            self.assertEqual(report["score_results"]["aggregate_scores"]["field_differences"][0]["field"], "mean_f1")
            self.assertEqual(report["score_results"]["item_scores"]["changed_question_ids"], ["q1"])

    def test_changed_aggregate_scores_match_when_per_question_behavior_matches(self):
        with self.workspace_tempdir() as tmp:
            expected = tmp / "expected"
            actual = tmp / "actual"
            self.write_run(expected)
            self.write_run(actual)
            (actual / "scores.json").write_text(
                json.dumps({"mean_f1": 1.0, "num_questions": 1, "run_completed_at": "2026-05-25T01:00:00+00:00"}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = compare_run_dirs(expected, actual, created_at=CREATED_AT)

            self.assertTrue(report["passed"])
            self.assertTrue(report["summary"]["replay_matched"])
            self.assertTrue(report["summary"]["behavior_equality_passed"])
            self.assertFalse(report["summary"]["aggregate_score_equality_passed"])
            self.assertFalse(report["summary"]["artifact_byte_equality_passed"])
            self.assertEqual(report["summary"]["operator_status"], "MATCHED_WITH_DIFFERENCES")
            self.assertEqual(report["summary"]["replay_outcome"], "matched_with_provenance_or_artifact_differences")
            self.assertEqual(report["summary"]["changed_question_count"], 0)
            self.assertEqual(report["summary"]["aggregate_score_difference_count"], 1)

    def test_missing_required_artifact_fails_clearly(self):
        with self.workspace_tempdir() as tmp:
            expected = tmp / "expected"
            actual = tmp / "actual"
            self.write_run(expected)
            self.write_run(actual)
            (actual / "predictions.jsonl").unlink()

            report = compare_run_dirs(expected, actual, created_at=CREATED_AT)

            self.assertFalse(report["passed"])
            self.assertEqual(report["summary"]["operator_status"], "INCOMPLETE")
            prediction_artifact = next(item for item in report["artifact_results"] if item["path"] == "predictions.jsonl")
            self.assertEqual(prediction_artifact["status"], "missing")
            self.assertEqual(report["prediction_results"]["missing"], ["actual"])

    def test_manifest_timestamp_only_differences_pass_when_behavior_matches(self):
        with self.workspace_tempdir() as tmp:
            expected = tmp / "expected"
            actual = tmp / "actual"
            self.write_run(expected)
            self.write_run(actual)
            self.rewrite_manifest(actual, created_at="2026-05-25T01:00:00+00:00")

            report = compare_run_dirs(expected, actual, created_at=CREATED_AT)

            self.assertTrue(report["passed"])
            self.assertTrue(report["manifest_result"]["only_volatile_differences"])
            self.assertEqual(report["summary"]["operator_status"], "MATCHED_WITH_DIFFERENCES")
            self.assertEqual(report["summary"]["replay_outcome"], "matched_with_provenance_or_artifact_differences")
            self.assertGreater(report["summary"]["provenance_sensitive_difference_count"], 0)
            self.assertEqual(report["summary"]["manifest_field_difference_count"], 0)

    def test_report_json_formatting_and_stable_ordering(self):
        with self.workspace_tempdir() as tmp:
            expected = tmp / "expected"
            actual = tmp / "actual"
            out = tmp / "replay_fidelity_report.json"
            self.write_run(expected)
            self.write_run(actual)

            report = compare_run_dirs(expected, actual, created_at=CREATED_AT, out_path=out)
            written = json.loads(out.read_text(encoding="utf-8"))

            self.assertTrue(out.read_text(encoding="utf-8").endswith("\n"))
            self.assertEqual(written["schema_version"], report["schema_version"])
            self.assertEqual(
                [item["path"] for item in report["artifact_results"]],
                ["action_traces.jsonl", "item_scores.jsonl", "metadata.json", "predictions.jsonl", "retrieval_replay_manifest.json", "scores.json"],
            )

    def test_cli_smoke_test(self):
        with self.workspace_tempdir() as tmp:
            expected = tmp / "expected"
            actual = tmp / "actual"
            out = tmp / "report.json"
            self.write_run(expected)
            shutil.copytree(expected, actual)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "retrieval_arena.cli",
                    "replay",
                    "compare",
                    "--expected",
                    str(expected),
                    "--actual",
                    str(actual),
                    "--out",
                    str(out),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Replay MATCHED_EXACTLY", result.stdout)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
