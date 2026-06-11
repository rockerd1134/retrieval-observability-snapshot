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

from retrieval_arena.measurements import collect_systems_measurements, timed_stage
from retrieval_arena.manifests import write_manifest


CREATED_AT = "2026-05-26T00:00:00+00:00"


class SystemsMeasurementsTests(unittest.TestCase):
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

    def write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    def write_snapshot(self, root: Path) -> None:
        root.mkdir(parents=True)
        write_manifest(
            root / "corpus_snapshot_manifest.json",
            {
                "schema_version": "retrieval_arena.snapshot_manifest.v1",
                "created_at": CREATED_AT,
                "manifest_type": "corpus_snapshot",
                "corpus_id": "toy_docs",
                "snapshot_id": "s1",
                "page_count": 2,
                "file_count": 2,
                "corpus_size_bytes": 30,
                "file_inventory": [
                    {"path": "a.md", "doc_id": "a", "size_bytes": 10, "sha256": "a" * 64},
                    {"path": "b.md", "doc_id": "b", "size_bytes": 20, "sha256": "b" * 64},
                ],
                "content_hash": "c" * 64,
            },
        )
        write_manifest(
            root / "graph_snapshot_manifest.json",
            {
                "schema_version": "retrieval_arena.snapshot_manifest.v1",
                "created_at": CREATED_AT,
                "manifest_type": "graph_snapshot",
                "corpus_id": "toy_docs",
                "snapshot_id": "s1",
                "corpus_snapshot_id": "s1",
                "graph_hash": "d" * 64,
                "node_count": 2,
                "edge_count": 3,
                "graph_metrics": {"weak_components": 1, "strong_components": [{"size": 1}, {"size": 1}], "largest_component_size": 2},
            },
        )
        write_manifest(
            root / "support_surface_manifest.json",
            {
                "schema_version": "retrieval_arena.snapshot_manifest.v1",
                "created_at": CREATED_AT,
                "manifest_type": "support_surface",
                "corpus_id": "toy_docs",
                "snapshot_id": "s1",
                "corpus_snapshot_id": "s1",
                "query_set_id": "toy-queries",
                "support_target_count": 2,
                "supported_question_ids": ["q1"],
                "support_targets_by_question": {"q1": ["a", "b"]},
                "support_target_doc_ids": ["a", "b"],
            },
        )

    def write_run(self, root: Path) -> None:
        root.mkdir(parents=True)
        (root / "input").mkdir()
        (root / "input" / "questions.jsonl").write_text('{"question_id":"q1","question":"Q?"}\n', encoding="utf-8")
        self.write_jsonl(
            root / "predictions.jsonl",
            [
                {"question_id": "q1", "retrieved_context": [{"doc_id": "a"}, {"doc_id": "b"}]},
                {"question_id": "q2", "retrieved_context": [{"doc_id": "b"}]},
            ],
        )
        self.write_jsonl(root / "item_scores.jsonl", [{"question_id": "q1", "f1": 1.0}, {"question_id": "q2", "f1": 0.5}])
        self.write_jsonl(
            root / "action_traces.jsonl",
            [
                {"question_id": "q1", "actions": [{"step": 1}, {"step": 2}]},
                {"question_id": "q2", "actions": [{"step": 1}]},
            ],
        )
        (root / "scores.json").write_text('{"mean_f1":0.75}\n', encoding="utf-8")

    def write_reports(self, root: Path) -> dict[str, Path]:
        root.mkdir(parents=True)
        snapshot_diff = root / "snapshot_diff.json"
        snapshot_diff.write_text(
            json.dumps(
                {
                    "schema_version": "retrieval_arena.snapshot_diff.v1",
                    "corpus_result": {"added_file_count": 1, "removed_file_count": 2, "changed_file_count": 3},
                    "graph_result": {"added_edge_count": 4, "removed_edge_count": 5},
                    "support_surface_result": {"added_target_reference_count": 6, "removed_target_reference_count": 7, "changed_question_count": 8},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        drift_jsonl = root / "retrieval_drift.jsonl"
        self.write_jsonl(drift_jsonl, [{"question_id": "q1"}, {"question_id": "q2"}])
        drift_summary = root / "retrieval_drift_summary.json"
        write_manifest(
            drift_summary,
            {
                "schema_version": "retrieval_arena.retrieval_drift_summary.v1",
                "created_at": CREATED_AT,
                "manifest_type": "retrieval_drift_summary",
                "query_count": 2,
            },
        )
        audit_jsonl = root / "regression_audit.jsonl"
        self.write_jsonl(
            audit_jsonl,
            [
                {
                    "question_id": "q1",
                    "cause_labels": ["retrieved_candidate_changed", "support_recall_regressed"],
                    "evidence_availability": {"support": {"available": True}, "graph": {"available": False}},
                },
                {
                    "question_id": "q2",
                    "cause_labels": ["support_recall_regressed"],
                    "evidence_availability": {"support": {"available": True}, "graph": {"available": False}},
                },
            ],
        )
        audit_summary = root / "regression_audit_summary.json"
        write_manifest(
            audit_summary,
            {
                "schema_version": "retrieval_arena.regression_audit_summary.v1",
                "created_at": CREATED_AT,
                "manifest_type": "regression_audit_summary",
                "query_count": 2,
                "cause_label_counts": {"retrieved_candidate_changed": 1, "support_recall_regressed": 2},
                "evidence_availability_counts": {"graph": {"unavailable": 2}, "support": {"available": 2}},
            },
        )
        return {
            "snapshot_diff": snapshot_diff,
            "drift_jsonl": drift_jsonl,
            "drift_summary": drift_summary,
            "audit_jsonl": audit_jsonl,
            "audit_summary": audit_summary,
        }

    def test_collects_workload_artifact_hashes_jsonl_counts_and_ratios(self):
        with self.workspace_tempdir() as tmp:
            snapshot = tmp / "snapshot"
            run = tmp / "run"
            reports = self.write_reports(tmp / "reports")
            self.write_snapshot(snapshot)
            self.write_run(run)

            report = collect_systems_measurements(
                tmp / "measurements",
                snapshot_dir=snapshot,
                run_dir=run,
                snapshot_diff_json=reports["snapshot_diff"],
                drift_jsonl=reports["drift_jsonl"],
                drift_summary_json=reports["drift_summary"],
                audit_jsonl=reports["audit_jsonl"],
                audit_summary_json=reports["audit_summary"],
                created_at=CREATED_AT,
            )

            workload = report["workload_metrics"]
            artifacts = report["artifact_metrics"]
            self.assertEqual(workload["page_count"], 2)
            self.assertEqual(workload["graph_edge_count"], 3)
            self.assertEqual(workload["query_count"], 2)
            self.assertEqual(workload["retrieved_context_count_per_query"], {"q1": 2, "q2": 1})
            self.assertEqual(workload["action_trace_step_count"], 3)
            self.assertEqual(workload["snapshot_diff_counts"]["changed_pages"], 3)
            self.assertEqual(workload["drift_row_count"], 2)
            self.assertEqual(workload["audit_row_count"], 2)
            self.assertEqual(workload["audit_cause_label_counts"]["support_recall_regressed"], 2)
            self.assertEqual(workload["evidence_availability_counts"]["graph"]["unavailable"], 2)
            self.assertGreater(artifacts["manifest_artifacts"]["artifact_count"], 0)
            self.assertGreater(artifacts["report_artifacts"]["artifact_count"], 0)
            self.assertIn("predictions.jsonl", artifacts["jsonl_row_counts"])
            self.assertEqual(artifacts["jsonl_row_counts"]["predictions.jsonl"], 2)
            self.assertTrue(artifacts["ratios"]["bytes_per_page"]["available"])
            self.assertEqual(artifacts["ratios"]["bytes_per_page"]["numerator"], 30)
            self.assertEqual(artifacts["ratios"]["bytes_per_page"]["denominator"], 2)
            self.assertEqual(artifacts["ratios"]["audit_rows_per_query"]["value"], 1.0)
            self.assertTrue((tmp / "measurements" / "systems_measurements.json").exists())
            self.assertTrue((tmp / "measurements" / "systems_measurements.md").exists())

    def test_missing_optional_artifact_families_are_unavailable_not_zero(self):
        with self.workspace_tempdir() as tmp:
            report = collect_systems_measurements(tmp / "measurements", created_at=CREATED_AT)

            workload = report["workload_metrics"]
            missing = {(item["artifact_family"], item["metric"]) for item in report["unavailable_metrics"]}
            self.assertIsNone(workload["page_count"])
            self.assertIsNone(workload["audit_row_count"])
            self.assertIn(("snapshot", "snapshot_metrics"), missing)
            self.assertIn(("run", "run_metrics"), missing)
            self.assertIn(("audit", "audit_row_count"), missing)

    def test_environment_and_timing_schema_are_present(self):
        with self.workspace_tempdir() as tmp:
            metric: dict[str, Any]
            with timed_stage("manifest_writing") as metric:
                pass

            report = collect_systems_measurements(tmp / "measurements", created_at=CREATED_AT, stage_runtime_metrics=[metric])

            environment = report["environment"]
            runtime = report["stage_runtime_metrics"][0]
            self.assertIn("python_version", environment)
            self.assertIn("retrieval_arena_git_provenance", environment)
            self.assertEqual(runtime["stage_name"], "manifest_writing")
            self.assertEqual(runtime["status"], "completed")
            self.assertIsInstance(runtime["duration_seconds"], float)

    def test_report_formatting_is_deterministic_and_cli_writes_reports(self):
        with self.workspace_tempdir() as tmp:
            snapshot = tmp / "snapshot"
            run = tmp / "run"
            self.write_snapshot(snapshot)
            self.write_run(run)
            out = tmp / "measurements"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "retrieval_arena.cli",
                    "measurements",
                    "collect",
                    "--snapshot-dir",
                    str(snapshot),
                    "--run-dir",
                    str(run),
                    "--out-dir",
                    str(out),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Systems measurements wrote 2 reports", result.stdout)
            json_text = (out / "systems_measurements.json").read_text(encoding="utf-8")
            markdown_text = (out / "systems_measurements.md").read_text(encoding="utf-8")
            self.assertTrue(json_text.endswith("\n"))
            self.assertIn("No efficiency score", markdown_text)
            written = json.loads(json_text)
            self.assertEqual(written["schema_version"], "retrieval_arena.systems_measurements.v1")


if __name__ == "__main__":
    unittest.main()
