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

from retrieval_arena.audit import build_regression_audit
from retrieval_arena.errors import ValidationError
from retrieval_arena.manifests import read_manifest, write_manifest


CREATED_AT = "2026-05-25T00:00:00+00:00"


class RegressionAuditTests(unittest.TestCase):
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

    def write_support_manifest(self, path: Path, targets: dict[str, list[str]]) -> None:
        all_targets = sorted({doc_id for docs in targets.values() for doc_id in docs})
        write_manifest(
            path,
            {
                "schema_version": "retrieval_arena.support_surface_manifest.v1",
                "created_at": CREATED_AT,
                "manifest_type": "support_surface",
                "snapshot_id": path.parent.name,
                "query_set_id": "toy-queries",
                "support_targets_by_question": dict(sorted(targets.items())),
                "support_target_doc_ids": all_targets,
                "support_target_count": len(all_targets),
            },
        )

    def write_run(
        self,
        run_dir: Path,
        *,
        qid: str = "q1",
        context: list[dict[str, Any]] | None = None,
        f1: float = 1.0,
        support_manifest: Path | None = None,
        traces: dict[str, Any] | None = None,
        retrieval_config_hash: str = "a" * 64,
        scoring_hash: str = "b" * 64,
    ) -> None:
        run_dir.mkdir(parents=True)
        context = context if context is not None else [{"doc_id": "install", "score": 1.0}]
        prediction = {
            "question_id": qid,
            "question": "How install?",
            "generated_answer": "Use pip.",
            "retrieved_context": context,
        }
        (run_dir / "predictions.jsonl").write_text(json.dumps(prediction, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "metadata.json").write_text('{"deterministic":true,"name":"toy"}\n', encoding="utf-8")
        (run_dir / "item_scores.jsonl").write_text(json.dumps({"question_id": qid, "f1": f1}, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "scores.json").write_text(json.dumps({"mean_f1": f1}, sort_keys=True) + "\n", encoding="utf-8")
        if traces is not None:
            (run_dir / "action_traces.jsonl").write_text(json.dumps({"question_id": qid, **traces}, sort_keys=True) + "\n", encoding="utf-8")
        refs = {"support_surface": {"path": str(support_manifest)}} if support_manifest else {}
        write_manifest(
            run_dir / "retrieval_replay_manifest.json",
            {
                "schema_version": "retrieval_arena.replay_manifest.v1",
                "created_at": CREATED_AT,
                "manifest_type": "retrieval_replay",
                "run_id": run_dir.name,
                "dataset": "toy",
                "test": "lexical",
                "query_set_id": "toy-queries",
                "corpus_snapshot_id": f"corpus-{run_dir.name}",
                "graph_snapshot_id": f"graph-{run_dir.name}",
                "support_surface_id": support_manifest.parent.name if support_manifest else None,
                "retrieval_config_id": "lexical",
                "retrieval_config_hash": retrieval_config_hash,
                "scoring_method": "lexical",
                "scoring_hash": scoring_hash,
                "snapshot_manifest_references": refs,
            },
        )

    def write_snapshot_diff(self, path: Path) -> None:
        report = {
            "schema_version": "retrieval_arena.snapshot_diff.v1",
            "created_at": CREATED_AT,
            "comparison_type": "snapshot_diff",
            "corpus_result": {
                "available": True,
                "added_files": [{"path": "new.md", "doc_id": "new"}],
                "removed_files": [{"path": "old.md", "doc_id": "old"}],
                "changed_files": [{"path": "install.md", "doc_id": "install", "before_doc_id": "install", "after_doc_id": "install"}],
            },
            "graph_result": {
                "available": True,
                "added_edges": [{"source": "new", "target": "install", "edge_id": "new->install"}],
                "removed_edges": [{"source": "old", "target": "install", "edge_id": "old->install"}],
            },
            "support_surface_result": {
                "available": True,
                "added_questions": [],
                "removed_questions": [],
                "changed_questions": [
                    {
                        "question_id": "q1",
                        "added_targets": ["new"],
                        "removed_targets": ["old"],
                        "before_targets": ["old", "install"],
                        "after_targets": ["new", "install"],
                    }
                ],
            },
        }
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def first_row(self, report: dict[str, Any]) -> dict[str, Any]:
        return report["rows"][0]

    def test_unchanged_drift_and_snapshot_diff_produce_no_regression_labels(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            out = tmp / "audit"
            self.write_run(before, context=[{"doc_id": "install", "score": 1.0}])
            self.write_run(after, context=[{"doc_id": "install", "score": 1.0}])

            row = self.first_row(build_regression_audit(before, after, out, created_at=CREATED_AT))

            self.assertEqual(row["cause_labels"], ["evidence_unavailable"])
            self.assertEqual(row["cause_label_schema_version"], "retrieval_arena.audit_cause_labels.v1")
            self.assertIn("support_exposure:support_targets_unavailable", row["missing_evidence"])

    def test_metric_specific_regression_labels_are_conservative(self):
        with self.workspace_tempdir() as tmp:
            before_support = tmp / "support-before" / "support_surface_manifest.json"
            after_support = tmp / "support-after" / "support_surface_manifest.json"
            self.write_support_manifest(before_support, {"q1": ["old", "install"]})
            self.write_support_manifest(after_support, {"q1": ["old", "install"]})
            before = tmp / "before"
            after = tmp / "after"
            out = tmp / "audit"
            self.write_run(
                before,
                support_manifest=before_support,
                f1=0.9,
                traces={"schema_version": "trace.v1", "actions": [{"step": 1, "action": "search"}], "final_context_doc_ids": ["install"]},
                context=[
                    {"doc_id": "install", "score": 1.0, "distance_to_support": 0, "is_evidence": True},
                    {"doc_id": "old", "score": 0.8, "distance_to_support": 1, "evidence_doc_ids": ["old"]},
                ],
            )
            self.write_run(
                after,
                support_manifest=after_support,
                f1=0.4,
                traces={
                    "schema_version": "trace.v1",
                    "actions": [{"step": 1, "action": "search"}, {"step": 2, "action": "stop"}],
                    "final_context_doc_ids": ["install", "new"],
                },
                context=[
                    {"doc_id": "new", "score": 0.1, "distance_to_support": 4},
                    {"doc_id": "install", "score": 0.7, "distance_to_support": 2},
                ],
            )

            labels = set(self.first_row(build_regression_audit(before, after, out, created_at=CREATED_AT))["cause_labels"])

            self.assertIn("retrieved_candidate_changed", labels)
            self.assertIn("rank_changed", labels)
            self.assertIn("retrieval_score_changed", labels)
            self.assertIn("answer_lexical_score_regressed", labels)
            self.assertIn("support_exposure_regressed", labels)
            self.assertIn("support_recall_regressed", labels)
            self.assertIn("evidence_coverage_regressed", labels)
            self.assertIn("distance_to_support_increased", labels)
            self.assertIn("action_trace_changed", labels)

    def test_snapshot_diff_labels_join_by_question_and_document_ids(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            snapshot_diff = tmp / "snapshot_diff.json"
            out = tmp / "audit"
            self.write_run(before, context=[{"doc_id": "old", "score": 1.0}, {"doc_id": "install", "score": 0.9}])
            self.write_run(after, context=[{"doc_id": "new", "score": 1.0}, {"doc_id": "install", "score": 0.8}])
            self.write_snapshot_diff(snapshot_diff)

            labels = set(self.first_row(build_regression_audit(before, after, out, snapshot_diff_json=snapshot_diff, created_at=CREATED_AT))["cause_labels"])

            self.assertIn("support_target_removed", labels)
            self.assertIn("support_target_added", labels)
            self.assertIn("support_target_changed", labels)
            self.assertIn("corpus_page_removed", labels)
            self.assertIn("corpus_page_added", labels)
            self.assertIn("corpus_page_changed", labels)
            self.assertIn("graph_edge_removed", labels)
            self.assertIn("graph_edge_added", labels)

    def test_manifest_hash_changes_label_retrieval_scoring_and_replay_changes(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            out = tmp / "audit"
            self.write_run(before, retrieval_config_hash="a" * 64, scoring_hash="b" * 64)
            self.write_run(after, retrieval_config_hash="c" * 64, scoring_hash="d" * 64)

            labels = set(self.first_row(build_regression_audit(before, after, out, created_at=CREATED_AT))["cause_labels"])

            self.assertIn("retrieval_config_changed", labels)
            self.assertIn("scoring_config_changed", labels)
            self.assertIn("replay_manifest_changed", labels)

    def test_malformed_or_unjoinable_inputs_fail_clearly(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            out = tmp / "audit"
            self.write_run(before)
            self.write_run(after)
            manifest = read_manifest(after / "retrieval_replay_manifest.json", verify_hash=False)
            manifest["dataset"] = "other"
            write_manifest(after / "retrieval_replay_manifest.json", manifest)

            with self.assertRaisesRegex(ValidationError, "dataset differs"):
                build_regression_audit(before, after, out, created_at=CREATED_AT)

    def test_cli_smoke_writes_all_audit_artifacts(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            out = tmp / "audit"
            self.write_run(before)
            self.write_run(after)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "retrieval_arena.cli",
                    "audit",
                    "report",
                    "--before-run",
                    str(before),
                    "--after-run",
                    str(after),
                    "--out-dir",
                    str(out),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Regression audit wrote 1 query rows", result.stdout)
            self.assertTrue((out / "regression_audit.jsonl").exists())
            self.assertTrue((out / "regression_audit_summary.json").exists())
            self.assertTrue((out / "regression_audit.md").exists())

    def test_report_formatting_is_deterministic_and_stably_ordered(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            out = tmp / "audit"
            self.write_run(before)
            self.write_run(after)

            build_regression_audit(before, after, out, created_at=CREATED_AT)

            jsonl_text = (out / "regression_audit.jsonl").read_text(encoding="utf-8")
            summary_text = (out / "regression_audit_summary.json").read_text(encoding="utf-8")
            self.assertTrue(jsonl_text.endswith("\n"))
            self.assertTrue(summary_text.endswith("\n"))
            self.assertEqual(json.loads(jsonl_text)["question_id"], "q1")
            summary = read_manifest(out / "regression_audit_summary.json")
            self.assertEqual(summary["schema_version"], "retrieval_arena.regression_audit_summary.v1")
            self.assertEqual(summary["cause_label_schema_version"], "retrieval_arena.audit_cause_labels.v1")
            self.assertIn("retrieved_candidate_changed", summary["cause_label_schema"])
            self.assertIn("Overall drift score: not computed", (out / "regression_audit.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
