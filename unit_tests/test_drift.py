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

from retrieval_arena.drift import compare_retrieval_runs
from retrieval_arena.errors import ValidationError
from retrieval_arena.manifests import read_manifest, write_manifest


CREATED_AT = "2026-05-25T00:00:00+00:00"


class RetrievalDriftTests(unittest.TestCase):
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
        precision: float = 1.0,
        support_manifest: Path | None = None,
        traces: dict[str, Any] | None = None,
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
        (run_dir / "item_scores.jsonl").write_text(
            json.dumps({"question_id": qid, "f1": f1, "precision": precision}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
                "retrieval_config_hash": "a" * 64,
                "snapshot_manifest_references": refs,
            },
        )

    def first_row(self, report: dict[str, Any]) -> dict[str, Any]:
        return report["rows"][0]

    def test_identical_run_directories_produce_zero_drift(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            self.write_run(before, context=[{"doc_id": "install", "score": 1.0}, {"doc_id": "usage", "score": 0.5}])
            self.write_run(after, context=[{"doc_id": "install", "score": 1.0}, {"doc_id": "usage", "score": 0.5}])

            report = compare_retrieval_runs(before, after, created_at=CREATED_AT)
            row = self.first_row(report)

            self.assertEqual(row["metrics"]["top_k_jaccard"]["value"], 1.0)
            self.assertEqual(row["metrics"]["ordered_top_k_overlap"]["value"], 1.0)
            self.assertEqual(row["metrics"]["rank_displacement"]["mean_absolute_delta"], 0.0)
            self.assertEqual(row["metrics"]["retained_score_delta"]["mean_delta"], 0.0)

    def test_changed_top_k_documents_changes_jaccard(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            self.write_run(before, context=[{"doc_id": "install"}, {"doc_id": "usage"}])
            self.write_run(after, context=[{"doc_id": "install"}, {"doc_id": "config"}])

            row = self.first_row(compare_retrieval_runs(before, after, created_at=CREATED_AT))

            self.assertAlmostEqual(row["metrics"]["top_k_jaccard"]["value"], 1 / 3)
            self.assertEqual(row["metrics"]["top_k_jaccard"]["intersection_count"], 1)

    def test_rank_swaps_report_displacement_and_ordered_overlap(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            self.write_run(before, context=[{"doc_id": "install"}, {"doc_id": "usage"}])
            self.write_run(after, context=[{"doc_id": "usage"}, {"doc_id": "install"}])

            row = self.first_row(compare_retrieval_runs(before, after, created_at=CREATED_AT))

            self.assertEqual(row["metrics"]["ordered_top_k_overlap"]["value"], 0.0)
            self.assertEqual(row["metrics"]["rank_displacement"]["mean_absolute_delta"], 1.0)

    def test_score_changes_report_retained_document_score_delta(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            self.write_run(before, context=[{"doc_id": "install", "score": 0.25}])
            self.write_run(after, context=[{"doc_id": "install", "score": 0.75}])

            row = self.first_row(compare_retrieval_runs(before, after, created_at=CREATED_AT))

            self.assertEqual(row["metrics"]["retained_score_delta"]["documents"][0]["score_delta"], 0.5)

    def test_lexical_item_score_deltas_are_reported(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            self.write_run(before, f1=0.25, precision=0.5)
            self.write_run(after, f1=0.75, precision=1.0)

            row = self.first_row(compare_retrieval_runs(before, after, created_at=CREATED_AT))

            self.assertTrue(row["metrics"]["lexical_score_delta"]["available"])
            self.assertEqual(row["metrics"]["lexical_score_delta"]["f1_delta"], 0.5)
            self.assertEqual(row["metrics"]["lexical_score_delta"]["precision_delta"], 0.5)

    def test_missing_optional_signals_are_unavailable(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            self.write_run(before, context=[{"doc_id": "install"}])
            self.write_run(after, context=[{"doc_id": "install"}])

            row = self.first_row(compare_retrieval_runs(before, after, created_at=CREATED_AT))

            self.assertFalse(row["metrics"]["support_exposure"]["available"])
            self.assertFalse(row["metrics"]["evidence_coverage"]["available"])
            self.assertFalse(row["metrics"]["distance_to_support"]["available"])
            self.assertFalse(row["metrics"]["action_trace"]["available"])

    def test_support_exposure_recall_evidence_and_distance_metrics_work(self):
        with self.workspace_tempdir() as tmp:
            before_support = tmp / "support-before" / "support_surface_manifest.json"
            after_support = tmp / "support-after" / "support_surface_manifest.json"
            self.write_support_manifest(before_support, {"q1": ["install", "usage"]})
            self.write_support_manifest(after_support, {"q1": ["install", "usage"]})
            before = tmp / "before"
            after = tmp / "after"
            self.write_run(
                before,
                support_manifest=before_support,
                context=[{"doc_id": "install", "score": 1.0, "distance_to_support": 1, "is_evidence": True}],
            )
            self.write_run(
                after,
                support_manifest=after_support,
                context=[{"doc_id": "usage", "score": 1.0, "distance_to_support": 0, "is_evidence": True}, {"doc_id": "other"}],
            )

            row = self.first_row(compare_retrieval_runs(before, after, created_at=CREATED_AT))

            self.assertEqual(row["metrics"]["support_exposure"]["before_exposed_count"], 1)
            self.assertEqual(row["metrics"]["support_exposure"]["after_exposed_count"], 1)
            self.assertEqual(row["metrics"]["support_recall"]["before_recall"], 0.5)
            self.assertEqual(row["metrics"]["support_recall"]["after_recall"], 0.5)
            self.assertTrue(row["metrics"]["evidence_coverage"]["available"])
            self.assertEqual(row["metrics"]["distance_to_support"]["min_distance_delta"], -1.0)

    def test_action_trace_length_and_final_context_drift_work(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            self.write_run(
                before,
                traces={"schema_version": "trace.v1", "actions": [{"step": 1, "action": "search"}, {"step": 2, "action": "stop"}], "final_context_doc_ids": ["install"]},
            )
            self.write_run(
                after,
                traces={
                    "schema_version": "trace.v1",
                    "actions": [{"step": 1, "action": "search"}, {"step": 2, "action": "add_context"}, {"step": 3, "action": "stop"}],
                    "final_context_doc_ids": ["usage"],
                },
            )

            row = self.first_row(compare_retrieval_runs(before, after, created_at=CREATED_AT))

            self.assertEqual(row["metrics"]["action_trace"]["action_count_delta"], 1)
            self.assertEqual(row["metrics"]["action_trace"]["final_context_jaccard"], 0.0)

    def test_incomparable_query_ids_fail_clearly(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            self.write_run(before, qid="q1")
            self.write_run(after, qid="q2")

            with self.assertRaisesRegex(ValidationError, "not comparable by query IDs"):
                compare_retrieval_runs(before, after, created_at=CREATED_AT)

    def test_cli_smoke_writes_all_report_artifacts(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            out_dir = tmp / "reports"
            self.write_run(before)
            self.write_run(after)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "retrieval_arena.cli",
                    "drift",
                    "compare",
                    "--before-run",
                    str(before),
                    "--after-run",
                    str(after),
                    "--out-dir",
                    str(out_dir),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Retrieval drift compared 1 queries", result.stdout)
            self.assertTrue((out_dir / "retrieval_drift.jsonl").exists())
            self.assertTrue((out_dir / "retrieval_drift_summary.json").exists())
            self.assertTrue((out_dir / "retrieval_drift_summary.md").exists())

    def test_report_formatting_is_deterministic_and_stably_ordered(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = tmp / "after"
            out_dir = tmp / "reports"
            self.write_run(before)
            self.write_run(after)

            compare_retrieval_runs(before, after, out_dir=out_dir, created_at=CREATED_AT)

            jsonl_text = (out_dir / "retrieval_drift.jsonl").read_text(encoding="utf-8")
            summary_text = (out_dir / "retrieval_drift_summary.json").read_text(encoding="utf-8")
            self.assertTrue(jsonl_text.endswith("\n"))
            self.assertTrue(summary_text.endswith("\n"))
            self.assertEqual(read_manifest(out_dir / "retrieval_drift_summary.json")["schema_version"], "retrieval_arena.retrieval_drift_summary.v1")
            self.assertEqual(json.loads(jsonl_text)["question_id"], "q1")


if __name__ == "__main__":
    unittest.main()
