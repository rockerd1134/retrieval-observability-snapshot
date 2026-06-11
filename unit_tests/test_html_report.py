from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from retrieval_arena.evidence_export import export_paper_evidence
from retrieval_arena.html_report import assemble_report_data, build_html_observability_report, _select_case_studies


CREATED_AT = "2026-05-26T00:00:00+00:00"


class HtmlReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = Path(__file__).resolve().parent / "_tmp_html_report" / uuid.uuid4().hex
        self.tmp_root.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def write_bundle(self, root: Path) -> Path:
        self.write_json(root / "pilot_manifest.json", {"pilot_id": "pilot_toy", "corpus_id": "toy_docs", "query_set_id": "toy-queries"})
        self.write_json(root / "plan_resolved.json", {"pilot_id": "pilot_toy", "corpus_id": "toy_docs"})
        self.write_json(
            root / "snapshot_comparison" / "snapshot_diff.json",
            {
                "schema_version": "retrieval_arena.snapshot_diff.v1",
                "summary": {
                    "page_count_delta": 1,
                    "added_file_count": 2,
                    "removed_file_count": 1,
                    "changed_file_count": 3,
                    "corpus_size_delta_bytes": 128,
                    "graph_available": True,
                    "added_edge_count": 4,
                    "removed_edge_count": 1,
                    "support_surface_available": True,
                    "changed_support_question_count": 1,
                    "stable_manifest_difference_count": 2,
                },
                "support_surface_result": {
                    "changed_questions": [{"question_id": "q1"}],
                },
            },
        )
        for side, pages in (("before", 10), ("after", 11)):
            self.write_json(
                root / "snapshot_comparison" / side / "snapshot_manifests" / "corpus_snapshot_manifest.json",
                {
                    "snapshot_id": side,
                    "page_count": pages,
                    "content_hash": f"{side}-content",
                    "manifest_hash": f"{side}-manifest",
                    "source_commit": f"{side}-commit",
                },
            )
            self.write_json(
                root / "snapshot_comparison" / side / "snapshot_manifests" / "graph_snapshot_manifest.json",
                {"node_count": pages, "edge_count": pages * 2},
            )
            self.write_json(
                root / "snapshot_comparison" / side / "snapshot_manifests" / "support_surface_manifest.json",
                {"support_target_count": 4, "supported_query_count": 2},
            )
        comparison = root / "comparisons" / "rag_lexical_topk"
        self.write_json(
            comparison / "retrieval_drift_summary.json",
            {
                "query_count": 2,
                "mean_top_k_jaccard": 0.5,
                "mean_ordered_top_k_overlap": 0.75,
                "mean_rank_displacement": 0.2,
                "support_recall_regression_count": 1,
                "support_exposure_regression_count": 1,
                "evidence_coverage_regression_count": 0,
                "optional_signal_availability": {"action_trace": {"available_count": 2, "unavailable_count": 0}},
            },
        )
        self.write_json(
            comparison / "regression_audit_summary.json",
            {
                "labeled_query_count": 2,
                "cause_label_counts": {"rank_changed": 2, "support_recall_regressed": 1},
                "evidence_availability_counts": {"graph": {"available": 2}, "support_surface": {"available": 2}},
            },
        )
        self.write_json(
            comparison / "systems_measurements.json",
            {
                "workload_metrics": {"page_count": 11, "graph_edge_count": 22, "audit_row_count": 2},
                "artifact_metrics": {
                    "ratios": {
                        "report_bytes_per_query": {"available": True, "value": 100.0},
                        "artifact_bytes_per_page": {"available": True, "value": 20.5},
                    }
                },
            },
        )
        self.write_jsonl(
            comparison / "regression_audit.jsonl",
            [
                {
                    "question_id": "q1",
                    "cause_labels": ["rank_changed", "support_recall_regressed"],
                    "before": {"doc_ids": ["guide/a.md", "guide/b.md"]},
                    "after": {"doc_ids": ["guide/c.md", "guide/a.md"]},
                    "associated_evidence": {
                        "corpus": {"changed_doc_ids": ["guide/a.md"], "added_doc_ids": ["guide/c.md"]},
                        "graph": {"added_edges": [{"source": "guide/c.md", "target": "guide/a.md", "relation": "links"}]},
                        "support_surface": {"changed_target_doc_ids": ["guide/a.md"]},
                    },
                    "drift_metrics": {
                        "top_k_jaccard": {"value": 0.33},
                        "ordered_top_k_overlap": {"value": 0.25},
                        "rank_displacement": {
                            "documents": [
                                {"doc_id": "guide/a.md", "before_rank": 1, "after_rank": 2, "rank_delta": 1},
                            ]
                        },
                        "retained_score_delta": {
                            "documents": [
                                {"doc_id": "guide/a.md", "before_score": 0.9, "after_score": 0.7, "score_delta": -0.2},
                            ]
                        },
                        "support_exposure": {"before_exposed_count": 2, "after_exposed_count": 1, "exposed_count_delta": -1},
                        "evidence_coverage": {"before_coverage": 1.0, "after_coverage": 0.5, "coverage_delta": -0.5},
                        "support_recall": {"before_recall": 1.0, "after_recall": 0.5, "recall_delta": -0.5},
                        "distance_to_support": {"before_min_distance": 0, "after_min_distance": 1, "min_distance_delta": 1},
                        "action_trace": {"before_action_count": 2, "after_action_count": 3, "action_count_delta": 1, "final_context_jaccard": 0.5},
                    },
                }
            ],
        )
        for side in ("before", "after"):
            self.write_json(
                root / "retrieval" / "rag_lexical_topk" / side / "retrieval_replay_manifest.json",
                {"run_id": side, "retrieval_config_id": "rag_lexical_topk"},
            )
        return root

    def test_report_data_derives_metric_backed_insights(self):
        root = self.write_bundle(self.tmp_root / "bundle")
        data = assemble_report_data(root, created_at=CREATED_AT)

        self.assertEqual(data["schema_version"], "retrieval_arena.html_observability_report.v1")
        self.assertEqual(data["pilot_id"], "pilot_toy")
        self.assertIn("Corpus page count changed by +1 pages", data["insights"][0])
        self.assertEqual(data["comparisons"][0]["comparison_id"], "rag_lexical_topk")
        self.assertEqual(data["comparisons"][0]["case_studies"][0]["question_id"], "q1")

    def test_report_writer_outputs_self_contained_html_sections(self):
        root = self.write_bundle(self.tmp_root / "bundle")
        out = self.tmp_root / "report.html"
        result = build_html_observability_report(root, out_path=out, created_at=CREATED_AT)
        html = out.read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("<style>", html)
        self.assertIn("Generated Insights", html)
        self.assertIn("Retrieval Family Matrix", html)
        self.assertIn("Query Case Studies", html)
        self.assertIn("Before retrieved docs", html)
        self.assertIn("Action trace steps", html)
        self.assertIn("guide/c.md", html)
        self.assertIn("rag_lexical_topk", html)
        self.assertNotIn("http://", html)

    def test_case_study_selection_is_deterministic_and_signal_weighted(self):
        audit_path = self.tmp_root / "audit.jsonl"
        self.write_jsonl(
            audit_path,
            [
                {"question_id": "q-b", "cause_labels": ["rank_changed"], "drift_metrics": {"top_k_jaccard": {"value": 0.9}}},
                {"question_id": "q-a", "cause_labels": ["rank_changed"], "drift_metrics": {"top_k_jaccard": {"value": 0.9}}},
                {"question_id": "q-c", "cause_labels": ["rank_changed"], "drift_metrics": {"support_recall": {"recall_delta": -1.0}}},
            ],
        )

        selected = _select_case_studies(audit_path, limit=3)

        self.assertEqual([row["question_id"] for row in selected], ["q-c", "q-a", "q-b"])

    def test_cli_smoke_writes_html_report(self):
        root = self.write_bundle(self.tmp_root / "bundle")
        out = self.tmp_root / "cli_report.html"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "retrieval_arena.cli",
                "report",
                "html",
                "--bundle",
                str(root),
                "--out",
                str(out),
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("HTML observability report wrote", result.stdout)
        self.assertTrue(out.exists())

    def test_paper_evidence_export_writes_report_derived_tables(self):
        root = self.write_bundle(self.tmp_root / "bundle")
        out_dir = self.tmp_root / "evidence"
        result = export_paper_evidence(root, out_dir)

        self.assertTrue(result["ok"])
        family_csv = (out_dir / "family_drift_matrix.csv").read_text(encoding="utf-8")
        case_csv = (out_dir / "selected_case_studies.csv").read_text(encoding="utf-8")
        provenance = (out_dir / "provenance_replay_summary.md").read_text(encoding="utf-8")
        self.assertIn("mean_top_k_jaccard", family_csv)
        self.assertIn("rag_lexical_topk", family_csv)
        self.assertIn("q1", case_csv)
        self.assertIn("pilot_manifest.json", provenance)

    def test_cli_smoke_writes_paper_evidence_export(self):
        root = self.write_bundle(self.tmp_root / "bundle")
        out_dir = self.tmp_root / "cli_evidence"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "retrieval_arena.cli",
                "report",
                "evidence",
                "--bundle",
                str(root),
                "--out-dir",
                str(out_dir),
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Paper evidence exported", result.stdout)
        self.assertTrue((out_dir / "corpus_graph_support_summary.csv").exists())

    def test_real_baseline_smoke_has_nonempty_key_sections(self):
        repo = Path(__file__).resolve().parents[1]
        baseline = repo / "calibration" / "review2026" / "pilot_express_docs"
        if not baseline.is_dir():
            self.skipTest("Express calibration baseline is not present.")
        out = self.tmp_root / "express_report.html"
        result = build_html_observability_report(baseline, out_path=out, created_at=CREATED_AT)
        html = out.read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("Corpus Snapshot Summary", html)
        self.assertIn("Graph And Support Drift", html)
        self.assertIn("Regression Audit Labels", html)
        self.assertIn("rag_iterative_search", html)


if __name__ == "__main__":
    unittest.main()
