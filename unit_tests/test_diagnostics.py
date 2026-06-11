from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from retrieval_arena.diagnostics import enrich_run_diagnostics
from retrieval_arena.schemas import read_jsonl, validate_action_traces


class DiagnosticsTests(unittest.TestCase):
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

    def test_enriches_predictions_with_support_distance_and_trace(self):
        with self.workspace_tempdir() as tmp:
            dataset = tmp / "dataset"
            run = tmp / "run"
            dataset.mkdir()
            run.mkdir()
            (dataset / "graph_edges.csv").write_text("source,target\nintro,install\nother,intro\n", encoding="utf-8")
            (dataset / "faq_support_audit.jsonl").write_text(
                json.dumps({"question_id": "q1", "top_docs": [{"doc_id": "install"}]}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (run / "predictions.jsonl").write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "question": "How install?",
                        "generated_answer": "Use install.",
                        "retrieved_context": [{"doc_id": "other", "score": 1.0}, {"doc_id": "install", "score": 0.5}],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            summary = enrich_run_diagnostics(run, dataset)

            self.assertTrue(summary["distance_overlay_available"])
            row = read_jsonl(run / "predictions.jsonl")[0]
            self.assertEqual(row["retrieved_context"][0]["distance_to_support"], 2)
            self.assertEqual(row["retrieved_context"][1]["distance_to_support"], 0)
            traces = validate_action_traces(run / "action_traces.jsonl", [{"question_id": "q1", "question": "How install?"}])
            self.assertEqual(traces[0]["trace_type"], "deterministic_retrieval_execution_trace")
            self.assertEqual(traces[0]["final_context_doc_ids"], ["other", "install"])


if __name__ == "__main__":
    unittest.main()
