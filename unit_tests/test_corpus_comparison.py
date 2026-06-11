from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from retrieval_arena.corpus.comparison import run_corpus_snapshot_comparison
from retrieval_arena.manifests import read_manifest


CREATED_AT = "2026-05-25T00:00:00+00:00"


class CorpusComparisonTests(unittest.TestCase):
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

    def write_descriptor(self, path: Path, *, source: Path, snapshot_id: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "retrieval_arena.corpus_source_descriptor.v1",
                    "corpus_id": "toy",
                    "snapshot_id": snapshot_id,
                    "source_type": "local",
                    "source_path": str(source),
                    "destination_workspace": str(path.parent / "imports"),
                    "include": ["*.md"],
                    "exclude": [],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_comparison_plan_imports_prepares_manifests_and_diffs(self):
        with self.workspace_tempdir() as tmp:
            before_source = tmp / "before_source"
            after_source = tmp / "after_source"
            before_source.mkdir()
            after_source.mkdir()
            (before_source / "install.md").write_text("Install MiniDocs with pip package manager.\n", encoding="utf-8")
            (before_source / "guide.md").write_text("Read [install](install.md).\n", encoding="utf-8")
            (after_source / "install.md").write_text("Install MiniDocs with uv package manager.\n", encoding="utf-8")
            (after_source / "guide.md").write_text("Read install.\n", encoding="utf-8")
            before_descriptor = tmp / "before.json"
            after_descriptor = tmp / "after.json"
            self.write_descriptor(before_descriptor, source=before_source, snapshot_id="before")
            self.write_descriptor(after_descriptor, source=after_source, snapshot_id="after")
            questions = tmp / "questions.jsonl"
            answers = tmp / "answers.jsonl"
            questions.write_text('{"question_id":"q1","question":"How install?"}\n', encoding="utf-8")
            answers.write_text('{"question_id":"q1","answer":"Install MiniDocs with pip package manager."}\n', encoding="utf-8")
            plan = tmp / "comparison.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema_version": "retrieval_arena.corpus_snapshot_comparison_plan.v1",
                        "comparison_id": "toy_pair",
                        "corpus_id": "toy",
                        "before_descriptor": str(before_descriptor),
                        "after_descriptor": str(after_descriptor),
                        "output_dir": str(tmp / "reports"),
                        "questions_path": str(questions),
                        "answers_path": str(answers),
                        "graph": {"enabled": True, "method": "markdown_links"},
                        "query_set_id": "toy-queries",
                        "support": {"enabled": True, "method": "answer_lexical_overlap"},
                        "overwrite_imports": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            first = run_corpus_snapshot_comparison(plan, created_at=CREATED_AT)
            second = run_corpus_snapshot_comparison(plan, created_at=CREATED_AT)

            self.assertEqual(first["snapshot_diff"]["summary"]["changed_file_count"], 2)
            self.assertTrue(second["manifest"]["before"]["import_reused"])
            self.assertTrue(second["manifest"]["after"]["import_reused"])
            self.assertEqual(read_manifest(tmp / "reports" / "before" / "snapshot_manifests" / "corpus_snapshot_manifest.json")["manifest_type"], "corpus_snapshot")
            graph = read_manifest(tmp / "reports" / "before" / "snapshot_manifests" / "graph_snapshot_manifest.json")
            self.assertEqual(graph["manifest_type"], "graph_snapshot")
            self.assertEqual(graph["edge_inventory"], [{"source": "guide", "target": "install"}])
            self.assertRegex(graph["source_dataset_identity_hash"], r"^[0-9a-f]{64}$")
            self.assertEqual(graph["graph_extraction_config"]["link_extraction_configuration"]["method"], "markdown_links")
            support = read_manifest(tmp / "reports" / "before" / "snapshot_manifests" / "support_surface_manifest.json")
            self.assertEqual(support["manifest_type"], "support_surface")
            self.assertIn("install", support["support_targets_by_question"]["q1"])
            self.assertEqual(support["support_label_counts"], {"supported": 1})
            self.assertRegex(support["source_dataset_identity_hash"], r"^[0-9a-f]{64}$")
            self.assertEqual(support["support_construction_config"]["support_construction_method"], "idf_weighted_answer_support_v1")
            preparation = read_manifest(tmp / "reports" / "before" / "dataset" / "dataset_preparation_manifest.json")
            self.assertEqual(preparation["query_set_id"], "toy-queries")
            self.assertEqual(preparation["validation_status"], "ok")


if __name__ == "__main__":
    unittest.main()
