from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from retrieval_arena.commands import replay_compare, snapshot_diff, snapshot_manifest
from retrieval_arena.manifests import read_manifest, write_manifest


CREATED_AT = "2026-05-25T00:00:00+00:00"


class CommandLayerTests(unittest.TestCase):
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

    def toy_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        corpus = dataset / "corpus"
        corpus.mkdir(parents=True)
        (corpus / "install.md").write_text("Use pip.\n", encoding="utf-8")
        (dataset / "questions.jsonl").write_text('{"question_id":"q1","question":"How install?"}\n', encoding="utf-8")
        (dataset / "answers.jsonl").write_text('{"question_id":"q1","answer":"Use pip."}\n', encoding="utf-8")
        return dataset

    def write_run(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True)
        (run_dir / "predictions.jsonl").write_text(
            '{"generated_answer":"Use pip.","question":"How install?","question_id":"q1","retrieved_context":[{"doc_id":"install","score":1.0}]}\n',
            encoding="utf-8",
        )
        (run_dir / "metadata.json").write_text('{"deterministic":true,"name":"toy"}\n', encoding="utf-8")
        (run_dir / "item_scores.jsonl").write_text('{"f1":1.0,"question_id":"q1"}\n', encoding="utf-8")
        (run_dir / "scores.json").write_text('{"mean_f1":1.0}\n', encoding="utf-8")
        write_manifest(
            run_dir / "retrieval_replay_manifest.json",
            {
                "schema_version": "retrieval_arena.replay_manifest.v1",
                "created_at": CREATED_AT,
                "manifest_type": "retrieval_replay",
                "run_id": "run123",
                "experiment_name": "toy",
            },
        )

    def test_snapshot_manifest_command_returns_structured_result(self):
        with self.workspace_tempdir() as tmp:
            result = snapshot_manifest(
                self.toy_dataset(tmp),
                tmp / "manifests",
                corpus_id="toy_docs",
                snapshot_id="s1",
                extraction_version="extract-v1",
                parser_version="parser-v1",
                manifest_types={"corpus"},
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["summary"], "retrieval audit snapshot manifests written: 1")
            self.assertEqual(read_manifest(tmp / "manifests" / "corpus_snapshot_manifest.json")["manifest_type"], "corpus_snapshot")

    def test_snapshot_diff_command_matches_report_summary(self):
        with self.workspace_tempdir() as tmp:
            before_dataset = self.toy_dataset(tmp / "before_src")
            after_dataset = self.toy_dataset(tmp / "after_src")
            (after_dataset / "corpus" / "usage.md").write_text("Run it.\n", encoding="utf-8")
            snapshot_manifest(before_dataset, tmp / "before", corpus_id="toy_docs", snapshot_id="s1", extraction_version="v", parser_version="v", manifest_types={"corpus"})
            snapshot_manifest(after_dataset, tmp / "after", corpus_id="toy_docs", snapshot_id="s2", extraction_version="v", parser_version="v", manifest_types={"corpus"})

            result = snapshot_diff(tmp / "before", tmp / "after", out_path=tmp / "diff.json")

            self.assertFalse(result["ok"])
            self.assertIn("files +1", result["summary"])
            self.assertEqual(json.loads((tmp / "diff.json").read_text(encoding="utf-8"))["summary"]["added_file_count"], 1)

    def test_replay_compare_command_returns_written_artifacts(self):
        with self.workspace_tempdir() as tmp:
            expected = tmp / "expected"
            actual = tmp / "actual"
            self.write_run(expected)
            shutil.copytree(expected, actual)

            result = replay_compare(expected, actual, out_path=tmp / "report.json")

            self.assertTrue(result["ok"])
            self.assertIn("Replay MATCHED_EXACTLY", result["summary"])
            self.assertEqual(result["written_artifacts"], [str(tmp / "report.json")])


if __name__ == "__main__":
    unittest.main()
