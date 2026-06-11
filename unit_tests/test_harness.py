from __future__ import annotations

import unittest
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from retrieval_arena.config import TestConfig, VolumeMount
from retrieval_arena.docker import run_container
from retrieval_arena.harness import assert_contract_negative_checks, prepare_input
from retrieval_arena.errors import ValidationError


class HarnessTests(unittest.TestCase):
    @contextmanager
    def workspace_tempdir(self) -> Iterator[str]:
        root = Path(".tmp_unit_tests")
        root.mkdir(exist_ok=True)
        path = root / f"tmp_{uuid.uuid4().hex}"
        path.mkdir()
        try:
            yield str(path)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_prepare_input_copies_optional_faq_support_audit(self):
        with self.workspace_tempdir() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            (dataset / "corpus").mkdir(parents=True)
            (dataset / "corpus" / "doc.md").write_text("# Doc\n", encoding="utf-8")
            (dataset / "questions.jsonl").write_text('{"question_id":"q1","question":"Q?"}\n', encoding="utf-8")
            (dataset / "answers.jsonl").write_text('{"question_id":"q1","answer":"A"}\n', encoding="utf-8")
            (dataset / "faq_support_audit.jsonl").write_text('{"question_id":"q1","top_docs":[]}\n', encoding="utf-8")
            input_dir = root / "input"

            prepare_input(dataset, TestConfig(name="oracle_graph_support", image="test", config={"experiment_id": "E005"}), input_dir)

            self.assertTrue((input_dir / "faq_support_audit.jsonl").exists())
            self.assertIn("E005", (input_dir / "config.yaml").read_text(encoding="utf-8"))

    def test_run_container_fails_fast_for_missing_volume_host_path(self):
        with self.workspace_tempdir() as tmp:
            root = Path(tmp)
            missing = root / "missing-model"
            test = TestConfig(name="rag_embedding_topk", image="no-such-image", volumes=[VolumeMount(missing, "/models/minilm")])
            with self.assertRaisesRegex(ValidationError, "volume host_path not found"):
                run_container(test, root / "input", root / "output")

    def test_contract_negative_checks_cover_adversarial_outputs(self):
        questions = [
            {"question_id": "q001", "question": "Question one?"},
            {"question_id": "q002", "question": "Question two?"},
        ]
        with self.workspace_tempdir() as tmp:
            assert_contract_negative_checks(Path(tmp), questions)


if __name__ == "__main__":
    unittest.main()
