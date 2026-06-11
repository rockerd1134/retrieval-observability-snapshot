from pathlib import Path
import shutil
import unittest
import uuid
from contextlib import contextmanager
from typing import Iterator

from retrieval_arena.errors import ValidationError
from retrieval_arena.schemas import validate_action_traces, validate_dataset, validate_metadata, validate_predictions, write_jsonl


class SchemaTests(unittest.TestCase):
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

    def test_validate_dataset_accepts_toy_dataset(self):
        questions, answers = validate_dataset(Path("examples/toy_docs"))
        self.assertEqual(len(questions), 3)
        self.assertEqual(set(answers), {"q001", "q002", "q003"})


    def test_validate_predictions_rejects_missing_question(self):
        questions, _ = validate_dataset(Path("examples/toy_docs"))
        with self.workspace_tempdir() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            write_jsonl(path, [{"question_id": "q001", "question": questions[0]["question"], "generated_answer": "x", "retrieved_context": []}])
            with self.assertRaisesRegex(ValidationError, "Missing predictions"):
                validate_predictions(path, questions)


    def test_validate_predictions_rejects_unknown_question(self):
        questions, _ = validate_dataset(Path("examples/toy_docs"))
        rows = [{"question_id": q["question_id"], "question": q["question"], "generated_answer": "x", "retrieved_context": []} for q in questions]
        rows[0]["question_id"] = "unknown"
        with self.workspace_tempdir() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValidationError, "unknown question_id"):
                validate_predictions(path, questions)


    def test_validate_predictions_rejects_duplicate_question(self):
        questions = [
            {"question_id": "q001", "question": "Question one?"},
            {"question_id": "q002", "question": "Question two?"},
        ]
        rows = [
            {"question_id": "q001", "question": "Question one?", "generated_answer": "x", "retrieved_context": []},
            {"question_id": "q001", "question": "Question one?", "generated_answer": "y", "retrieved_context": []},
        ]
        with self.workspace_tempdir() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValidationError, "Duplicate prediction"):
                validate_predictions(path, questions)

    def test_validate_predictions_rejects_invalid_jsonl(self):
        questions, _ = validate_dataset(Path("examples/toy_docs"))
        with self.workspace_tempdir() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text('{"question_id": "q001"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "Invalid JSON"):
                validate_predictions(path, questions)

    def test_validate_predictions_rejects_missing_generated_answer(self):
        questions, _ = validate_dataset(Path("examples/toy_docs"))
        rows = [{"question_id": q["question_id"], "question": q["question"], "generated_answer": "x", "retrieved_context": []} for q in questions]
        del rows[0]["generated_answer"]
        with self.workspace_tempdir() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValidationError, "generated_answer"):
                validate_predictions(path, questions)

    def test_validate_predictions_rejects_mismatched_question_text(self):
        questions, _ = validate_dataset(Path("examples/toy_docs"))
        rows = [{"question_id": q["question_id"], "question": q["question"], "generated_answer": "x", "retrieved_context": []} for q in questions]
        rows[0]["question"] = questions[1]["question"]
        with self.workspace_tempdir() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValidationError, "question text"):
                validate_predictions(path, questions)

    def test_validate_predictions_rejects_missing_question_field(self):
        questions, _ = validate_dataset(Path("examples/toy_docs"))
        rows = [{"question_id": q["question_id"], "question": q["question"], "generated_answer": "x", "retrieved_context": []} for q in questions]
        del rows[0]["question"]
        with self.workspace_tempdir() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValidationError, "question string"):
                validate_predictions(path, questions)

    def test_validate_predictions_rejects_malformed_context(self):
        questions, _ = validate_dataset(Path("examples/toy_docs"))
        rows = [{"question_id": q["question_id"], "question": q["question"], "generated_answer": "x", "retrieved_context": []} for q in questions]
        rows[0]["retrieved_context"] = [{"doc_id": "d1", "score": "high"}]
        with self.workspace_tempdir() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValidationError, "score must be numeric"):
                validate_predictions(path, questions)

    def test_validate_metadata_requires_name_and_deterministic_flag(self):
        with self.workspace_tempdir() as tmp:
            path = Path(tmp) / "metadata.json"
            path.write_text('{"name":"bad"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "deterministic"):
                validate_metadata(path)

            path.write_text('{"deterministic":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "name"):
                validate_metadata(path)

            path.write_text('{"name":"ok","deterministic":true}\n', encoding="utf-8")
            self.assertEqual(validate_metadata(path)["name"], "ok")

    def test_validate_action_traces_accepts_e007a_trace_schema(self):
        questions = [{"question_id": "q001", "question": "Question one?"}]
        rows = [
            {
                "question_id": "q001",
                "schema_version": "e007a.action_trace.v1",
                "policy": "deterministic_bm25_link_walk_v1",
                "actions": [
                    {"step": 1, "action": "search", "budget_remaining": 2, "details": {"top_doc_ids": ["a"]}},
                    {"step": 2, "action": "inspect_page", "budget_remaining": 1, "details": {"doc_id": "a"}},
                    {"step": 3, "action": "stop", "budget_remaining": 0, "details": {"reason": "test"}},
                ],
                "final_context_doc_ids": ["a"],
            }
        ]
        with self.workspace_tempdir() as tmp:
            path = Path(tmp) / "action_traces.jsonl"
            write_jsonl(path, rows)
            self.assertEqual(validate_action_traces(path, questions)[0]["question_id"], "q001")

    def test_validate_action_traces_rejects_bad_action_and_missing_stop(self):
        questions = [{"question_id": "q001", "question": "Question one?"}]
        rows = [
            {
                "question_id": "q001",
                "schema_version": "e007a.action_trace.v1",
                "actions": [
                    {"step": 1, "action": "search", "budget_remaining": 1},
                    {"step": 2, "action": "teleport", "budget_remaining": 0},
                ],
            }
        ]
        with self.workspace_tempdir() as tmp:
            path = Path(tmp) / "action_traces.jsonl"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValidationError, "unsupported action"):
                validate_action_traces(path, questions)


if __name__ == "__main__":
    unittest.main()
