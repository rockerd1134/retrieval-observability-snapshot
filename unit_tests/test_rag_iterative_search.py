from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def load_runner():
    path = Path("tests/rag_iterative_search/runner.py")
    spec = importlib.util.spec_from_file_location("rag_iterative_search_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RagIterativeSearchTests(unittest.TestCase):
    @contextmanager
    def workspace_tempdir(self) -> Iterator[Path]:
        root = Path(".tmp_unit_tests")
        root.mkdir(exist_ok=True)
        path = root / f"tmp_{uuid.uuid4().hex}"
        path.mkdir()
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def write_input(self, root: Path, *, max_actions: int = 12) -> tuple[Path, Path]:
        input_dir = root / "input"
        output_dir = root / "output"
        (input_dir / "corpus").mkdir(parents=True)
        (input_dir / "corpus" / "install.md").write_text("# Install\nUse plugins with collections.\n", encoding="utf-8")
        (input_dir / "corpus" / "collections.md").write_text("# Collections\nCollections package plugins and modules.\n", encoding="utf-8")
        (input_dir / "corpus" / "inventory.md").write_text("# Inventory\nInventory lists managed hosts.\n", encoding="utf-8")
        (input_dir / "questions.jsonl").write_text(
            '{"question_id":"q1","question":"How do plugins work with collections?"}\n',
            encoding="utf-8",
        )
        (input_dir / "answers.jsonl").write_text('{"question_id":"q1","answer":"Collections package plugins."}\n', encoding="utf-8")
        (input_dir / "graph_edges.csv").write_text("source,target\ninstall,collections\ncollections,inventory\n", encoding="utf-8")
        (input_dir / "config.yaml").write_text(
            json.dumps(
                {
                    "search_top_k": 1,
                    "inspect_budget": 4,
                    "follow_budget_per_page": 2,
                    "final_top_k": 2,
                    "max_actions": max_actions,
                    "max_context_chars_per_doc": 200,
                    "directed": False,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return input_dir, output_dir

    def read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_runner_outputs_valid_trace_and_prediction(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root)
            runner.run(input_dir, output_dir)

            predictions = self.read_jsonl(output_dir / "predictions.jsonl")
            traces = self.read_jsonl(output_dir / "action_traces.jsonl")
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(predictions[0]["question_id"], "q1")
            self.assertEqual(traces[0]["schema_version"], "e007a.action_trace.v1")
            self.assertEqual(traces[0]["actions"][-1]["action"], "stop")
            self.assertLessEqual(len(traces[0]["final_context_doc_ids"]), 2)
            self.assertFalse(metadata["uses_llm"])
            self.assertEqual(metadata["provider"], "none")

    def test_runner_is_deterministic(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_one = self.write_input(root)
            output_two = root / "output_two"
            runner.run(input_dir, output_one)
            runner.run(input_dir, output_two)

            self.assertEqual(
                (output_one / "predictions.jsonl").read_text(encoding="utf-8"),
                (output_two / "predictions.jsonl").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (output_one / "action_traces.jsonl").read_text(encoding="utf-8"),
                (output_two / "action_traces.jsonl").read_text(encoding="utf-8"),
            )

    def test_runner_honors_action_budget(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root, max_actions=5)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")
            self.assertLessEqual(len(traces[0]["actions"]), 5)
            self.assertEqual(traces[0]["actions"][-1]["action"], "stop")


if __name__ == "__main__":
    unittest.main()
