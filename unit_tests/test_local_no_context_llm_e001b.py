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
    path = Path("tests/local_no_context_llm_e001b/runner.py")
    spec = importlib.util.spec_from_file_location("local_no_context_llm_e001b_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LocalNoContextLlmE001bTests(unittest.TestCase):
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

    def write_input(self, root: Path, config: dict[str, object] | None = None) -> tuple[Path, Path]:
        input_dir = root / "input"
        output_dir = root / "output"
        (input_dir / "corpus").mkdir(parents=True)
        (input_dir / "corpus" / "install.md").write_text("# Install\nUse the installer.\n", encoding="utf-8")
        (input_dir / "questions.jsonl").write_text('{"question_id":"q1","question":"How do I install it?"}\n', encoding="utf-8")
        (input_dir / "answers.jsonl").write_text('{"question_id":"q1","answer":"Use the installer."}\n', encoding="utf-8")
        (input_dir / "config.yaml").write_text(json.dumps(config or {"provider": "mock", "local_only": True, "allow_network": False}, sort_keys=True), encoding="utf-8")
        return input_dir, output_dir

    def read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_mock_e001b_outputs_no_context_and_policy_metadata(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root)
            runner.run(input_dir, output_dir)

            predictions = self.read_jsonl(output_dir / "predictions.jsonl")
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(predictions[0]["generated_answer"], "Mock no-context answer.")
            self.assertEqual(predictions[0]["retrieved_context"], [])
            self.assertEqual(predictions[0]["metadata"]["experiment_id"], "E001b")
            self.assertFalse(predictions[0]["metadata"]["paper_facing_evidence"])
            self.assertEqual(predictions[0]["metadata"]["provider_provenance"]["provider"], "mock")
            self.assertEqual(metadata["experiment_id"], "E001b")
            self.assertEqual(metadata["name"], "local_no_context_llm_e001b")
            self.assertFalse(metadata["uses_retrieval"])
            self.assertFalse(metadata["allow_network"])

    def test_live_e001b_refuses_without_local_execution_gate(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(
                root,
                {
                    "provider": "openai_compatible",
                    "model_id": "qwen3:14b",
                    "base_url": "http://localhost:11434/v1",
                    "local_only": True,
                    "allow_network": False,
                    "allow_local_provider_execution": False,
                },
            )

            with self.assertRaisesRegex(RuntimeError, "allow_local_provider_execution"):
                runner.run(input_dir, output_dir)

    def test_live_e001b_refuses_remote_or_network_policy(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(
                root,
                {
                    "provider": "openai_compatible",
                    "model_id": "remote/model",
                    "base_url": "https://api.openai.com/v1",
                    "local_only": False,
                    "allow_network": False,
                    "allow_local_provider_execution": True,
                },
            )

            with self.assertRaisesRegex(RuntimeError, "local_only=true"):
                runner.run(input_dir, output_dir)


if __name__ == "__main__":
    unittest.main()
