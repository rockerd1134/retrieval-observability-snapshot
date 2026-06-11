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
    path = Path("tests/rag_random_seed_graph_traversal/runner.py")
    spec = importlib.util.spec_from_file_location("rag_random_seed_graph_traversal_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RandomSeedGraphTraversalTests(unittest.TestCase):
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

    def write_input(self, root: Path, *, rng_seed: int = 20260504) -> tuple[Path, Path]:
        input_dir = root / "input"
        output_dir = root / "output"
        corpus_dir = input_dir / "corpus"
        corpus_dir.mkdir(parents=True)
        (corpus_dir / "a.md").write_text("# A\nAlpha start page.", encoding="utf-8")
        (corpus_dir / "b.md").write_text("# B\nBeta page with plugins.", encoding="utf-8")
        (corpus_dir / "c.md").write_text("# C\nCollections package plugins.", encoding="utf-8")
        (corpus_dir / "d.md").write_text("# D\nDeployment unrelated.", encoding="utf-8")
        (input_dir / "questions.jsonl").write_text(
            '{"question_id":"q1","question":"How do collections package plugins?"}\n',
            encoding="utf-8",
        )
        (input_dir / "answers.jsonl").write_text('{"question_id":"q1","answer":"Collections package plugins."}\n', encoding="utf-8")
        (input_dir / "graph_edges.csv").write_text("source,target\na,b\nb,c\nc,d\n", encoding="utf-8")
        (input_dir / "config.yaml").write_text(
            json.dumps(
                {
                    "num_trials": 3,
                    "rng_seed": rng_seed,
                    "max_hops": 2,
                    "neighbor_budget": 1,
                    "candidate_budget": 3,
                    "final_top_k": 2,
                    "distance_penalty": 0.10,
                    "max_context_chars_per_doc": 200,
                    "directed": False,
                    "experiment_id": "E010",
                    "diagnostic_role": "random_seed_graph_navigability",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return input_dir, output_dir

    def read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_repeated_output_is_deterministic_with_same_rng_seed(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_one = self.write_input(root)
            output_two = root / "output_two"
            runner.run(input_dir, output_one)
            runner.run(input_dir, output_two)

            self.assertEqual((output_one / "predictions.jsonl").read_text(encoding="utf-8"), (output_two / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual((output_one / "metadata.json").read_text(encoding="utf-8"), (output_two / "metadata.json").read_text(encoding="utf-8"))

    def test_different_rng_seed_changes_sampled_seed_metadata(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_one = self.write_input(root, rng_seed=1)
            runner.run(input_dir, output_one)
            (input_dir / "config.yaml").write_text(
                json.dumps(
                    {
                        "num_trials": 3,
                        "rng_seed": 2,
                        "max_hops": 2,
                        "neighbor_budget": 1,
                        "candidate_budget": 3,
                        "final_top_k": 2,
                        "distance_penalty": 0.10,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            output_two = root / "output_two"
            runner.run(input_dir, output_two)

            audit_one = self.read_jsonl(output_one / "predictions.jsonl")[0]["random_seed_graph_traversal"]
            audit_two = self.read_jsonl(output_two / "predictions.jsonl")[0]["random_seed_graph_traversal"]

            self.assertNotEqual(audit_one["sampled_seed_doc_ids"], audit_two["sampled_seed_doc_ids"])

    def test_trial_metadata_includes_rng_config_and_no_llm_provider(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root)
            runner.run(input_dir, output_dir)

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            audit = self.read_jsonl(output_dir / "predictions.jsonl")[0]["random_seed_graph_traversal"]

            self.assertEqual(metadata["experiment_id"], "E010")
            self.assertEqual(metadata["rng_seed"], 20260504)
            self.assertFalse(metadata["uses_llm"])
            self.assertEqual(metadata["provider"], "none")
            self.assertEqual(metadata["llm_provider"], "none")
            self.assertEqual(audit["rng_seed"], 20260504)
            self.assertEqual(audit["num_trials"], 3)
            self.assertIn("question_rng_seed", audit)
            self.assertIn("trials", audit)

    def test_traversal_respects_hop_candidate_and_final_budgets(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root)
            runner.run(input_dir, output_dir)

            audit = self.read_jsonl(output_dir / "predictions.jsonl")[0]["random_seed_graph_traversal"]

            self.assertLessEqual(len(self.read_jsonl(output_dir / "predictions.jsonl")[0]["retrieved_context"]), 2)
            for trial in audit["trials"]:
                self.assertLessEqual(trial["candidate_count"], 3)
                self.assertLessEqual(len(trial["final_doc_ids"]), 2)
                self.assertTrue(all(distance <= 2 for distance in trial["graph_distances"].values()))


if __name__ == "__main__":
    unittest.main()
