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
    path = Path("tests/rag_semantic_graph_rerank/runner.py")
    spec = importlib.util.spec_from_file_location("rag_semantic_graph_rerank_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SemanticGraphRerankTests(unittest.TestCase):
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

    def write_input(self, root: Path) -> tuple[Path, Path]:
        input_dir = root / "input"
        output_dir = root / "output"
        corpus_dir = input_dir / "corpus"
        corpus_dir.mkdir(parents=True)
        (corpus_dir / "install.md").write_text("# Install\nInstall collections with the command line.\n", encoding="utf-8")
        (corpus_dir / "collections.md").write_text("# Collections\nCollections package plugins and modules for automation.\n", encoding="utf-8")
        (corpus_dir / "inventory.md").write_text("# Inventory\nInventory lists managed hosts and groups.\n", encoding="utf-8")
        (input_dir / "questions.jsonl").write_text(
            '{"question_id":"q1","question":"How do plugins work with collections?"}\n',
            encoding="utf-8",
        )
        (input_dir / "answers.jsonl").write_text('{"question_id":"q1","answer":"Collections package plugins."}\n', encoding="utf-8")
        (input_dir / "graph_edges.csv").write_text("source,target\ninstall,collections\ncollections,inventory\n", encoding="utf-8")
        (input_dir / "config.yaml").write_text(
            json.dumps(
                {
                    "seed_top_k": 1,
                    "graph_hops": 1,
                    "neighbor_budget": 2,
                    "candidate_budget": 4,
                    "final_top_k": 2,
                    "distance_penalty": 0.0,
                    "max_context_chars_per_doc": 200,
                    "embedding_backend": "local_hashing_tfidf",
                    "model_id": "local-hashing-tfidf-v1",
                    "embedding_dimensions": 128,
                    "variant_id": "E009b",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return input_dir, output_dir

    def read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_runner_produces_predictions_and_metadata(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root)
            runner.run(input_dir, output_dir)

            predictions = self.read_jsonl(output_dir / "predictions.jsonl")
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(predictions[0]["question_id"], "q1")
            self.assertLessEqual(len(predictions[0]["retrieved_context"]), 2)
            self.assertEqual(metadata["name"], "rag_semantic_graph_rerank")
            self.assertEqual(metadata["experiment_id"], "E009")
            self.assertEqual(metadata["variant_id"], "E009b")
            self.assertEqual(metadata["seed_ranker"], "minilm_embedding")
            self.assertEqual(metadata["embedding_backend"], "local_hashing_tfidf")
            self.assertFalse(metadata["uses_llm"])
            self.assertEqual(metadata["provider"], "none")

    def test_semantic_seeds_are_expanded_before_final_rerank(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root)
            runner.run(input_dir, output_dir)

            row = self.read_jsonl(output_dir / "predictions.jsonl")[0]
            audit = row["semantic_graph_rerank"]

            self.assertEqual(audit["seed_doc_ids"], ["collections"])
            self.assertIn("install", audit["expanded_candidate_doc_ids"])
            self.assertIn("install", audit["final_context_doc_ids"])
            self.assertEqual(audit["graph_distances"]["collections"], 0)
            self.assertEqual(audit["graph_distances"]["install"], 1)

    def test_graph_neighbors_can_enter_final_context_when_relevant(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root)
            (input_dir / "config.yaml").write_text(
                json.dumps(
                    {
                        "seed_top_k": 1,
                        "graph_hops": 1,
                        "neighbor_budget": 2,
                        "candidate_budget": 4,
                        "final_top_k": 2,
                        "distance_penalty": 0.0,
                        "embedding_backend": "local_hashing_tfidf",
                        "embedding_dimensions": 128,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            runner.run(input_dir, output_dir)

            context = self.read_jsonl(output_dir / "predictions.jsonl")[0]["retrieved_context"]
            roles_by_doc = {item["doc_id"]: item["selection_role"] for item in context}

            self.assertEqual(roles_by_doc.get("install"), "expanded_neighbor")

    def test_output_is_deterministic_across_repeated_runs(self):
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
                (output_one / "metadata.json").read_text(encoding="utf-8"),
                (output_two / "metadata.json").read_text(encoding="utf-8"),
            )

    def test_metadata_records_provenance_and_ranking_fields(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root)
            runner.run(input_dir, output_dir)

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            row = self.read_jsonl(output_dir / "predictions.jsonl")[0]
            audit = row["semantic_graph_rerank"]

            self.assertIn("model_id", metadata)
            self.assertIn("model_revision", metadata)
            self.assertIn("model_lock_path", metadata)
            self.assertIn("model_cache_path", metadata)
            self.assertEqual(audit["expansion_policy"], "semantic_seeded_graph_rerank_v1")
            self.assertEqual(audit["final_ranker"], "minilm_embedding_plus_graph_distance")
            self.assertIn("final_context_doc_ids", audit)
            self.assertIn("pagerank_prior_enabled", audit)
            self.assertEqual(audit["variant_id"], "E009b")


if __name__ == "__main__":
    unittest.main()
