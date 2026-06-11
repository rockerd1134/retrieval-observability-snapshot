from __future__ import annotations

import importlib.util
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import unittest


def load_runner():
    path = Path("tests/rag_directed_spectral_rerank/runner.py")
    spec = importlib.util.spec_from_file_location("rag_directed_spectral_rerank_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class DirectedSpectralRerankTests(unittest.TestCase):
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

    def test_runner_uses_directed_spectral_components(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir = root / "input"
            output_dir = root / "output"
            corpus_dir = input_dir / "corpus"
            corpus_dir.mkdir(parents=True)
            write_jsonl(input_dir / "questions.jsonl", [{"question_id": "q1", "question": "How do I install collections?"}])
            (input_dir / "config.yaml").write_text(
                json.dumps(
                    {
                        "seed_top_k": 1,
                        "final_top_k": 2,
                        "directed_neighbor_budget": 2,
                        "global_prior_top_k": 1,
                        "pagerank_weight": 0.2,
                        "authority_weight": 0.3,
                    }
                ),
                encoding="utf-8",
            )
            (input_dir / "graph_edges.csv").write_text(
                "source,target\ninstall,collections\ncollections,plugins\nplugins,collections\n",
                encoding="utf-8",
            )
            (corpus_dir / "install.md").write_text("# Install\nInstall collections with the command line.", encoding="utf-8")
            (corpus_dir / "collections.md").write_text("# Collections\nCollections package plugins and modules.", encoding="utf-8")
            (corpus_dir / "plugins.md").write_text("# Plugins\nPlugins extend collections.", encoding="utf-8")

            runner.run(input_dir, output_dir)

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["experiment_id"], "E008")
            row = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["question_id"], "q1")
            self.assertLessEqual(len(row["retrieved_context"]), 2)
            self.assertIn("directed_spectral", row)
            self.assertTrue(all("pagerank" in item and "authority" in item for item in row["retrieved_context"]))


if __name__ == "__main__":
    unittest.main()
