from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

from retrieval_arena.config import load_config
from retrieval_arena.harness import run_experiment
from retrieval_arena.manifests import compute_manifest_hash, read_manifest, write_manifest
from retrieval_arena.replay_manifests import build_experiment_manifest, build_run_replay_manifest
from retrieval_arena.snapshots import (
    build_corpus_snapshot_manifest,
    build_graph_snapshot_manifest,
    build_support_surface_manifest,
)


CREATED_AT = "2026-05-25T00:00:00+00:00"


class ReplayManifestTests(unittest.TestCase):
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
        (corpus / "install.md").write_text("# Install\nUse pip.\n", encoding="utf-8")
        (dataset / "questions.jsonl").write_text('{"question_id":"q1","question":"How install?"}\n', encoding="utf-8")
        (dataset / "answers.jsonl").write_text('{"question_id":"q1","answer":"Use pip."}\n', encoding="utf-8")
        (dataset / "graph_edges.csv").write_text("source,target\ninstall,install\n", encoding="utf-8")
        (dataset / "faq_support_audit.jsonl").write_text('{"question_id":"q1","top_docs":["install"]}\n', encoding="utf-8")
        return dataset

    def write_snapshot_manifests(self, dataset: Path, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir()
        corpus = write_manifest(
            out_dir / "corpus_snapshot_manifest.json",
            build_corpus_snapshot_manifest(
                dataset,
                corpus_id="toy",
                snapshot_id="corpus-v1",
                extraction_version="extract-v1",
                parser_version="parser-v1",
                created_at=CREATED_AT,
            ),
        )
        self.assertEqual(corpus["snapshot_id"], "corpus-v1")
        write_manifest(
            out_dir / "graph_snapshot_manifest.json",
            build_graph_snapshot_manifest(
                dataset,
                corpus_id="toy",
                snapshot_id="graph-v1",
                corpus_snapshot_id="corpus-v1",
                graph_extraction_version="graph-v1",
                created_at=CREATED_AT,
            ),
        )
        write_manifest(
            out_dir / "support_surface_manifest.json",
            build_support_surface_manifest(
                dataset,
                corpus_id="toy",
                snapshot_id="support-v1",
                corpus_snapshot_id="corpus-v1",
                query_set_id="toy-queries",
                created_at=CREATED_AT,
            ),
        )
        return {
            "corpus": out_dir / "corpus_snapshot_manifest.json",
            "graph": out_dir / "graph_snapshot_manifest.json",
            "support": out_dir / "support_surface_manifest.json",
        }

    def write_config(self, root: Path, dataset: Path, manifests: dict[str, Path]) -> Path:
        config = root / "experiment.yaml"
        config.write_text(
            "\n".join(
                [
                    "experiment_name: toy_replay",
                    "datasets:",
                    "  - name: toy",
                    f"    path: {dataset}",
                    "    query_set_id: toy-queries",
                    f"    corpus_snapshot_manifest: {manifests['corpus']}",
                    f"    graph_snapshot_manifest: {manifests['graph']}",
                    f"    support_surface_manifest: {manifests['support']}",
                    "tests:",
                    "  - name: oracle",
                    "    image: toy-image",
                    "scoring:",
                    "  method: lexical_baseline",
                    "  match_threshold: 0.5",
                    f"output_dir: {root / 'results'}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return config

    def populate_run_dir(self, run_dir: Path, dataset: Path) -> None:
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True)
        shutil.copytree(dataset / "corpus", input_dir / "corpus")
        for filename in ["questions.jsonl", "answers.jsonl", "graph_edges.csv", "faq_support_audit.jsonl"]:
            shutil.copy2(dataset / filename, input_dir / filename)
        (input_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
        (run_dir / "predictions.jsonl").write_text(
            '{"question_id":"q1","question":"How install?","generated_answer":"Use pip.","retrieved_context":[{"doc_id":"install","score":1.0}]}\n',
            encoding="utf-8",
        )
        (run_dir / "metadata.json").write_text('{"name":"toy","version":"1","deterministic":true}\n', encoding="utf-8")
        (run_dir / "item_scores.jsonl").write_text('{"question_id":"q1","f1":1.0}\n', encoding="utf-8")
        (run_dir / "scores.json").write_text('{"mean_f1":1.0}\n', encoding="utf-8")

    def test_run_manifest_hashes_config_query_inputs_outputs_scoring_and_git(self):
        with self.workspace_tempdir() as tmp:
            dataset = self.toy_dataset(tmp)
            manifests = self.write_snapshot_manifests(dataset, tmp / "manifests")
            config = load_config(self.write_config(tmp, dataset, manifests))
            run_dir = tmp / "run"
            self.populate_run_dir(run_dir, dataset)

            manifest = build_run_replay_manifest(
                config=config,
                dataset=config.datasets[0],
                test=config.tests[0],
                run_id="run123",
                run_dir=run_dir,
                metadata={"name": "toy", "version": "1", "deterministic": True},
                run_started_at=CREATED_AT,
                run_completed_at=CREATED_AT,
                created_at=CREATED_AT,
            )
            written = write_manifest(run_dir / "retrieval_replay_manifest.json", manifest)

            self.assertEqual(written["manifest_type"], "retrieval_replay")
            self.assertEqual(written["query_set_id"], "toy-queries")
            self.assertEqual(written["corpus_snapshot_id"], "corpus-v1")
            self.assertEqual(written["graph_snapshot_id"], "graph-v1")
            self.assertEqual(written["support_surface_id"], "support-v1")
            self.assertIn("questions.jsonl", written["input_artifact_hashes"])
            self.assertIn("predictions.jsonl", written["output_artifact_hashes"])
            self.assertRegex(written["retrieval_config_hash"], r"^[0-9a-f]{64}$")
            self.assertRegex(written["scoring_hash"], r"^[0-9a-f]{64}$")
            self.assertIn("is_git_worktree", written["retrieval_arena_git_provenance"])
            self.assertEqual(written["manifest_hash"], compute_manifest_hash(written))

    def test_experiment_manifest_indexes_run_manifests(self):
        with self.workspace_tempdir() as tmp:
            dataset = self.toy_dataset(tmp)
            manifests = self.write_snapshot_manifests(dataset, tmp / "manifests")
            config = load_config(self.write_config(tmp, dataset, manifests))
            experiment_dir = tmp / "experiment"
            run_dir = experiment_dir / "runs" / "toy__oracle__run123"
            self.populate_run_dir(run_dir, dataset)
            run_manifest = write_manifest(
                run_dir / "retrieval_replay_manifest.json",
                build_run_replay_manifest(
                    config=config,
                    dataset=config.datasets[0],
                    test=config.tests[0],
                    run_id="run123",
                    run_dir=run_dir,
                    metadata={"name": "toy", "version": "1", "deterministic": True},
                    run_started_at=CREATED_AT,
                    run_completed_at=CREATED_AT,
                    created_at=CREATED_AT,
                ),
            )

            experiment = build_experiment_manifest(
                config=config,
                experiment_dir=experiment_dir,
                run_manifests=[run_dir / "retrieval_replay_manifest.json"],
                created_at=CREATED_AT,
            )

            self.assertEqual(experiment["run_manifest_count"], 1)
            self.assertEqual(experiment["run_manifests"][0]["manifest_hash"], run_manifest["manifest_hash"])
            self.assertEqual(experiment["run_manifests"][0]["path"], "runs/toy__oracle__run123/retrieval_replay_manifest.json")

    def test_harness_writes_run_and_experiment_manifests_without_changing_summary(self):
        with self.workspace_tempdir() as tmp:
            dataset = self.toy_dataset(tmp)
            manifests = self.write_snapshot_manifests(dataset, tmp / "manifests")
            config = load_config(self.write_config(tmp, dataset, manifests))

            def fake_run_container(test, input_dir, output_dir):
                output_dir.mkdir(parents=True)
                (output_dir / "metadata.json").write_text(
                    json.dumps({"name": "toy", "version": "1", "deterministic": True}) + "\n",
                    encoding="utf-8",
                )
                (output_dir / "predictions.jsonl").write_text(
                    '{"question_id":"q1","question":"How install?","generated_answer":"Use pip.","retrieved_context":[{"doc_id":"install","score":1.0}]}\n',
                    encoding="utf-8",
                )

            with patch("retrieval_arena.harness.build_image"), patch("retrieval_arena.harness.run_container", fake_run_container):
                rows = run_experiment(config)

            self.assertEqual(len(rows), 1)
            experiment_dir = config.output_dir / "toy_replay"
            run_manifest_path = next((experiment_dir / "runs").glob("*/retrieval_replay_manifest.json"))
            self.assertEqual(read_manifest(run_manifest_path)["manifest_type"], "retrieval_replay")
            self.assertEqual(read_manifest(experiment_dir / "experiment_manifest.json")["run_manifest_count"], 1)
            self.assertTrue((experiment_dir / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
