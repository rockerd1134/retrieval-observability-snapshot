from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from retrieval_arena.cli import main
from retrieval_arena.errors import ValidationError
from retrieval_arena.hashing import sha256_file, sha256_jsonl
from retrieval_arena.manifests import read_manifest, write_manifest
from retrieval_arena.snapshots import (
    build_corpus_snapshot_manifest,
    build_graph_snapshot_manifest,
    build_support_surface_manifest,
)


CREATED_AT = "2026-05-25T00:00:00+00:00"


class SnapshotManifestTests(unittest.TestCase):
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

    def toy_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        corpus = dataset / "corpus"
        corpus.mkdir(parents=True)
        (corpus / "install.md").write_text("# Install\n\nUse pip.\n", encoding="utf-8")
        (corpus / "guide").mkdir()
        (corpus / "guide" / "config.md").write_text("# Config\n\nSet cache_size.\n", encoding="utf-8")
        (dataset / "questions.jsonl").write_text(
            '{"question_id":"q1","question":"How install?"}\n'
            '{"question_id":"q2","question":"How configure?"}\n',
            encoding="utf-8",
        )
        (dataset / "answers.jsonl").write_text(
            '{"question_id":"q1","answer":"Use pip."}\n'
            '{"question_id":"q2","answer":"Set cache_size."}\n',
            encoding="utf-8",
        )
        (dataset / "graph_edges.csv").write_text("source,target\ninstall,guide/config\n", encoding="utf-8")
        (dataset / "faq_support_audit.jsonl").write_text(
            '{"question_id":"q1","top_docs":[{"doc_id":"install","score":1.0}]}\n'
            '{"question_id":"q2","top_docs":[{"doc_id":"guide/config"},{"doc_id":"install"}]}\n',
            encoding="utf-8",
        )
        return dataset

    def test_corpus_snapshot_manifest_inventory_counts_and_hashes_are_stable(self):
        with self.workspace_tempdir() as tmp:
            dataset = self.toy_dataset(tmp)

            first = build_corpus_snapshot_manifest(
                dataset,
                corpus_id="toy_docs",
                snapshot_id="2026-05-25",
                extraction_version="extract-v1",
                parser_version="parser-v1",
                created_at=CREATED_AT,
                source_name="Toy Docs",
            )
            second = build_corpus_snapshot_manifest(
                dataset,
                corpus_id="toy_docs",
                snapshot_id="2026-05-25",
                extraction_version="extract-v1",
                parser_version="parser-v1",
                created_at=CREATED_AT,
                source_name="Toy Docs",
            )

            self.assertEqual(first, second)
            self.assertEqual(first["manifest_type"], "corpus_snapshot")
            self.assertEqual(first["page_count"], 2)
            self.assertEqual(first["file_count"], 2)
            self.assertEqual([item["path"] for item in first["file_inventory"]], ["guide/config.md", "install.md"])
            self.assertEqual([item["doc_id"] for item in first["file_inventory"]], ["guide/config", "install"])
            self.assertEqual(first["corpus_size_bytes"], 52)
            self.assertEqual(first["source_name"], "Toy Docs")

            manifest_path = tmp / "corpus_snapshot_manifest.json"
            written = write_manifest(manifest_path, first)
            self.assertEqual(read_manifest(manifest_path), written)
            self.assertEqual(read_manifest(manifest_path)["content_hash"], first["content_hash"])

    def test_graph_snapshot_manifest_counts_edges_and_optional_metrics(self):
        with self.workspace_tempdir() as tmp:
            dataset = self.toy_dataset(tmp)
            (dataset / "graph_metrics.json").write_text('{"weak_components":1,"largest_component_size":2}\n', encoding="utf-8")

            manifest = build_graph_snapshot_manifest(
                dataset,
                corpus_id="toy_docs",
                snapshot_id="graph-2026-05-25",
                corpus_snapshot_id="2026-05-25",
                graph_extraction_version="graph-v1",
                created_at=CREATED_AT,
            )

            self.assertEqual(manifest["manifest_type"], "graph_snapshot")
            self.assertEqual(manifest["edge_file"], "graph_edges.csv")
            self.assertEqual(manifest["edge_file_hash"], sha256_file(dataset / "graph_edges.csv"))
            self.assertEqual(manifest["node_count"], 2)
            self.assertEqual(manifest["edge_count"], 1)
            self.assertEqual(manifest["graph_metrics_file_hash"], sha256_file(dataset / "graph_metrics.json"))
            self.assertEqual(manifest["graph_metrics"]["weak_components"], 1)

    def test_support_surface_manifest_extracts_targets_and_hashes_artifacts(self):
        with self.workspace_tempdir() as tmp:
            dataset = self.toy_dataset(tmp)

            manifest = build_support_surface_manifest(
                dataset,
                corpus_id="toy_docs",
                snapshot_id="support-2026-05-25",
                corpus_snapshot_id="2026-05-25",
                query_set_id="toy-faq-v1",
                created_at=CREATED_AT,
            )

            self.assertEqual(manifest["manifest_type"], "support_surface")
            self.assertEqual(manifest["query_set_hash"], sha256_jsonl(dataset / "questions.jsonl"))
            self.assertEqual(manifest["support_audit_file_hash"], sha256_jsonl(dataset / "faq_support_audit.jsonl"))
            self.assertEqual(manifest["supported_question_ids"], ["q1", "q2"])
            self.assertEqual(manifest["support_target_doc_ids"], ["guide/config", "install"])
            self.assertEqual(manifest["support_target_count"], 2)
            self.assertEqual(manifest["support_targets_by_question"]["q2"], ["guide/config", "install"])

    def test_corpus_manifest_rejects_missing_dataset_corpus(self):
        with self.workspace_tempdir() as tmp:
            dataset = self.toy_dataset(tmp)
            shutil.rmtree(dataset / "corpus")

            with self.assertRaisesRegex(ValidationError, "required path"):
                build_corpus_snapshot_manifest(
                    dataset,
                    corpus_id="toy_docs",
                    snapshot_id="2026-05-25",
                    extraction_version="extract-v1",
                    parser_version="parser-v1",
                    created_at=CREATED_AT,
                )

    def test_snapshot_builders_reject_missing_required_ids(self):
        with self.workspace_tempdir() as tmp:
            dataset = self.toy_dataset(tmp)

            with self.assertRaisesRegex(ValidationError, "snapshot_id"):
                build_corpus_snapshot_manifest(
                    dataset,
                    corpus_id="toy_docs",
                    snapshot_id="",
                    extraction_version="extract-v1",
                    parser_version="parser-v1",
                    created_at=CREATED_AT,
                )

            with self.assertRaisesRegex(ValidationError, "corpus_snapshot_id"):
                build_graph_snapshot_manifest(
                    dataset,
                    corpus_id="toy_docs",
                    snapshot_id="graph",
                    corpus_snapshot_id="",
                    graph_extraction_version="graph-v1",
                    created_at=CREATED_AT,
                )

    def test_graph_manifest_rejects_malformed_graph_file(self):
        with self.workspace_tempdir() as tmp:
            dataset = self.toy_dataset(tmp)
            (dataset / "graph_edges.csv").write_text("from,to\ninstall,guide/config\n", encoding="utf-8")

            with self.assertRaisesRegex(ValidationError, "source,target"):
                build_graph_snapshot_manifest(
                    dataset,
                    corpus_id="toy_docs",
                    snapshot_id="graph",
                    corpus_snapshot_id="corpus",
                    graph_extraction_version="graph-v1",
                    created_at=CREATED_AT,
                )

    def test_support_manifest_rejects_malformed_audit_rows(self):
        with self.workspace_tempdir() as tmp:
            dataset = self.toy_dataset(tmp)
            (dataset / "faq_support_audit.jsonl").write_text('{"question_id":"q1","top_docs":"install"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValidationError, "support docs must be a list"):
                build_support_surface_manifest(
                    dataset,
                    corpus_id="toy_docs",
                    snapshot_id="support",
                    corpus_snapshot_id="corpus",
                    query_set_id="queries",
                    created_at=CREATED_AT,
                )

    def test_cli_snapshot_manifest_writes_requested_manifests(self):
        with self.workspace_tempdir() as tmp:
            dataset = self.toy_dataset(tmp)
            out_dir = tmp / "manifests"

            exit_code = main(
                [
                    "snapshot",
                    "manifest",
                    "--dataset",
                    str(dataset),
                    "--out-dir",
                    str(out_dir),
                    "--corpus-id",
                    "toy_docs",
                    "--snapshot-id",
                    "2026-05-25",
                    "--query-set-id",
                    "toy-faq-v1",
                    "--extraction-version",
                    "extract-v1",
                    "--parser-version",
                    "parser-v1",
                    "--graph-extraction-version",
                    "graph-v1",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(read_manifest(out_dir / "corpus_snapshot_manifest.json")["manifest_type"], "corpus_snapshot")
            self.assertEqual(read_manifest(out_dir / "graph_snapshot_manifest.json")["manifest_type"], "graph_snapshot")
            self.assertEqual(read_manifest(out_dir / "support_surface_manifest.json")["manifest_type"], "support_surface")


if __name__ == "__main__":
    unittest.main()
