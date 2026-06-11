from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from retrieval_arena.errors import ValidationError
from retrieval_arena.hashing import sha256_json
from retrieval_arena.manifests import read_manifest, write_manifest
from retrieval_arena.snapshot_diff import compare_snapshot_bundles


CREATED_AT = "2026-05-25T00:00:00+00:00"


class SnapshotDiffTests(unittest.TestCase):
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

    def file_item(self, path: str, text: str, *, doc_id: str | None = None) -> dict[str, Any]:
        return {
            "path": path,
            "doc_id": doc_id or Path(path).with_suffix("").as_posix(),
            "size_bytes": len(text.encode("utf-8")),
            "sha256": sha256_json(text),
        }

    def corpus_manifest(self, snapshot_id: str, files: list[dict[str, Any]], *, created_at: str = CREATED_AT) -> dict[str, Any]:
        return {
            "schema_version": "retrieval_arena.snapshot_manifest.v1",
            "created_at": created_at,
            "manifest_type": "corpus_snapshot",
            "corpus_id": "toy_docs",
            "snapshot_id": snapshot_id,
            "extraction_version": "extract-v1",
            "parser_version": "parser-v1",
            "page_count": len(files),
            "file_count": len(files),
            "corpus_size_bytes": sum(item["size_bytes"] for item in files),
            "file_inventory": sorted(files, key=lambda item: item["path"]),
            "content_hash": sha256_json({"files": sorted(files, key=lambda item: item["path"])}),
        }

    def graph_manifest(
        self,
        snapshot_id: str,
        edges: list[tuple[str, str]],
        *,
        metrics: dict[str, Any] | None = None,
        created_at: str = CREATED_AT,
    ) -> dict[str, Any]:
        nodes = sorted({node for edge in edges for node in edge})
        edge_inventory = [{"source": source, "target": target} for source, target in sorted(edges)]
        manifest = {
            "schema_version": "retrieval_arena.snapshot_manifest.v1",
            "created_at": created_at,
            "manifest_type": "graph_snapshot",
            "corpus_id": "toy_docs",
            "snapshot_id": snapshot_id,
            "corpus_snapshot_id": snapshot_id,
            "graph_extraction_version": "graph-v1",
            "edge_inventory": edge_inventory,
            "graph_hash": sha256_json({"edges": edge_inventory}),
            "node_count": len(nodes),
            "edge_count": len(edge_inventory),
        }
        if metrics is not None:
            manifest["graph_metrics"] = metrics
        return manifest

    def support_manifest(self, snapshot_id: str, targets_by_question: dict[str, list[str]], *, created_at: str = CREATED_AT) -> dict[str, Any]:
        all_targets = sorted({target for targets in targets_by_question.values() for target in targets})
        supported = sorted(qid for qid, targets in targets_by_question.items() if targets)
        return {
            "schema_version": "retrieval_arena.snapshot_manifest.v1",
            "created_at": created_at,
            "manifest_type": "support_surface",
            "corpus_id": "toy_docs",
            "snapshot_id": snapshot_id,
            "corpus_snapshot_id": snapshot_id,
            "query_set_id": "toy-queries",
            "query_set_hash": "a" * 64,
            "support_audit_file_hash": "b" * 64,
            "supported_question_ids": supported,
            "support_targets_by_question": dict(sorted(targets_by_question.items())),
            "support_target_doc_ids": all_targets,
            "support_target_count": len(all_targets),
        }

    def write_bundle(
        self,
        root: Path,
        *,
        snapshot_id: str,
        files: list[dict[str, Any]],
        edges: list[tuple[str, str]] | None = None,
        metrics: dict[str, Any] | None = None,
        support: dict[str, list[str]] | None = None,
        created_at: str = CREATED_AT,
    ) -> Path:
        root.mkdir()
        write_manifest(root / "corpus_snapshot_manifest.json", self.corpus_manifest(snapshot_id, files, created_at=created_at))
        if edges is not None:
            write_manifest(root / "graph_snapshot_manifest.json", self.graph_manifest(snapshot_id, edges, metrics=metrics, created_at=created_at))
        if support is not None:
            write_manifest(root / "support_surface_manifest.json", self.support_manifest(snapshot_id, support, created_at=created_at))
        return root

    def base_files(self) -> list[dict[str, Any]]:
        return [self.file_item("guide/config.md", "Set cache_size."), self.file_item("install.md", "Use pip.")]

    def test_identical_snapshot_bundles_pass_with_zero_deltas(self):
        with self.workspace_tempdir() as tmp:
            before = self.write_bundle(
                tmp / "before",
                snapshot_id="s1",
                files=self.base_files(),
                edges=[("install", "guide/config")],
                metrics={"weak_components": 1},
                support={"q1": ["install"], "q2": ["guide/config"]},
            )
            after = self.write_bundle(
                tmp / "after",
                snapshot_id="s1",
                files=self.base_files(),
                edges=[("install", "guide/config")],
                metrics={"weak_components": 1},
                support={"q1": ["install"], "q2": ["guide/config"]},
            )

            report = compare_snapshot_bundles(before, after, created_at=CREATED_AT)

            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["total_delta_count"], 0)
            self.assertEqual(report["corpus_result"]["changed_files"], [])

    def test_added_corpus_file_is_reported_with_path_and_hash(self):
        with self.workspace_tempdir() as tmp:
            before = self.write_bundle(tmp / "before", snapshot_id="s1", files=self.base_files())
            new_file = self.file_item("usage.md", "Run the command.")
            after = self.write_bundle(tmp / "after", snapshot_id="s2", files=[*self.base_files(), new_file])

            report = compare_snapshot_bundles(before, after, created_at=CREATED_AT)

            self.assertFalse(report["passed"])
            self.assertEqual(report["corpus_result"]["added_files"], [new_file])
            self.assertEqual(report["corpus_result"]["page_count_delta"], 1)

    def test_removed_corpus_file_is_reported_with_path_and_hash(self):
        with self.workspace_tempdir() as tmp:
            removed = self.file_item("install.md", "Use pip.")
            before = self.write_bundle(tmp / "before", snapshot_id="s1", files=[self.file_item("guide/config.md", "Set cache_size."), removed])
            after = self.write_bundle(tmp / "after", snapshot_id="s2", files=[self.file_item("guide/config.md", "Set cache_size.")])

            report = compare_snapshot_bundles(before, after, created_at=CREATED_AT)

            self.assertEqual(report["corpus_result"]["removed_files"], [removed])
            self.assertEqual(report["summary"]["removed_file_count"], 1)

    def test_changed_corpus_file_reports_before_and_after_hashes(self):
        with self.workspace_tempdir() as tmp:
            before_item = self.file_item("install.md", "Use pip.")
            after_item = self.file_item("install.md", "Use uv.")
            before = self.write_bundle(tmp / "before", snapshot_id="s1", files=[before_item])
            after = self.write_bundle(tmp / "after", snapshot_id="s2", files=[after_item])

            report = compare_snapshot_bundles(before, after, created_at=CREATED_AT)
            changed = report["corpus_result"]["changed_files"][0]

            self.assertEqual(changed["path"], "install.md")
            self.assertEqual(changed["before_sha256"], before_item["sha256"])
            self.assertEqual(changed["after_sha256"], after_item["sha256"])

    def test_graph_edge_additions_and_removals_are_stably_ordered(self):
        with self.workspace_tempdir() as tmp:
            before = self.write_bundle(
                tmp / "before",
                snapshot_id="s1",
                files=self.base_files(),
                edges=[("a", "b"), ("c", "d")],
            )
            after = self.write_bundle(
                tmp / "after",
                snapshot_id="s2",
                files=self.base_files(),
                edges=[("a", "b"), ("a", "c")],
            )

            report = compare_snapshot_bundles(before, after, created_at=CREATED_AT)

            self.assertEqual(report["graph_result"]["added_edges"], [{"source": "a", "target": "c", "edge_id": "a->c"}])
            self.assertEqual(report["graph_result"]["removed_edges"], [{"source": "c", "target": "d", "edge_id": "c->d"}])

    def test_graph_descriptor_field_changes_are_reported(self):
        with self.workspace_tempdir() as tmp:
            before = self.write_bundle(
                tmp / "before",
                snapshot_id="s1",
                files=self.base_files(),
                edges=[("a", "b")],
                metrics={"weak_components": 1, "largest_component_size": 2},
            )
            after = self.write_bundle(
                tmp / "after",
                snapshot_id="s2",
                files=self.base_files(),
                edges=[("a", "b")],
                metrics={"weak_components": 2, "largest_component_size": 1},
            )

            report = compare_snapshot_bundles(before, after, created_at=CREATED_AT)
            fields = [item["field"] for item in report["graph_result"]["descriptor_deltas"]]

            self.assertEqual(fields, ["graph_metrics.largest_component_size", "graph_metrics.weak_components"])

    def test_support_surface_question_and_target_changes_are_reported(self):
        with self.workspace_tempdir() as tmp:
            before = self.write_bundle(
                tmp / "before",
                snapshot_id="s1",
                files=self.base_files(),
                support={"q1": ["install"], "q2": ["guide/config"], "q3": ["install"]},
            )
            after = self.write_bundle(
                tmp / "after",
                snapshot_id="s2",
                files=self.base_files(),
                support={"q1": ["install", "guide/config"], "q2": [], "q4": ["install"]},
            )

            report = compare_snapshot_bundles(before, after, created_at=CREATED_AT)

            self.assertEqual([item["question_id"] for item in report["support_surface_result"]["added_questions"]], ["q4"])
            self.assertEqual([item["question_id"] for item in report["support_surface_result"]["removed_questions"]], ["q3"])
            self.assertEqual(report["support_surface_result"]["changed_questions"][0]["question_id"], "q1")
            self.assertEqual(report["support_surface_result"]["changed_questions"][0]["added_targets"], ["guide/config"])

    def test_missing_required_corpus_manifest_fails_clearly(self):
        with self.workspace_tempdir() as tmp:
            before = tmp / "before"
            after = self.write_bundle(tmp / "after", snapshot_id="s2", files=self.base_files())
            before.mkdir()

            with self.assertRaisesRegex(ValidationError, "Missing required corpus snapshot manifest"):
                compare_snapshot_bundles(before, after, created_at=CREATED_AT)

    def test_missing_optional_manifests_are_reported_as_unavailable(self):
        with self.workspace_tempdir() as tmp:
            before = self.write_bundle(tmp / "before", snapshot_id="s1", files=self.base_files())
            after = self.write_bundle(tmp / "after", snapshot_id="s2", files=self.base_files())

            report = compare_snapshot_bundles(before, after, created_at=CREATED_AT)

            self.assertTrue(report["passed"])
            self.assertFalse(report["graph_result"]["available"])
            self.assertFalse(report["support_surface_result"]["available"])
            self.assertEqual(report["summary"]["unavailable_optional_manifests"], ["graph", "support_surface"])

    def test_manifest_only_timestamp_differences_are_categorized_separately(self):
        with self.workspace_tempdir() as tmp:
            before = self.write_bundle(tmp / "before", snapshot_id="s1", files=self.base_files())
            after = self.write_bundle(tmp / "after", snapshot_id="s1", files=self.base_files(), created_at="2026-05-25T01:00:00+00:00")

            report = compare_snapshot_bundles(before, after, created_at=CREATED_AT)

            self.assertTrue(report["passed"])
            self.assertTrue(report["manifest_results"][0]["only_volatile_differences"])
            self.assertEqual(report["summary"]["stable_manifest_difference_count"], 0)

    def test_report_json_formatting_and_stable_ordering(self):
        with self.workspace_tempdir() as tmp:
            before = self.write_bundle(tmp / "before", snapshot_id="s1", files=[self.file_item("b.md", "b")])
            after = self.write_bundle(
                tmp / "after",
                snapshot_id="s2",
                files=[self.file_item("a.md", "a"), self.file_item("b.md", "changed"), self.file_item("c.md", "c")],
            )
            out = tmp / "snapshot_diff.json"
            markdown_out = tmp / "snapshot_diff.md"

            report = compare_snapshot_bundles(before, after, created_at=CREATED_AT, out_path=out, markdown_out_path=markdown_out)
            written = json.loads(out.read_text(encoding="utf-8"))

            self.assertTrue(out.read_text(encoding="utf-8").endswith("\n"))
            self.assertEqual(written["schema_version"], report["schema_version"])
            self.assertEqual([item["path"] for item in report["corpus_result"]["added_files"]], ["a.md", "c.md"])
            self.assertIn("# Snapshot Diff", markdown_out.read_text(encoding="utf-8"))

    def test_cli_smoke_test_for_snapshot_diff(self):
        with self.workspace_tempdir() as tmp:
            before = self.write_bundle(tmp / "before", snapshot_id="s1", files=self.base_files())
            after = self.write_bundle(tmp / "after", snapshot_id="s1", files=self.base_files())
            out = tmp / "snapshot_diff.json"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "retrieval_arena.cli",
                    "snapshot",
                    "diff",
                    "--before",
                    str(before),
                    "--after",
                    str(after),
                    "--out",
                    str(out),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Snapshot diff PASSED", result.stdout)
            self.assertEqual(read_manifest(before / "corpus_snapshot_manifest.json")["manifest_type"], "corpus_snapshot")
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
