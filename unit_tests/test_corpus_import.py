from __future__ import annotations

import shutil
import subprocess
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from retrieval_arena.corpus.dataset_export import build_graph_transformation, build_support_surface, prepare_dataset
from retrieval_arena.corpus.importers import import_corpus_source
from retrieval_arena.corpus.sources import CorpusSourceDescriptor
from retrieval_arena.errors import ValidationError
from retrieval_arena.manifests import read_manifest
from studies.review2026.toy_study import run_toy_study


CREATED_AT = "2026-05-25T00:00:00+00:00"


class CorpusImportTests(unittest.TestCase):
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

    def write_query_files(self, root: Path) -> tuple[Path, Path]:
        questions = root / "questions.jsonl"
        answers = root / "answers.jsonl"
        questions.write_text('{"question_id":"q1","question":"How install?"}\n', encoding="utf-8")
        answers.write_text('{"question_id":"q1","answer":"Use pip."}\n', encoding="utf-8")
        return questions, answers

    def test_local_source_import_selects_files_and_doc_ids_deterministically(self):
        with self.workspace_tempdir() as tmp:
            source = tmp / "source" / "docs"
            source.mkdir(parents=True)
            (source / "install.md").write_text("Use pip.\n", encoding="utf-8")
            (source / "draft.tmp").write_text("skip\n", encoding="utf-8")
            (source / "guide").mkdir()
            (source / "guide" / "config.md").write_text("Set cache.\n", encoding="utf-8")
            descriptor = CorpusSourceDescriptor(
                corpus_id="toy",
                snapshot_id="s1",
                source_type="local",
                source_path=source,
                destination_workspace=tmp / "workspace",
                include=("*.md", "**/*.md"),
                exclude=("draft.*",),
            )

            result = import_corpus_source(descriptor, created_at=CREATED_AT)
            manifest = read_manifest(result["manifest_path"])

            self.assertEqual([item["path"] for item in manifest["file_inventory"]], ["guide/config.md", "install.md"])
            self.assertEqual([item["doc_id"] for item in manifest["file_inventory"]], ["guide/config", "install"])
            self.assertEqual([item["path"] for item in manifest["ignored_files"]], ["draft.tmp"])
            self.assertEqual(manifest["source_provenance"]["source_kind"], "local_directory")
            self.assertRegex(manifest["source_provenance"]["source_hash"], r"^[0-9a-f]{64}$")
            self.assertRegex(manifest["import_config_hash"], r"^[0-9a-f]{64}$")
            self.assertRegex(manifest["source_descriptor_hash"], r"^[0-9a-f]{64}$")
            self.assertRegex(manifest["snapshot_identity_hash"], r"^[0-9a-f]{64}$")
            self.assertEqual(manifest["parser_version"], "retrieval_arena.corpus_import.copy-v1")
            self.assertTrue((result["documents_path"] / "guide" / "config.md").exists())
            self.assertTrue((result["manifest_path"]).read_text(encoding="utf-8").endswith("\n"))

    def test_local_source_import_records_requested_snapshot_revision(self):
        with self.workspace_tempdir() as tmp:
            source = tmp / "source"
            source.mkdir()
            (source / "install.md").write_text("Use pip.\n", encoding="utf-8")
            descriptor = CorpusSourceDescriptor(
                corpus_id="toy",
                snapshot_id="s1",
                source_type="local",
                source_url="https://example.com/docs",
                source_path=source,
                requested_ref="upstream-commit-1",
                destination_workspace=tmp / "workspace",
                include=("*.md",),
            )

            result = import_corpus_source(descriptor, created_at=CREATED_AT)
            manifest = read_manifest(result["manifest_path"])

            self.assertEqual(manifest["requested_ref"], "upstream-commit-1")
            self.assertEqual(manifest["source_url"], "https://example.com/docs")
            self.assertEqual(manifest["source_provenance"]["requested_ref"], "upstream-commit-1")
            self.assertEqual(manifest["source_provenance"]["resolved_revision"], "upstream-commit-1")
            self.assertEqual(manifest["source_provenance"]["source_url"], "https://example.com/docs")
            self.assertRegex(manifest["snapshot_identity_hash"], r"^[0-9a-f]{64}$")

    def test_git_source_import_uses_explicit_commits_without_mutating_checkout(self):
        with self.workspace_tempdir() as tmp:
            repo = tmp / "repo"
            repo.mkdir()
            self.git(repo, "init")
            self.git(repo, "config", "user.email", "toy.invalid")
            self.git(repo, "config", "user.name", "Toy User")
            self.git(repo, "config", "commit.gpgsign", "false")
            self.git(repo, "config", "tag.gpgSign", "false")
            (repo / "docs").mkdir()
            (repo / "docs" / "install.md").write_text("Use pip.\n", encoding="utf-8")
            (repo / "outside.md").write_text("Not documentation.\n", encoding="utf-8")
            self.git(repo, "add", "docs/install.md", "outside.md")
            self.git(repo, "commit", "-m", "first")
            first = self.git(repo, "rev-parse", "HEAD").stdout.strip()
            (repo / "docs" / "install.md").write_text("Use uv.\n", encoding="utf-8")
            self.git(repo, "commit", "-am", "second")
            second = self.git(repo, "rev-parse", "HEAD").stdout.strip()

            before = import_corpus_source(
                CorpusSourceDescriptor(
                    corpus_id="toy",
                    snapshot_id="before",
                    source_type="git",
                    source_path=repo,
                    requested_ref=first,
                    docs_root="docs",
                    destination_workspace=tmp / "workspace",
                    include=("*.md",),
                ),
                created_at=CREATED_AT,
            )
            after = import_corpus_source(
                CorpusSourceDescriptor(
                    corpus_id="toy",
                    snapshot_id="after",
                    source_type="git",
                    source_path=repo,
                    requested_ref=second,
                    docs_root="docs",
                    destination_workspace=tmp / "workspace",
                    include=("*.md",),
                ),
                created_at=CREATED_AT,
            )

            self.assertEqual(self.git(repo, "rev-parse", "HEAD").stdout.strip(), second)
            self.assertEqual((before["documents_path"] / "install.md").read_text(encoding="utf-8"), "Use pip.\n")
            self.assertEqual((after["documents_path"] / "install.md").read_text(encoding="utf-8"), "Use uv.\n")
            self.assertFalse((before["manifest_path"].parent / "source" / "outside.md").exists())
            self.assertEqual(read_manifest(before["manifest_path"])["source_provenance"]["resolved_commit"], first)

    def test_git_source_import_can_reuse_clean_existing_snapshot(self):
        with self.workspace_tempdir() as tmp:
            repo, commit = self.git_repo_with_two_commits(tmp)
            descriptor = CorpusSourceDescriptor(
                corpus_id="toy",
                snapshot_id="before",
                source_type="git",
                source_path=repo,
                requested_ref=commit,
                docs_root="docs",
                destination_workspace=tmp / "workspace",
                include=("*.md",),
            )
            first = import_corpus_source(descriptor, created_at=CREATED_AT)
            sentinel = tmp / "workspace" / "toy" / "before" / "source" / "sentinel.txt"
            sentinel.write_text("kept on reuse\n", encoding="utf-8")

            second = import_corpus_source(descriptor, created_at=CREATED_AT, overwrite=False)

            self.assertTrue(second["reused"])
            self.assertTrue(sentinel.exists())
            self.assertEqual(first["manifest"]["manifest_hash"], second["manifest"]["manifest_hash"])

    def test_git_source_import_refuses_reuse_when_source_repo_is_dirty(self):
        with self.workspace_tempdir() as tmp:
            repo, commit = self.git_repo_with_two_commits(tmp)
            descriptor = CorpusSourceDescriptor(
                corpus_id="toy",
                snapshot_id="before",
                source_type="git",
                source_path=repo,
                requested_ref=commit,
                docs_root="docs",
                destination_workspace=tmp / "workspace",
                include=("*.md",),
            )
            import_corpus_source(descriptor, created_at=CREATED_AT)
            (repo / "untracked.md").write_text("dirty\n", encoding="utf-8")

            with self.assertRaisesRegex(ValidationError, "must be clean"):
                import_corpus_source(descriptor, created_at=CREATED_AT, overwrite=False)

    def test_import_rebuilds_incomplete_workspace_when_reuse_requested(self):
        with self.workspace_tempdir() as tmp:
            source = tmp / "source"
            source.mkdir()
            (source / "install.md").write_text("Use pip.\n", encoding="utf-8")
            workspace = tmp / "workspace"
            incomplete = workspace / "toy" / "s1"
            incomplete.mkdir(parents=True)
            (incomplete / "partial.txt").write_text("interrupted\n", encoding="utf-8")
            descriptor = CorpusSourceDescriptor(
                corpus_id="toy",
                snapshot_id="s1",
                source_type="local",
                source_path=source,
                destination_workspace=workspace,
                include=("*.md",),
            )

            result = import_corpus_source(descriptor, created_at=CREATED_AT, overwrite=False)

            self.assertFalse(result["reused"])
            self.assertFalse((incomplete / "partial.txt").exists())
            self.assertTrue((result["documents_path"] / "install.md").exists())

    def test_dataset_preparation_and_conditional_hooks_write_manifests(self):
        with self.workspace_tempdir() as tmp:
            source = tmp / "source"
            source.mkdir()
            (source / "install.md").write_text("Use pip.\n", encoding="utf-8")
            imported = import_corpus_source(
                CorpusSourceDescriptor(
                    corpus_id="toy",
                    snapshot_id="s1",
                    source_type="local",
                    source_path=source,
                    destination_workspace=tmp / "workspace",
                    include=("*.md",),
                ),
                created_at=CREATED_AT,
            )
            questions, answers = self.write_query_files(tmp)
            prepared = prepare_dataset(imported["manifest_path"], tmp / "dataset", questions_path=questions, answers_path=answers, query_set_id="toy-queries", created_at=CREATED_AT)

            preparation_manifest = read_manifest(prepared["manifest_path"])
            self.assertEqual(preparation_manifest["validation_status"], "ok")
            self.assertEqual(preparation_manifest["input_snapshot_identity_hash"], imported["manifest"]["snapshot_identity_hash"])
            self.assertEqual(preparation_manifest["generated_document_ids"], ["install"])
            self.assertRegex(preparation_manifest["output_dataset_identity_hash"], r"^[0-9a-f]{64}$")
            self.assertEqual(preparation_manifest["chunking_config"]["method"], "one_file_per_document")
            self.assertIsNone(build_graph_transformation(prepared["dataset_path"], prepared["manifest_path"], created_at=CREATED_AT))
            self.assertIsNone(build_support_surface(prepared["dataset_path"], prepared["manifest_path"], created_at=CREATED_AT))

            graph_edges = tmp / "graph_edges.csv"
            graph_edges.write_text("source,target\ninstall,install\n", encoding="utf-8")
            support = tmp / "support.jsonl"
            support.write_text('{"question_id":"q1","top_docs":[{"doc_id":"install"}]}\n', encoding="utf-8")
            graph = build_graph_transformation(prepared["dataset_path"], prepared["manifest_path"], graph_edges_path=graph_edges, created_at=CREATED_AT)
            support_result = build_support_surface(prepared["dataset_path"], prepared["manifest_path"], support_records_path=support, query_set_id="toy-queries", created_at=CREATED_AT)

            self.assertEqual(read_manifest(graph["manifest_path"])["edge_count"], 1)
            self.assertEqual(read_manifest(graph["manifest_path"])["source_dataset_identity_hash"], preparation_manifest["output_dataset_identity_hash"])
            self.assertEqual(read_manifest(support_result["manifest_path"])["support_target_doc_ids"], ["install"])
            self.assertEqual(read_manifest(support_result["manifest_path"])["source_dataset_identity_hash"], preparation_manifest["output_dataset_identity_hash"])

    def test_review_toy_study_runs_source_to_snapshot_diff(self):
        with self.workspace_tempdir() as tmp:
            before_source = tmp / "before_source"
            after_source = tmp / "after_source"
            before_source.mkdir()
            after_source.mkdir()
            (before_source / "install.md").write_text("Use pip.\n", encoding="utf-8")
            (after_source / "install.md").write_text("Use uv.\n", encoding="utf-8")
            questions, answers = self.write_query_files(tmp)

            result = run_toy_study(
                CorpusSourceDescriptor("toy", "before", "local", tmp / "imports", source_path=before_source, include=("*.md",)),
                CorpusSourceDescriptor("toy", "after", "local", tmp / "imports", source_path=after_source, include=("*.md",)),
                tmp / "study",
                before_questions=questions,
                before_answers=answers,
                created_at=CREATED_AT,
            )

            self.assertEqual(read_manifest(result["manifest_path"])["manifest_type"], "review2026_toy_study")
            self.assertEqual(result["snapshot_diff"]["summary"]["changed_file_count"], 1)

    def git(self, repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(["git", "-C", str(repo), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def git_repo_with_two_commits(self, tmp: Path) -> tuple[Path, str]:
        repo = tmp / "repo"
        repo.mkdir()
        self.git(repo, "init")
        self.git(repo, "config", "user.email", "toy.invalid")
        self.git(repo, "config", "user.name", "Toy User")
        self.git(repo, "config", "commit.gpgsign", "false")
        self.git(repo, "config", "tag.gpgSign", "false")
        (repo / "docs").mkdir()
        (repo / "docs" / "install.md").write_text("Use pip.\n", encoding="utf-8")
        self.git(repo, "add", "docs/install.md")
        self.git(repo, "commit", "-m", "first")
        first = self.git(repo, "rev-parse", "HEAD").stdout.strip()
        (repo / "docs" / "install.md").write_text("Use uv.\n", encoding="utf-8")
        self.git(repo, "commit", "-am", "second")
        return repo, first


if __name__ == "__main__":
    unittest.main()
