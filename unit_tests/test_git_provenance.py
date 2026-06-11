from __future__ import annotations

import shutil
import subprocess
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from retrieval_arena.errors import RetrievalAuditError
from retrieval_arena.git_provenance import describe_git_provenance, plan_git_comparison, resolve_ref


class GitProvenanceTests(unittest.TestCase):
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

    def git(self, repo: Path, *args: str) -> str:
        result = subprocess.run(["git", "-C", str(repo), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            self.fail(result.stderr)
        return result.stdout.strip()

    def init_repo(self, repo: Path) -> str:
        self.git(repo, "init")
        self.git(repo, "config", "user.email", "unit.invalid")
        self.git(repo, "config", "user.name", "Unit Test")
        self.git(repo, "config", "commit.gpgsign", "false")
        self.git(repo, "config", "tag.gpgSign", "false")
        self.git(repo, "checkout", "-b", "main")
        (repo / "tracked.txt").write_text("one\n", encoding="utf-8")
        self.git(repo, "add", "tracked.txt")
        self.git(repo, "commit", "-m", "initial")
        self.git(repo, "tag", "v1")
        return self.git(repo, "rev-parse", "HEAD")

    def test_non_git_path_returns_explicit_empty_block(self):
        provenance = describe_git_provenance(Path("C:/"))

        self.assertFalse(provenance["is_git_worktree"])
        self.assertIsNone(provenance["repo_root"])
        self.assertIsNone(provenance["commit"])
        self.assertEqual(provenance["untracked_files"], [])

    def test_captures_branch_commit_dirty_diff_hash_and_untracked_files(self):
        with self.workspace_tempdir() as tmp:
            repo = tmp / "repo"
            repo.mkdir()
            commit = self.init_repo(repo)
            (repo / "tracked.txt").write_text("two\n", encoding="utf-8")
            (repo / "new.txt").write_text("new\n", encoding="utf-8")

            provenance = describe_git_provenance(repo)

            self.assertTrue(provenance["is_git_worktree"])
            self.assertEqual(provenance["repo_root"], str(repo.resolve()))
            self.assertEqual(provenance["branch"], "main")
            self.assertEqual(provenance["commit"], commit)
            self.assertEqual(provenance["tag"], "v1")
            self.assertTrue(provenance["dirty"])
            self.assertRegex(provenance["diff_hash"], r"^[0-9a-f]{64}$")
            self.assertEqual(provenance["untracked_files"], ["new.txt"])

    def test_resolves_branch_and_commit_refs_without_checkout(self):
        with self.workspace_tempdir() as tmp:
            repo = tmp / "repo"
            repo.mkdir()
            commit = self.init_repo(repo)

            self.assertEqual(resolve_ref(repo, "main"), commit)
            self.assertEqual(resolve_ref(repo, commit), commit)
            provenance = describe_git_provenance(repo, ref="main")
            self.assertEqual(provenance["ref"], "main")
            self.assertEqual(provenance["ref_commit"], commit)

    def test_unresolved_ref_fails_clearly(self):
        with self.workspace_tempdir() as tmp:
            repo = tmp / "repo"
            repo.mkdir()
            self.init_repo(repo)

            with self.assertRaisesRegex(RetrievalAuditError, "does not resolve"):
                resolve_ref(repo, "missing-ref")

    def test_compare_plan_is_dry_run_and_resolves_both_refs(self):
        with self.workspace_tempdir() as tmp:
            repo = tmp / "repo"
            repo.mkdir()
            left = self.init_repo(repo)
            self.git(repo, "checkout", "-b", "feature")
            (repo / "tracked.txt").write_text("feature\n", encoding="utf-8")
            self.git(repo, "add", "tracked.txt")
            self.git(repo, "commit", "-m", "feature")
            right = self.git(repo, "rev-parse", "HEAD")

            plan = plan_git_comparison(repo, left_ref="main", right_ref="feature", output_dir=tmp / "planned")

            self.assertEqual(plan["left_commit"], left)
            self.assertEqual(plan["right_commit"], right)
            self.assertFalse(plan["mutates_current_worktree"])
            self.assertEqual(plan["checkout_strategy"], "dry_run_only")


if __name__ == "__main__":
    unittest.main()
