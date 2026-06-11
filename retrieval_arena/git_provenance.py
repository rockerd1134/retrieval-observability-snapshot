from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from .errors import RetrievalAuditError


GIT_PROVENANCE_FIELDS = (
    "is_git_worktree",
    "repo_root",
    "branch",
    "commit",
    "tag",
    "dirty",
    "diff_hash",
    "untracked_files",
    "remote_url",
    "ref",
    "ref_commit",
)


def _empty_provenance(*, ref: str | None = None) -> dict[str, Any]:
    return {
        "is_git_worktree": False,
        "repo_root": None,
        "branch": None,
        "commit": None,
        "tag": None,
        "dirty": None,
        "diff_hash": None,
        "untracked_files": [],
        "remote_url": None,
        "ref": ref,
        "ref_commit": None,
    }


def _run_git(path: Path, args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise RetrievalAuditError("Git executable is not available for provenance capture.") from exc
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise RetrievalAuditError(detail)
    return result


def _run_git_bytes(path: Path, args: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise RetrievalAuditError("Git executable is not available for provenance capture.") from exc


def is_git_worktree(path: Path) -> bool:
    return _run_git(path, ["rev-parse", "--is-inside-work-tree"]).stdout.strip() == "true"


def repository_root(path: Path) -> Path | None:
    result = _run_git(path, ["rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def resolve_ref(path: Path, ref: str) -> str:
    root = repository_root(path)
    if root is None:
        raise RetrievalAuditError(f"Cannot resolve Git ref outside a worktree: {ref}")
    result = _run_git(root, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RetrievalAuditError(f"Git ref does not resolve to a commit: {ref}. {detail}".strip())
    return result.stdout.strip()


def _optional_git_output(root: Path, args: list[str]) -> str | None:
    result = _run_git(root, args)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _branch_name(root: Path) -> str | None:
    branch = _optional_git_output(root, ["branch", "--show-current"])
    if branch:
        return branch
    abbrev = _optional_git_output(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if abbrev and abbrev != "HEAD":
        return abbrev
    return None


def _diff_hash(root: Path) -> str:
    result = _run_git_bytes(root, ["diff", "--no-ext-diff", "--binary", "HEAD", "--"])
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RetrievalAuditError(f"Unable to capture Git diff for provenance: {detail}")
    return hashlib.sha256(result.stdout).hexdigest()


def _untracked_files(root: Path) -> list[str]:
    result = _run_git(root, ["status", "--porcelain", "--untracked-files=all"])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RetrievalAuditError(f"Unable to capture Git status for provenance: {detail}")
    files: list[str] = []
    for line in result.stdout.splitlines():
        if not line.startswith("?? "):
            continue
        path = line[3:].strip()
        if path:
            files.append(path.replace("\\", "/"))
    return sorted(files)


def describe_git_provenance(path: Path, *, ref: str | None = None) -> dict[str, Any]:
    root = repository_root(path)
    if root is None:
        if ref:
            raise RetrievalAuditError(f"Cannot validate Git ref outside a worktree: {ref}")
        return _empty_provenance(ref=ref)

    ref_commit = resolve_ref(root, ref) if ref else None
    status = _run_git(root, ["status", "--porcelain", "--untracked-files=all"], check=True)
    untracked_files = _untracked_files(root)
    commit = _optional_git_output(root, ["rev-parse", "HEAD"])
    return {
        "is_git_worktree": True,
        "repo_root": str(root),
        "branch": _branch_name(root),
        "commit": commit,
        "tag": _optional_git_output(root, ["describe", "--tags", "--exact-match", "HEAD"]),
        "dirty": bool(status.stdout.strip()),
        "diff_hash": _diff_hash(root),
        "untracked_files": untracked_files,
        "remote_url": _optional_git_output(root, ["config", "--get", "remote.origin.url"]),
        "ref": ref,
        "ref_commit": ref_commit,
    }


def plan_git_comparison(repo_path: Path, *, left_ref: str, right_ref: str, output_dir: Path | None = None) -> dict[str, Any]:
    left_commit = resolve_ref(repo_path, left_ref)
    right_commit = resolve_ref(repo_path, right_ref)
    root = repository_root(repo_path)
    assert root is not None
    base_output = output_dir or (root / "results" / "git_comparisons")
    return {
        "schema_version": "retrieval_arena.git_comparison_plan.v1",
        "repo_root": str(root),
        "left_ref": left_ref,
        "left_commit": left_commit,
        "right_ref": right_ref,
        "right_commit": right_commit,
        "checkout_strategy": "dry_run_only",
        "mutates_current_worktree": False,
        "planned_run_directories": {
            "left": str((base_output / f"left_{left_commit[:12]}").resolve()),
            "right": str((base_output / f"right_{right_commit[:12]}").resolve()),
        },
    }
