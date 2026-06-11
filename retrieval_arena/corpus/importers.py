from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..errors import RetrievalAuditError, ValidationError
from ..hashing import sha256_file, sha256_json
from ..manifests import read_manifest, write_manifest
from ..snapshots import corpus_doc_id, utc_now_iso
from .sources import CorpusSourceDescriptor


IMPORT_SCHEMA_VERSION = "retrieval_arena.corpus_import_manifest.v1"
IMPORT_PARSER_VERSION = "retrieval_arena.corpus_import.copy-v1"


def import_corpus_source(descriptor: CorpusSourceDescriptor, *, created_at: str | None = None, overwrite: bool = True) -> dict[str, Any]:
    snapshot_workspace = descriptor.destination_workspace / descriptor.corpus_id / descriptor.snapshot_id
    source_workspace = snapshot_workspace / "source"
    documents_workspace = snapshot_workspace / "documents"
    if snapshot_workspace.exists():
        if not overwrite:
            reused = _reuse_existing_import(descriptor, snapshot_workspace)
            if reused is not None:
                return reused
        shutil.rmtree(snapshot_workspace)
    source_workspace.mkdir(parents=True)
    documents_workspace.mkdir(parents=True)

    provenance = _materialize_source(descriptor, source_workspace)
    selection_root = _selection_root(source_workspace, descriptor.docs_root)
    selected_files = _select_files(selection_root, include=descriptor.include, exclude=descriptor.exclude)
    ignored_files = _ignored_files(selection_root, selected_files)
    inventory: list[dict[str, Any]] = []
    for source_file in selected_files:
        relative_path = source_file.relative_to(selection_root).as_posix()
        destination = documents_workspace / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)
        inventory.append(
            {
                "path": relative_path,
                "doc_id": corpus_doc_id(relative_path),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    inventory = sorted(inventory, key=lambda item: item["path"])
    descriptor_payload = _descriptor_payload(descriptor)
    import_config = {
        "docs_root": descriptor.docs_root,
        "include": list(descriptor.include),
        "exclude": list(descriptor.exclude),
        "parser_version": IMPORT_PARSER_VERSION,
    }
    snapshot_identity = {
        "corpus_id": descriptor.corpus_id,
        "snapshot_id": descriptor.snapshot_id,
        "source_provenance": provenance,
        "selected_files": inventory,
        "import_config_hash": sha256_json(import_config),
    }
    manifest = {
        "schema_version": IMPORT_SCHEMA_VERSION,
        "created_at": created_at or utc_now_iso(),
        "manifest_type": "corpus_import",
        "corpus_id": descriptor.corpus_id,
        "snapshot_id": descriptor.snapshot_id,
        "source_type": descriptor.source_type,
        "source_url": descriptor.source_url,
        "source_path": str(descriptor.source_path) if descriptor.source_path else None,
        "requested_ref": descriptor.requested_ref,
        "docs_root": descriptor.docs_root,
        "include": list(descriptor.include),
        "exclude": list(descriptor.exclude),
        "import_config": import_config,
        "import_config_hash": sha256_json(import_config),
        "parser_version": IMPORT_PARSER_VERSION,
        "source_descriptor": descriptor_payload,
        "source_descriptor_hash": sha256_json(descriptor_payload),
        "snapshot_workspace": str(snapshot_workspace.resolve()),
        "materialized_source_path": str(source_workspace.resolve()),
        "documents_path": str(documents_workspace.resolve()),
        "source_provenance": provenance,
        "selected_file_count": len(inventory),
        "selected_size_bytes": sum(int(item["size_bytes"]) for item in inventory),
        "file_inventory": inventory,
        "selected_files": inventory,
        "ignored_files": ignored_files,
        "ignored_file_count": len(ignored_files),
        "content_hash": sha256_json({"files": inventory}),
        "snapshot_identity_hash": sha256_json(snapshot_identity),
        "stage_status": "ok",
    }
    written = write_manifest(snapshot_workspace / "corpus_import_manifest.json", manifest)
    return {
        "manifest": written,
        "manifest_path": snapshot_workspace / "corpus_import_manifest.json",
        "documents_path": documents_workspace,
        "reused": False,
    }


def _reuse_existing_import(descriptor: CorpusSourceDescriptor, snapshot_workspace: Path) -> dict[str, Any] | None:
    manifest_path = snapshot_workspace / "corpus_import_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_manifest(manifest_path)
    documents_path = Path(str(manifest.get("documents_path", snapshot_workspace / "documents")))
    _validate_reusable_manifest(descriptor, manifest, documents_path)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "documents_path": documents_path,
        "reused": True,
    }


def _validate_reusable_manifest(descriptor: CorpusSourceDescriptor, manifest: dict[str, Any], documents_path: Path) -> None:
    expected = {
        "manifest_type": "corpus_import",
        "corpus_id": descriptor.corpus_id,
        "snapshot_id": descriptor.snapshot_id,
        "source_type": descriptor.source_type,
        "source_url": descriptor.source_url,
        "source_path": str(descriptor.source_path) if descriptor.source_path else None,
        "requested_ref": descriptor.requested_ref,
        "docs_root": descriptor.docs_root,
        "include": list(descriptor.include),
        "exclude": list(descriptor.exclude),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValidationError(f"Existing import does not match requested descriptor field: {key}")
    if not documents_path.is_dir():
        raise ValidationError(f"Existing import documents directory is missing: {documents_path}")
    inventory = _documents_inventory(documents_path)
    if inventory != manifest.get("file_inventory"):
        raise ValidationError("Existing import documents no longer match corpus_import_manifest.json.")
    if sha256_json({"files": inventory}) != manifest.get("content_hash"):
        raise ValidationError("Existing import content hash does not match corpus_import_manifest.json.")
    if descriptor.source_type == "git":
        assert descriptor.source_path is not None
        if not _git_worktree_clean(descriptor.source_path):
            raise ValidationError(f"Git source worktree must be clean to reuse an existing import: {descriptor.source_path}")
        ref = descriptor.requested_ref or "HEAD"
        resolved_commit = _git_output(descriptor.source_path, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
        if manifest.get("source_provenance", {}).get("resolved_commit") != resolved_commit:
            raise ValidationError("Existing import resolved commit does not match requested Git ref.")


def _materialize_source(descriptor: CorpusSourceDescriptor, source_workspace: Path) -> dict[str, Any]:
    if descriptor.source_type == "local":
        assert descriptor.source_path is not None
        if not descriptor.source_path.is_dir():
            raise ValidationError(f"Local source path must be a directory: {descriptor.source_path}")
        _copy_tree_contents(descriptor.source_path, source_workspace)
        source_hash = _source_tree_hash(descriptor.source_path)
        return {
            "source_kind": "local_directory",
            "source_path": str(descriptor.source_path),
            "source_url": descriptor.source_url,
            "requested_ref": descriptor.requested_ref,
            "resolved_revision": descriptor.requested_ref,
            "source_hash": source_hash,
            "mutates_active_checkout": False,
        }
    return _materialize_git_source(descriptor, source_workspace)


def _materialize_git_source(descriptor: CorpusSourceDescriptor, source_workspace: Path) -> dict[str, Any]:
    source = descriptor.source_path or Path(str(descriptor.source_url))
    if descriptor.source_path is None:
        raise RetrievalAuditError("Remote Git clone support is scaffolded; provide source_path for offline imports.")
    if not descriptor.source_path.is_dir():
        raise ValidationError(f"Git source path must be a directory: {descriptor.source_path}")
    ref = descriptor.requested_ref or "HEAD"
    commit = _git_output(descriptor.source_path, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
    archive_args = ["git", "-C", str(descriptor.source_path), "archive", "--format=tar", commit]
    if descriptor.docs_root:
        archive_args.append(descriptor.docs_root)
    archive = subprocess.run(
        archive_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if archive.returncode != 0:
        detail = archive.stderr.decode("utf-8", errors="replace").strip()
        raise RetrievalAuditError(f"Unable to archive Git source {source}: {detail}")
    extract = subprocess.run(
        ["tar", "-xf", "-", "-C", str(source_workspace)],
        input=archive.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if extract.returncode != 0:
        detail = extract.stderr.decode("utf-8", errors="replace").strip()
        raise RetrievalAuditError(f"Unable to extract Git archive: {detail}")
    return {
        "source_kind": "git_local_archive",
        "source_path": str(descriptor.source_path),
        "source_url": descriptor.source_url,
        "requested_ref": descriptor.requested_ref,
        "resolved_commit": commit,
        "source_worktree_clean": _git_worktree_clean(descriptor.source_path),
        "mutates_active_checkout": False,
    }


def _copy_tree_contents(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _select_files(root: Path, *, include: tuple[str, ...], exclude: tuple[str, ...]) -> list[Path]:
    if not root.is_dir():
        raise ValidationError(f"docs_root does not exist in materialized source: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if not any(Path(relative).match(pattern) for pattern in include):
            continue
        if any(Path(relative).match(pattern) for pattern in exclude):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _ignored_files(root: Path, selected_files: list[Path]) -> list[dict[str, Any]]:
    selected = {path.resolve() for path in selected_files}
    ignored = []
    for path in root.rglob("*"):
        if not path.is_file() or path.resolve() in selected:
            continue
        ignored.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sorted(ignored, key=lambda item: item["path"])


def _descriptor_payload(descriptor: CorpusSourceDescriptor) -> dict[str, Any]:
    return {
        "corpus_id": descriptor.corpus_id,
        "snapshot_id": descriptor.snapshot_id,
        "source_type": descriptor.source_type,
        "source_url": descriptor.source_url,
        "source_path": str(descriptor.source_path) if descriptor.source_path else None,
        "requested_ref": descriptor.requested_ref,
        "docs_root": descriptor.docs_root,
        "include": list(descriptor.include),
        "exclude": list(descriptor.exclude),
        "destination_workspace": str(descriptor.destination_workspace),
    }


def _selection_root(source_workspace: Path, docs_root: str | None) -> Path:
    return source_workspace / docs_root if docs_root else source_workspace


def _source_tree_hash(path: Path) -> str:
    return sha256_json({"files": _documents_inventory(path)})


def _documents_inventory(path: Path) -> list[dict[str, Any]]:
    entries = []
    for item in path.rglob("*"):
        if item.is_file():
            relative_path = item.relative_to(path).as_posix()
            entries.append(
                {
                    "path": relative_path,
                    "doc_id": corpus_doc_id(relative_path),
                    "size_bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
    return sorted(entries, key=lambda item: item["path"])


def _git_output(repo: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise RetrievalAuditError("Git executable is not available for corpus import.") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RetrievalAuditError(detail)
    return result.stdout.strip()


def _git_worktree_clean(repo: Path) -> bool:
    return _git_output(repo, ["status", "--porcelain", "--untracked-files=all"]) == ""
