from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import DatasetConfig, ExperimentConfig, TestConfig
from .errors import ValidationError
from .git_provenance import describe_git_provenance
from .hashing import directory_inventory, sha256_file, sha256_json, sha256_jsonl
from .manifests import read_manifest, write_manifest


REPLAY_SCHEMA_VERSION = "retrieval_arena.replay_manifest.v1"
EXPERIMENT_SCHEMA_VERSION = "retrieval_arena.experiment_manifest.v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def artifact_hashes(root: Path) -> dict[str, dict[str, Any]]:
    if not root.exists():
        return {}
    return {item["path"]: {"sha256": item["sha256"], "size_bytes": item["size_bytes"]} for item in directory_inventory(root)}


def selected_file_hashes(root: Path, filenames: list[str]) -> dict[str, dict[str, Any]]:
    hashes: dict[str, dict[str, Any]] = {}
    for filename in filenames:
        path = root / filename
        if not path.exists() or not path.is_file():
            continue
        hashes[filename] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return hashes


def scoring_hash(scoring: Any) -> str:
    scoring_path = Path(__file__).resolve().parent / "scoring.py"
    return sha256_json(
        {
            "scoring_config": {
                "method": scoring.method,
                "match_threshold": scoring.match_threshold,
            },
            "scoring_code_hash": sha256_file(scoring_path),
        }
    )


def resolved_run_identity_hash(*, dataset: DatasetConfig, test: TestConfig, scoring: Any) -> str:
    """Hash the resolved inputs that define one dataset/test run."""
    return sha256_json(
        {
            "dataset": {
                "name": dataset.name,
                "path": str(dataset.path),
                "query_set_id": dataset.query_set_id or dataset.name,
                "corpus_snapshot_manifest": _manifest_reference(dataset.corpus_snapshot_manifest),
                "graph_snapshot_manifest": _manifest_reference(dataset.graph_snapshot_manifest),
                "support_surface_manifest": _manifest_reference(dataset.support_surface_manifest),
            },
            "test": {
                "name": test.name,
                "image": test.image,
                "build_context_hash": sha256_json({"build_context": artifact_hashes(test.build_context)}) if test.build_context else sha256_json({"build_context": None}),
                "config": test.config,
                "volumes": [
                    {
                        "host_path": str(volume.host_path),
                        "container_path": volume.container_path,
                        "read_only": volume.read_only,
                    }
                    for volume in test.volumes
                ],
                "network_disabled": test.network_disabled,
            },
            "scoring": {
                "method": scoring.method,
                "match_threshold": scoring.match_threshold,
                "scoring_hash": scoring_hash(scoring),
            },
        }
    )


def container_metadata_hash(metadata: dict[str, Any]) -> str:
    return sha256_json(metadata)


def _manifest_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest = read_manifest(path)
    return {
        "path": str(path),
        "manifest_type": manifest.get("manifest_type"),
        "manifest_hash": manifest["manifest_hash"],
        "snapshot_id": manifest.get("snapshot_id"),
    }


def _snapshot_id(path: Path | None) -> str | None:
    if path is None:
        return None
    return read_manifest(path).get("snapshot_id")


def build_run_replay_manifest(
    *,
    config: ExperimentConfig,
    dataset: DatasetConfig,
    test: TestConfig,
    run_id: str,
    run_dir: Path,
    metadata: dict[str, Any],
    run_started_at: str,
    run_completed_at: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    config_path = config.config_path
    query_set_path = dataset.path / "questions.jsonl"
    if not query_set_path.exists():
        raise ValidationError(f"Dataset missing questions.jsonl for replay manifest: {query_set_path}")

    retrieval_arena_root = Path(__file__).resolve().parent.parent
    output_hashes = selected_file_hashes(
        run_dir,
        ["predictions.jsonl", "metadata.json", "action_traces.jsonl", "item_scores.jsonl", "scores.json"],
    )
    corpus_ref = _manifest_reference(dataset.corpus_snapshot_manifest)
    graph_ref = _manifest_reference(dataset.graph_snapshot_manifest)
    support_ref = _manifest_reference(dataset.support_surface_manifest)
    build_context_hash = sha256_json({"build_context": None})
    if test.build_context:
        build_context_hash = sha256_json({"build_context": artifact_hashes(test.build_context)})

    manifest: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "created_at": created_at or utc_now_iso(),
        "manifest_type": "retrieval_replay",
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "dataset": dataset.name,
        "test": test.name,
        "query_set_id": dataset.query_set_id or dataset.name,
        "query_set_hash": sha256_jsonl(query_set_path),
        "corpus_snapshot_id": _snapshot_id(dataset.corpus_snapshot_manifest),
        "graph_snapshot_id": _snapshot_id(dataset.graph_snapshot_manifest),
        "support_surface_id": _snapshot_id(dataset.support_surface_manifest),
        "snapshot_manifest_references": {
            "corpus": corpus_ref,
            "graph": graph_ref,
            "support_surface": support_ref,
        },
        "retrieval_config_id": test.name,
        "retrieval_config_hash": sha256_file(config_path),
        "resolved_run_identity_hash": resolved_run_identity_hash(dataset=dataset, test=test, scoring=config.scoring),
        "container_image": test.image,
        "container_metadata_hash": container_metadata_hash(metadata),
        "build_context_hash": build_context_hash,
        "scoring_method": config.scoring.method,
        "scoring_hash": scoring_hash(config.scoring),
        "retrieval_arena_version": __version__,
        "retrieval_arena_git_provenance": describe_git_provenance(retrieval_arena_root, ref=config.retrieval_arena_git_ref),
        "run_started_at": run_started_at,
        "run_completed_at": run_completed_at,
        "input_artifact_hashes": artifact_hashes(run_dir / "input"),
        "output_artifact_hashes": output_hashes,
    }
    dataset_provenance = describe_git_provenance(dataset.path, ref=dataset.git_ref)
    if dataset_provenance["is_git_worktree"]:
        manifest["dataset_git_provenance"] = dataset_provenance
    config_provenance = describe_git_provenance(config_path.parent, ref=config.config_git_ref)
    if config_provenance["is_git_worktree"]:
        manifest["config_git_provenance"] = config_provenance
    return manifest


def write_run_replay_manifest(output_path: Path, **kwargs: Any) -> dict[str, Any]:
    return write_manifest(output_path, build_run_replay_manifest(**kwargs))


def build_experiment_manifest(
    *,
    config: ExperimentConfig,
    experiment_dir: Path,
    run_manifests: list[Path],
    created_at: str | None = None,
) -> dict[str, Any]:
    indexed_runs = []
    for path in sorted(run_manifests, key=lambda item: item.as_posix()):
        manifest = read_manifest(path)
        indexed_runs.append(
            {
                "path": path.relative_to(experiment_dir).as_posix(),
                "manifest_hash": manifest["manifest_hash"],
                "run_id": manifest["run_id"],
                "dataset": manifest["dataset"],
                "test": manifest["test"],
                "query_set_id": manifest["query_set_id"],
                "corpus_snapshot_id": manifest.get("corpus_snapshot_id"),
                "graph_snapshot_id": manifest.get("graph_snapshot_id"),
                "support_surface_id": manifest.get("support_surface_id"),
            }
        )
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "created_at": created_at or utc_now_iso(),
        "manifest_type": "experiment",
        "experiment_name": config.experiment_name,
        "retrieval_config_hash": sha256_file(config.config_path),
        "retrieval_arena_version": __version__,
        "retrieval_arena_git_provenance": describe_git_provenance(
            Path(__file__).resolve().parent.parent,
            ref=config.retrieval_arena_git_ref,
        ),
        "run_manifest_count": len(indexed_runs),
        "run_manifests": indexed_runs,
    }


def write_experiment_manifest(output_path: Path, **kwargs: Any) -> dict[str, Any]:
    return write_manifest(output_path, build_experiment_manifest(**kwargs))


def manifest_json_text(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
