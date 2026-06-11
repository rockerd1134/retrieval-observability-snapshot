from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ValidationError


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    query_set_id: str | None = None
    corpus_snapshot_manifest: Path | None = None
    graph_snapshot_manifest: Path | None = None
    support_surface_manifest: Path | None = None
    git_ref: str | None = None


@dataclass(frozen=True)
class TestConfig:
    name: str
    image: str
    build_context: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)
    volumes: list["VolumeMount"] = field(default_factory=list)
    network_disabled: bool = False


@dataclass(frozen=True)
class VolumeMount:
    host_path: Path
    container_path: str
    read_only: bool = True


@dataclass(frozen=True)
class ScoringConfig:
    method: str = "lexical_baseline"
    match_threshold: float = 0.5


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    datasets: list[DatasetConfig]
    tests: list[TestConfig]
    scoring: ScoringConfig
    output_dir: Path
    config_path: Path
    retrieval_arena_git_ref: str | None = None
    config_git_ref: str | None = None


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path).resolve()
    if not path.exists():
        raise ValidationError(f"Config file not found: {path}")
    raw = _load_simple_yaml(path)
    return parse_config(raw, path)


def _scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"\'')


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the small YAML subset used by Retrieval Audit Framework example configs.

    This keeps the v1 package dependency-free while supporting mappings, lists of
    mappings, nested config mappings, strings, ints, floats, and booleans.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_key: tuple[int, dict[str, Any], str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        text = raw_line.strip()
        if pending_key and indent > pending_key[0]:
            p_indent, p_map, p_key = pending_key
            container: Any = [] if text.startswith("-") else {}
            p_map[p_key] = container
            stack.append((p_indent, container))
            parent = container
            pending_key = None
        else:
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
        if text.startswith("- "):
            if not isinstance(parent, list):
                raise ValidationError(f"Unsupported YAML list placement in {path}: {raw_line}")
            item_text = text[2:].strip()
            item: dict[str, Any] = {}
            parent.append(item)
            stack.append((indent, item))
            if item_text:
                key, sep, value = item_text.partition(":")
                if not sep:
                    raise ValidationError(f"Unsupported YAML list item in {path}: {raw_line}")
                if value.strip():
                    item[key.strip()] = _scalar(value)
                else:
                    pending_key = (indent, item, key.strip())
            continue
        key, sep, value = text.partition(":")
        if not sep or not isinstance(parent, dict):
            raise ValidationError(f"Unsupported YAML line in {path}: {raw_line}")
        if value.strip():
            parent[key.strip()] = _scalar(value)
            pending_key = None
        else:
            pending_key = (indent, parent, key.strip())
    return root


def parse_config(raw: dict[str, Any], config_path: Path) -> ExperimentConfig:
    base = config_path.parent
    if not isinstance(raw, dict):
        raise ValidationError("Experiment config must be a mapping.")
    experiment_name = raw.get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValidationError("experiment_name is required and must be a non-empty string.")

    datasets_raw = raw.get("datasets")
    if not isinstance(datasets_raw, list) or not datasets_raw:
        raise ValidationError("datasets must be a non-empty list.")
    datasets: list[DatasetConfig] = []
    seen_datasets: set[str] = set()
    for item in datasets_raw:
        if not isinstance(item, dict):
            raise ValidationError("Each dataset must be a mapping.")
        name = item.get("name")
        path_value = item.get("path")
        if not isinstance(name, str) or not name:
            raise ValidationError("Each dataset requires a non-empty name.")
        if name in seen_datasets:
            raise ValidationError(f"Duplicate dataset name: {name}")
        if not isinstance(path_value, str) or not path_value:
            raise ValidationError(f"Dataset {name} requires path.")
        query_set_id = item.get("query_set_id")
        if query_set_id is not None and not isinstance(query_set_id, str):
            raise ValidationError(f"Dataset {name} query_set_id must be a string when present.")
        git_ref = item.get("git_ref")
        if git_ref is not None and not isinstance(git_ref, str):
            raise ValidationError(f"Dataset {name} git_ref must be a string when present.")
        manifest_paths: dict[str, Path | None] = {}
        for key in ["corpus_snapshot_manifest", "graph_snapshot_manifest", "support_surface_manifest"]:
            value = item.get(key)
            if value is not None and not isinstance(value, str):
                raise ValidationError(f"Dataset {name} {key} must be a string when present.")
            manifest_paths[key] = _resolve_path(value, base) if value else None
        seen_datasets.add(name)
        datasets.append(
            DatasetConfig(
                name=name,
                path=_resolve_path(path_value, base),
                query_set_id=query_set_id,
                corpus_snapshot_manifest=manifest_paths["corpus_snapshot_manifest"],
                graph_snapshot_manifest=manifest_paths["graph_snapshot_manifest"],
                support_surface_manifest=manifest_paths["support_surface_manifest"],
                git_ref=git_ref,
            )
        )

    tests_raw = raw.get("tests")
    if not isinstance(tests_raw, list) or not tests_raw:
        raise ValidationError("tests must be a non-empty list.")
    tests: list[TestConfig] = []
    seen_tests: set[str] = set()
    for item in tests_raw:
        if not isinstance(item, dict):
            raise ValidationError("Each test must be a mapping.")
        name = item.get("name")
        image = item.get("image")
        if not isinstance(name, str) or not name:
            raise ValidationError("Each test requires a non-empty name.")
        if name in seen_tests:
            raise ValidationError(f"Duplicate test name: {name}")
        if not isinstance(image, str) or not image:
            raise ValidationError(f"Test {name} requires image.")
        build_context = item.get("build_context")
        if build_context is not None and not isinstance(build_context, str):
            raise ValidationError(f"Test {name} build_context must be a string when present.")
        network_disabled = item.get("network_disabled", False)
        if not isinstance(network_disabled, bool):
            raise ValidationError(f"Test {name} network_disabled must be boolean when present.")
        test_config = item.get("config", {})
        if not isinstance(test_config, dict):
            raise ValidationError(f"Test {name} config must be a mapping.")
        volumes_raw = item.get("volumes", [])
        if not isinstance(volumes_raw, list):
            raise ValidationError(f"Test {name} volumes must be a list when present.")
        volumes: list[VolumeMount] = []
        for volume in volumes_raw:
            if not isinstance(volume, dict):
                raise ValidationError(f"Test {name} volume entries must be mappings.")
            host_path = volume.get("host_path")
            container_path = volume.get("container_path")
            read_only = volume.get("read_only", True)
            if not isinstance(host_path, str) or not host_path:
                raise ValidationError(f"Test {name} volume requires host_path.")
            if not isinstance(container_path, str) or not container_path.startswith("/"):
                raise ValidationError(f"Test {name} volume requires absolute container_path.")
            if not isinstance(read_only, bool):
                raise ValidationError(f"Test {name} volume read_only must be boolean.")
            volumes.append(VolumeMount(host_path=_resolve_path(host_path, base), container_path=container_path, read_only=read_only))
        seen_tests.add(name)
        tests.append(TestConfig(name=name, image=image, build_context=_resolve_path(build_context, base) if build_context else None, config=test_config, volumes=volumes, network_disabled=network_disabled))

    scoring_raw = raw.get("scoring", {})
    if not isinstance(scoring_raw, dict):
        raise ValidationError("scoring must be a mapping.")
    method = scoring_raw.get("method", "lexical_baseline")
    if method != "lexical_baseline":
        raise ValidationError("Only scoring.method=lexical_baseline is supported in v1.")
    threshold = scoring_raw.get("match_threshold", 0.5)
    if not isinstance(threshold, (int, float)) or not 0 <= float(threshold) <= 1:
        raise ValidationError("scoring.match_threshold must be a number in [0, 1].")

    output_dir_raw = raw.get("output_dir", "results")
    if not isinstance(output_dir_raw, str) or not output_dir_raw:
        raise ValidationError("output_dir must be a non-empty string.")
    retrieval_arena_git_ref = raw.get("retrieval_arena_git_ref")
    if retrieval_arena_git_ref is not None and not isinstance(retrieval_arena_git_ref, str):
        raise ValidationError("retrieval_arena_git_ref must be a string when present.")
    config_git_ref = raw.get("config_git_ref")
    if config_git_ref is not None and not isinstance(config_git_ref, str):
        raise ValidationError("config_git_ref must be a string when present.")

    return ExperimentConfig(
        experiment_name=experiment_name,
        datasets=datasets,
        tests=tests,
        scoring=ScoringConfig(method=method, match_threshold=float(threshold)),
        output_dir=_resolve_path(output_dir_raw, base),
        config_path=config_path.resolve(),
        retrieval_arena_git_ref=retrieval_arena_git_ref,
        config_git_ref=config_git_ref,
    )
