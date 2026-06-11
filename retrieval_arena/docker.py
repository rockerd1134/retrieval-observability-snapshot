from __future__ import annotations

import subprocess
from pathlib import Path

from .config import TestConfig
from .errors import RetrievalAuditError, ValidationError


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RetrievalAuditError("Command failed ({}):\n{}".format(" ".join(command), completed.stdout))


def build_image(test: TestConfig) -> None:
    if not test.build_context:
        return
    if not test.build_context.exists():
        raise ValidationError(f"Docker build_context not found for {test.name}: {test.build_context}")
    _run_command(["docker", "build", "-t", test.image, str(test.build_context)])


def run_container(test: TestConfig, input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = ["docker", "run", "--rm", "-v", f"{input_dir.resolve()}:/input:ro", "-v", f"{output_dir.resolve()}:/output"]
    if test.network_disabled:
        command.extend(["--network", "none"])
    for volume in test.volumes:
        if not volume.host_path.exists():
            raise ValidationError(f"Docker volume host_path not found for {test.name}: {volume.host_path}")
        suffix = ":ro" if volume.read_only else ""
        command.extend(["-v", f"{volume.host_path.resolve()}:{volume.container_path}{suffix}"])
    command.append(test.image)
    _run_command(command)
