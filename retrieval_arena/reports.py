from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_run_scores(run_dir: Path, item_scores: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    with (run_dir / "item_scores.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in item_scores:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (run_dir / "scores.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary(experiment_dir: Path, rows: list[dict[str, Any]]) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "test", "run_id", "image", "build_context", "match_percent", "mean_f1", "mean_precision", "mean_recall", "mean_lexical_overlap", "num_questions", "run_dir"]
    with (experiment_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    lines = ["# Retrieval Audit Summary", "", "| Dataset | Test | Match % | Mean F1 | Questions |", "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['test']} | {row['match_percent']:.3f} | {row['mean_f1']:.3f} | {row['num_questions']} |")
    lines.append("")
    (experiment_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")