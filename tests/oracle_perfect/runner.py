from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


input_dir = Path("/input")
output_dir = Path("/output")
output_dir.mkdir(parents=True, exist_ok=True)
questions = read_jsonl(input_dir / "questions.jsonl")
answers = {row["question_id"]: row["answer"] for row in read_jsonl(input_dir / "answers.jsonl")}
with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for q in questions:
        handle.write(json.dumps({"question_id": q["question_id"], "question": q["question"], "generated_answer": answers[q["question_id"]], "retrieved_context": []}, sort_keys=True) + "\n")
(output_dir / "metadata.json").write_text(json.dumps({"name": "oracle_perfect", "deterministic": True}, indent=2) + "\n", encoding="utf-8")