from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_outputs(predictions: list[dict], metadata: dict) -> None:
    output = Path("/output")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_input() -> tuple[list[dict], dict[str, str], dict[str, str]]:
    input_dir = Path("/input")
    questions = read_jsonl(input_dir / "questions.jsonl")
    answers = {row["question_id"]: row["answer"] for row in read_jsonl(input_dir / "answers.jsonl")}
    corpus = {}
    for path in (input_dir / "corpus").glob("**/*"):
        if path.is_file():
            corpus[path.stem] = path.read_text(encoding="utf-8")
    return questions, answers, corpus


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def best_docs(question: str, corpus: dict[str, str], top_k: int = 2) -> list[dict]:
    q = Counter(tokens(question))
    scored = []
    for doc_id, text in corpus.items():
        c = Counter(tokens(text))
        score = float(sum((q & c).values()))
        scored.append((score, doc_id, text))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{"doc_id": doc_id, "text": text, "score": score, "source": "lexical"} for score, doc_id, text in scored[:top_k]]


def mode_from_env(default: str) -> str:
    return os.environ.get("RETRIEVAL_ARENA_MODE", default)