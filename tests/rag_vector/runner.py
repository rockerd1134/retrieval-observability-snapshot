from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


input_dir = Path("/input")
output_dir = Path("/output")
output_dir.mkdir(parents=True, exist_ok=True)
questions = read_jsonl(input_dir / "questions.jsonl")
corpus = {p.stem: p.read_text(encoding="utf-8") for p in (input_dir / "corpus").glob("*.md")}
with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for q in questions:
        qtokens = Counter(tokens(q["question"]))
        scored = sorted(((sum((qtokens & Counter(tokens(text))).values()), doc_id, text) for doc_id, text in corpus.items()), key=lambda x: (-x[0], x[1]))
        score, doc_id, text = scored[0]
        generated = " ".join(line.strip("# ") for line in text.splitlines() if line.strip() and not line.startswith("#"))
        context = [{"doc_id": doc_id, "text": text, "score": float(score), "source": "vector"}]
        handle.write(json.dumps({"question_id": q["question_id"], "question": q["question"], "generated_answer": generated, "retrieved_context": context}, sort_keys=True) + "\n")
(output_dir / "metadata.json").write_text(json.dumps({"name": "rag_vector", "deterministic": True}, indent=2) + "\n", encoding="utf-8")