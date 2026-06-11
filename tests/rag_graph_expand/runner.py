from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
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
edges = defaultdict(set)
edge_path = input_dir / "graph_edges.csv"
if edge_path.exists():
    for row in csv.DictReader(edge_path.open("r", encoding="utf-8")):
        edges[row["source"]].add(row["target"])
        edges[row["target"]].add(row["source"])
with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for q in questions:
        qtokens = Counter(tokens(q["question"]))
        scored = sorted(((sum((qtokens & Counter(tokens(text))).values()), doc_id) for doc_id, text in corpus.items()), key=lambda x: (-x[0], x[1]))
        selected = [scored[0][1]] + sorted(edges.get(scored[0][1], set()))[:1]
        context = [{"doc_id": doc_id, "text": corpus[doc_id], "score": float(i == 0), "source": "graph_expand"} for i, doc_id in enumerate(selected) if doc_id in corpus]
        generated = " ".join(line.strip("# ") for item in context for line in item["text"].splitlines() if line.strip() and not line.startswith("#"))
        handle.write(json.dumps({"question_id": q["question_id"], "question": q["question"], "generated_answer": generated, "retrieved_context": context}, sort_keys=True) + "\n")
(output_dir / "metadata.json").write_text(json.dumps({"name": "rag_graph_expand", "deterministic": True}, indent=2) + "\n", encoding="utf-8")