from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def answer_text(text: str, *, max_chars: int) -> str:
    lines = [line.strip("# ").strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    snippet = " ".join(lines)
    if max_chars > 0 and len(snippet) > max_chars:
        return snippet[:max_chars].rsplit(" ", 1)[0].strip()
    return snippet


def bm25_scores(query_tokens: list[str], corpus_tokens: dict[str, list[str]], candidates: set[str] | None = None) -> list[tuple[float, str]]:
    num_docs = len(corpus_tokens)
    avg_len = sum(len(doc_tokens) for doc_tokens in corpus_tokens.values()) / num_docs if num_docs else 0.0
    doc_freq: Counter[str] = Counter()
    for doc_tokens in corpus_tokens.values():
        doc_freq.update(set(doc_tokens))

    candidate_ids = candidates if candidates is not None else set(corpus_tokens)
    k1 = 1.5
    b = 0.75
    query = Counter(query_tokens)
    scored: list[tuple[float, str]] = []
    for doc_id in candidate_ids:
        doc_tokens = corpus_tokens.get(doc_id, [])
        tf = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        score = 0.0
        for token, query_count in query.items():
            if tf[token] == 0:
                continue
            idf = math.log(1.0 + (num_docs - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            denom = tf[token] + k1 * (1.0 - b + b * (doc_len / avg_len if avg_len else 0.0))
            score += query_count * idf * ((tf[token] * (k1 + 1.0)) / denom)
        scored.append((score, doc_id))
    return sorted(scored, key=lambda item: (-item[0], item[1]))


def read_edges(input_dir: Path) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    edge_path = input_dir / "graph_edges.csv"
    if not edge_path.exists():
        return edges
    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source", "")
            target = row.get("target", "")
            if source and target:
                edges[source].add(target)
                edges[target].add(source)
    return edges


input_dir = Path("/input")
output_dir = Path("/output")
output_dir.mkdir(parents=True, exist_ok=True)

config = read_config(input_dir / "config.yaml")
seed_top_k = max(1, int(config.get("seed_top_k", 5)))
final_top_k = max(1, int(config.get("final_top_k", config.get("top_k", 5))))
max_context_chars = max(0, int(config.get("max_context_chars_per_doc", 1200)))

questions = read_jsonl(input_dir / "questions.jsonl")
corpus = {path.stem: path.read_text(encoding="utf-8") for path in (input_dir / "corpus").glob("*.md")}
corpus_tokens = {doc_id: tokens(text) for doc_id, text in corpus.items()}
edges = read_edges(input_dir)

with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for question in questions:
        query_tokens = tokens(question["question"])
        seeds = bm25_scores(query_tokens, corpus_tokens)[:seed_top_k]
        seed_ids = {doc_id for _, doc_id in seeds}
        expanded_ids = set(seed_ids)
        expanded_ids.update(*(edges.get(doc_id, set()) for doc_id in seed_ids))
        expanded_ids &= set(corpus)
        reranked = bm25_scores(query_tokens, corpus_tokens, expanded_ids)[:final_top_k]
        context = [
            {
                "doc_id": doc_id,
                "text": answer_text(corpus[doc_id], max_chars=max_context_chars),
                "score": float(score),
                "source": "graph_rerank",
                "selection_role": "seed" if doc_id in seed_ids else "expanded_neighbor",
                "rank": rank,
            }
            for rank, (score, doc_id) in enumerate(reranked, start=1)
        ]
        generated = " ".join(item["text"] for item in context if item["text"])
        handle.write(
            json.dumps(
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "generated_answer": generated,
                    "retrieved_context": context,
                },
                sort_keys=True,
            )
            + "\n"
        )

(output_dir / "metadata.json").write_text(
    json.dumps(
        {
            "name": "rag_graph_rerank",
            "deterministic": True,
            "seed_ranker": "bm25",
            "final_ranker": "bm25",
            "seed_top_k": seed_top_k,
            "final_top_k": final_top_k,
            "max_context_chars_per_doc": max_context_chars,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
