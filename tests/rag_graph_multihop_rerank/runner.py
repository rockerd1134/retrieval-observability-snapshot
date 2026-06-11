from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict, deque
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
    query = Counter(query_tokens)
    scored: list[tuple[float, str]] = []
    k1 = 1.5
    b = 0.75
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


def read_edges(input_dir: Path, *, directed: bool) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    edge_path = input_dir / "graph_edges.csv"
    if not edge_path.exists():
        return edges
    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source", "")
            target = row.get("target", "")
            if not source or not target:
                continue
            edges[source].add(target)
            if not directed:
                edges[target].add(source)
    return edges


def multihop_distances(seeds: list[str], edges: dict[str, set[str]], *, max_hops: int, frontier_budget: int) -> dict[str, int]:
    distances: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for seed in seeds:
        if seed in distances:
            continue
        distances[seed] = 0
        queue.append((seed, 0))
    while queue:
        node, distance = queue.popleft()
        if distance >= max_hops:
            continue
        neighbors = sorted(edges.get(node, set()))
        if frontier_budget > 0:
            neighbors = neighbors[:frontier_budget]
        for neighbor in neighbors:
            if neighbor in distances:
                continue
            distances[neighbor] = distance + 1
            queue.append((neighbor, distance + 1))
    return distances


input_dir = Path("/input")
output_dir = Path("/output")
output_dir.mkdir(parents=True, exist_ok=True)

config = read_config(input_dir / "config.yaml")
seed_top_k = max(1, int(config.get("seed_top_k", 5)))
max_hops = max(1, int(config.get("max_hops", 2)))
frontier_budget = max(0, int(config.get("frontier_budget_per_node", 12)))
candidate_budget = max(1, int(config.get("candidate_budget", 60)))
final_top_k = max(1, int(config.get("final_top_k", 5)))
distance_penalty = float(config.get("distance_penalty", 0.15))
reserved_deep_slots = max(0, int(config.get("reserved_deep_slots", 0)))
reserved_min_graph_distance = max(1, int(config.get("reserved_min_graph_distance", 2)))
max_context_chars = max(0, int(config.get("max_context_chars_per_doc", 1200)))
directed = bool(config.get("directed", False))

questions = read_jsonl(input_dir / "questions.jsonl")
corpus = {path.stem: path.read_text(encoding="utf-8") for path in (input_dir / "corpus").glob("*.md")}
corpus_tokens = {doc_id: tokens(text) for doc_id, text in corpus.items()}
edges = read_edges(input_dir, directed=directed)

with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for question in questions:
        query_tokens = tokens(question["question"])
        seeds = bm25_scores(query_tokens, corpus_tokens)[:seed_top_k]
        seed_ids = [doc_id for _, doc_id in seeds]
        distances = multihop_distances(seed_ids, edges, max_hops=max_hops, frontier_budget=frontier_budget)
        candidates = set(distances) & set(corpus)
        if len(candidates) > candidate_budget:
            ranked_candidates = bm25_scores(query_tokens, corpus_tokens, candidates)[:candidate_budget]
            candidates = {doc_id for _, doc_id in ranked_candidates}
        bm25_by_doc = {doc_id: score for score, doc_id in bm25_scores(query_tokens, corpus_tokens, candidates)}
        ranked_candidates = sorted(
            candidates,
            key=lambda doc_id: (-(bm25_by_doc.get(doc_id, 0.0) - (distance_penalty * distances.get(doc_id, max_hops + 1))), distances.get(doc_id, max_hops + 1), doc_id),
        )
        reserved_docs = [
            doc_id
            for doc_id in ranked_candidates
            if distances.get(doc_id, max_hops + 1) >= reserved_min_graph_distance
        ][: min(reserved_deep_slots, final_top_k)]
        fill_budget = final_top_k - len(reserved_docs)
        selected_docs: list[str] = []
        fill_count = 0
        for doc_id in ranked_candidates:
            if doc_id in reserved_docs:
                selected_docs.append(doc_id)
            elif fill_count < fill_budget:
                selected_docs.append(doc_id)
                fill_count += 1
            if len(selected_docs) >= final_top_k:
                break
        reranked = selected_docs
        context = [
            {
                "doc_id": doc_id,
                "text": answer_text(corpus[doc_id], max_chars=max_context_chars),
                "score": float(bm25_by_doc.get(doc_id, 0.0) - (distance_penalty * distances.get(doc_id, max_hops + 1))),
                "source": "graph_multihop_rerank",
                "selection_role": "seed" if doc_id in seed_ids else "multihop_neighbor",
                "graph_distance": distances.get(doc_id),
                "rank": rank,
            }
            for rank, doc_id in enumerate(reranked, start=1)
        ]
        generated = " ".join(item["text"] for item in context if item["text"])
        handle.write(
            json.dumps(
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "generated_answer": generated,
                    "retrieved_context": context,
                    "graph_multihop": {
                        "seed_doc_ids": seed_ids,
                        "candidate_count": len(candidates),
                        "max_hops": max_hops,
                        "reserved_deep_slots": reserved_deep_slots,
                        "reserved_min_graph_distance": reserved_min_graph_distance,
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )

(output_dir / "metadata.json").write_text(
    json.dumps(
        {
            "name": "rag_graph_multihop_rerank",
            "deterministic": True,
            "seed_ranker": "bm25",
            "final_ranker": "bm25_minus_distance_penalty",
            "seed_top_k": seed_top_k,
            "max_hops": max_hops,
            "frontier_budget_per_node": frontier_budget,
            "candidate_budget": candidate_budget,
            "final_top_k": final_top_k,
            "distance_penalty": distance_penalty,
            "reserved_deep_slots": reserved_deep_slots,
            "reserved_min_graph_distance": reserved_min_graph_distance,
            "directed": directed,
            "max_context_chars_per_doc": max_context_chars,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
