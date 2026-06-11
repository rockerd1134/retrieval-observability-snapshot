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
    if not path.exists():
        return []
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


def shortest_distances(seeds: list[str], edges: dict[str, set[str]], *, max_hops: int) -> dict[str, int]:
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
        for neighbor in sorted(edges.get(node, set())):
            if neighbor in distances:
                continue
            distances[neighbor] = distance + 1
            queue.append((neighbor, distance + 1))
    return distances


def support_targets(audit: dict[str, Any], *, target_top_k: int) -> list[str]:
    docs = audit.get("top_docs", [])
    targets = []
    for doc in docs:
        doc_id = doc.get("doc_id", "")
        if doc_id and doc_id not in targets:
            targets.append(doc_id)
        if len(targets) >= target_top_k:
            break
    return targets


input_dir = Path("/input")
output_dir = Path("/output")
output_dir.mkdir(parents=True, exist_ok=True)

config = read_config(input_dir / "config.yaml")
seed_top_k = max(1, int(config.get("seed_top_k", 5)))
target_top_k = max(1, int(config.get("target_top_k", 5)))
hop_budget = max(0, int(config.get("hop_budget", 2)))
final_top_k = max(1, int(config.get("final_top_k", seed_top_k + target_top_k)))
max_context_chars = max(0, int(config.get("max_context_chars_per_doc", 1200)))
directed = bool(config.get("directed", False))

questions = read_jsonl(input_dir / "questions.jsonl")
audit_by_question = {row.get("question_id", ""): row for row in read_jsonl(input_dir / "faq_support_audit.jsonl")}
corpus = {path.stem: path.read_text(encoding="utf-8") for path in (input_dir / "corpus").glob("*.md")}
corpus_tokens = {doc_id: tokens(text) for doc_id, text in corpus.items()}
edges = read_edges(input_dir, directed=directed)

diagnostics: list[dict[str, Any]] = []
with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for question in questions:
        query_tokens = tokens(question["question"])
        seed_ranked = bm25_scores(query_tokens, corpus_tokens)[:seed_top_k]
        seed_ids = [doc_id for _, doc_id in seed_ranked]
        targets = [doc_id for doc_id in support_targets(audit_by_question.get(question["question_id"], {}), target_top_k=target_top_k) if doc_id in corpus]
        distances = shortest_distances(seed_ids, edges, max_hops=hop_budget)
        reachable_targets = [doc_id for doc_id in targets if doc_id in distances]
        unreachable_targets = [doc_id for doc_id in targets if doc_id not in distances]

        selected: list[tuple[float, str, str, int, int | None]] = []
        for rank, (score, doc_id) in enumerate(seed_ranked, start=1):
            selected.append((score, doc_id, "seed", rank, 0))
        target_scores = {doc_id: score for score, doc_id in bm25_scores(query_tokens, corpus_tokens, set(reachable_targets))} if reachable_targets else {}
        for rank, doc_id in enumerate(sorted(reachable_targets, key=lambda value: (distances[value], -target_scores.get(value, 0.0), value)), start=1):
            if doc_id not in seed_ids:
                selected.append((target_scores.get(doc_id, 0.0), doc_id, "oracle_reachable_support", rank, distances[doc_id]))
        deduped: list[tuple[float, str, str, int, int | None]] = []
        seen: set[str] = set()
        for item in selected:
            if item[1] in seen:
                continue
            seen.add(item[1])
            deduped.append(item)
        deduped = deduped[:final_top_k]

        context = [
            {
                "doc_id": doc_id,
                "text": answer_text(corpus[doc_id], max_chars=max_context_chars),
                "score": float(score),
                "source": "oracle_graph_support",
                "selection_role": role,
                "rank": rank,
                "graph_distance": distance,
            }
            for score, doc_id, role, rank, distance in deduped
        ]
        generated = " ".join(item["text"] for item in context if item["text"])
        diagnostics.append(
            {
                "question_id": question["question_id"],
                "seed_doc_ids": seed_ids,
                "support_target_doc_ids": targets,
                "reachable_support_doc_ids": reachable_targets,
                "unreachable_support_doc_ids": unreachable_targets,
                "num_support_targets": len(targets),
                "num_reachable_support_targets": len(reachable_targets),
                "support_reachable": bool(reachable_targets),
                "min_support_distance": min((distances[doc_id] for doc_id in reachable_targets), default=None),
            }
        )
        handle.write(
            json.dumps(
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "generated_answer": generated,
                    "retrieved_context": context,
                    "oracle_graph_support": diagnostics[-1],
                },
                sort_keys=True,
            )
            + "\n"
        )

reachable_questions = sum(1 for row in diagnostics if row["support_reachable"])
target_count = sum(row["num_support_targets"] for row in diagnostics)
reachable_target_count = sum(row["num_reachable_support_targets"] for row in diagnostics)
(output_dir / "oracle_graph_support_diagnostics.jsonl").write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in diagnostics),
    encoding="utf-8",
)
(output_dir / "metadata.json").write_text(
    json.dumps(
        {
            "name": "oracle_graph_support",
            "deterministic": True,
            "diagnostic": True,
            "oracle_inputs": ["faq_support_audit.jsonl"],
            "seed_ranker": "bm25",
            "seed_top_k": seed_top_k,
            "target_top_k": target_top_k,
            "hop_budget": hop_budget,
            "final_top_k": final_top_k,
            "directed": directed,
            "max_context_chars_per_doc": max_context_chars,
            "num_questions": len(diagnostics),
            "support_reachability_rate": reachable_questions / len(diagnostics) if diagnostics else 0.0,
            "support_target_reachability_rate": reachable_target_count / target_count if target_count else 0.0,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
