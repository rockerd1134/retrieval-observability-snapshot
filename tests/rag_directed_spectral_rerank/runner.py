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


def read_directed_edges(input_dir: Path, nodes: set[str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    edge_path = input_dir / "graph_edges.csv"
    if not edge_path.exists():
        return outgoing, incoming
    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source", "")
            target = row.get("target", "")
            if source in nodes and target in nodes and source != target:
                outgoing[source].add(target)
                incoming[target].add(source)
    return outgoing, incoming


def normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def l2_normalize(values: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in values.values()))
    if norm == 0.0:
        return values
    return {key: value / norm for key, value in values.items()}


def pagerank(nodes: set[str], outgoing: dict[str, set[str]], *, iterations: int, damping: float) -> dict[str, float]:
    if not nodes:
        return {}
    n = len(nodes)
    ranks = {node: 1.0 / n for node in nodes}
    base = (1.0 - damping) / n
    for _ in range(iterations):
        next_ranks = {node: base for node in nodes}
        dangling = sum(ranks[node] for node in nodes if not outgoing.get(node))
        dangling_share = damping * dangling / n
        for node in nodes:
            next_ranks[node] += dangling_share
        for source, targets in outgoing.items():
            if source not in nodes or not targets:
                continue
            share = damping * ranks[source] / len(targets)
            for target in targets:
                if target in nodes:
                    next_ranks[target] += share
        ranks = next_ranks
    return normalize(ranks)


def hits(nodes: set[str], outgoing: dict[str, set[str]], incoming: dict[str, set[str]], *, iterations: int) -> tuple[dict[str, float], dict[str, float]]:
    authority = {node: 1.0 for node in nodes}
    hub = {node: 1.0 for node in nodes}
    for _ in range(iterations):
        authority = {
            node: sum(hub.get(source, 0.0) for source in incoming.get(node, set()))
            for node in nodes
        }
        authority = l2_normalize(authority)
        hub = {
            node: sum(authority.get(target, 0.0) for target in outgoing.get(node, set()))
            for node in nodes
        }
        hub = l2_normalize(hub)
    return normalize(authority), normalize(hub)


def candidate_set(seed_ids: list[str], outgoing: dict[str, set[str]], incoming: dict[str, set[str]], nodes: set[str], *, high_authority: list[str], high_pagerank: list[str], neighbor_budget: int) -> set[str]:
    candidates = set(seed_ids)
    for seed in seed_ids:
        candidates.update(sorted(outgoing.get(seed, set()))[:neighbor_budget])
        candidates.update(sorted(incoming.get(seed, set()))[:neighbor_budget])
    candidates.update(high_authority)
    candidates.update(high_pagerank)
    return candidates & nodes


def run(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_config(input_dir / "config.yaml")
    seed_top_k = max(1, int(config.get("seed_top_k", 5)))
    final_top_k = max(1, int(config.get("final_top_k", config.get("top_k", 5))))
    neighbor_budget = max(0, int(config.get("directed_neighbor_budget", 8)))
    global_prior_top_k = max(0, int(config.get("global_prior_top_k", 10)))
    max_context_chars = max(0, int(config.get("max_context_chars_per_doc", 1200)))
    pagerank_weight = float(config.get("pagerank_weight", 0.18))
    authority_weight = float(config.get("authority_weight", 0.22))
    hub_weight = float(config.get("hub_weight", -0.04))
    forward_link_bonus = float(config.get("forward_link_bonus", 0.08))
    backward_link_bonus = float(config.get("backward_link_bonus", 0.03))
    iterations = max(1, int(config.get("spectral_iterations", 40)))
    damping = float(config.get("pagerank_damping", 0.85))

    questions = read_jsonl(input_dir / "questions.jsonl")
    corpus = {path.stem: path.read_text(encoding="utf-8") for path in (input_dir / "corpus").glob("*.md")}
    nodes = set(corpus)
    corpus_tokens = {doc_id: tokens(text) for doc_id, text in corpus.items()}
    outgoing, incoming = read_directed_edges(input_dir, nodes)
    pr = pagerank(nodes, outgoing, iterations=iterations, damping=damping)
    authority, hub = hits(nodes, outgoing, incoming, iterations=iterations)
    high_authority = [doc_id for doc_id, _ in sorted(authority.items(), key=lambda item: (-item[1], item[0]))[:global_prior_top_k]]
    high_pagerank = [doc_id for doc_id, _ in sorted(pr.items(), key=lambda item: (-item[1], item[0]))[:global_prior_top_k]]

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            query_tokens = tokens(question["question"])
            all_bm25 = bm25_scores(query_tokens, corpus_tokens)
            bm25_norm = normalize({doc_id: score for score, doc_id in all_bm25})
            seed_ids = [doc_id for _, doc_id in all_bm25[:seed_top_k]]
            candidates = candidate_set(
                seed_ids,
                outgoing,
                incoming,
                nodes,
                high_authority=high_authority,
                high_pagerank=high_pagerank,
                neighbor_budget=neighbor_budget,
            )
            forward_neighbors = set().union(*(outgoing.get(seed, set()) for seed in seed_ids)) if seed_ids else set()
            backward_neighbors = set().union(*(incoming.get(seed, set()) for seed in seed_ids)) if seed_ids else set()
            ranked = sorted(
                candidates,
                key=lambda doc_id: (
                    -(
                        bm25_norm.get(doc_id, 0.0)
                        + pagerank_weight * pr.get(doc_id, 0.0)
                        + authority_weight * authority.get(doc_id, 0.0)
                        + hub_weight * hub.get(doc_id, 0.0)
                        + (forward_link_bonus if doc_id in forward_neighbors else 0.0)
                        + (backward_link_bonus if doc_id in backward_neighbors else 0.0)
                    ),
                    doc_id,
                ),
            )
            selected = ranked[:final_top_k]
            context = []
            for rank, doc_id in enumerate(selected, start=1):
                spectral_score = (
                    pagerank_weight * pr.get(doc_id, 0.0)
                    + authority_weight * authority.get(doc_id, 0.0)
                    + hub_weight * hub.get(doc_id, 0.0)
                    + (forward_link_bonus if doc_id in forward_neighbors else 0.0)
                    + (backward_link_bonus if doc_id in backward_neighbors else 0.0)
                )
                context.append(
                    {
                        "doc_id": doc_id,
                        "text": answer_text(corpus[doc_id], max_chars=max_context_chars),
                        "score": float(bm25_norm.get(doc_id, 0.0) + spectral_score),
                        "source": "directed_spectral_rerank",
                        "selection_role": "seed" if doc_id in seed_ids else "directed_or_spectral_candidate",
                        "rank": rank,
                        "bm25_score_norm": float(bm25_norm.get(doc_id, 0.0)),
                        "pagerank": float(pr.get(doc_id, 0.0)),
                        "authority": float(authority.get(doc_id, 0.0)),
                        "hub": float(hub.get(doc_id, 0.0)),
                        "spectral_prior_score": float(spectral_score),
                        "is_forward_neighbor": doc_id in forward_neighbors,
                        "is_backward_neighbor": doc_id in backward_neighbors,
                    }
                )
            handle.write(
                json.dumps(
                    {
                        "question_id": question["question_id"],
                        "question": question["question"],
                        "generated_answer": " ".join(item["text"] for item in context if item["text"]),
                        "retrieved_context": context,
                        "directed_spectral": {
                            "seed_doc_ids": seed_ids,
                            "candidate_count": len(candidates),
                            "policy": "bm25_plus_directed_pagerank_hits_v1",
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "name": "rag_directed_spectral_rerank",
                "experiment_id": "E008",
                "deterministic": True,
                "policy": "bm25_plus_directed_pagerank_hits_v1",
                "uses_llm": False,
                "provider": "none",
                "seed_ranker": "bm25",
                "final_ranker": "bm25_plus_directed_pagerank_hits",
                "seed_top_k": seed_top_k,
                "final_top_k": final_top_k,
                "directed_neighbor_budget": neighbor_budget,
                "global_prior_top_k": global_prior_top_k,
                "pagerank_weight": pagerank_weight,
                "authority_weight": authority_weight,
                "hub_weight": hub_weight,
                "forward_link_bonus": forward_link_bonus,
                "backward_link_bonus": backward_link_bonus,
                "spectral_iterations": iterations,
                "pagerank_damping": damping,
                "max_context_chars_per_doc": max_context_chars,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run(Path("/input"), Path("/output"))
