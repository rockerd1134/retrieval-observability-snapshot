from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
POLICY_NAME = "random_seed_graph_traversal_v1"


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


def stable_question_seed(rng_seed: int, question_id: str) -> int:
    digest = hashlib.blake2b(f"{rng_seed}:{question_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def document_frequencies(corpus_tokens: dict[str, list[str]]) -> Counter[str]:
    doc_freq: Counter[str] = Counter()
    for doc_tokens in corpus_tokens.values():
        doc_freq.update(set(doc_tokens))
    return doc_freq


def bm25_score(query_tokens: list[str], doc_tokens: list[str], *, doc_freq: Counter[str], num_docs: int, avg_len: float) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    tf = Counter(doc_tokens)
    query = Counter(query_tokens)
    k1 = 1.5
    b = 0.75
    score = 0.0
    for token, query_count in query.items():
        if tf[token] == 0:
            continue
        idf = math.log(1.0 + (num_docs - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
        denom = tf[token] + k1 * (1.0 - b + b * (len(doc_tokens) / avg_len if avg_len else 0.0))
        score += query_count * idf * ((tf[token] * (k1 + 1.0)) / denom)
    return score


def read_edges(input_dir: Path, nodes: set[str], *, directed: bool) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    edge_path = input_dir / "graph_edges.csv"
    if not edge_path.exists():
        return edges
    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source", "")
            target = row.get("target", "")
            if source in nodes and target in nodes and source != target:
                edges[source].add(target)
                if not directed:
                    edges[target].add(source)
    return edges


def graph_distances(seed: str, edges: dict[str, set[str]], *, max_hops: int, neighbor_budget: int) -> dict[str, int]:
    distances = {seed: 0}
    queue: deque[tuple[str, int]] = deque([(seed, 0)])
    while queue:
        doc_id, distance = queue.popleft()
        if distance >= max_hops:
            continue
        neighbors = sorted(edges.get(doc_id, set()))
        if neighbor_budget > 0:
            neighbors = neighbors[:neighbor_budget]
        for neighbor in neighbors:
            if neighbor in distances:
                continue
            distances[neighbor] = distance + 1
            queue.append((neighbor, distance + 1))
    return distances


def sample_seed_docs(doc_ids: list[str], *, question_id: str, rng_seed: int, num_trials: int) -> list[str]:
    rng = random.Random(stable_question_seed(rng_seed, question_id))
    if not doc_ids:
        return []
    if num_trials <= len(doc_ids):
        return rng.sample(doc_ids, num_trials)
    sampled = list(doc_ids)
    while len(sampled) < num_trials:
        sampled.append(rng.choice(doc_ids))
    return sampled


def variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def run(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_config(input_dir / "config.yaml")
    num_trials = max(1, int(config.get("num_trials", 8)))
    rng_seed = int(config.get("rng_seed", 20260504))
    max_hops = max(0, int(config.get("max_hops", 2)))
    neighbor_budget = max(0, int(config.get("neighbor_budget", 12)))
    candidate_budget = max(1, int(config.get("candidate_budget", 60)))
    final_top_k = max(1, int(config.get("final_top_k", int(config.get("top_k", 5)))))
    distance_penalty = float(config.get("distance_penalty", 0.10))
    max_context_chars = max(0, int(config.get("max_context_chars_per_doc", 1200)))
    directed = bool(config.get("directed", False))
    experiment_id = str(config.get("experiment_id", "E010"))
    diagnostic_role = str(config.get("diagnostic_role", "random_seed_graph_navigability"))

    questions = read_jsonl(input_dir / "questions.jsonl")
    corpus = {path.stem: path.read_text(encoding="utf-8") for path in (input_dir / "corpus").glob("*.md")}
    doc_ids = sorted(corpus)
    nodes = set(doc_ids)
    edges = read_edges(input_dir, nodes, directed=directed)
    corpus_tokens = {doc_id: tokens(text) for doc_id, text in corpus.items()}
    doc_freq = document_frequencies(corpus_tokens)
    num_docs = len(corpus_tokens)
    avg_len = sum(len(doc_tokens) for doc_tokens in corpus_tokens.values()) / num_docs if num_docs else 0.0

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            qid = str(question["question_id"])
            query_tokens = tokens(str(question["question"]))
            relevance_scores = {
                doc_id: bm25_score(query_tokens, doc_tokens, doc_freq=doc_freq, num_docs=num_docs, avg_len=avg_len)
                for doc_id, doc_tokens in corpus_tokens.items()
            }
            sampled_seed_docs = sample_seed_docs(doc_ids, question_id=qid, rng_seed=rng_seed, num_trials=num_trials)
            trials: list[dict[str, Any]] = []
            for trial_index, seed_doc_id in enumerate(sampled_seed_docs):
                distances = graph_distances(seed_doc_id, edges, max_hops=max_hops, neighbor_budget=neighbor_budget)
                candidates = set(distances) & nodes
                if len(candidates) > candidate_budget:
                    ranked_for_budget = sorted(
                        candidates,
                        key=lambda doc_id: (-relevance_scores.get(doc_id, 0.0), distances.get(doc_id, max_hops + 1), doc_id),
                    )
                    candidates = set(ranked_for_budget[:candidate_budget])
                    candidates.add(seed_doc_id)
                ranked = sorted(
                    candidates,
                    key=lambda doc_id: (
                        -(relevance_scores.get(doc_id, 0.0) - distance_penalty * distances.get(doc_id, max_hops + 1)),
                        distances.get(doc_id, max_hops + 1),
                        doc_id,
                    ),
                )
                final_docs = ranked[:final_top_k]
                final_scores = {
                    doc_id: relevance_scores.get(doc_id, 0.0) - distance_penalty * distances.get(doc_id, max_hops + 1)
                    for doc_id in final_docs
                }
                trials.append(
                    {
                        "trial_index": trial_index,
                        "seed_doc_id": seed_doc_id,
                        "candidate_count": len(candidates),
                        "reached_candidate_doc_ids": sorted(candidates),
                        "final_doc_ids": final_docs,
                        "graph_distances": {doc_id: distances[doc_id] for doc_id in sorted(distances) if doc_id in nodes},
                        "final_scores": {doc_id: float(final_scores[doc_id]) for doc_id in final_docs},
                        "trial_relevance_score": float(sum(final_scores.values()) / len(final_scores) if final_scores else 0.0),
                    }
                )
            best_trial = max(trials, key=lambda trial: (trial["trial_relevance_score"], -int(trial["trial_index"]))) if trials else None
            selected = list(best_trial["final_doc_ids"] if best_trial else [])
            context = [
                {
                    "doc_id": doc_id,
                    "text": answer_text(corpus[doc_id], max_chars=max_context_chars),
                    "score": float((best_trial or {}).get("final_scores", {}).get(doc_id, 0.0)),
                    "bm25_score": float(relevance_scores.get(doc_id, 0.0)),
                    "source": "random_seed_graph_traversal",
                    "selection_role": "random_seed" if best_trial and doc_id == best_trial["seed_doc_id"] else "traversed_candidate",
                    "graph_distance": (best_trial or {}).get("graph_distances", {}).get(doc_id),
                    "rank": rank,
                }
                for rank, doc_id in enumerate(selected, start=1)
            ]
            trial_scores = [float(trial["trial_relevance_score"]) for trial in trials]
            handle.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "question": question["question"],
                        "generated_answer": " ".join(item["text"] for item in context if item["text"]),
                        "retrieved_context": context,
                        "random_seed_graph_traversal": {
                            "experiment_id": experiment_id,
                            "diagnostic_role": diagnostic_role,
                            "policy": POLICY_NAME,
                            "rng_seed": rng_seed,
                            "question_rng_seed": stable_question_seed(rng_seed, qid),
                            "num_trials": num_trials,
                            "sampled_seed_doc_ids": sampled_seed_docs,
                            "max_hops": max_hops,
                            "neighbor_budget": neighbor_budget,
                            "candidate_budget": candidate_budget,
                            "final_top_k": final_top_k,
                            "distance_penalty": distance_penalty,
                            "directed": directed,
                            "ranking_policy": "bm25_minus_graph_distance_penalty",
                            "best_trial_selection_rule": "max_mean_final_bm25_minus_distance_penalty_then_lowest_trial_index",
                            "best_trial_index": best_trial["trial_index"] if best_trial else None,
                            "best_trial_seed_doc_id": best_trial["seed_doc_id"] if best_trial else "",
                            "best_trial_relevance_score": best_trial["trial_relevance_score"] if best_trial else 0.0,
                            "mean_trial_relevance_score": mean(trial_scores) if trial_scores else 0.0,
                            "trial_relevance_score_variance": variance(trial_scores),
                            "trials": trials,
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "name": "rag_random_seed_graph_traversal",
                "experiment_id": experiment_id,
                "diagnostic_role": diagnostic_role,
                "deterministic": False,
                "stochastic_diagnostic": True,
                "rng_seed": rng_seed,
                "policy": POLICY_NAME,
                "uses_llm": False,
                "provider": "none",
                "seed_policy": "uniform_random_pages_per_question",
                "seed_ranker": "random_uniform",
                "final_ranker": "bm25_minus_graph_distance_penalty",
                "ranking_policy": "bm25_minus_graph_distance_penalty",
                "best_trial_selection_rule": "max_mean_final_bm25_minus_distance_penalty_then_lowest_trial_index",
                "num_trials": num_trials,
                "max_hops": max_hops,
                "neighbor_budget": neighbor_budget,
                "candidate_budget": candidate_budget,
                "final_top_k": final_top_k,
                "distance_penalty": distance_penalty,
                "directed": directed,
                "max_context_chars_per_doc": max_context_chars,
                "model_id": "",
                "embedding_backend": "",
                "llm_provider": "none",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run(Path("/input"), Path("/output"))
