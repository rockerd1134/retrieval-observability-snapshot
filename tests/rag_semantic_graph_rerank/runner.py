from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
POLICY_NAME = "semantic_seeded_graph_rerank_v1"


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


def signed_hash(token: str, *, dimensions: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    index = value % dimensions
    sign = 1.0 if (value >> 63) == 0 else -1.0
    return index, sign


def document_frequencies(corpus_tokens: dict[str, list[str]]) -> Counter[str]:
    doc_freq: Counter[str] = Counter()
    for doc_tokens in corpus_tokens.values():
        doc_freq.update(set(doc_tokens))
    return doc_freq


def hashed_tfidf_vector(doc_tokens: list[str], *, doc_freq: Counter[str], num_docs: int, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    token_counts = Counter(doc_tokens)
    if not token_counts:
        return vector
    for token, count in token_counts.items():
        idf = math.log(1.0 + (num_docs + 1.0) / (doc_freq[token] + 1.0))
        index, sign = signed_hash(token, dimensions=dimensions)
        vector[index] += sign * (1.0 + math.log(count)) * idf
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def set_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_lock(lock_path: Path, model_path: Path, *, expected_model_id: str) -> dict[str, Any]:
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing model lock: {lock_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing local model snapshot: {model_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    matches = [model for model in lock.get("models", []) if model.get("model_id") == expected_model_id]
    if not matches:
        raise RuntimeError(f"Model lock does not contain expected model_id: {expected_model_id}")
    model = matches[0]
    if not model.get("offline_required"):
        raise RuntimeError("Model lock must require offline execution.")
    mismatches: list[str] = []
    for row in model.get("files", []):
        path = model_path / row["path"]
        if not path.exists():
            mismatches.append(f"missing:{row['path']}")
            continue
        if path.stat().st_size != int(row["size_bytes"]) or hash_file(path) != row["sha256"]:
            mismatches.append(f"hash:{row['path']}")
    if mismatches:
        raise RuntimeError(f"Model snapshot does not match lock ({len(mismatches)} mismatch(es)): {', '.join(mismatches[:5])}")
    return model


def sentence_transformer_vectors(texts: list[str], *, model_path: Path, batch_size: int) -> tuple[list[list[float]], int]:
    set_offline_environment()
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("sentence_transformers is required for embedding_backend=sentence_transformers_local") from exc
    model = SentenceTransformer(str(model_path))
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    vectors = [list(map(float, vector)) for vector in embeddings]
    return vectors, len(vectors[0]) if vectors else 0


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


def graph_distances(seeds: list[str], edges: dict[str, set[str]], *, hop_budget: int, neighbor_budget: int) -> dict[str, int]:
    distances: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for seed in seeds:
        if seed not in distances:
            distances[seed] = 0
            queue.append((seed, 0))
    while queue:
        doc_id, distance = queue.popleft()
        if distance >= hop_budget:
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


def normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def pagerank(nodes: set[str], edges: dict[str, set[str]], *, iterations: int, damping: float) -> dict[str, float]:
    if not nodes:
        return {}
    n = len(nodes)
    ranks = {node: 1.0 / n for node in nodes}
    base = (1.0 - damping) / n
    for _ in range(iterations):
        next_ranks = {node: base for node in nodes}
        dangling = sum(ranks[node] for node in nodes if not edges.get(node))
        dangling_share = damping * dangling / n
        for node in nodes:
            next_ranks[node] += dangling_share
        for source, targets in edges.items():
            if source not in nodes or not targets:
                continue
            share = damping * ranks[source] / len(targets)
            for target in targets:
                if target in nodes:
                    next_ranks[target] += share
        ranks = next_ranks
    return normalize(ranks)


def build_vectors(
    questions: list[dict[str, Any]],
    corpus: dict[str, str],
    *,
    embedding_backend: str,
    dimensions: int,
    model_id: str,
    model_lock_path: Path,
    model_cache_path: Path,
    batch_size: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]], int, dict[str, Any]]:
    model_provenance: dict[str, Any] = {}
    if embedding_backend == "local_hashing_tfidf":
        corpus_tokens = {doc_id: tokens(text) for doc_id, text in corpus.items()}
        doc_freq = document_frequencies(corpus_tokens)
        num_docs = len(corpus_tokens)
        doc_vectors = {
            doc_id: hashed_tfidf_vector(doc_tokens, doc_freq=doc_freq, num_docs=num_docs, dimensions=dimensions)
            for doc_id, doc_tokens in corpus_tokens.items()
        }
        question_vectors = {
            question["question_id"]: hashed_tfidf_vector(tokens(question["question"]), doc_freq=doc_freq, num_docs=num_docs, dimensions=dimensions)
            for question in questions
        }
        return doc_vectors, question_vectors, dimensions, model_provenance
    if embedding_backend == "sentence_transformers_local":
        model_provenance = validate_model_lock(model_lock_path, model_cache_path, expected_model_id=model_id)
        doc_ids = sorted(corpus)
        doc_embeddings, dimensions = sentence_transformer_vectors([corpus[doc_id] for doc_id in doc_ids], model_path=model_cache_path, batch_size=batch_size)
        question_embeddings, _ = sentence_transformer_vectors([question["question"] for question in questions], model_path=model_cache_path, batch_size=batch_size)
        return dict(zip(doc_ids, doc_embeddings)), {question["question_id"]: vector for question, vector in zip(questions, question_embeddings)}, dimensions, model_provenance
    raise RuntimeError(f"Unknown embedding backend: {embedding_backend}")


def run(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_config(input_dir / "config.yaml")
    seed_top_k = max(1, int(config.get("seed_top_k", config.get("top_k", 5))))
    hop_budget = max(0, int(config.get("graph_hops", config.get("hop_budget", 1))))
    neighbor_budget = max(0, int(config.get("neighbor_budget", 12)))
    candidate_budget = max(1, int(config.get("candidate_budget", 60)))
    final_top_k = max(1, int(config.get("final_top_k", config.get("top_k", 5))))
    max_context_chars = max(0, int(config.get("max_context_chars_per_doc", 1200)))
    distance_penalty = float(config.get("distance_penalty", 0.02))
    directed = bool(config.get("directed", False))
    use_pagerank_prior = bool(config.get("use_pagerank_prior", False))
    pagerank_weight = float(config.get("pagerank_weight", 0.0))
    pagerank_iterations = max(1, int(config.get("pagerank_iterations", 30)))
    pagerank_damping = float(config.get("pagerank_damping", 0.85))
    dimensions = max(64, int(config.get("embedding_dimensions", 512)))
    model_id = str(config.get("model_id", "local-hashing-tfidf-v1"))
    embedding_backend = str(config.get("embedding_backend", "local_hashing_tfidf"))
    model_revision = str(config.get("model_revision", ""))
    model_lock_path = Path(str(config.get("model_lock_path", "/model_provenance/embedding_models.lock.json")))
    model_cache_path = Path(str(config.get("model_cache_path", "/models/minilm")))
    batch_size = max(1, int(config.get("batch_size", 32)))
    variant_id = str(config.get("variant_id", ""))

    questions = read_jsonl(input_dir / "questions.jsonl")
    corpus = {path.stem: path.read_text(encoding="utf-8") for path in (input_dir / "corpus").glob("*.md")}
    nodes = set(corpus)
    edges = read_edges(input_dir, nodes, directed=directed)
    doc_vectors, question_vectors, dimensions, model_provenance = build_vectors(
        questions,
        corpus,
        embedding_backend=embedding_backend,
        dimensions=dimensions,
        model_id=model_id,
        model_lock_path=model_lock_path,
        model_cache_path=model_cache_path,
        batch_size=batch_size,
    )
    model_revision = str(model_provenance.get("revision", model_revision))
    pagerank_scores = pagerank(nodes, edges, iterations=pagerank_iterations, damping=pagerank_damping) if use_pagerank_prior else {}

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            query_vector = question_vectors[question["question_id"]]
            semantic_scores = {doc_id: cosine(query_vector, doc_vector) for doc_id, doc_vector in doc_vectors.items()}
            semantic_ranked = sorted(semantic_scores.items(), key=lambda item: (-item[1], item[0]))
            seed_ids = [doc_id for doc_id, _ in semantic_ranked[:seed_top_k]]
            distances = graph_distances(seed_ids, edges, hop_budget=hop_budget, neighbor_budget=neighbor_budget)
            candidates = set(distances) & nodes
            if len(candidates) > candidate_budget:
                seed_set = set(seed_ids)
                ranked_expanded = [
                    doc_id
                    for doc_id in sorted(
                        candidates - seed_set,
                        key=lambda doc_id: (-semantic_scores.get(doc_id, 0.0), distances.get(doc_id, hop_budget + 1), doc_id),
                    )
                ]
                candidates = seed_set | set(ranked_expanded[: max(0, candidate_budget - len(seed_set))])
            ranked = sorted(
                candidates,
                key=lambda doc_id: (
                    -(
                        semantic_scores.get(doc_id, 0.0)
                        - distance_penalty * distances.get(doc_id, hop_budget + 1)
                        + pagerank_weight * pagerank_scores.get(doc_id, 0.0)
                    ),
                    distances.get(doc_id, hop_budget + 1),
                    doc_id,
                ),
            )
            selected = ranked[:final_top_k]
            context = []
            for rank, doc_id in enumerate(selected, start=1):
                final_score = (
                    semantic_scores.get(doc_id, 0.0)
                    - distance_penalty * distances.get(doc_id, hop_budget + 1)
                    + pagerank_weight * pagerank_scores.get(doc_id, 0.0)
                )
                context.append(
                    {
                        "doc_id": doc_id,
                        "text": answer_text(corpus[doc_id], max_chars=max_context_chars),
                        "score": float(final_score),
                        "semantic_score": float(semantic_scores.get(doc_id, 0.0)),
                        "source": "semantic_graph_rerank",
                        "selection_role": "seed" if doc_id in seed_ids else "expanded_neighbor",
                        "graph_distance": distances.get(doc_id),
                        "pagerank": float(pagerank_scores.get(doc_id, 0.0)),
                        "rank": rank,
                    }
                )
            expanded_ids = sorted(doc_id for doc_id, distance in distances.items() if distance > 0 and doc_id in nodes)
            handle.write(
                json.dumps(
                    {
                        "question_id": question["question_id"],
                        "question": question["question"],
                        "generated_answer": " ".join(item["text"] for item in context if item["text"]),
                        "retrieved_context": context,
                        "semantic_graph_rerank": {
                            "seed_ranker": "minilm_embedding",
                            "expansion_policy": POLICY_NAME,
                            "final_ranker": "minilm_embedding_plus_graph_distance",
                            "seed_doc_ids": seed_ids,
                            "expanded_candidate_doc_ids": expanded_ids,
                            "expanded_candidate_count": len(expanded_ids),
                            "candidate_count": len(candidates),
                            "final_context_doc_ids": selected,
                            "graph_distances": {doc_id: distances[doc_id] for doc_id in sorted(distances) if doc_id in nodes},
                            "dropped_candidate_count": max(0, len(candidates) - len(selected)),
                            "pagerank_prior_enabled": use_pagerank_prior,
                            "variant_id": variant_id,
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "name": "rag_semantic_graph_rerank",
                "experiment_id": "E009",
                "variant_id": variant_id,
                "deterministic": True,
                "policy": POLICY_NAME,
                "uses_llm": False,
                "provider": "none",
                "seed_ranker": "minilm_embedding",
                "expansion_policy": POLICY_NAME,
                "final_ranker": "minilm_embedding_plus_graph_distance",
                "embedding_backend": embedding_backend,
                "model_id": model_id,
                "model_revision": model_revision,
                "model_lock_path": str(model_lock_path) if embedding_backend == "sentence_transformers_local" else "",
                "model_cache_path": str(model_cache_path) if embedding_backend == "sentence_transformers_local" else "",
                "model_provenance": model_provenance,
                "embedding_dimensions": dimensions,
                "seed_top_k": seed_top_k,
                "graph_hops": hop_budget,
                "neighbor_budget": neighbor_budget,
                "candidate_budget": candidate_budget,
                "final_top_k": final_top_k,
                "distance_penalty": distance_penalty,
                "directed": directed,
                "pagerank_prior_enabled": use_pagerank_prior,
                "pagerank_weight": pagerank_weight if use_pagerank_prior else 0.0,
                "pagerank_iterations": pagerank_iterations if use_pagerank_prior else 0,
                "pagerank_damping": pagerank_damping if use_pagerank_prior else 0.0,
                "hits_prior_enabled": False,
                "max_context_chars_per_doc": max_context_chars,
                "offline_required": bool(model_provenance.get("offline_required", False)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run(Path("/input"), Path("/output"))
