from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
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


input_dir = Path("/input")
output_dir = Path("/output")
output_dir.mkdir(parents=True, exist_ok=True)

config = read_config(input_dir / "config.yaml")
top_k = max(1, int(config.get("top_k", 5)))
max_context_chars = max(0, int(config.get("max_context_chars_per_doc", 1500)))
dimensions = max(64, int(config.get("embedding_dimensions", 512)))
model_id = str(config.get("model_id", "local-hashing-tfidf-v1"))
embedding_backend = str(config.get("embedding_backend", "local_hashing_tfidf"))
model_revision = str(config.get("model_revision", ""))
model_lock_path = Path(str(config.get("model_lock_path", "/model_provenance/embedding_models.lock.json")))
model_cache_path = Path(str(config.get("model_cache_path", "/models/minilm")))
batch_size = max(1, int(config.get("batch_size", 32)))

questions = read_jsonl(input_dir / "questions.jsonl")
corpus = {path.stem: path.read_text(encoding="utf-8") for path in (input_dir / "corpus").glob("*.md")}
model_provenance: dict[str, Any] = {}

if embedding_backend == "local_hashing_tfidf":
    corpus_tokens = {doc_id: tokens(text) for doc_id, text in corpus.items()}
    doc_freq = document_frequencies(corpus_tokens)
    num_docs = len(corpus_tokens)
    doc_vectors = {
        doc_id: hashed_tfidf_vector(doc_tokens, doc_freq=doc_freq, num_docs=num_docs, dimensions=dimensions)
        for doc_id, doc_tokens in corpus_tokens.items()
    }
elif embedding_backend == "sentence_transformers_local":
    model_provenance = validate_model_lock(model_lock_path, model_cache_path, expected_model_id=model_id)
    model_revision = str(model_provenance.get("revision", model_revision))
    doc_ids = sorted(corpus)
    doc_embeddings, dimensions = sentence_transformer_vectors([corpus[doc_id] for doc_id in doc_ids], model_path=model_cache_path, batch_size=batch_size)
    doc_vectors = dict(zip(doc_ids, doc_embeddings))
    question_embeddings, _ = sentence_transformer_vectors([question["question"] for question in questions], model_path=model_cache_path, batch_size=batch_size)
    question_vectors = {question["question_id"]: vector for question, vector in zip(questions, question_embeddings)}
else:
    raise RuntimeError(f"Unknown embedding backend: {embedding_backend}")

with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for question in questions:
        if embedding_backend == "local_hashing_tfidf":
            query_vector = hashed_tfidf_vector(tokens(question["question"]), doc_freq=doc_freq, num_docs=num_docs, dimensions=dimensions)
        else:
            query_vector = question_vectors[question["question_id"]]
        ranked = sorted(
            ((cosine(query_vector, doc_vector), doc_id) for doc_id, doc_vector in doc_vectors.items()),
            key=lambda item: (-item[0], item[1]),
        )[:top_k]
        context = [
            {
                "doc_id": doc_id,
                "text": answer_text(corpus[doc_id], max_chars=max_context_chars),
                "score": float(score),
                "source": "embedding_topk",
                "rank": rank,
            }
            for rank, (score, doc_id) in enumerate(ranked, start=1)
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
            "name": "rag_embedding_topk",
            "deterministic": True,
            "ranker": "cosine",
            "embedding_backend": embedding_backend,
            "model_id": model_id,
            "model_revision": model_revision,
            "model_lock_path": str(model_lock_path) if embedding_backend == "sentence_transformers_local" else "",
            "model_cache_path": str(model_cache_path) if embedding_backend == "sentence_transformers_local" else "",
            "model_provenance": model_provenance,
            "embedding_dimensions": dimensions,
            "top_k": top_k,
            "max_context_chars_per_doc": max_context_chars,
            "offline_required": bool(model_provenance.get("offline_required", False)),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
