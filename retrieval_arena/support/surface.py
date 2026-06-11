from __future__ import annotations

import re
import math
from collections import Counter
from pathlib import Path
from typing import Any

from ..schemas import read_jsonl, write_jsonl
from ..snapshots import corpus_doc_id


SUPPORT_CONSTRUCTION_METHOD = "idf_weighted_answer_support_v1"
TOKEN_RE = re.compile(r"[a-z0-9_]+(?:-[a-z0-9_]+)*")
STOPWORDS = {
    "a",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "each",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "must",
    "not",
    "of",
    "on",
    "or",
    "see",
    "set",
    "that",
    "the",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "use",
    "with",
    "you",
    "your",
    "we",
    "when",
    "where",
    "which",
    "will",
}
CLEAN_QA_SURFACES = {"dedicated_faq", "troubleshooting_or_support"}
ALLOWED_QA_SURFACES = CLEAN_QA_SURFACES | {
    "extracted_questions",
    "generated_topic_questions",
    "policy_or_account_surface",
    "diagnostic_fallback",
    "unknown",
}


def write_answer_overlap_support(
    dataset_path: Path,
    *,
    top_k: int = 5,
    qa_surface_type: str = "unknown",
    supported_threshold: float = 0.6,
    partial_threshold: float = 0.25,
) -> dict[str, Any]:
    qa_surface_type = validate_qa_surface_type(qa_surface_type)
    answers = read_jsonl(dataset_path / "answers.jsonl")
    questions = {row["question_id"]: row["question"] for row in read_jsonl(dataset_path / "questions.jsonl")}
    docs = _document_tokens(dataset_path / "corpus")
    answers_by_id = {row["question_id"]: row for row in answers}
    weights = _idf_weights(docs, [_tokens(str(row.get("answer", ""))) for row in answers])
    rows: list[dict[str, Any]] = []
    for question_id in sorted(questions):
        answer_row = answers_by_id[question_id]
        answer = str(answer_row.get("answer", ""))
        answer_tokens = _tokens(answer)
        question_tokens = _tokens(str(questions.get(question_id, "")))
        scored = []
        combined_matched: set[str] = set()
        for doc in docs:
            matched = answer_tokens & doc["tokens"]
            answer_coverage = 0.0 if not answer_tokens else len(matched) / len(answer_tokens)
            weighted_answer_coverage = _weighted_coverage(answer_tokens, matched, weights)
            question_overlap = 0.0 if not question_tokens else len(question_tokens & doc["tokens"]) / len(question_tokens)
            score = weighted_answer_coverage + (0.25 * question_overlap)
            if score > 0:
                scored.append(
                    {
                        "doc_id": doc["doc_id"],
                        "score": round(score, 6),
                        "answer_token_coverage": round(answer_coverage, 6),
                        "weighted_answer_token_coverage": round(weighted_answer_coverage, 6),
                        "matched_answer_tokens": sorted(matched),
                    }
                )
        scored = sorted(scored, key=lambda item: (-float(item["score"]), str(item["doc_id"])))[:top_k]
        for doc in scored:
            combined_matched.update(str(token) for token in doc["matched_answer_tokens"])
        coverage = 0.0 if not answer_tokens else len(combined_matched) / len(answer_tokens)
        weighted = _weighted_coverage(answer_tokens, combined_matched, weights)
        label, reasons = _classify_item(
            answer,
            answer_tokens,
            coverage,
            weighted,
            supported_threshold=supported_threshold,
            partial_threshold=partial_threshold,
        )
        rows.append(
            {
                "question_id": question_id,
                "question": questions.get(question_id),
                "support_construction_method": SUPPORT_CONSTRUCTION_METHOD,
                "qa_surface_type": qa_surface_type,
                "is_clean_qa_surface": is_clean_qa_surface(qa_surface_type),
                "support_label": label,
                "answer_token_coverage": round(coverage, 6),
                "weighted_answer_token_coverage": round(weighted, 6),
                "answer_content_tokens": len(answer_tokens),
                "matched_answer_tokens": sorted(combined_matched),
                "missing_answer_tokens": sorted(answer_tokens - combined_matched),
                "top_docs": scored,
                "reasons": reasons,
            }
        )
    destination = dataset_path / "faq_support_audit.jsonl"
    write_jsonl(destination, rows)
    target_doc_ids = sorted({doc["doc_id"] for row in rows for doc in row["top_docs"]})
    label_counts = Counter(str(row["support_label"]) for row in rows)
    return {
        "support_construction_method": SUPPORT_CONSTRUCTION_METHOD,
        "support_audit_file": "faq_support_audit.jsonl",
        "supported_question_ids": sorted(row["question_id"] for row in rows if row["top_docs"]),
        "support_target_doc_ids": target_doc_ids,
        "support_target_count": len(target_doc_ids),
        "support_query_count": len(rows),
        "support_label_counts": dict(sorted(label_counts.items())),
        "qa_surface_type": qa_surface_type,
        "is_clean_qa_surface": is_clean_qa_surface(qa_surface_type),
        "top_k": top_k,
        "supported_threshold": supported_threshold,
        "partial_threshold": partial_threshold,
    }


def _document_tokens(corpus_dir: Path) -> list[dict[str, Any]]:
    docs = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(corpus_dir).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        docs.append({"doc_id": corpus_doc_id(relative), "path": relative, "tokens": _tokens(text)})
    return docs


def _tokens(text: str) -> set[str]:
    return {token for token in (match.group(0).lower() for match in TOKEN_RE.finditer(text.lower())) if token not in STOPWORDS and len(token) > 1}


def _idf_weights(docs: list[dict[str, Any]], answer_token_sets: list[set[str]]) -> dict[str, float]:
    answer_vocab: set[str] = set()
    for tokens in answer_token_sets:
        answer_vocab.update(tokens)
    document_frequency: Counter[str] = Counter()
    for doc in docs:
        doc_tokens = doc["tokens"]
        for token in answer_vocab & doc_tokens:
            document_frequency[token] += 1
    num_docs = len(docs)
    return {token: math.log((num_docs + 1) / (document_frequency[token] + 1)) + 1.0 for token in answer_vocab}


def _weighted_coverage(answer_tokens: set[str], matched_tokens: set[str], weights: dict[str, float]) -> float:
    denominator = sum(weights.get(token, 1.0) for token in answer_tokens)
    if denominator == 0:
        return 0.0
    numerator = sum(weights.get(token, 1.0) for token in matched_tokens)
    return numerator / denominator


def _classify_item(
    answer: str,
    answer_tokens: set[str],
    coverage: float,
    weighted: float,
    *,
    supported_threshold: float,
    partial_threshold: float,
) -> tuple[str, list[str]]:
    if not answer.strip() or len(answer_tokens) < 3:
        return "bad_extraction", ["answer has too few content tokens"]
    if weighted >= supported_threshold and coverage >= 0.45:
        return "supported", ["weighted answer token coverage >= 0.60 and raw coverage >= 0.45"]
    if weighted >= partial_threshold or coverage >= partial_threshold:
        return "partially_supported", ["weighted or raw answer token coverage >= 0.25"]
    if re.search(r"https?://|slack|discord|github issue|github project|zoom", answer, flags=re.IGNORECASE):
        return "external_or_meta", ["low coverage and answer references external/project metadata"]
    return "unsupported", ["answer token coverage < 0.25"]


def validate_qa_surface_type(value: str) -> str:
    if value not in ALLOWED_QA_SURFACES:
        allowed = ", ".join(sorted(ALLOWED_QA_SURFACES))
        raise ValueError(f"qa_surface_type must be one of: {allowed}; got {value!r}")
    return value


def is_clean_qa_surface(value: str) -> bool:
    return value in CLEAN_QA_SURFACES
