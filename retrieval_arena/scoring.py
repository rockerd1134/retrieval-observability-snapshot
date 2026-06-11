from __future__ import annotations

import re
from collections import Counter
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def score_pair(prediction: str, reference: str, match_threshold: float = 0.5) -> dict[str, Any]:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap_count = sum((pred_counts & ref_counts).values())
    precision = overlap_count / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap_count / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    union = pred_set | ref_set
    lexical_overlap = len(pred_set & ref_set) / len(union) if union else 1.0
    return {"precision": precision, "recall": recall, "f1": f1, "lexical_overlap": lexical_overlap, "match": f1 >= match_threshold}


def score_predictions(predictions: list[dict[str, Any]], answers: dict[str, str], match_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    item_scores: list[dict[str, Any]] = []
    for row in predictions:
        qid = row["question_id"]
        metrics = score_pair(row.get("generated_answer", ""), answers[qid], match_threshold)
        item_scores.append({"question_id": qid, "question": row.get("question", ""), "reference_answer": answers[qid], "generated_answer": row.get("generated_answer", ""), **metrics})
    n = len(item_scores)
    aggregate = {
        "match_percent": sum(1 for item in item_scores if item["match"]) / n if n else 0.0,
        "mean_f1": sum(item["f1"] for item in item_scores) / n if n else 0.0,
        "mean_precision": sum(item["precision"] for item in item_scores) / n if n else 0.0,
        "mean_recall": sum(item["recall"] for item in item_scores) / n if n else 0.0,
        "mean_lexical_overlap": sum(item["lexical_overlap"] for item in item_scores) / n if n else 0.0,
        "num_questions": n,
    }
    return item_scores, aggregate