from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from .schemas import read_jsonl, write_jsonl
from .snapshots import support_doc_ids_from_row


DIAGNOSTIC_TRACE_SCHEMA_VERSION = "retrieval_arena.diagnostic_execution_trace.v1"


def enrich_run_diagnostics(run_dir: Path, dataset_path: Path) -> dict[str, Any]:
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        return {"distance_overlay_available": False, "diagnostic_trace_available": False}

    predictions = read_jsonl(predictions_path)
    support_targets = _support_targets_by_question(dataset_path / "faq_support_audit.jsonl")
    graph = _read_graph(dataset_path / "graph_edges.csv")
    distance_overlay_available = bool(support_targets and graph)
    if distance_overlay_available:
        predictions = [_enrich_prediction(row, support_targets.get(str(row.get("question_id")), []), graph) for row in predictions]
        write_jsonl(predictions_path, predictions)

    trace_path = run_dir / "action_traces.jsonl"
    diagnostic_trace_available = False
    if not trace_path.exists():
        write_jsonl(trace_path, [_diagnostic_trace(row) for row in predictions])
        diagnostic_trace_available = True

    return {
        "distance_overlay_available": distance_overlay_available,
        "diagnostic_trace_available": diagnostic_trace_available,
        "support_question_count": len(support_targets),
        "graph_node_count": len(graph),
    }


def _enrich_prediction(row: dict[str, Any], targets: list[str], graph: dict[str, set[str]]) -> dict[str, Any]:
    target_set = set(targets)
    distances = _distances_from_targets(targets, graph)
    context = []
    for item in row.get("retrieved_context", []) if isinstance(row.get("retrieved_context"), list) else []:
        if not isinstance(item, dict):
            context.append(item)
            continue
        enriched = dict(item)
        doc_id = enriched.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            if doc_id in distances:
                enriched["graph_distance_to_support"] = distances[doc_id]
                enriched["distance_to_support"] = distances[doc_id]
            enriched["is_support"] = doc_id in target_set
            enriched["is_evidence"] = doc_id in target_set
            enriched["support_target_doc_ids"] = targets
            enriched["evidence_doc_ids"] = targets
        context.append(enriched)
    enriched_row = dict(row)
    enriched_row["retrieved_context"] = context
    enriched_row["support_target_doc_ids"] = targets
    enriched_row["evidence_doc_ids"] = targets
    enriched_row["diagnostic_overlays"] = {
        "support_distance_overlay": {
            "available": bool(targets and graph),
            "method": "undirected_shortest_path_to_query_support_v1",
            "support_target_count": len(targets),
        }
    }
    return enriched_row


def _diagnostic_trace(row: dict[str, Any]) -> dict[str, Any]:
    docs = [
        item["doc_id"]
        for item in row.get("retrieved_context", [])
        if isinstance(item, dict) and isinstance(item.get("doc_id"), str) and item["doc_id"]
    ]
    return {
        "schema_version": DIAGNOSTIC_TRACE_SCHEMA_VERSION,
        "trace_type": "deterministic_retrieval_execution_trace",
        "question_id": row.get("question_id"),
        "actions": [
            {"step": 1, "action": "search", "budget_remaining": 2},
            {"step": 2, "action": "add_context", "budget_remaining": 1, "doc_ids": docs},
            {"step": 3, "action": "stop", "budget_remaining": 0},
        ],
        "final_context_doc_ids": docs,
    }


def _support_targets_by_question(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    targets: dict[str, list[str]] = {}
    for line_number, row in enumerate(read_jsonl(path), start=1):
        question_id = row.get("question_id")
        if isinstance(question_id, str) and question_id:
            targets[question_id] = support_doc_ids_from_row(row, line_number=line_number)
    return targets


def _read_graph(path: Path) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = row.get("source")
            target = row.get("target")
            if not source or not target:
                continue
            graph[source].add(target)
            graph[target].add(source)
    return dict(graph)


def _distances_from_targets(targets: list[str], graph: dict[str, set[str]]) -> dict[str, int]:
    distances: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for target in targets:
        if target in distances:
            continue
        distances[target] = 0
        queue.append((target, 0))
    while queue:
        node, distance = queue.popleft()
        for neighbor in sorted(graph.get(node, set())):
            if neighbor in distances:
                continue
            distances[neighbor] = distance + 1
            queue.append((neighbor, distance + 1))
    return distances
