from __future__ import annotations

import argparse
import csv
import http.client
import json
import math
import re
import socket
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
TRACE_SCHEMA_VERSION = "e007b.action_trace.v1"
COMPATIBLE_ACTION_SCHEMA_VERSION = "e007a.action_trace.v1"
POLICY_NAME = "mock_llm_guided_fixture_controller_v1"
PROVIDER_POLICY_NAME = "bounded_provider_action_controller_v1"
ALLOWED_ACTIONS = {"search", "inspect_page", "follow_link", "add_context", "stop"}
OPENAI_COMPATIBLE_PROVIDER = "openai_compatible"
FAKE_SEQUENCE_PROVIDER = "fake_sequence"
LOCAL_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ProviderActionError(ValueError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


@dataclass(frozen=True)
class SearchConfig:
    controller: str = "mock"
    search_top_k: int = 5
    inspect_budget: int = 8
    follow_budget_per_page: int = 2
    max_actions: int = 30
    final_top_k: int = 5
    max_context_chars_per_doc: int = 1200
    directed: bool = False
    provider: dict[str, Any] | None = None


def context_decision_required(inspected: set[str], context_doc_ids: list[str]) -> bool:
    return len(inspected) >= 3 and not context_doc_ids


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def parse_config(raw: dict[str, Any]) -> SearchConfig:
    provider = raw.get("provider")
    if provider is not None and not isinstance(provider, dict):
        provider = {"provider": str(provider)}
    return SearchConfig(
        controller=str(raw.get("controller", "mock")),
        search_top_k=max(1, int(raw.get("search_top_k", raw.get("top_k", 5)))),
        inspect_budget=max(1, int(raw.get("inspect_budget", 8))),
        follow_budget_per_page=max(0, int(raw.get("follow_budget_per_page", 2))),
        max_actions=max(3, int(raw.get("max_actions", 30))),
        final_top_k=max(1, int(raw.get("final_top_k", raw.get("top_k", 5)))),
        max_context_chars_per_doc=max(0, int(raw.get("max_context_chars_per_doc", 1200))),
        directed=bool(raw.get("directed", False)),
        provider=provider,
    )


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
    query = Counter(query_tokens)
    scored: list[tuple[float, str]] = []
    for doc_id in candidates if candidates is not None else set(corpus_tokens):
        doc_tokens = corpus_tokens.get(doc_id, [])
        tf = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        score = 0.0
        for token, query_count in query.items():
            if tf[token] == 0:
                continue
            idf = math.log(1.0 + (num_docs - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            denom = tf[token] + 1.5 * (1.0 - 0.75 + 0.75 * (doc_len / avg_len if avg_len else 0.0))
            score += query_count * idf * ((tf[token] * 2.5) / denom)
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


def trace_action(actions: list[dict[str, Any]], action: str, *, budget_remaining: int, details: dict[str, Any]) -> None:
    actions.append({"step": len(actions) + 1, "action": action, "budget_remaining": budget_remaining, "details": details})


def provider_manifest(config: SearchConfig) -> dict[str, Any]:
    provider = config.provider or {}
    provider_type = str(provider.get("provider", provider.get("type", "mock")))
    base_url = str(provider.get("base_url", ""))
    parsed = urlparse(base_url)
    is_loopback = parsed.scheme == "http" and parsed.hostname in LOCAL_LOOPBACK_HOSTS
    return {
        "provider": provider_type,
        "model_id": provider.get("model_id", "mock-e007b-controller"),
        "prompt_version": provider.get("prompt_version", "e007b_next_action_prompt_v1"),
        "response_schema_version": "e007b.next_action.v1",
        "local_only": bool(provider.get("local_only", True)),
        "allow_network": bool(provider.get("allow_network", False)),
        "dry_run": bool(provider.get("dry_run", provider_type not in {FAKE_SEQUENCE_PROVIDER, OPENAI_COMPATIBLE_PROVIDER})),
        "base_url_category": "local_loopback" if is_loopback else ("remote_or_network" if base_url else "none"),
        "mock": provider_type not in {FAKE_SEQUENCE_PROVIDER, OPENAI_COMPATIBLE_PROVIDER},
        "temperature": provider.get("temperature", 0.0),
        "max_tokens": provider.get("max_tokens", 256),
        "timeout_seconds": provider.get("timeout_seconds", 60),
        "max_retries": provider.get("max_retries", 0),
        "allow_local_provider_execution": bool(provider.get("allow_local_provider_execution", False)),
    }


def action_schema() -> dict[str, Any]:
    return {
        "schema_version": "e007b.next_action.v1",
        "compatible_action_schema_version": COMPATIBLE_ACTION_SCHEMA_VERSION,
        "type": "object",
        "required": ["action"],
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
            "query": {"type": "string"},
            "doc_id": {"type": "string"},
            "from_doc_id": {"type": "string"},
            "to_doc_id": {"type": "string"},
            "reason": {"type": "string"},
        },
    }


def validate_structured_action(value: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["response must be a JSON object"]
    for field_name in schema.get("required", []):
        if field_name not in value:
            errors.append(f"{field_name} is required")
    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        for field_name in value:
            if field_name not in properties:
                errors.append(f"{field_name} is not allowed")
    for field_name, field_schema in properties.items():
        if field_name not in value or not isinstance(field_schema, dict):
            continue
        if field_schema.get("type") == "string" and not isinstance(value[field_name], str):
            errors.append(f"{field_name} must be a string")
    return errors


def validate_local_provider_gate(provider: dict[str, Any]) -> None:
    parsed = urlparse(str(provider.get("base_url", "")))
    host = parsed.hostname or ""
    if not bool(provider.get("allow_local_provider_execution", False)):
        raise ValueError("missing_explicit_local_execution_flag")
    if str(provider.get("provider", provider.get("type", ""))) != OPENAI_COMPATIBLE_PROVIDER:
        raise ValueError("provider_must_be_openai_compatible")
    if not bool(provider.get("local_only", True)):
        raise ValueError("local_only_must_be_true")
    if bool(provider.get("allow_network", False)):
        raise ValueError("allow_network_must_be_false")
    if parsed.scheme != "http":
        raise ValueError("scheme_must_be_http_loopback")
    if host not in LOCAL_LOOPBACK_HOSTS:
        raise ValueError("base_url_host_must_be_loopback")


def compact_snippet(text: str, *, max_chars: int = 300) -> str:
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def sanitize_provider_debug(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            if key in {"reasoning", "reasoning_content", "thinking"}:
                sanitized[key] = f"<redacted:{len(child) if isinstance(child, str) else type(child).__name__}>"
            else:
                sanitized[key] = sanitize_provider_debug(child)
        return sanitized
    if isinstance(value, list):
        return [sanitize_provider_debug(item) for item in value]
    return value


def raw_response_debug(raw_response: dict[str, Any], *, raw_text: str = "") -> dict[str, Any]:
    sanitized = sanitize_provider_debug(raw_response)
    debug: dict[str, Any] = {
        "raw_response_keys": sorted(raw_response.keys()),
        "raw_response_snippet": compact_snippet(json.dumps(sanitized, sort_keys=True), max_chars=1000),
        "raw_text_length": len(raw_text),
    }
    choices = raw_response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        first = choices[0]
        debug["choice_keys"] = sorted(first.keys())
        if "finish_reason" in first:
            debug["finish_reason"] = first.get("finish_reason")
        message = first.get("message")
        if isinstance(message, dict):
            debug["message_keys"] = sorted(message.keys())
            content = message.get("content")
            if isinstance(content, str):
                debug["message_content_length"] = len(content)
                debug["message_content_repr"] = repr(content[:500])
                debug["message_content_snippet"] = compact_snippet(content, max_chars=500)
            else:
                debug["message_content_type"] = type(content).__name__
            for key in ("reasoning", "reasoning_content", "thinking", "tool_calls"):
                if key in message:
                    value = message[key]
                    debug[f"message_{key}_type"] = type(value).__name__
                    if isinstance(value, str):
                        debug[f"message_{key}_length"] = len(value)
        text = first.get("text")
        if isinstance(text, str):
            debug["choice_text_length"] = len(text)
            debug["choice_text_snippet"] = compact_snippet(text, max_chars=500)
    usage = raw_response.get("usage")
    if isinstance(usage, dict):
        debug["usage"] = usage
    return debug


def extract_json_content(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(stripped[start : index + 1])
        start = stripped.find("{", start + 1)
    raise ProviderActionError(
        f"provider_response_content_is_not_json:{compact_snippet(text)}",
        {"message_content_length": len(text), "message_content_repr": repr(text[:500])},
    )


def parse_openai_compatible_response(raw_response: dict[str, Any]) -> Any:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProviderActionError("provider_response_missing_choices", raw_response_debug(raw_response))
    message = choices[0].get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        try:
            return extract_json_content(message["content"])
        except ProviderActionError as exc:
            exc.details = {**raw_response_debug(raw_response), **exc.details}
            raise
    if isinstance(choices[0].get("text"), str):
        try:
            return extract_json_content(choices[0]["text"])
        except ProviderActionError as exc:
            exc.details = {**raw_response_debug(raw_response), **exc.details}
            raise
    raise ProviderActionError("provider_response_missing_message_content", raw_response_debug(raw_response))


def build_provider_messages(observation: dict[str, Any]) -> list[dict[str, str]]:
    allowed_actions = ", ".join(observation["allowed_actions"])
    return [
        {
            "role": "system",
            "content": (
                "/no_think\n"
                "You are the E007b next-action JSON controller for a bounded "
                "documentation search environment. You must return exactly one JSON "
                "object and nothing else. Do not include markdown, code fences, XML "
                "tags, commentary, chain-of-thought, or prose outside the JSON object. "
                "The only allowed action values are: "
                f"{allowed_actions}. "
                "If candidate_queue is empty, choose search with a short query derived "
                "from the question. If candidate_queue is non-empty, do not choose "
                "search; choose inspect_page for one candidate, or stop if no valid "
                "candidate should be inspected. Choose inspect_page only for a doc_id shown in "
                "search_results or candidate_queue. Choose follow_link only from an "
                "inspected page to one of that page's allowed_links. Choose add_context "
                "only for an inspected doc_id. add_context means save that inspected "
                "page as evidence for the final answer; without add_context the final "
                "answer will be empty. Watch the budget. If actions_remaining is small "
                "or you have inspected several pages, add the best inspected page even "
                "if it is only partially relevant. An acceptable weak answer grounded "
                "in partial local evidence is better than no context. If three pages "
                "have been inspected and no context has been added, you must choose "
                "add_context for the best inspected page or stop with a no relevant "
                "local context reason. Choose stop only when enough context was already "
                "added or no valid action remains. Never use FAQ reference "
                "answers, human labels, support labels, graph/spectral predictors, "
                "downstream metrics, or hidden knowledge."
            ),
        },
        {
            "role": "user",
            "content": (
                "/no_think\n"
                "Return one JSON object matching this schema:\n"
                '{"action":"search","query":"short search query","reason":"brief reason"}\n\n'
                "Current observation and schema:\n"
                + json.dumps({"observation": observation, "response_schema": action_schema()}, sort_keys=True)
            ),
        },
    ]


def openai_compatible_next_action(config: SearchConfig, step_index: int, observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    provider = dict(config.provider or {})
    manifest = provider_manifest(config)
    request_schema = action_schema()
    messages = build_provider_messages(observation)
    base_url = str(provider.get("base_url", "")).rstrip("/")
    request_body = {
        "model": str(provider.get("model_id", provider.get("model", "qwen3:14b"))),
        "messages": messages,
        "temperature": float(provider.get("temperature", 0.0)),
        "max_tokens": int(provider.get("max_tokens", 256)),
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "e007b_next_action_v1", "schema": request_schema, "strict": True},
        },
    }
    request_metadata = {
        "request_index": step_index + 1,
        "method": "POST",
        "url": f"{base_url}/chat/completions",
        "body": request_body,
        "timeout_seconds": float(provider.get("timeout_seconds", 60)),
        "max_retries": int(provider.get("max_retries", 0)),
        "local_only": bool(provider.get("local_only", True)),
        "allow_network": bool(provider.get("allow_network", False)),
    }
    validate_local_provider_gate(provider)
    last_error = ""
    for attempt in range(request_metadata["max_retries"] + 1):
        try:
            payload = json.dumps(request_body).encode("utf-8")
            request = urllib.request.Request(
                request_metadata["url"],
                data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=request_metadata["timeout_seconds"]) as response:
                raw_text = response.read().decode("utf-8")
            raw_response = json.loads(raw_text)
            if not isinstance(raw_response, dict):
                raise ProviderActionError("provider_response_must_be_object", {"raw_text_length": len(raw_text), "raw_text_snippet": compact_snippet(raw_text, max_chars=1000)})
            content = parse_openai_compatible_response(raw_response)
            schema_errors = validate_structured_action(content, request_schema)
            if schema_errors:
                raise ProviderActionError("structured_response_invalid:" + ";".join(schema_errors), {"content": content, **raw_response_debug(raw_response, raw_text=raw_text)})
            return dict(content), {
                **manifest,
                "request_index": step_index + 1,
                "request_metadata": request_metadata,
                "usage": raw_response.get("usage") if isinstance(raw_response.get("usage"), dict) else {},
                "dry_run": False,
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, socket.timeout, http.client.HTTPException, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            if attempt >= request_metadata["max_retries"]:
                details = getattr(exc, "details", {})
                raise ProviderActionError(f"provider_request_failed:{last_error}", {"request_metadata": request_metadata, **details}) from exc
    raise ProviderActionError(f"provider_request_failed:{last_error}", {"request_metadata": request_metadata})


def excerpt(text: str, *, max_chars: int = 500) -> str:
    return answer_text(text, max_chars=max_chars)


def build_observation(
    question: dict[str, str],
    corpus: dict[str, str],
    edges: dict[str, set[str]],
    ranked: list[tuple[float, str]],
    *,
    inspected: set[str],
    context_doc_ids: list[str],
    queue: list[str],
    action_budget_remaining: int,
    config: SearchConfig,
) -> dict[str, Any]:
    inspected_pages = [
        {"doc_id": doc_id, "excerpt": excerpt(corpus[doc_id]), "allowed_links": sorted(edges.get(doc_id, set()) & set(corpus))}
        for doc_id in sorted(inspected)
        if doc_id in corpus
    ]
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "search_results": [{"doc_id": doc_id, "score": float(score)} for score, doc_id in ranked[:10]],
        "candidate_queue": list(queue),
        "inspected_pages": inspected_pages,
        "context_doc_ids": list(context_doc_ids),
        "budget": {
            "actions_remaining": action_budget_remaining,
            "inspect_slots_remaining": max(0, config.inspect_budget - len(inspected)),
            "context_slots_remaining": max(0, config.final_top_k - len(context_doc_ids)),
            "inspected_count": len(inspected),
            "context_count": len(context_doc_ids),
            "max_actions": config.max_actions,
            "final_top_k": config.final_top_k,
            "context_decision_required": context_decision_required(inspected, context_doc_ids),
        },
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "forbidden_inputs": [
            "faq_reference_answers",
            "human_review_labels",
            "faq_support_audit_labels",
            "graph_spectral_predictor_summaries",
            "downstream_analysis_metrics",
        ],
    }


def fake_provider_next_action(config: SearchConfig, step_index: int, observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    provider = config.provider or {}
    provider_type = str(provider.get("provider", provider.get("type", "mock")))
    manifest = provider_manifest(config)
    if provider_type == OPENAI_COMPATIBLE_PROVIDER:
        return openai_compatible_next_action(config, step_index, observation)
    if provider_type != FAKE_SEQUENCE_PROVIDER:
        return {"action": "stop", "reason": "non_fake_provider_execution_disabled"}, {**manifest, "refused_live_execution": True}
    responses = provider.get("responses", [])
    if not isinstance(responses, list):
        responses = []
    if step_index < len(responses) and isinstance(responses[step_index], dict):
        return dict(responses[step_index]), {**manifest, "request_index": step_index + 1}
    if observation["candidate_queue"]:
        return {"action": "inspect_page", "doc_id": observation["candidate_queue"][0], "reason": "fake_sequence_default_inspect"}, {**manifest, "request_index": step_index + 1}
    return {"action": "stop", "reason": "fake_sequence_exhausted"}, {**manifest, "request_index": step_index + 1}


def validate_provider_action(
    action: dict[str, Any],
    observation: dict[str, Any],
    *,
    inspected: set[str],
    context_doc_ids: list[str],
    follow_counts: dict[str, int],
    edges: dict[str, set[str]],
    corpus: dict[str, str],
    config: SearchConfig,
) -> tuple[bool, str]:
    action_name = action.get("action")
    if action_name not in ALLOWED_ACTIONS:
        return False, f"unsupported_action:{action_name}"
    must_commit_or_stop = context_decision_required(inspected, context_doc_ids)
    if action_name == "search":
        if must_commit_or_stop:
            return False, "context_decision_required_after_three_inspections"
        if observation["candidate_queue"]:
            return False, "search_not_allowed_when_candidate_queue_non_empty"
        ok = isinstance(action.get("query"), str) and bool(action["query"].strip())
        return ok, "search_requires_query"
    if action_name == "inspect_page":
        if must_commit_or_stop:
            return False, "context_decision_required_after_three_inspections"
        doc_id = action.get("doc_id")
        allowed = {row["doc_id"] for row in observation["search_results"]} | set(observation["candidate_queue"])
        ok = len(inspected) < config.inspect_budget and isinstance(doc_id, str) and doc_id in allowed and doc_id in corpus
        return ok, "inspect_page_doc_not_allowed"
    if action_name == "follow_link":
        from_doc_id = action.get("from_doc_id")
        to_doc_id = action.get("to_doc_id")
        allowed_links = edges.get(str(from_doc_id), set()) & set(corpus)
        ok = (
            isinstance(from_doc_id, str)
            and follow_counts.get(from_doc_id, 0) < config.follow_budget_per_page
            and from_doc_id in inspected
            and isinstance(to_doc_id, str)
            and to_doc_id in allowed_links
        )
        return ok, "follow_link_not_allowed"
    if action_name == "add_context":
        doc_id = action.get("doc_id")
        ok = len(context_doc_ids) < config.final_top_k and isinstance(doc_id, str) and doc_id in inspected and doc_id not in context_doc_ids
        return ok, "add_context_doc_not_inspected_or_duplicate"
    return True, "ok"


def run_question_provider_controller(
    question: dict[str, str],
    corpus: dict[str, str],
    corpus_tokens: dict[str, list[str]],
    edges: dict[str, set[str]],
    config: SearchConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_tokens = tokens(question["question"])
    ranked = bm25_scores(query_tokens, corpus_tokens)
    bm25_by_doc = {doc_id: score for score, doc_id in ranked}
    actions: list[dict[str, Any]] = []
    queue: list[str] = []
    seen: set[str] = set()
    inspected: set[str] = set()
    follow_counts: dict[str, int] = {}
    context_doc_ids: list[str] = []
    provider_calls: list[dict[str, Any]] = []
    step_index = 0

    while len(actions) < config.max_actions - 1:
        observation = build_observation(
            question,
            corpus,
            edges,
            ranked,
            inspected=inspected,
            context_doc_ids=context_doc_ids,
            queue=queue,
            action_budget_remaining=max(0, config.max_actions - len(actions) - 1),
            config=config,
        )
        try:
            requested, provenance = fake_provider_next_action(config, step_index, observation)
        except (ProviderActionError, ValueError) as exc:
            details = getattr(exc, "details", {})
            provenance = {**provider_manifest(config), "request_index": step_index + 1, "provider_error": str(exc), "provider_error_details": details}
            provider_calls.append({"request_index": step_index + 1, "observation": observation, "response_schema": action_schema(), "provider_provenance": provenance, "provider_error": str(exc), "provider_error_details": details})
            trace_action(
                actions,
                "stop",
                budget_remaining=max(0, config.max_actions - len(actions) - 1),
                details={"reason": "provider_execution_refused_or_failed", "validation_error": str(exc), "provider": provenance},
            )
            break
        provider_calls.append({"request_index": step_index + 1, "observation": observation, "response_schema": action_schema(), "provider_provenance": provenance, "requested_action": requested})
        step_index += 1
        ok, reason = validate_provider_action(requested, observation, inspected=inspected, context_doc_ids=context_doc_ids, follow_counts=follow_counts, edges=edges, corpus=corpus, config=config)
        if not ok:
            trace_action(actions, "stop", budget_remaining=max(0, config.max_actions - len(actions) - 1), details={"reason": "invalid_provider_action_refused", "validation_error": reason, "requested_action": requested, "provider": provenance})
            break

        action_name = requested["action"]
        if action_name == "search":
            top_doc_ids = [doc_id for _, doc_id in ranked[: config.search_top_k]]
            trace_action(actions, "search", budget_remaining=config.max_actions - len(actions) - 1, details={"query": requested["query"], "top_doc_ids": top_doc_ids, "provider": provenance})
            for doc_id in top_doc_ids:
                if doc_id not in seen:
                    queue.append(doc_id)
                    seen.add(doc_id)
        elif action_name == "inspect_page":
            doc_id = requested["doc_id"]
            if doc_id in queue:
                queue.remove(doc_id)
            inspected.add(doc_id)
            trace_action(actions, "inspect_page", budget_remaining=config.max_actions - len(actions) - 1, details={"doc_id": doc_id, "score": float(bm25_by_doc.get(doc_id, 0.0)), "provider": provenance})
        elif action_name == "follow_link":
            to_doc_id = requested["to_doc_id"]
            from_doc_id = requested["from_doc_id"]
            if to_doc_id not in seen:
                queue.append(to_doc_id)
                seen.add(to_doc_id)
            follow_counts[from_doc_id] = follow_counts.get(from_doc_id, 0) + 1
            trace_action(actions, "follow_link", budget_remaining=config.max_actions - len(actions) - 1, details={"from_doc_id": from_doc_id, "to_doc_ids": [to_doc_id], "provider": provenance})
        elif action_name == "add_context":
            doc_id = requested["doc_id"]
            if len(context_doc_ids) < config.final_top_k:
                context_doc_ids.append(doc_id)
            trace_action(actions, "add_context", budget_remaining=config.max_actions - len(actions) - 1, details={"doc_id": doc_id, "context_rank": len(context_doc_ids), "reason": requested.get("reason", "provider_selected_context"), "provider": provenance})
            if len(context_doc_ids) >= config.final_top_k:
                trace_action(actions, "stop", budget_remaining=max(0, config.max_actions - len(actions) - 1), details={"reason": "context_budget_exhausted", "provider": provenance})
                break
        elif action_name == "stop":
            trace_action(actions, "stop", budget_remaining=max(0, config.max_actions - len(actions) - 1), details={"reason": requested.get("reason", "provider_stop"), "provider": provenance})
            break

    if not actions or actions[-1]["action"] != "stop":
        trace_action(actions, "stop", budget_remaining=max(0, config.max_actions - len(actions) - 1), details={"reason": "search_budget_exhausted", "provider": provider_manifest(config)})

    context = [
        {"doc_id": doc_id, "text": answer_text(corpus[doc_id], max_chars=config.max_context_chars_per_doc), "score": float(bm25_by_doc.get(doc_id, 0.0)), "source": "llm_guided_search_provider", "selection_role": "provider_agent_context", "rank": rank}
        for rank, doc_id in enumerate(context_doc_ids, start=1)
    ]
    prediction = {"question_id": question["question_id"], "question": question["question"], "generated_answer": " ".join(item["text"] for item in context if item["text"]), "retrieved_context": context, "action_trace": actions}
    trace = {
        "question_id": question["question_id"],
        "schema_version": TRACE_SCHEMA_VERSION,
        "compatible_action_schema_version": COMPATIBLE_ACTION_SCHEMA_VERSION,
        "policy": PROVIDER_POLICY_NAME,
        "controller": {**provider_manifest(config), "type": "provider_action"},
        "provider_calls": provider_calls,
        "actions": actions,
        "final_context_doc_ids": context_doc_ids,
        "paper_facing_evidence": False,
    }
    return prediction, trace


def run_question_mock(question: dict[str, str], corpus: dict[str, str], corpus_tokens: dict[str, list[str]], edges: dict[str, set[str]], config: SearchConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    query_tokens = tokens(question["question"])
    ranked = bm25_scores(query_tokens, corpus_tokens)
    bm25_by_doc = {doc_id: score for score, doc_id in ranked}
    actions: list[dict[str, Any]] = []
    queue = [doc_id for _, doc_id in ranked[: config.search_top_k]]
    seen = set(queue)
    inspected: set[str] = set()
    context_doc_ids: list[str] = []
    trace_action(actions, "search", budget_remaining=config.max_actions - 1, details={"query": question["question"], "top_doc_ids": queue, "controller": "mock_provider_fixture"})
    while queue and len(actions) < config.max_actions - 1 and len(context_doc_ids) < config.final_top_k:
        if len(inspected) >= config.inspect_budget:
            break
        doc_id = queue.pop(0)
        if doc_id in inspected or doc_id not in corpus:
            continue
        inspected.add(doc_id)
        trace_action(actions, "inspect_page", budget_remaining=config.max_actions - len(actions) - 1, details={"doc_id": doc_id, "controller_decision": "inspect_highest_ranked_candidate"})
        if doc_id not in context_doc_ids:
            context_doc_ids.append(doc_id)
            trace_action(actions, "add_context", budget_remaining=config.max_actions - len(actions) - 1, details={"doc_id": doc_id, "reason": "mock_controller_selected_relevant_page"})
        neighbors = sorted(edges.get(doc_id, set()) & set(corpus))
        followed = [candidate for _, candidate in bm25_scores(query_tokens, corpus_tokens, set(neighbors)) if candidate not in seen][: config.follow_budget_per_page]
        if followed and len(actions) < config.max_actions - 1 and len(context_doc_ids) < config.final_top_k:
            trace_action(actions, "follow_link", budget_remaining=config.max_actions - len(actions) - 1, details={"from_doc_id": doc_id, "to_doc_ids": followed, "controller_decision": "follow_ranked_links"})
            queue.extend(followed)
            seen.update(followed)
    trace_action(actions, "stop", budget_remaining=max(0, config.max_actions - len(actions) - 1), details={"reason": "mock_controller_stop", "context_count": len(context_doc_ids)})
    context = [
        {"doc_id": doc_id, "text": answer_text(corpus[doc_id], max_chars=config.max_context_chars_per_doc), "score": float(bm25_by_doc.get(doc_id, 0.0)), "source": "llm_guided_search_mock", "selection_role": "mock_agent_context", "rank": rank}
        for rank, doc_id in enumerate(context_doc_ids, start=1)
    ]
    prediction = {"question_id": question["question_id"], "question": question["question"], "generated_answer": " ".join(item["text"] for item in context if item["text"]), "retrieved_context": context, "action_trace": actions}
    trace = {
        "question_id": question["question_id"],
        "schema_version": TRACE_SCHEMA_VERSION,
        "compatible_action_schema_version": COMPATIBLE_ACTION_SCHEMA_VERSION,
        "policy": POLICY_NAME,
        "controller": {"type": "mock", "local_only": True, "allow_network": False},
        "actions": actions,
        "final_context_doc_ids": context_doc_ids,
        "paper_facing_evidence": False,
    }
    return prediction, trace


def run_question(question: dict[str, str], corpus: dict[str, str], corpus_tokens: dict[str, list[str]], edges: dict[str, set[str]], config: SearchConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.controller == "provider":
        return run_question_provider_controller(question, corpus, corpus_tokens, edges, config)
    return run_question_mock(question, corpus, corpus_tokens, edges, config)


def run(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = parse_config(read_config(input_dir / "config.yaml"))
    questions = read_jsonl(input_dir / "questions.jsonl")
    corpus = {path.stem: path.read_text(encoding="utf-8") for path in (input_dir / "corpus").glob("*.md")}
    corpus_tokens = {doc_id: tokens(text) for doc_id, text in corpus.items()}
    edges = read_edges(input_dir, directed=config.directed)
    predictions = []
    traces = []
    for question in questions:
        prediction, trace = run_question(question, corpus, corpus_tokens, edges, config)
        predictions.append(prediction)
        traces.append(trace)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "action_traces.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in traces:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    provider = provider_manifest(config)
    provider_request_count = sum(len(trace.get("provider_calls", [])) for trace in traces)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "name": "rag_llm_guided_search_mock",
                "experiment_id": "E007b",
                "deterministic": provider["provider"] != OPENAI_COMPATIBLE_PROVIDER,
                "policy": PROVIDER_POLICY_NAME if config.controller == "provider" else POLICY_NAME,
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "compatible_action_schema_version": COMPATIBLE_ACTION_SCHEMA_VERSION,
                "uses_llm": provider["provider"] == OPENAI_COMPATIBLE_PROVIDER,
                "uses_llm_controller": config.controller,
                "provider": provider["provider"],
                "provider_provenance": provider,
                "model_id": provider["model_id"],
                "prompt_version": provider["prompt_version"],
                "response_schema_version": provider["response_schema_version"],
                "local_only": provider["local_only"],
                "allow_network": provider["allow_network"],
                "base_url_category": provider["base_url_category"],
                "mock": provider["mock"],
                "dry_run": provider["dry_run"],
                "provider_request_count": provider_request_count,
                "paper_facing_evidence": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the E007b LLM-guided search test runner.")
    parser.add_argument("--input-dir", default="/input")
    parser.add_argument("--output-dir", default="/output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.input_dir), Path(args.output_dir))
