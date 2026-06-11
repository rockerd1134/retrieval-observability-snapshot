from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROMPT_VERSION = "review_e001b_no_context_prompt_v1"
RESPONSE_SCHEMA_VERSION = "review_e001b_no_context_answer_v1"
POLICY_NAME = "e001b_local_no_context_llm_v1"
MOCK_ANSWER = "Mock no-context answer."


@dataclass(frozen=True)
class ProviderConfig:
    provider: str = "mock"
    model_id: str = "mock/review-e001b-no-context-v1"
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_seconds: float = 60.0
    max_retries: int = 0
    local_only: bool = True
    allow_network: bool = False
    allow_local_provider_execution: bool = False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def provider_config(raw: dict[str, Any]) -> ProviderConfig:
    provider_raw = raw.get("provider", raw.get("provider_mode", "mock"))
    return ProviderConfig(
        provider=str(provider_raw),
        model_id=str(raw.get("model_id", raw.get("model", "mock/review-e001b-no-context-v1"))),
        base_url=str(raw.get("base_url", "")).rstrip("/"),
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 256)),
        timeout_seconds=float(raw.get("timeout_seconds", 60.0)),
        max_retries=int(raw.get("max_retries", 0)),
        local_only=bool(raw.get("local_only", True)),
        allow_network=bool(raw.get("allow_network", False)),
        allow_local_provider_execution=bool(raw.get("allow_local_provider_execution", False)),
    )


def base_url_category(config: ProviderConfig) -> str:
    parsed = urlparse(config.base_url)
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return "local_loopback"
    if not config.base_url:
        return ""
    return "remote_or_network"


def validate_local_gate(config: ProviderConfig) -> None:
    parsed = urlparse(config.base_url)
    host = parsed.hostname or ""
    if not config.allow_local_provider_execution:
        raise RuntimeError("Refusing E001b live provider execution without allow_local_provider_execution=true")
    if config.provider != "openai_compatible":
        raise RuntimeError("E001b live execution requires provider=openai_compatible")
    if not config.local_only:
        raise RuntimeError("E001b live execution requires local_only=true")
    if config.allow_network:
        raise RuntimeError("E001b live execution requires allow_network=false")
    if base_url_category(config) != "local_loopback":
        raise RuntimeError("E001b live execution refuses non-loopback base_url")
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("E001b live execution host must be localhost, 127.0.0.1, or ::1")
    if parsed.scheme != "http":
        raise RuntimeError("E001b live execution requires an http loopback URL")


def provenance(config: ProviderConfig) -> dict[str, Any]:
    return {
        "provider": config.provider,
        "model_id": config.model_id,
        "base_url_category": base_url_category(config),
        "prompt_version": PROMPT_VERSION,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "local_only": config.local_only,
        "allow_network": config.allow_network,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def messages(question: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are running the the retrieval audit E001b no-context baseline. "
                "Answer without retrieved context and return JSON."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    f"Question ID: {question['question_id']}",
                    f"Question: {question['question']}",
                    "Retrieved context: none",
                ]
            ),
        },
    ]


def live_complete(config: ProviderConfig, question: dict[str, Any]) -> dict[str, str]:
    validate_local_gate(config)
    request_body = {
        "model": config.model_id,
        "messages": messages(question),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": RESPONSE_SCHEMA_VERSION,
                "schema": {
                    "schema_version": RESPONSE_SCHEMA_VERSION,
                    "type": "object",
                    "required": ["schema_version", "answer", "rationale"],
                    "properties": {
                        "schema_version": {"type": "string", "const": RESPONSE_SCHEMA_VERSION},
                        "answer": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                },
            },
        },
    }
    request = urllib.request.Request(
        f"{config.base_url}/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"E001b local provider request failed: {exc}") from exc
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("E001b provider response missing choices")
    content = choices[0].get("message", {}).get("content")
    if isinstance(content, str):
        content = json.loads(content)
    if not isinstance(content, dict) or content.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise RuntimeError("E001b provider response failed schema_version validation")
    return {"answer": str(content.get("answer", "")), "rationale": str(content.get("rationale", ""))}


def answer_question(config: ProviderConfig, question: dict[str, Any]) -> dict[str, str]:
    if config.provider == "mock":
        return {"answer": MOCK_ANSWER, "rationale": "Deterministic mock response for Retrieval Audit Framework E001b contract tests."}
    return live_complete(config, question)


def run(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_config = read_config(input_dir / "config.yaml")
    config = provider_config(raw_config)
    questions = read_jsonl(input_dir / "questions.jsonl")
    rows = []
    for question in questions:
        result = answer_question(config, question)
        rows.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "generated_answer": result["answer"],
                "retrieved_context": [],
                "metadata": {
                    "experiment_id": "E001b",
                    "baseline_policy": POLICY_NAME,
                    "rationale": result["rationale"],
                    "retrieval_context": "none",
                    "paper_facing_evidence": False,
                    "provider_provenance": provenance(config),
                },
            }
        )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "name": "local_no_context_llm_e001b",
                "experiment_id": "E001b",
                "deterministic": config.provider == "mock",
                "uses_retrieval": False,
                "retrieval_context": "none",
                "uses_llm": config.provider != "mock",
                "provider": config.provider,
                "local_only": config.local_only,
                "allow_network": config.allow_network,
                "paper_facing_evidence": False,
                "baseline_policy": POLICY_NAME,
                "prompt_version": PROMPT_VERSION,
                "response_schema_version": RESPONSE_SCHEMA_VERSION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run(Path("/input"), Path("/output"))
