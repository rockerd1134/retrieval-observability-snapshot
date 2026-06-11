from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import threading
import unittest
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator


def load_runner():
    path = Path("tests/rag_llm_guided_search_mock/runner.py")
    spec = importlib.util.spec_from_file_location("rag_llm_guided_search_mock_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RagLlmGuidedSearchMockTests(unittest.TestCase):
    @contextmanager
    def workspace_tempdir(self) -> Iterator[Path]:
        root = Path(".tmp_unit_tests")
        root.mkdir(exist_ok=True)
        path = root / f"tmp_{uuid.uuid4().hex}"
        path.mkdir()
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def write_input(self, root: Path, *, config: dict | None = None) -> tuple[Path, Path]:
        input_dir = root / "input"
        output_dir = root / "output"
        (input_dir / "corpus").mkdir(parents=True)
        (input_dir / "corpus" / "install.md").write_text("# Install\nUse plugins with collections.\n", encoding="utf-8")
        (input_dir / "corpus" / "collections.md").write_text("# Collections\nCollections package plugins and modules.\n", encoding="utf-8")
        (input_dir / "corpus" / "modules.md").write_text("# Modules\nModules can be packaged with plugins.\n", encoding="utf-8")
        (input_dir / "corpus" / "inventory.md").write_text("# Inventory\nInventory configures hosts for modules and plugins.\n", encoding="utf-8")
        (input_dir / "questions.jsonl").write_text('{"question_id":"q1","question":"How do plugins work with collections?"}\n', encoding="utf-8")
        (input_dir / "answers.jsonl").write_text('{"question_id":"q1","answer":"Collections package plugins."}\n', encoding="utf-8")
        (input_dir / "graph_edges.csv").write_text("source,target\ninstall,collections\ncollections,modules\nmodules,inventory\n", encoding="utf-8")
        raw_config = {"search_top_k": 1, "final_top_k": 2, "max_actions": 12}
        if config:
            raw_config.update(config)
        (input_dir / "config.yaml").write_text(json.dumps(raw_config, sort_keys=True), encoding="utf-8")
        return input_dir, output_dir

    def read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_mock_agent_outputs_e007b_trace_without_network_provider(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(traces[0]["schema_version"], "e007b.action_trace.v1")
            self.assertEqual(traces[0]["actions"][-1]["action"], "stop")
            self.assertEqual(metadata["experiment_id"], "E007b")
            self.assertEqual(metadata["provider"], "mock")
            self.assertFalse(metadata["allow_network"])
            self.assertFalse(metadata["paper_facing_evidence"])

    def test_mock_agent_is_deterministic(self):
        runner = load_runner()
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root)
            output_two = root / "output_two"
            runner.run(input_dir, output_dir)
            runner.run(input_dir, output_two)

            self.assertEqual((output_dir / "predictions.jsonl").read_text(encoding="utf-8"), (output_two / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual((output_dir / "action_traces.jsonl").read_text(encoding="utf-8"), (output_two / "action_traces.jsonl").read_text(encoding="utf-8"))

    def test_provider_controller_accepts_valid_fake_sequence(self):
        runner = load_runner()
        config = {
            "controller": "provider",
            "provider": {
                "provider": "fake_sequence",
                "model_id": "fake-qwen3-14b",
                "local_only": True,
                "allow_network": False,
                "responses": [
                    {"action": "search", "query": "plugins collections"},
                    {"action": "inspect_page", "doc_id": "install"},
                    {"action": "follow_link", "from_doc_id": "install", "to_doc_id": "collections"},
                    {"action": "inspect_page", "doc_id": "collections"},
                    {"action": "add_context", "doc_id": "collections", "reason": "contains answer"},
                    {"action": "stop", "reason": "done"},
                ],
            },
        }
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root, config=config)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual([action["action"] for action in traces[0]["actions"]], ["search", "inspect_page", "follow_link", "inspect_page", "add_context", "stop"])
            self.assertEqual(traces[0]["final_context_doc_ids"], ["collections"])
            self.assertEqual(traces[0]["provider_calls"][0]["response_schema"]["compatible_action_schema_version"], "e007a.action_trace.v1")
            self.assertEqual(metadata["uses_llm_controller"], "provider")
            self.assertEqual(metadata["provider"], "fake_sequence")
            self.assertFalse(metadata["allow_network"])

    def test_provider_controller_refuses_invalid_fake_action(self):
        runner = load_runner()
        config = {
            "controller": "provider",
            "provider": {
                "provider": "fake_sequence",
                "local_only": True,
                "allow_network": False,
                "responses": [
                    {"action": "search", "query": "plugins collections"},
                    {"action": "inspect_page", "doc_id": "install"},
                    {"action": "follow_link", "from_doc_id": "install", "to_doc_id": "missing"},
                ],
            },
        }
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root, config=config)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")

            self.assertEqual(traces[0]["actions"][-1]["action"], "stop")
            self.assertEqual(traces[0]["actions"][-1]["details"]["reason"], "invalid_provider_action_refused")
            self.assertEqual(traces[0]["actions"][-1]["details"]["validation_error"], "follow_link_not_allowed")

    def test_provider_controller_refuses_remote_openai_compatible_url_before_http(self):
        runner = load_runner()
        config = {
            "controller": "provider",
            "provider": {
                "provider": "openai_compatible",
                "model_id": "qwen3:14b",
                "base_url": "http://example.com/v1",
                "allow_local_provider_execution": True,
                "local_only": True,
                "allow_network": False,
            },
        }
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root, config=config)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")
            self.assertEqual(traces[0]["actions"][-1]["action"], "stop")
            self.assertEqual(traces[0]["actions"][-1]["details"]["reason"], "provider_execution_refused_or_failed")
            self.assertIn("base_url_host_must_be_loopback", traces[0]["actions"][-1]["details"]["validation_error"])

    def test_provider_controller_refuses_local_only_false_before_http(self):
        runner = load_runner()
        config = {
            "controller": "provider",
            "provider": {
                "provider": "openai_compatible",
                "model_id": "qwen3:14b",
                "base_url": "http://localhost:11434/v1",
                "allow_local_provider_execution": True,
                "local_only": False,
                "allow_network": False,
            },
        }
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root, config=config)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")
            self.assertIn("local_only_must_be_true", traces[0]["actions"][-1]["details"]["validation_error"])

    def test_provider_controller_refuses_allow_network_true_before_http(self):
        runner = load_runner()
        config = {
            "controller": "provider",
            "provider": {
                "provider": "openai_compatible",
                "model_id": "qwen3:14b",
                "base_url": "http://localhost:11434/v1",
                "allow_local_provider_execution": True,
                "local_only": True,
                "allow_network": True,
            },
        }
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root, config=config)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")
            self.assertIn("allow_network_must_be_false", traces[0]["actions"][-1]["details"]["validation_error"])

    def test_provider_controller_refuses_missing_explicit_local_execution_flag(self):
        runner = load_runner()
        config = {
            "controller": "provider",
            "provider": {
                "provider": "openai_compatible",
                "model_id": "qwen3:14b",
                "base_url": "http://localhost:11434/v1",
                "local_only": True,
                "allow_network": False,
            },
        }
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root, config=config)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertIn("missing_explicit_local_execution_flag", traces[0]["actions"][-1]["details"]["validation_error"])
            self.assertFalse(metadata["paper_facing_evidence"])

    def test_openai_compatible_request_construction_with_loopback_handler(self):
        runner = load_runner()
        requests: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                requests.append({"path": self.path, "body": json.loads(self.rfile.read(length).decode("utf-8"))})
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({"action": "stop", "reason": "fake_loopback_handler"})
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                }
                payload = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):  # noqa: A002
                return

        server = HTTPServer(("localhost", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = {
                "controller": "provider",
                "provider": {
                    "provider": "openai_compatible",
                    "model_id": "qwen3:14b",
                    "base_url": f"http://localhost:{server.server_port}/v1",
                    "temperature": 0.1,
                    "max_tokens": 77,
                    "timeout_seconds": 5,
                    "max_retries": 0,
                    "allow_local_provider_execution": True,
                    "local_only": True,
                    "allow_network": False,
                },
            }
            with self.workspace_tempdir() as root:
                input_dir, output_dir = self.write_input(root, config=config)
                runner.run(input_dir, output_dir)

                traces = self.read_jsonl(output_dir / "action_traces.jsonl")
                metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

                self.assertEqual(requests[0]["path"], "/v1/chat/completions")
                self.assertEqual(requests[0]["body"]["model"], "qwen3:14b")
                self.assertEqual(requests[0]["body"]["temperature"], 0.1)
                self.assertEqual(requests[0]["body"]["max_tokens"], 77)
                self.assertEqual(requests[0]["body"]["response_format"]["type"], "json_schema")
                self.assertIn("/no_think", requests[0]["body"]["messages"][0]["content"])
                self.assertIn("If candidate_queue is empty, choose search", requests[0]["body"]["messages"][0]["content"])
                self.assertIn("If candidate_queue is non-empty, do not choose search", requests[0]["body"]["messages"][0]["content"])
                self.assertIn("without add_context the final answer will be empty", requests[0]["body"]["messages"][0]["content"])
                self.assertIn("An acceptable weak answer grounded in partial local evidence is better than no context", requests[0]["body"]["messages"][0]["content"])
                self.assertIn("If three pages have been inspected and no context has been added", requests[0]["body"]["messages"][0]["content"])
                self.assertIn('"action":"search"', requests[0]["body"]["messages"][1]["content"])
                self.assertEqual(traces[0]["actions"][-1]["action"], "stop")
                self.assertEqual(traces[0]["controller"]["base_url_category"], "local_loopback")
                self.assertEqual(metadata["provider"], "openai_compatible")
                self.assertEqual(metadata["model_id"], "qwen3:14b")
                self.assertEqual(metadata["provider_request_count"], 1)
                self.assertFalse(metadata["mock"])
                self.assertFalse(metadata["dry_run"])
                self.assertFalse(metadata["allow_network"])
                self.assertFalse(metadata["paper_facing_evidence"])
        finally:
            server.shutdown()
            server.server_close()

    def test_provider_observation_records_forbidden_downstream_inputs(self):
        runner = load_runner()
        config = {
            "controller": "provider",
            "provider": {
                "provider": "fake_sequence",
                "local_only": True,
                "allow_network": False,
                "responses": [{"action": "stop", "reason": "observation_only"}],
            },
        }
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root, config=config)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")
            observation = traces[0]["provider_calls"][0]["observation"]

            self.assertIn("question", observation)
            self.assertIn("search_results", observation)
            self.assertEqual(observation["budget"]["actions_remaining"], 11)
            self.assertEqual(observation["budget"]["context_slots_remaining"], 2)
            self.assertFalse(observation["budget"]["context_decision_required"])
            self.assertNotIn("reference_answer", observation)
            self.assertIn("graph_spectral_predictor_summaries", observation["forbidden_inputs"])

    def test_provider_controller_refuses_repeated_search_when_queue_is_populated(self):
        runner = load_runner()
        config = {
            "controller": "provider",
            "provider": {
                "provider": "fake_sequence",
                "local_only": True,
                "allow_network": False,
                "responses": [
                    {"action": "search", "query": "plugins collections"},
                    {"action": "search", "query": "repeat search"},
                ],
            },
        }
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root, config=config)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")

            self.assertEqual([action["action"] for action in traces[0]["actions"]], ["search", "stop"])
            self.assertEqual(traces[0]["actions"][-1]["details"]["reason"], "invalid_provider_action_refused")
            self.assertEqual(traces[0]["actions"][-1]["details"]["validation_error"], "search_not_allowed_when_candidate_queue_non_empty")

    def test_provider_controller_requires_context_decision_after_three_inspections(self):
        runner = load_runner()
        config = {
            "controller": "provider",
            "search_top_k": 4,
            "provider": {
                "provider": "fake_sequence",
                "local_only": True,
                "allow_network": False,
                "responses": [
                    {"action": "search", "query": "plugins collections modules inventory"},
                    {"action": "inspect_page", "doc_id": "install"},
                    {"action": "inspect_page", "doc_id": "collections"},
                    {"action": "inspect_page", "doc_id": "modules"},
                    {"action": "inspect_page", "doc_id": "inventory"},
                ],
            },
        }
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root, config=config)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")

            self.assertEqual([action["action"] for action in traces[0]["actions"]], ["search", "inspect_page", "inspect_page", "inspect_page", "stop"])
            self.assertEqual(traces[0]["actions"][-1]["details"]["reason"], "invalid_provider_action_refused")
            self.assertEqual(traces[0]["actions"][-1]["details"]["validation_error"], "context_decision_required_after_three_inspections")
            self.assertTrue(traces[0]["provider_calls"][-1]["observation"]["budget"]["context_decision_required"])

    def test_provider_controller_allows_add_context_after_three_inspections(self):
        runner = load_runner()
        config = {
            "controller": "provider",
            "search_top_k": 4,
            "provider": {
                "provider": "fake_sequence",
                "local_only": True,
                "allow_network": False,
                "responses": [
                    {"action": "search", "query": "plugins collections modules inventory"},
                    {"action": "inspect_page", "doc_id": "install"},
                    {"action": "inspect_page", "doc_id": "collections"},
                    {"action": "inspect_page", "doc_id": "modules"},
                    {"action": "add_context", "doc_id": "collections", "reason": "best inspected page"},
                    {"action": "stop", "reason": "done"},
                ],
            },
        }
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root, config=config)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")

            self.assertEqual([action["action"] for action in traces[0]["actions"]], ["search", "inspect_page", "inspect_page", "inspect_page", "add_context", "stop"])
            self.assertEqual(traces[0]["final_context_doc_ids"], ["collections"])

    def test_provider_controller_allows_stop_after_three_inspections(self):
        runner = load_runner()
        config = {
            "controller": "provider",
            "search_top_k": 4,
            "provider": {
                "provider": "fake_sequence",
                "local_only": True,
                "allow_network": False,
                "responses": [
                    {"action": "search", "query": "plugins collections modules inventory"},
                    {"action": "inspect_page", "doc_id": "install"},
                    {"action": "inspect_page", "doc_id": "collections"},
                    {"action": "inspect_page", "doc_id": "modules"},
                    {"action": "stop", "reason": "no relevant local context found"},
                ],
            },
        }
        with self.workspace_tempdir() as root:
            input_dir, output_dir = self.write_input(root, config=config)
            runner.run(input_dir, output_dir)

            traces = self.read_jsonl(output_dir / "action_traces.jsonl")

            self.assertEqual([action["action"] for action in traces[0]["actions"]], ["search", "inspect_page", "inspect_page", "inspect_page", "stop"])
            self.assertEqual(traces[0]["actions"][-1]["details"]["reason"], "no relevant local context found")


if __name__ == "__main__":
    unittest.main()
