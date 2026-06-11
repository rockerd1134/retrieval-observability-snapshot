from pathlib import Path
import unittest

from retrieval_arena.config import load_config, parse_config
from retrieval_arena.errors import ValidationError


class ConfigTests(unittest.TestCase):
    def test_load_example_config_resolves_paths(self):
        config = load_config("configs/experiment.example.yaml")
        self.assertEqual(config.experiment_name, "toy_faq_reconstruction")
        self.assertTrue(config.datasets[0].path.exists())
        self.assertTrue(config.tests[0].build_context.exists())


    def test_loads_docs_pilot_configs_with_oracle_graph_support(self):
        for path in sorted(Path("configs").glob("docs_pilot_*.yaml")):
            config = load_config(path)
            tests = {test.name: test for test in config.tests}
            self.assertIn("oracle_graph_support", tests)
            self.assertEqual(tests["oracle_graph_support"].config["experiment_id"], "E005")
            self.assertTrue(tests["oracle_graph_support"].build_context.exists())
            self.assertIn("rag_graph_multihop_rerank", tests)
            self.assertEqual(tests["rag_graph_multihop_rerank"].config["experiment_id"], "E006")
            self.assertTrue(tests["rag_graph_multihop_rerank"].build_context.exists())
            if path.name in {"docs_pilot_django_docs.yaml", "docs_pilot_python312.yaml"}:
                self.assertIn("rag_directed_spectral_rerank", tests)
                self.assertEqual(tests["rag_directed_spectral_rerank"].config["experiment_id"], "E008")
                self.assertTrue(tests["rag_directed_spectral_rerank"].build_context.exists())
            variants = {
                "rag_semantic_graph_rerank_e009a": ("E009a", 5, 1),
                "rag_semantic_graph_rerank_e009b": ("E009b", 2, 1),
                "rag_semantic_graph_rerank_e009c": ("E009c", 2, 2),
            }
            for test_name, (variant_id, seed_top_k, graph_hops) in variants.items():
                self.assertIn(test_name, tests)
                self.assertEqual(tests[test_name].config["experiment_id"], "E009")
                self.assertEqual(tests[test_name].config["variant_id"], variant_id)
                self.assertEqual(tests[test_name].config["seed_top_k"], seed_top_k)
                self.assertEqual(tests[test_name].config["graph_hops"], graph_hops)
                self.assertEqual(tests[test_name].config["final_top_k"], 5)
                self.assertEqual(tests[test_name].config["embedding_backend"], "sentence_transformers_local")
                self.assertTrue(tests[test_name].network_disabled)
                self.assertTrue(tests[test_name].build_context.exists())
            if path.name == "docs_pilot_django_docs.yaml":
                self.assertIn("rag_random_seed_graph_traversal", tests)
                e010 = tests["rag_random_seed_graph_traversal"]
                self.assertEqual(e010.config["experiment_id"], "E010")
                self.assertEqual(e010.config["diagnostic_role"], "random_seed_graph_navigability")
                self.assertEqual(e010.config["rng_seed"], 20260504)
                self.assertEqual(e010.config["num_trials"], 8)
                self.assertEqual(e010.config["max_hops"], 2)
                self.assertEqual(e010.config["neighbor_budget"], 12)
                self.assertEqual(e010.config["candidate_budget"], 60)
                self.assertEqual(e010.config["final_top_k"], 5)
                self.assertTrue(e010.network_disabled)
                self.assertTrue(e010.build_context.exists())

    def test_loads_minilm_smoke_with_e009(self):
        config = load_config("configs/minilm_smoke.yaml")
        tests = {test.name: test for test in config.tests}

        self.assertIn("rag_embedding_topk", tests)
        self.assertIn("rag_semantic_graph_rerank", tests)
        self.assertEqual(tests["rag_semantic_graph_rerank"].config["experiment_id"], "E009")
        self.assertEqual(tests["rag_semantic_graph_rerank"].config["model_id"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertTrue(tests["rag_semantic_graph_rerank"].network_disabled)

    def test_loads_e001b_local_no_context_config(self):
        config = load_config("configs/e001b_local_no_context_minidocs.yaml")
        tests = {test.name: test for test in config.tests}

        self.assertIn("local_no_context_llm_e001b", tests)
        e001b = tests["local_no_context_llm_e001b"]
        self.assertEqual(e001b.config["experiment_id"], "E001b")
        self.assertEqual(e001b.config["provider"], "mock")
        self.assertTrue(e001b.config["local_only"])
        self.assertFalse(e001b.config["allow_network"])
        self.assertFalse(e001b.config["allow_local_provider_execution"])
        self.assertTrue(e001b.network_disabled)
        self.assertTrue(e001b.build_context.exists())

    def test_parse_config_loads_read_only_test_volumes(self):
        raw = {
            "experiment_name": "x",
            "datasets": [{"name": "d", "path": "."}],
            "tests": [
                {
                    "name": "rag_embedding_topk",
                    "image": "i",
                    "volumes": [
                        {
                            "host_path": ".",
                            "container_path": "/models/minilm",
                            "read_only": True,
                        }
                    ],
                }
            ],
        }
        config = parse_config(raw, Path("config.yaml"))

        self.assertEqual(len(config.tests[0].volumes), 1)
        self.assertEqual(config.tests[0].volumes[0].container_path, "/models/minilm")
        self.assertTrue(config.tests[0].volumes[0].read_only)

    def test_parse_config_loads_network_disabled_flag(self):
        raw = {
            "experiment_name": "x",
            "datasets": [{"name": "d", "path": "."}],
            "tests": [{"name": "t", "image": "i", "network_disabled": True}],
        }
        config = parse_config(raw, Path("config.yaml"))

        self.assertTrue(config.tests[0].network_disabled)


    def test_parse_config_rejects_duplicate_tests(self):
        raw = {
            "experiment_name": "x",
            "datasets": [{"name": "d", "path": "."}],
            "tests": [{"name": "t", "image": "i"}, {"name": "t", "image": "i2"}],
        }
        with self.assertRaisesRegex(ValidationError, "Duplicate test"):
            parse_config(raw, Path("config.yaml"))


    def test_parse_config_rejects_bad_threshold(self):
        raw = {"experiment_name": "x", "datasets": [{"name": "d", "path": "."}], "tests": [{"name": "t", "image": "i"}], "scoring": {"match_threshold": 2}}
        with self.assertRaisesRegex(ValidationError, "match_threshold"):
            parse_config(raw, Path("config.yaml"))


if __name__ == "__main__":
    unittest.main()
