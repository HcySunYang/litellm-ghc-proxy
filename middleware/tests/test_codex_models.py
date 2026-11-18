import unittest

from app.codex_models import (
    GPT_5_6_AUTO_COMPACT_TOKEN_LIMIT,
    GPT_5_6_CONTEXT_WINDOW,
    GPT_5_6_EFFECTIVE_CONTEXT_WINDOW_PERCENT,
    GPT_5_6_MAX_PROMPT_TOKENS,
    GPT_6_ASTRA_AUTO_COMPACT_TOKEN_LIMIT,
    GPT_6_ASTRA_CONTEXT_WINDOW,
    GPT_6_ASTRA_EFFECTIVE_CONTEXT_WINDOW_PERCENT,
    GPT_6_ASTRA_MAX_PROMPT_TOKENS,
    build_codex_models_response,
)


class CodexModelsTests(unittest.TestCase):
    def test_hides_models_unavailable_from_litellm(self):
        response = build_codex_models_response(
            ["gpt-5.6-sol", "gpt-5.4"], "0.144.0"
        )
        models = {model["slug"]: model for model in response["models"]}

        self.assertEqual(len(models), 6)
        self.assertEqual(models["gpt-5.6-sol"]["visibility"], "list")
        self.assertTrue(models["gpt-5.6-sol"]["supported_in_api"])
        self.assertEqual(models["gpt-5.6-terra"]["visibility"], "none")
        self.assertFalse(models["gpt-5.6-terra"]["supported_in_api"])
        self.assertEqual(models["gpt-5.6-luna"]["visibility"], "none")
        self.assertFalse(models["gpt-5.6-luna"]["supported_in_api"])
        self.assertEqual(models["gpt-6-astra"]["visibility"], "none")
        self.assertFalse(models["gpt-6-astra"]["supported_in_api"])
        self.assertEqual(models["gpt-6.1-sol"]["visibility"], "none")
        self.assertFalse(models["gpt-6.1-sol"]["supported_in_api"])
        self.assertEqual(models["gpt-6.2-sol"]["visibility"], "none")
        self.assertFalse(models["gpt-6.2-sol"]["supported_in_api"])

    def test_lists_gpt_6_2_sol_with_safe_context_limits(self):
        response = build_codex_models_response(["gpt-6.2-sol"], "0.144.0")
        models = {model["slug"]: model for model in response["models"]}
        model = models["gpt-6.2-sol"]

        self.assertEqual(model["display_name"], "GPT-6.2 Sol")
        self.assertEqual(model["visibility"], "list")
        self.assertTrue(model["supported_in_api"])
        self.assertEqual(
            [item["slug"] for item in models.values() if item["supported_in_api"]],
            ["gpt-6.2-sol"],
        )
        self.assertEqual(model["context_window"], 1_050_000)
        self.assertEqual(model["max_context_window"], 1_050_000)
        self.assertEqual(model["auto_compact_token_limit"], 900_000)
        effective_window = (
            model["context_window"] * model["effective_context_window_percent"] // 100
        )
        self.assertEqual(effective_window, 913_500)
        self.assertLess(model["auto_compact_token_limit"], effective_window)
        self.assertLessEqual(effective_window + 128_000, model["context_window"])

    def test_lists_gpt_6_1_sol_when_available(self):
        response = build_codex_models_response(["gpt-6.1-sol"], "0.144.0")
        models = {model["slug"]: model for model in response["models"]}
        model = models["gpt-6.1-sol"]

        self.assertEqual(model["display_name"], "GPT-6.1 Sol")
        self.assertEqual(model["visibility"], "list")
        self.assertTrue(model["supported_in_api"])
        self.assertEqual(
            [model["slug"] for model in models.values() if model["supported_in_api"]],
            ["gpt-6.1-sol"],
        )

    def test_reports_gpt_6_1_sol_context_limits(self):
        response = build_codex_models_response(["gpt-6.1-sol"], "0.144.0")
        models = {model["slug"]: model for model in response["models"]}
        model = models["gpt-6.1-sol"]

        self.assertEqual(model["context_window"], 1_050_000)
        self.assertEqual(model["max_context_window"], 1_050_000)
        self.assertEqual(model["auto_compact_token_limit"], 900_000)
        self.assertEqual(model["effective_context_window_percent"], 87)
        effective_window = (
            model["context_window"] * model["effective_context_window_percent"] // 100
        )
        self.assertEqual(effective_window, 913_500)
        self.assertLessEqual(effective_window, 922_000)
        self.assertLess(model["auto_compact_token_limit"], effective_window)
        self.assertLessEqual(effective_window + 128_000, model["context_window"])

    def test_gpt_6_sol_reasoning_levels_respect_client_version(self):
        for client_version, expected in (
            ("0.144.0", ["low", "medium", "high", "xhigh", "max"]),
            ("0.144.0-alpha.4", ["low", "medium", "high", "xhigh", "max"]),
            ("0.125.0", ["low", "medium", "high", "xhigh"]),
            (None, ["low", "medium", "high", "xhigh"]),
            ("invalid", ["low", "medium", "high", "xhigh"]),
        ):
            for slug in ("gpt-6.1-sol", "gpt-6.2-sol"):
                with self.subTest(client_version=client_version, slug=slug):
                    response = build_codex_models_response([slug], client_version)
                    models = {model["slug"]: model for model in response["models"]}
                    model = models[slug]

                    self.assertEqual(
                        [level["effort"] for level in model["supported_reasoning_levels"]],
                        expected,
                    )
                    self.assertEqual(model["default_reasoning_level"], "medium")
                    self.assertEqual(model["multi_agent_version"], "v1")

    def test_lists_gpt_6_astra_when_available(self):
        response = build_codex_models_response(["gpt-6-astra"], "0.144.0")
        models = {model["slug"]: model for model in response["models"]}
        model = models["gpt-6-astra"]

        self.assertEqual(model["display_name"], "GPT-6 Astra")
        self.assertEqual(model["visibility"], "list")
        self.assertTrue(model["supported_in_api"])
        self.assertEqual(
            [model["slug"] for model in models.values() if model["supported_in_api"]],
            ["gpt-6-astra"],
        )

    def test_reports_model_specific_context_limits(self):
        response = build_codex_models_response(
            ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
            "0.144.0",
        )
        models = {model["slug"]: model for model in response["models"]}
        model = models["gpt-6-astra"]

        self.assertEqual(model["context_window"], GPT_6_ASTRA_CONTEXT_WINDOW)
        self.assertEqual(model["max_context_window"], GPT_6_ASTRA_CONTEXT_WINDOW)
        self.assertEqual(
            model["auto_compact_token_limit"],
            GPT_6_ASTRA_AUTO_COMPACT_TOKEN_LIMIT,
        )
        self.assertEqual(
            model["effective_context_window_percent"],
            GPT_6_ASTRA_EFFECTIVE_CONTEXT_WINDOW_PERCENT,
        )
        effective_window = (
            model["context_window"] * model["effective_context_window_percent"] // 100
        )
        self.assertEqual(effective_window, 870_000)
        self.assertLessEqual(effective_window, GPT_6_ASTRA_MAX_PROMPT_TOKENS)
        self.assertLess(model["auto_compact_token_limit"], effective_window)
        self.assertLessEqual(
            effective_window + 128_000, model["context_window"]
        )

        for slug in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            with self.subTest(slug=slug):
                self.assertEqual(models[slug]["context_window"], GPT_5_6_CONTEXT_WINDOW)
                self.assertEqual(
                    models[slug]["auto_compact_token_limit"],
                    GPT_5_6_AUTO_COMPACT_TOKEN_LIMIT,
                )

    def test_gpt_6_astra_reasoning_levels_respect_client_version(self):
        for client_version, expected in (
            ("0.144.0", ["low", "medium", "high", "xhigh", "max"]),
            ("0.144.0-alpha.4", ["low", "medium", "high", "xhigh", "max"]),
            ("0.125.0", ["low", "medium", "high", "xhigh"]),
            (None, ["low", "medium", "high", "xhigh"]),
            ("invalid", ["low", "medium", "high", "xhigh"]),
        ):
            with self.subTest(client_version=client_version):
                response = build_codex_models_response(["gpt-6-astra"], client_version)
                models = {model["slug"]: model for model in response["models"]}
                model = models["gpt-6-astra"]

                self.assertEqual(
                    [level["effort"] for level in model["supported_reasoning_levels"]],
                    expected,
                )
                self.assertIn(model["default_reasoning_level"], expected)

    def test_keeps_existing_model_priorities(self):
        response = build_codex_models_response(
            ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
            "0.144.0",
        )

        self.assertEqual(
            [(model["slug"], model["priority"]) for model in response["models"]],
            [
                ("gpt-5.6-sol", 1),
                ("gpt-5.6-terra", 2),
                ("gpt-5.6-luna", 3),
                ("gpt-6-astra", 4),
                ("gpt-6.1-sol", 5),
                ("gpt-6.2-sol", 6),
            ],
        )

    def test_reports_gpt_5_6_context_limits(self):
        response = build_codex_models_response(["gpt-5.6-luna"], "0.144.0")
        model = response["models"][0]

        self.assertEqual(model["context_window"], GPT_5_6_CONTEXT_WINDOW)
        self.assertEqual(model["max_context_window"], GPT_5_6_CONTEXT_WINDOW)
        self.assertEqual(
            model["auto_compact_token_limit"],
            GPT_5_6_AUTO_COMPACT_TOKEN_LIMIT,
        )
        self.assertEqual(
            model["effective_context_window_percent"],
            GPT_5_6_EFFECTIVE_CONTEXT_WINDOW_PERCENT,
        )
        effective_window = (
            model["context_window"] * model["effective_context_window_percent"] // 100
        )
        self.assertLessEqual(effective_window, GPT_5_6_MAX_PROMPT_TOKENS)

    def test_returns_complete_instruction_and_personality_metadata(self):
        response = build_codex_models_response(["gpt-5.6-sol"], "0.144.0")
        model = response["models"][0]

        self.assertGreater(len(model["base_instructions"]), 300)
        self.assertIn(
            "{{ personality }}",
            model["model_messages"]["instructions_template"],
        )
        self.assertEqual(
            set(model["model_messages"]["instructions_variables"]),
            {
                "personality_default",
                "personality_friendly",
                "personality_pragmatic",
            },
        )

    def test_omits_new_reasoning_levels_for_older_codex_clients(self):
        response = build_codex_models_response(["gpt-5.6-sol"], "0.125.0")
        efforts = [
            level["effort"]
            for level in response["models"][0]["supported_reasoning_levels"]
        ]

        self.assertEqual(efforts, ["low", "medium", "high", "xhigh"])

    def test_includes_current_reasoning_levels_for_newer_codex_clients(self):
        response = build_codex_models_response(
            ["gpt-5.6-sol"], "0.144.0-alpha.4"
        )
        efforts = [
            level["effort"]
            for level in response["models"][0]["supported_reasoning_levels"]
        ]

        self.assertEqual(
            efforts,
            ["low", "medium", "high", "xhigh", "max", "ultra"],
        )

    def test_preserves_multi_agent_metadata_for_ultra_mode(self):
        response = build_codex_models_response(
            ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
            "0.144.0",
        )
        models = {model["slug"]: model for model in response["models"]}

        self.assertEqual(models["gpt-5.6-sol"]["multi_agent_version"], "v2")
        self.assertEqual(models["gpt-5.6-terra"]["multi_agent_version"], "v2")
        self.assertEqual(models["gpt-5.6-luna"]["multi_agent_version"], "v1")
        self.assertTrue(
            all(model["tool_mode"] == "code_mode_only" for model in models.values())
        )


if __name__ == "__main__":
    unittest.main()
