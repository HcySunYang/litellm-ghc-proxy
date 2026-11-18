"""Codex-compatible model metadata exposed by the proxy."""

import re
from collections.abc import Iterable
from typing import Any


GPT_5_6_CONTEXT_WINDOW = 1_050_000
GPT_5_6_MAX_PROMPT_TOKENS = 922_000
GPT_5_6_AUTO_COMPACT_TOKEN_LIMIT = 900_000
GPT_5_6_EFFECTIVE_CONTEXT_WINDOW_PERCENT = 87

GPT_6_ASTRA_CONTEXT_WINDOW = 1_000_000
GPT_6_ASTRA_MAX_PROMPT_TOKENS = 872_000
GPT_6_ASTRA_AUTO_COMPACT_TOKEN_LIMIT = 850_000
GPT_6_ASTRA_EFFECTIVE_CONTEXT_WINDOW_PERCENT = 87

_GPT_1_05M_CONTEXT_LIMITS = {
    "context_window": GPT_5_6_CONTEXT_WINDOW,
    "max_context_window": GPT_5_6_CONTEXT_WINDOW,
    "auto_compact_token_limit": GPT_5_6_AUTO_COMPACT_TOKEN_LIMIT,
    "effective_context_window_percent": GPT_5_6_EFFECTIVE_CONTEXT_WINDOW_PERCENT,
}

_BASE_INSTRUCTIONS = (
    "You are Codex, a coding agent running in a terminal. Collaborate with the "
    "user until their software-engineering task is fully handled. Inspect the "
    "repository and its instruction files before acting, use the available "
    "tools efficiently, and make precise changes that preserve unrelated work. "
    "Do not rewrite git history or create commits unless the user asks. Run the "
    "smallest relevant validation for changes you make, surface failures "
    "clearly, and report the outcome concisely."
)
_INSTRUCTIONS_TEMPLATE = f"{_BASE_INSTRUCTIONS}\n\n{{{{ personality }}}}"

_COMMON_REASONING_LEVELS = [
    {"effort": "low", "description": "Fast responses with lighter reasoning"},
    {
        "effort": "medium",
        "description": "Balances speed and reasoning depth for everyday tasks",
    },
    {"effort": "high", "description": "Greater reasoning depth for complex problems"},
    {
        "effort": "xhigh",
        "description": "Extra high reasoning depth for complex problems",
    },
]

_MODEL_SPECS = (
    {
        "slug": "gpt-5.6-sol",
        "display_name": "GPT-5.6-Sol",
        "description": "Latest frontier agentic coding model.",
        "default_reasoning_level": "low",
        "priority": 1,
        "supports_ultra": True,
        "multi_agent_version": "v2",
        "context_limits": _GPT_1_05M_CONTEXT_LIMITS,
    },
    {
        "slug": "gpt-5.6-terra",
        "display_name": "GPT-5.6-Terra",
        "description": "Balanced agentic coding model for everyday work.",
        "default_reasoning_level": "medium",
        "priority": 2,
        "supports_ultra": True,
        "multi_agent_version": "v2",
        "context_limits": _GPT_1_05M_CONTEXT_LIMITS,
    },
    {
        "slug": "gpt-5.6-luna",
        "display_name": "GPT-5.6-Luna",
        "description": "Fast and affordable agentic coding model.",
        "default_reasoning_level": "medium",
        "priority": 3,
        "supports_ultra": False,
        "multi_agent_version": "v1",
        "context_limits": _GPT_1_05M_CONTEXT_LIMITS,
    },
    {
        "slug": "gpt-6-astra",
        "display_name": "GPT-6 Astra",
        "description": "Frontier coding model with a one-million-token context window.",
        "default_reasoning_level": "medium",
        "priority": 4,
        "supports_ultra": False,
        "multi_agent_version": "v1",
        "context_limits": {
            "context_window": GPT_6_ASTRA_CONTEXT_WINDOW,
            "max_context_window": GPT_6_ASTRA_CONTEXT_WINDOW,
            "auto_compact_token_limit": GPT_6_ASTRA_AUTO_COMPACT_TOKEN_LIMIT,
            "effective_context_window_percent": (
                GPT_6_ASTRA_EFFECTIVE_CONTEXT_WINDOW_PERCENT
            ),
        },
    },
    {
        "slug": "gpt-6.1-sol",
        "display_name": "GPT-6.1 Sol",
        "description": "Frontier agentic coding model with a 1.05-million-token context window.",
        "default_reasoning_level": "medium",
        "priority": 5,
        "supports_ultra": False,
        "multi_agent_version": "v1",
        "context_limits": _GPT_1_05M_CONTEXT_LIMITS,
    },
    {
        "slug": "gpt-6.2-sol",
        "display_name": "GPT-6.2 Sol",
        "description": "Frontier agentic coding model with a 1.05-million-token context window.",
        "default_reasoning_level": "medium",
        "priority": 6,
        "supports_ultra": False,
        "multi_agent_version": "v1",
        "context_limits": _GPT_1_05M_CONTEXT_LIMITS,
    },
)


def _parse_client_version(client_version: str | None) -> tuple[int, int, int]:
    if not client_version:
        return (0, 0, 0)

    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", client_version)
    if match is None:
        return (0, 0, 0)

    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))


def _reasoning_levels(
    client_version: str | None, supports_ultra: bool
) -> list[dict[str, str]]:
    levels = [level.copy() for level in _COMMON_REASONING_LEVELS]
    if _parse_client_version(client_version) >= (0, 144, 0):
        levels.append(
            {
                "effort": "max",
                "description": "Maximum reasoning depth for the hardest problems",
            }
        )
        if supports_ultra:
            levels.append(
                {
                    "effort": "ultra",
                    "description": "Maximum reasoning with automatic task delegation",
                }
            )
    return levels


def _build_model_info(
    spec: dict[str, Any], client_version: str | None, available: bool
) -> dict[str, Any]:
    return {
        "slug": spec["slug"],
        "display_name": spec["display_name"],
        "description": spec["description"],
        "default_reasoning_level": spec["default_reasoning_level"],
        "supported_reasoning_levels": _reasoning_levels(
            client_version, spec["supports_ultra"]
        ),
        "shell_type": "shell_command",
        "visibility": "list" if available else "none",
        "supported_in_api": available,
        "priority": spec["priority"],
        "additional_speed_tiers": [],
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": _BASE_INSTRUCTIONS,
        "model_messages": {
            "instructions_template": _INSTRUCTIONS_TEMPLATE,
            "instructions_variables": {
                "personality_default": "",
                "personality_friendly": (
                    "Be warm, encouraging, and collaborative while remaining precise."
                ),
                "personality_pragmatic": (
                    "Be direct, practical, and focused on the fastest reliable solution."
                ),
            },
            "approvals": None,
            "auto_review": None,
            "permissions": None,
        },
        "supports_reasoning_summaries": True,
        "supports_reasoning_summary_parameter": True,
        "default_reasoning_summary": "none",
        "support_verbosity": True,
        "default_verbosity": "low",
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "tokens", "limit": 10_000},
        "supports_parallel_tool_calls": True,
        "supports_image_detail_original": True,
        **spec["context_limits"],
        "comp_hash": "3000",
        "experimental_supported_tools": [],
        "input_modalities": ["text", "image"],
        "supports_search_tool": False,
        "use_responses_lite": False,
        "tool_mode": "code_mode_only",
        "multi_agent_version": spec["multi_agent_version"],
    }


def build_codex_models_response(
    available_model_ids: Iterable[str], client_version: str | None
) -> dict[str, list[dict[str, Any]]]:
    """Build the model catalog shape expected by Codex's `/models` client."""
    available = set(available_model_ids)
    return {
        "models": [
            _build_model_info(
                spec,
                client_version,
                available=spec["slug"] in available,
            )
            for spec in _MODEL_SPECS
        ]
    }
