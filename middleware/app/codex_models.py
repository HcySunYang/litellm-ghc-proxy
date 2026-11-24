"""Codex-compatible model metadata exposed by the proxy."""

import logging
import re
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


GPT_5_6_CONTEXT_WINDOW = 1_050_000
GPT_5_6_MAX_PROMPT_TOKENS = 922_000
GPT_5_6_AUTO_COMPACT_TOKEN_LIMIT = 900_000
GPT_5_6_EFFECTIVE_CONTEXT_WINDOW_PERCENT = 87

GPT_6_ASTRA_CONTEXT_WINDOW = 1_000_000
GPT_6_ASTRA_MAX_PROMPT_TOKENS = 872_000
GPT_6_ASTRA_AUTO_COMPACT_TOKEN_LIMIT = 850_000
GPT_6_ASTRA_EFFECTIVE_CONTEXT_WINDOW_PERCENT = 87

GEMINI_4_CARBON_CONTEXT_WINDOW = 1_000_000
GEMINI_4_CARBON_MAX_PROMPT_TOKENS = 936_000
GEMINI_4_CARBON_MAX_OUTPUT_TOKENS = 64_000
GEMINI_4_CARBON_AUTO_COMPACT_TOKEN_LIMIT = 900_000
GEMINI_4_CARBON_EFFECTIVE_CONTEXT_WINDOW_PERCENT = 93

GEMINI_4_NEON_CONTEXT_WINDOW = 1_048_576
GEMINI_4_NEON_MAX_PROMPT_TOKENS = 983_040
GEMINI_4_NEON_MAX_OUTPUT_TOKENS = 65_536
GEMINI_4_NEON_AUTO_COMPACT_TOKEN_LIMIT = 950_000
GEMINI_4_NEON_EFFECTIVE_CONTEXT_WINDOW_PERCENT = 93

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

_THREE_TIER_REASONING_LEVELS = [
    {"effort": "low", "description": "Fast responses with lighter reasoning"},
    {
        "effort": "medium",
        "description": "Balances speed and reasoning depth for everyday tasks",
    },
    {"effort": "high", "description": "Greater reasoning depth for complex problems"},
]

_COMMON_REASONING_LEVELS = [
    *_THREE_TIER_REASONING_LEVELS,
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
    {
        "slug": "gemini-4-carbon",
        "display_name": "Gemini 4 Carbon",
        "description": "Frontier Google coding model with a one-million-token context window.",
        "default_reasoning_level": "medium",
        "fixed_reasoning_levels": _THREE_TIER_REASONING_LEVELS,
        "priority": 7,
        "supports_ultra": False,
        "support_verbosity": False,
        "supports_image_detail_original": False,
        "apply_patch_tool_type": "freeform",
        "tool_mode": "direct",
        "multi_agent_version": "v1",
        "context_limits": {
            "context_window": GEMINI_4_CARBON_CONTEXT_WINDOW,
            "max_context_window": GEMINI_4_CARBON_CONTEXT_WINDOW,
            "auto_compact_token_limit": GEMINI_4_CARBON_AUTO_COMPACT_TOKEN_LIMIT,
            "effective_context_window_percent": (
                GEMINI_4_CARBON_EFFECTIVE_CONTEXT_WINDOW_PERCENT
            ),
        },
    },
    {
        "slug": "gemini-4-neon",
        "display_name": "Gemini 4 Neon",
        "description": "Google coding model with a one-million-token context window and extra-high reasoning.",
        "default_reasoning_level": "medium",
        "fixed_reasoning_levels": _COMMON_REASONING_LEVELS,
        "priority": 8,
        "supports_ultra": False,
        "support_verbosity": False,
        "supports_image_detail_original": False,
        "apply_patch_tool_type": "freeform",
        "tool_mode": "direct",
        "multi_agent_version": "v1",
        "context_limits": {
            "context_window": GEMINI_4_NEON_CONTEXT_WINDOW,
            "max_context_window": GEMINI_4_NEON_CONTEXT_WINDOW,
            "auto_compact_token_limit": GEMINI_4_NEON_AUTO_COMPACT_TOKEN_LIMIT,
            "effective_context_window_percent": (
                GEMINI_4_NEON_EFFECTIVE_CONTEXT_WINDOW_PERCENT
            ),
        },
    },
)

_LEGACY_SAFE_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)
_VALID_APPLY_PATCH_TOOL_TYPES = frozenset({"freeform"})
_VALID_SHELL_TYPES = frozenset({"shell_command", "disabled"})
_VALID_VISIBILITIES = frozenset({"list", "hide", "none"})
_VALID_REASONING_SUMMARIES = frozenset({"auto", "concise", "detailed", "none"})
_VALID_VERBOSITIES = frozenset({"low", "medium", "high"})
_VALID_TRUNCATION_MODES = frozenset({"bytes", "tokens"})
_LEGACY_SAFE_INPUT_MODALITIES = frozenset({"text", "image"})
_MODERN_INPUT_MODALITIES = frozenset({"text", "image", "audio"})
_VALID_TOOL_MODES = frozenset({"direct", "code_mode", "code_mode_only"})
_VALID_MULTI_AGENT_VERSIONS = frozenset({"v1", "v2"})


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
    fixed_levels = spec.get("fixed_reasoning_levels")
    if fixed_levels is not None:
        reasoning_levels = [level.copy() for level in fixed_levels]
    else:
        reasoning_levels = _reasoning_levels(
            client_version, spec["supports_ultra"]
        )

    return {
        "slug": spec["slug"],
        "display_name": spec["display_name"],
        "description": spec["description"],
        "default_reasoning_level": spec["default_reasoning_level"],
        "supported_reasoning_levels": reasoning_levels,
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
        "support_verbosity": spec.get("support_verbosity", True),
        "default_verbosity": "low",
        "apply_patch_tool_type": spec.get("apply_patch_tool_type", "freeform"),
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "tokens", "limit": 10_000},
        "supports_parallel_tool_calls": True,
        "supports_image_detail_original": spec.get("supports_image_detail_original", True),
        **spec["context_limits"],
        "comp_hash": "3000",
        "experimental_supported_tools": [],
        "input_modalities": ["text", "image"],
        "supports_search_tool": False,
        "use_responses_lite": False,
        "tool_mode": spec.get("tool_mode", "code_mode_only"),
        "multi_agent_version": spec["multi_agent_version"],
    }


def _sanitize_model_info_for_client(
    model_info: dict[str, Any], client_version: str | None
) -> dict[str, Any]:
    """
    Normalize enum fields to values supported by the target Codex `client_version`
    so a single unsupported enum variant never breaks `/models` deserialization
    for the entire catalog.
    """
    parsed_version = _parse_client_version(client_version)
    is_modern_client = parsed_version >= (0, 144, 0)
    slug = model_info.get("slug", "<unknown>")

    patch_type = model_info.get("apply_patch_tool_type")
    if patch_type is not None and patch_type not in _VALID_APPLY_PATCH_TOOL_TYPES:
        logger.warning(
            "Downgrading unsupported apply_patch_tool_type=%r to 'freeform' for model %s",
            patch_type,
            slug,
        )
        model_info["apply_patch_tool_type"] = "freeform"

    shell_type = model_info.get("shell_type")
    if shell_type not in _VALID_SHELL_TYPES:
        logger.warning(
            "Normalizing unsupported shell_type=%r to 'shell_command' for model %s",
            shell_type,
            slug,
        )
        model_info["shell_type"] = "shell_command"

    visibility = model_info.get("visibility")
    if visibility not in _VALID_VISIBILITIES:
        model_info["visibility"] = "none"

    web_search_type = model_info.get("web_search_tool_type")
    valid_web_search = (
        {"text", "text_and_image"} if is_modern_client else {"text"}
    )
    if web_search_type not in valid_web_search:
        model_info["web_search_tool_type"] = "text"

    if model_info.get("default_reasoning_summary") not in _VALID_REASONING_SUMMARIES:
        model_info["default_reasoning_summary"] = "none"

    default_verbosity = model_info.get("default_verbosity")
    if default_verbosity is not None and default_verbosity not in _VALID_VERBOSITIES:
        model_info["default_verbosity"] = "low"

    truncation = model_info.get("truncation_policy")
    if not isinstance(truncation, dict) or (
        truncation.get("mode") not in _VALID_TRUNCATION_MODES
    ):
        model_info["truncation_policy"] = {"mode": "tokens", "limit": 10_000}

    allowed_modalities = (
        _MODERN_INPUT_MODALITIES
        if is_modern_client
        else _LEGACY_SAFE_INPUT_MODALITIES
    )
    raw_modalities = model_info.get("input_modalities")
    if isinstance(raw_modalities, list):
        filtered_modalities = [
            m for m in raw_modalities if m in allowed_modalities
        ]
        model_info["input_modalities"] = filtered_modalities or ["text", "image"]
    else:
        model_info["input_modalities"] = ["text", "image"]

    if not is_modern_client:
        raw_levels = model_info.get("supported_reasoning_levels")
        if isinstance(raw_levels, list):
            safe_levels = [
                lvl
                for lvl in raw_levels
                if isinstance(lvl, dict)
                and lvl.get("effort") in _LEGACY_SAFE_REASONING_EFFORTS
            ]
            if safe_levels:
                model_info["supported_reasoning_levels"] = safe_levels
        default_effort = model_info.get("default_reasoning_level")
        if (
            default_effort is not None
            and default_effort not in _LEGACY_SAFE_REASONING_EFFORTS
        ):
            model_info["default_reasoning_level"] = "high"

    tool_mode = model_info.get("tool_mode")
    if tool_mode is not None and tool_mode not in _VALID_TOOL_MODES:
        model_info["tool_mode"] = "direct"

    multi_agent_version = model_info.get("multi_agent_version")
    if (
        multi_agent_version is not None
        and multi_agent_version not in _VALID_MULTI_AGENT_VERSIONS
    ):
        model_info["multi_agent_version"] = "v1"

    return model_info


def build_codex_models_response(
    available_model_ids: Iterable[str], client_version: str | None
) -> dict[str, list[dict[str, Any]]]:
    """Build the model catalog shape expected by Codex's `/models` client."""
    available = set(available_model_ids)
    models: list[dict[str, Any]] = []
    for spec in _MODEL_SPECS:
        try:
            entry = _build_model_info(
                spec,
                client_version,
                available=spec["slug"] in available,
            )
            models.append(
                _sanitize_model_info_for_client(entry, client_version)
            )
        except Exception:
            logger.exception(
                "Skipping malformed Codex model spec %r during /models build",
                spec.get("slug"),
            )
    return {"models": models}
