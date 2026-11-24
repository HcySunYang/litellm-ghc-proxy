"""
Bridge OpenAI Responses API requests to LiteLLM `/chat/completions` for upstream
models (such as GitHub Copilot's `gemini-4-carbon`) that only expose
`/chat/completions` and reject `/responses`.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_REASONING_EFFORTS = {
    "gemini-4-carbon": ("low", "medium", "high"),
    "gemini-4-neon": ("low", "medium", "high", "xhigh"),
}
_MAX_CACHED_RESPONSES = 128
_RESPONSE_HISTORY_CACHE: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

_FREEFORM_JSON_CLAUSE_RE = re.compile(
    r"\s*This is a FREEFORM tool,\s*so do not wrap [^.]+in JSON\.?",
    re.IGNORECASE,
)

_APPLY_PATCH_BRIDGE_DESCRIPTION = (
    "Use the `apply_patch` tool to create, update, move, or delete files. "
    "Pass the raw patch text in the `input` string parameter.\n\n"
    "Patch format:\n"
    "*** Begin Patch\n"
    "[*** Add File: <path>\n"
    "+<added line>]\n"
    "[*** Update File: <path>\n"
    "[*** Move to: <new_path>]\n"
    "@@ [optional context header]\n"
    " <context line>\n"
    "-<removed line>\n"
    "+<added line>]\n"
    "[*** Delete File: <path>]\n"
    "*** End Patch"
)


@dataclass
class BridgeContext:
    """Per-request metadata needed when translating Chat Completions back to Responses."""

    model: str
    input_messages: list[dict[str, Any]]
    custom_tool_names: set[str] = field(default_factory=set)
    namespaced_tools: dict[str, tuple[str, str]] = field(default_factory=dict)
    include_encrypted_reasoning: bool = False
    emit_reasoning_summary: bool = True


def _strip_provider_prefix(model: str) -> str:
    stripped = model.strip()
    if stripped.startswith("github_copilot/"):
        return stripped.split("/", 1)[1]
    return stripped


def is_chat_completions_responses_model(model: Any) -> bool:
    """Return True if `/responses` requests for `model` should bridge via `/chat/completions`."""
    if not isinstance(model, str) or not model.strip():
        return False
    normalized = _strip_provider_prefix(model).lower()
    return normalized.startswith("gemini-")


def _store_response_history(
    response_id: str, messages: list[dict[str, Any]]
) -> None:
    _RESPONSE_HISTORY_CACHE[response_id] = copy.deepcopy(messages)
    _RESPONSE_HISTORY_CACHE.move_to_end(response_id)
    while len(_RESPONSE_HISTORY_CACHE) > _MAX_CACHED_RESPONSES:
        _RESPONSE_HISTORY_CACHE.popitem(last=False)


def _normalize_reasoning_effort(model: str, body: dict[str, Any]) -> str | None:
    effort: Any = None
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    if effort is None:
        effort = body.get("reasoning_effort")
    if effort is None:
        return None

    effort_str = str(effort).strip().lower()
    if effort_str in ("", "none"):
        return None

    normalized_model = _strip_provider_prefix(model).lower()
    supported_efforts = _MODEL_REASONING_EFFORTS.get(normalized_model)
    if supported_efforts is not None:
        if effort_str in supported_efforts:
            return effort_str
        if effort_str == "minimal":
            logger.info(
                "Clamping reasoning effort %r to 'low' for %s",
                effort_str,
                model,
            )
            return "low"
        if effort_str in ("xhigh", "max", "ultra"):
            logger.info(
                "Clamping reasoning effort %r to %r for %s",
                effort_str,
                supported_efforts[-1],
                model,
            )
            return supported_efforts[-1]

    return effort_str


def _extract_image_part(part: dict[str, Any]) -> dict[str, Any] | None:
    raw_image = part.get("image_url")
    detail = part.get("detail")
    if isinstance(raw_image, str) and raw_image:
        image_obj: dict[str, Any] = {"url": raw_image}
    elif isinstance(raw_image, dict) and isinstance(raw_image.get("url"), str):
        image_obj = dict(raw_image)
    else:
        return None
    if isinstance(detail, str) and detail and "detail" not in image_obj:
        image_obj["detail"] = detail
    # Responses supports "original", but Copilot's Gemini Chat Completions rejects it.
    if image_obj.get("detail") == "original":
        image_obj["detail"] = "high"
    return {"type": "image_url", "image_url": image_obj}


def _convert_message_content(
    content: Any, *, force_text: bool = False
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: list[dict[str, Any]] = []
    text_chunks: list[str] = []
    has_non_text = False

    for part in content:
        if isinstance(part, str):
            text_chunks.append(part)
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue

        p_type = part.get("type")
        if p_type in ("input_text", "output_text", "text"):
            text = part.get("text", "")
            if isinstance(text, str):
                text_chunks.append(text)
                parts.append({"type": "text", "text": text})
        elif p_type == "refusal":
            refusal = part.get("refusal", "")
            if isinstance(refusal, str) and refusal:
                text_chunks.append(refusal)
                parts.append({"type": "text", "text": refusal})
        elif p_type in ("input_image", "image_url") and not force_text:
            image_part = _extract_image_part(part)
            if image_part is not None:
                has_non_text = True
                parts.append(image_part)
        elif p_type in ("input_file", "file") and not force_text:
            file_data = part.get("file")
            if isinstance(file_data, dict):
                has_non_text = True
                parts.append({"type": "file", "file": file_data})

    if force_text or not has_non_text:
        return "".join(text_chunks)
    return parts


def _convert_tool_output(
    raw_output: Any,
) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(raw_output, str):
        return raw_output, []
    if isinstance(raw_output, list):
        text_chunks: list[str] = []
        images: list[dict[str, Any]] = []
        for part in raw_output:
            if isinstance(part, str):
                text_chunks.append(part)
            elif isinstance(part, dict):
                p_type = part.get("type")
                if p_type in ("input_text", "output_text", "text"):
                    text = part.get("text", "")
                    if isinstance(text, str):
                        text_chunks.append(text)
                elif p_type in ("input_image", "image_url"):
                    img = _extract_image_part(part)
                    if img is not None:
                        images.append(img)
                else:
                    text_chunks.append(json.dumps(part))
        joined = "\n".join(chunk for chunk in text_chunks if chunk)
        if not joined and images:
            joined = "[Image attached in following user message]"
        return joined, images
    if raw_output is None:
        return "", []
    return json.dumps(raw_output), []


def _encode_namespaced_tool_name(namespace: str, name: str) -> str:
    safe_ns = re.sub(r"[^a-zA-Z0-9_.-]", "_", namespace)
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    combined = f"{safe_ns}__{safe_name}"
    return combined[:64]


def _build_function_tool_def(
    src: dict[str, Any],
    *,
    wire_name: str,
    namespace_description: str | None = None,
) -> dict[str, Any]:
    fn_def: dict[str, Any] = {"name": wire_name}
    description = src.get("description")
    desc_parts: list[str] = []
    if isinstance(namespace_description, str) and namespace_description.strip():
        desc_parts.append(namespace_description.strip())
    if isinstance(description, str) and description.strip():
        desc_parts.append(description.strip())
    if desc_parts:
        fn_def["description"] = "\n\n".join(desc_parts)

    parameters = src.get("parameters")
    if isinstance(parameters, dict):
        params_copy = copy.deepcopy(parameters)
        params_copy.setdefault("type", "object")
        params_copy.setdefault("properties", {})
        fn_def["parameters"] = params_copy
    else:
        fn_def["parameters"] = {"type": "object", "properties": {}}
    return {"type": "function", "function": fn_def}


def _build_custom_tool_def(
    tool: dict[str, Any],
    *,
    original_name: str,
    wire_name: str,
    namespace_description: str | None = None,
) -> dict[str, Any]:
    fmt = tool.get("format")
    grammar_def = (
        fmt.get("definition")
        if isinstance(fmt, dict) and isinstance(fmt.get("definition"), str)
        else ""
    )

    if original_name == "apply_patch":
        description = _APPLY_PATCH_BRIDGE_DESCRIPTION
        if "environment_id" in grammar_def:
            description += (
                "\n(When targeting a specific environment, include "
                "`*** Environment ID: <id>` immediately after `*** Begin Patch`.)"
            )
        input_desc = (
            "Complete patch text starting with `*** Begin Patch` and ending "
            "with `*** End Patch`."
        )
    else:
        raw_desc = tool.get("description")
        cleaned_desc = (
            _FREEFORM_JSON_CLAUSE_RE.sub("", raw_desc).strip()
            if isinstance(raw_desc, str)
            else ""
        )
        desc_parts = []
        if isinstance(namespace_description, str) and namespace_description.strip():
            desc_parts.append(namespace_description.strip())
        if cleaned_desc:
            desc_parts.append(cleaned_desc)
        desc_parts.append(
            "Pass the raw tool input text in the `input` string parameter."
        )
        if grammar_def.strip() and grammar_def.strip() not in cleaned_desc:
            desc_parts.append(f"Expected input grammar:\n{grammar_def.strip()}")
        description = "\n\n".join(desc_parts)
        input_desc = "Raw input text for the tool."

    fn_def: dict[str, Any] = {
        "name": wire_name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": input_desc,
                }
            },
            "required": ["input"],
        },
    }
    return {"type": "function", "function": fn_def}


def _convert_tools(
    raw_tools: Any,
) -> tuple[list[dict[str, Any]], set[str], dict[str, tuple[str, str]]]:
    if not isinstance(raw_tools, list):
        return [], set(), {}

    converted: list[dict[str, Any]] = []
    custom_tool_names: set[str] = set()
    namespaced_tools: dict[str, tuple[str, str]] = {}

    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        t_type = tool.get("type")
        if t_type == "function":
            nested = tool.get("function")
            src = nested if isinstance(nested, dict) else tool
            name = src.get("name")
            if not isinstance(name, str) or not name:
                continue
            converted.append(_build_function_tool_def(src, wire_name=name))
        elif t_type == "custom":
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                continue
            custom_tool_names.add(name)
            converted.append(
                _build_custom_tool_def(
                    tool, original_name=name, wire_name=name
                )
            )
        elif t_type == "namespace":
            ns_name = tool.get("name")
            ns_tools = tool.get("tools")
            if (
                not isinstance(ns_name, str)
                or not ns_name
                or not isinstance(ns_tools, list)
            ):
                continue
            ns_desc = (
                tool.get("description")
                if isinstance(tool.get("description"), str)
                else None
            )
            for inner in ns_tools:
                if not isinstance(inner, dict):
                    continue
                inner_type = inner.get("type")
                inner_name = inner.get("name")
                if not isinstance(inner_name, str) or not inner_name:
                    continue
                wire_name = (
                    inner_name
                    if ns_name == "functions"
                    else _encode_namespaced_tool_name(ns_name, inner_name)
                )
                if ns_name != "functions":
                    namespaced_tools[wire_name] = (ns_name, inner_name)
                if inner_type == "function":
                    converted.append(
                        _build_function_tool_def(
                            inner,
                            wire_name=wire_name,
                            namespace_description=ns_desc,
                        )
                    )
                elif inner_type == "custom":
                    custom_tool_names.add(wire_name)
                    converted.append(
                        _build_custom_tool_def(
                            inner,
                            original_name=inner_name,
                            wire_name=wire_name,
                            namespace_description=ns_desc,
                        )
                    )
        else:
            # Hosted Responses-only tool definitions (such as `{"type": "web_search"}`)
            # have no client function schema on Chat Completions.
            logger.debug("Skipping non-function Responses tool type=%r", t_type)

    return converted, custom_tool_names, namespaced_tools


def _convert_tool_choice(
    raw_choice: Any,
    reverse_namespaced: dict[tuple[str, str], str] | None = None,
) -> Any:
    if raw_choice in ("auto", "none", "required"):
        return raw_choice
    if isinstance(raw_choice, dict):
        c_type = raw_choice.get("type")
        if c_type in ("function", "custom"):
            fn_obj = raw_choice.get("function")
            name = (
                fn_obj.get("name")
                if isinstance(fn_obj, dict)
                else raw_choice.get("name")
            )
            namespace = raw_choice.get("namespace")
            if isinstance(name, str) and name:
                wire_name = name
                if (
                    isinstance(namespace, str)
                    and namespace
                    and namespace != "functions"
                ):
                    wire_name = (
                        reverse_namespaced.get((namespace, name))
                        if reverse_namespaced
                        else None
                    ) or _encode_namespaced_tool_name(namespace, name)
                return {"type": "function", "function": {"name": wire_name}}
        elif c_type == "allowed_tools":
            mode = raw_choice.get("mode")
            if mode in ("auto", "none", "required"):
                return mode
    return None


def responses_request_to_chat_completions(
    body: dict[str, Any], *, stream: bool
) -> tuple[dict[str, Any], BridgeContext]:
    """Translate an OpenAI Responses API request payload into `/chat/completions` shape."""
    model = str(body.get("model", ""))
    messages: list[dict[str, Any]] = []
    tools, custom_tool_names, namespaced_tools = _convert_tools(
        body.get("tools")
    )
    reverse_namespaced = {
        (ns, inner): wire for wire, (ns, inner) in namespaced_tools.items()
    }

    instructions = body.get("instructions")
    has_instructions = isinstance(instructions, str) and bool(instructions)
    if has_instructions:
        # Use role="developer" rather than "system" so LiteLLM's Copilot adapter
        # preserves instructions as developer context instead of rewriting system->assistant.
        messages.append({"role": "developer", "content": instructions})

    prev_id = body.get("previous_response_id")
    if isinstance(prev_id, str) and prev_id in _RESPONSE_HISTORY_CACHE:
        cached = copy.deepcopy(_RESPONSE_HISTORY_CACHE[prev_id])
        if (
            has_instructions
            and cached
            and cached[0].get("role") == "developer"
        ):
            cached = cached[1:]
        messages.extend(cached)

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        pending_reasoning_opaque: str | None = None
        deferred_user_images: list[dict[str, Any]] = []

        for item in raw_input:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            if (
                item_type not in ("function_call_output", "custom_tool_call_output")
                and deferred_user_images
            ):
                messages.append(
                    {"role": "user", "content": deferred_user_images.copy()}
                )
                deferred_user_images.clear()

            if item_type == "reasoning":
                encrypted = item.get("encrypted_content")
                if isinstance(encrypted, str) and encrypted:
                    pending_reasoning_opaque = encrypted
                continue

            if item_type == "message" or (item_type is None and "role" in item):
                role = str(item.get("role", "user"))
                if role == "system":
                    role = "developer"

                if role == "assistant":
                    text_content = _convert_message_content(
                        item.get("content"), force_text=True
                    )
                    msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": text_content,
                    }
                    if pending_reasoning_opaque:
                        msg["reasoning_opaque"] = pending_reasoning_opaque
                        pending_reasoning_opaque = None
                    messages.append(msg)
                else:
                    msg_content = _convert_message_content(item.get("content"))
                    messages.append({"role": role, "content": msg_content})
                continue

            if item_type in ("function_call", "custom_tool_call"):
                call_id = (
                    item.get("call_id")
                    or item.get("id")
                    or f"call_{uuid.uuid4().hex[:16]}"
                )
                name = str(item.get("name", ""))
                namespace = item.get("namespace")
                if (
                    isinstance(namespace, str)
                    and namespace
                    and namespace != "functions"
                    and name
                ):
                    name = reverse_namespaced.get(
                        (namespace, name)
                    ) or _encode_namespaced_tool_name(namespace, name)
                if item_type == "custom_tool_call":
                    args_str = json.dumps({"input": item.get("input", "")})
                else:
                    raw_args = item.get("arguments", "{}")
                    args_str = (
                        raw_args
                        if isinstance(raw_args, str)
                        else json.dumps(raw_args)
                    )

                tool_call = {
                    "id": str(call_id),
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }

                if messages and messages[-1].get("role") == "assistant":
                    prev_msg = messages[-1]
                    prev_msg.setdefault("tool_calls", []).append(tool_call)
                    if pending_reasoning_opaque and "reasoning_opaque" not in prev_msg:
                        prev_msg["reasoning_opaque"] = pending_reasoning_opaque
                        pending_reasoning_opaque = None
                else:
                    msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call],
                    }
                    if pending_reasoning_opaque:
                        msg["reasoning_opaque"] = pending_reasoning_opaque
                        pending_reasoning_opaque = None
                    messages.append(msg)
                continue

            if item_type in ("function_call_output", "custom_tool_call_output"):
                call_id = str(item.get("call_id") or "")
                output_text, image_parts = _convert_tool_output(item.get("output"))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output_text,
                    }
                )
                if image_parts:
                    deferred_user_images.extend(image_parts)

        if deferred_user_images:
            messages.append(
                {"role": "user", "content": deferred_user_images.copy()}
            )

    if not messages:
        messages.append({"role": "user", "content": ""})

    input_messages = copy.deepcopy(messages)
    if (
        is_chat_completions_responses_model(model)
        and messages[-1].get("role") not in ("user", "tool")
    ):
        logger.info(
            "Adding Gemini continuation turn after role=%s",
            messages[-1].get("role"),
        )
        # Gemini cannot prefill assistant turns; keep this adapter-only turn out of history.
        messages.append(
            {
                "role": "user",
                "content": "Continue from the conversation above and complete the user's request.",
            }
        )

    chat_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if stream:
        chat_body["stream_options"] = {"include_usage": True}

    if tools:
        chat_body["tools"] = tools
        tool_choice = _convert_tool_choice(
            body.get("tool_choice"), reverse_namespaced
        )
        if tool_choice is not None:
            chat_body["tool_choice"] = tool_choice
        if isinstance(body.get("parallel_tool_calls"), bool):
            chat_body["parallel_tool_calls"] = body["parallel_tool_calls"]

    effort = _normalize_reasoning_effort(model, body)
    if effort is not None:
        chat_body["reasoning_effort"] = effort

    max_output_tokens = body.get("max_output_tokens")
    if isinstance(max_output_tokens, int) and max_output_tokens > 0:
        chat_body["max_tokens"] = max_output_tokens

    for passthrough_key in ("temperature", "top_p", "user"):
        if body.get(passthrough_key) is not None:
            chat_body[passthrough_key] = body[passthrough_key]

    include = body.get("include")
    include_encrypted = (
        isinstance(include, list)
        and "reasoning.encrypted_content" in include
    )
    reasoning_cfg = body.get("reasoning")
    summary_pref = (
        reasoning_cfg.get("summary")
        if isinstance(reasoning_cfg, dict)
        else None
    )
    emit_summary = summary_pref != "none"

    ctx = BridgeContext(
        model=model,
        input_messages=input_messages,
        custom_tool_names=custom_tool_names,
        namespaced_tools=namespaced_tools,
        include_encrypted_reasoning=include_encrypted,
        emit_reasoning_summary=emit_summary,
    )
    return chat_body, ctx


def _extract_reasoning_fields(obj: dict[str, Any]) -> tuple[str, str]:
    psf = obj.get("provider_specific_fields")
    psf_dict = psf if isinstance(psf, dict) else {}

    r_text = (
        obj.get("reasoning_text")
        or psf_dict.get("reasoning_text")
        or obj.get("reasoning_content")
        or psf_dict.get("reasoning_content")
        or ""
    )
    r_opaque = (
        obj.get("reasoning_opaque")
        or psf_dict.get("reasoning_opaque")
        or ""
    )
    return (
        r_text if isinstance(r_text, str) else "",
        r_opaque if isinstance(r_opaque, str) else "",
    )


def _convert_usage(raw_usage: Any) -> dict[str, Any] | None:
    if not isinstance(raw_usage, dict):
        return None
    prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
    completion_tokens = int(raw_usage.get("completion_tokens") or 0)
    total_tokens = int(
        raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens)
    )

    prompt_details = raw_usage.get("prompt_tokens_details")
    cached_tokens = 0
    if isinstance(prompt_details, dict):
        cached_tokens = int(prompt_details.get("cached_tokens") or 0)

    completion_details = raw_usage.get("completion_tokens_details")
    reasoning_tokens = int(raw_usage.get("reasoning_tokens") or 0)
    if isinstance(completion_details, dict) and completion_details.get(
        "reasoning_tokens"
    ) is not None:
        reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)

    return {
        "input_tokens": prompt_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": completion_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": total_tokens,
    }


def _unwrap_custom_tool_input(arguments_str: str) -> str:
    try:
        parsed = json.loads(arguments_str)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("input"), str):
                return parsed["input"]
            if len(parsed) == 1:
                sole_value = next(iter(parsed.values()))
                if isinstance(sole_value, str):
                    return sole_value
        elif isinstance(parsed, str):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return arguments_str


def _resolve_output_tool_identity(
    wire_name: str, ctx: BridgeContext
) -> tuple[str | None, str]:
    ns_entry = ctx.namespaced_tools.get(wire_name)
    if ns_entry is not None:
        return ns_entry[0], ns_entry[1]
    return None, wire_name


def _build_response_envelope(
    *,
    response_id: str,
    created_at: int,
    model: str,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    incomplete_details: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": error,
        "incomplete_details": incomplete_details,
        "model": model,
        "output": output,
        "usage": usage,
    }


def _record_assistant_turn_in_history(
    response_id: str,
    ctx: BridgeContext,
    text: str,
    tool_calls: list[dict[str, Any]],
    reasoning_opaque: str,
) -> None:
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": text if text or not tool_calls else None,
    }
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    if reasoning_opaque:
        assistant_msg["reasoning_opaque"] = reasoning_opaque
    _store_response_history(
        response_id, [*ctx.input_messages, assistant_msg]
    )


def chat_completion_to_responses_object(
    chat_resp: dict[str, Any], ctx: BridgeContext
) -> dict[str, Any]:
    """Convert a non-streaming `/chat/completions` JSON response into a `response` object."""
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created_at = int(chat_resp.get("created") or time.time())

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_opaque = ""
    raw_tool_calls: list[dict[str, Any]] = []
    finish_reason: str | None = None

    for choice in chat_resp.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        if choice.get("finish_reason"):
            finish_reason = str(choice["finish_reason"])
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            text_parts.append(content)
        r_text, r_opaque = _extract_reasoning_fields(msg)
        if r_text:
            reasoning_parts.append(r_text)
        if r_opaque:
            reasoning_opaque = r_opaque
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                raw_tool_calls.append(tc)

    full_text = "".join(text_parts)
    full_reasoning = "".join(reasoning_parts)
    output: list[dict[str, Any]] = []

    show_summary = bool(ctx.emit_reasoning_summary and full_reasoning)
    show_encrypted = bool(ctx.include_encrypted_reasoning and reasoning_opaque)
    if show_summary or show_encrypted:
        rs_item: dict[str, Any] = {
            "id": f"rs_{uuid.uuid4().hex[:24]}",
            "type": "reasoning",
            "summary": (
                [{"type": "summary_text", "text": full_reasoning}]
                if show_summary
                else []
            ),
        }
        if show_encrypted:
            rs_item["encrypted_content"] = reasoning_opaque
        output.append(rs_item)

    if full_text or not raw_tool_calls:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": full_text,
                        "annotations": [],
                    }
                ],
            }
        )

    history_tool_calls: list[dict[str, Any]] = []
    for tc in raw_tool_calls:
        call_id = str(tc.get("id") or f"call_{uuid.uuid4().hex[:16]}")
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        fn_name = str(fn.get("name") or "")
        raw_args = fn.get("arguments")
        args_str = (
            raw_args
            if isinstance(raw_args, str) and raw_args
            else (json.dumps(raw_args) if raw_args is not None else "{}")
        )
        history_tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": fn_name, "arguments": args_str},
            }
        )

        ns_name, out_name = _resolve_output_tool_identity(fn_name, ctx)
        if fn_name in ctx.custom_tool_names:
            tc_item: dict[str, Any] = {
                "id": f"ctc_{uuid.uuid4().hex[:24]}",
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": call_id,
                "name": out_name,
                "input": _unwrap_custom_tool_input(args_str),
            }
        else:
            tc_item = {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": out_name,
                "arguments": args_str,
            }
        if ns_name is not None:
            tc_item["namespace"] = ns_name
        output.append(tc_item)

    _record_assistant_turn_in_history(
        response_id, ctx, full_text, history_tool_calls, reasoning_opaque
    )

    is_incomplete = finish_reason == "length"
    status = "incomplete" if is_incomplete else "completed"
    incomplete_details = (
        {"reason": "max_output_tokens"} if is_incomplete else None
    )
    usage = _convert_usage(chat_resp.get("usage"))

    return _build_response_envelope(
        response_id=response_id,
        created_at=created_at,
        model=ctx.model,
        status=status,
        output=output,
        usage=usage,
        incomplete_details=incomplete_details,
    )


async def iter_responses_events_from_chat_sse(
    lines: AsyncIterator[str], ctx: BridgeContext
) -> AsyncIterator[dict[str, Any]]:
    """
    Consume `/chat/completions` SSE lines and yield OpenAI Responses API event dicts.
    """
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    seq = 0

    def next_seq() -> int:
        nonlocal seq
        current = seq
        seq += 1
        return current

    initial_resp = _build_response_envelope(
        response_id=response_id,
        created_at=created_at,
        model=ctx.model,
        status="in_progress",
        output=[],
    )
    yield {
        "type": "response.created",
        "sequence_number": next_seq(),
        "response": initial_resp,
    }
    yield {
        "type": "response.in_progress",
        "sequence_number": next_seq(),
        "response": copy.deepcopy(initial_resp),
    }

    final_output: list[dict[str, Any]] = []

    reasoning_item_id = f"rs_{uuid.uuid4().hex[:24]}"
    reasoning_output_index: int | None = None
    reasoning_open = False
    reasoning_summary_part_open = False
    reasoning_chunks: list[str] = []
    reasoning_opaque = ""

    msg_item_id = f"msg_{uuid.uuid4().hex[:24]}"
    msg_output_index: int | None = None
    msg_open = False
    text_chunks: list[str] = []

    # Keyed by (choice_index, tc_index, generation) so parallel tool calls across
    # either tool_calls[].index or multiple choices never collide.
    tool_states: OrderedDict[tuple[int, int, int], dict[str, Any]] = OrderedDict()
    slot_generations: dict[tuple[int, int], int] = {}

    finish_reason: str | None = None
    usage_obj: dict[str, Any] | None = None

    def open_reasoning_if_needed() -> list[dict[str, Any]]:
        nonlocal reasoning_output_index, reasoning_open
        events: list[dict[str, Any]] = []
        if reasoning_output_index is None:
            reasoning_output_index = len(final_output)
            placeholder: dict[str, Any] = {
                "id": reasoning_item_id,
                "type": "reasoning",
                "summary": [],
            }
            final_output.append(placeholder)
            reasoning_open = True
            events.append(
                {
                    "type": "response.output_item.added",
                    "sequence_number": next_seq(),
                    "output_index": reasoning_output_index,
                    "item": copy.deepcopy(placeholder),
                }
            )
        return events

    def close_reasoning_if_open() -> list[dict[str, Any]]:
        nonlocal reasoning_open, reasoning_summary_part_open
        events: list[dict[str, Any]] = []
        if not reasoning_open or reasoning_output_index is None:
            return events

        full_reasoning = "".join(reasoning_chunks)
        summary_list: list[dict[str, Any]] = []
        if reasoning_summary_part_open:
            part_obj = {"type": "summary_text", "text": full_reasoning}
            summary_list.append(part_obj)
            events.append(
                {
                    "type": "response.reasoning_summary_text.done",
                    "sequence_number": next_seq(),
                    "item_id": reasoning_item_id,
                    "output_index": reasoning_output_index,
                    "summary_index": 0,
                    "text": full_reasoning,
                }
            )
            events.append(
                {
                    "type": "response.reasoning_summary_part.done",
                    "sequence_number": next_seq(),
                    "item_id": reasoning_item_id,
                    "output_index": reasoning_output_index,
                    "summary_index": 0,
                    "part": part_obj,
                }
            )
            reasoning_summary_part_open = False

        completed_rs: dict[str, Any] = {
            "id": reasoning_item_id,
            "type": "reasoning",
            "summary": summary_list,
        }
        if ctx.include_encrypted_reasoning and reasoning_opaque:
            completed_rs["encrypted_content"] = reasoning_opaque
        final_output[reasoning_output_index] = completed_rs
        reasoning_open = False
        events.append(
            {
                "type": "response.output_item.done",
                "sequence_number": next_seq(),
                "output_index": reasoning_output_index,
                "item": copy.deepcopy(completed_rs),
            }
        )
        return events

    def close_message_if_open() -> list[dict[str, Any]]:
        nonlocal msg_open
        events: list[dict[str, Any]] = []
        if not msg_open or msg_output_index is None:
            return events

        full_text = "".join(text_chunks)
        part_obj = {"type": "output_text", "text": full_text, "annotations": []}
        completed_msg = {
            "id": msg_item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [part_obj],
        }
        final_output[msg_output_index] = completed_msg
        msg_open = False
        events.append(
            {
                "type": "response.output_text.done",
                "sequence_number": next_seq(),
                "item_id": msg_item_id,
                "output_index": msg_output_index,
                "content_index": 0,
                "text": full_text,
            }
        )
        events.append(
            {
                "type": "response.content_part.done",
                "sequence_number": next_seq(),
                "item_id": msg_item_id,
                "output_index": msg_output_index,
                "content_index": 0,
                "part": copy.deepcopy(part_obj),
            }
        )
        events.append(
            {
                "type": "response.output_item.done",
                "sequence_number": next_seq(),
                "output_index": msg_output_index,
                "item": copy.deepcopy(completed_msg),
            }
        )
        return events

    async for raw_line in lines:
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload_str = line[5:].strip()
        if payload_str == "[DONE]":
            break

        try:
            chunk = json.loads(payload_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue

        if isinstance(chunk.get("error"), dict):
            err = chunk["error"]
            failed_resp = _build_response_envelope(
                response_id=response_id,
                created_at=created_at,
                model=ctx.model,
                status="failed",
                output=final_output,
                error={
                    "code": str(err.get("code") or "server_error"),
                    "message": str(err.get("message") or "Upstream stream error"),
                },
            )
            yield {
                "type": "response.failed",
                "sequence_number": next_seq(),
                "response": failed_resp,
            }
            return

        if isinstance(chunk.get("usage"), dict):
            usage_obj = _convert_usage(chunk["usage"])

        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            c_idx = int(choice.get("index") or 0)
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])

            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue

            r_text, r_opaque = _extract_reasoning_fields(delta)
            if r_opaque:
                reasoning_opaque = r_opaque
                if (
                    ctx.include_encrypted_reasoning
                    and reasoning_output_index is not None
                    and not reasoning_open
                ):
                    final_output[reasoning_output_index][
                        "encrypted_content"
                    ] = reasoning_opaque

            if r_text:
                reasoning_chunks.append(r_text)
                if ctx.emit_reasoning_summary and msg_output_index is None and not tool_states:
                    for ev in open_reasoning_if_needed():
                        yield ev
                    if not reasoning_summary_part_open and reasoning_output_index is not None:
                        reasoning_summary_part_open = True
                        yield {
                            "type": "response.reasoning_summary_part.added",
                            "sequence_number": next_seq(),
                            "item_id": reasoning_item_id,
                            "output_index": reasoning_output_index,
                            "summary_index": 0,
                            "part": {"type": "summary_text", "text": ""},
                        }
                    if reasoning_output_index is not None:
                        yield {
                            "type": "response.reasoning_summary_text.delta",
                            "sequence_number": next_seq(),
                            "item_id": reasoning_item_id,
                            "output_index": reasoning_output_index,
                            "summary_index": 0,
                            "delta": r_text,
                        }

            content_delta = delta.get("content")
            if isinstance(content_delta, str) and content_delta:
                for ev in close_reasoning_if_open():
                    yield ev
                if not msg_open:
                    msg_output_index = len(final_output)
                    in_progress_msg = {
                        "id": msg_item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                    final_output.append(in_progress_msg)
                    msg_open = True
                    yield {
                        "type": "response.output_item.added",
                        "sequence_number": next_seq(),
                        "output_index": msg_output_index,
                        "item": copy.deepcopy(in_progress_msg),
                    }
                    yield {
                        "type": "response.content_part.added",
                        "sequence_number": next_seq(),
                        "item_id": msg_item_id,
                        "output_index": msg_output_index,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    }
                text_chunks.append(content_delta)
                yield {
                    "type": "response.output_text.delta",
                    "sequence_number": next_seq(),
                    "item_id": msg_item_id,
                    "output_index": msg_output_index,
                    "content_index": 0,
                    "delta": content_delta,
                }

            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                for ev in close_reasoning_if_open():
                    yield ev
                for ev in close_message_if_open():
                    yield ev

                tc_idx = int(tc.get("index") or 0)
                slot = (c_idx, tc_idx)
                incoming_id = tc.get("id")
                gen = slot_generations.get(slot, 0)
                key = (c_idx, tc_idx, gen)

                if (
                    key in tool_states
                    and isinstance(incoming_id, str)
                    and incoming_id
                    and incoming_id != tool_states[key]["call_id"]
                ):
                    gen += 1
                    slot_generations[slot] = gen
                    key = (c_idx, tc_idx, gen)

                fn_delta = (
                    tc.get("function")
                    if isinstance(tc.get("function"), dict)
                    else {}
                )
                name_piece = fn_delta.get("name")
                args_piece = fn_delta.get("arguments")

                if key not in tool_states:
                    slot_generations[slot] = gen
                    call_id = (
                        str(incoming_id)
                        if isinstance(incoming_id, str) and incoming_id
                        else f"call_{uuid.uuid4().hex[:16]}"
                    )
                    fn_name = str(name_piece or "")
                    ns_name, out_name = _resolve_output_tool_identity(
                        fn_name, ctx
                    )
                    is_custom = fn_name in ctx.custom_tool_names
                    item_id = (
                        f"ctc_{uuid.uuid4().hex[:24]}"
                        if is_custom
                        else f"fc_{uuid.uuid4().hex[:24]}"
                    )
                    out_idx = len(final_output)
                    state = {
                        "item_id": item_id,
                        "output_index": out_idx,
                        "call_id": call_id,
                        "name": fn_name,
                        "out_name": out_name,
                        "namespace": ns_name,
                        "is_custom": is_custom,
                        "args_chunks": [],
                    }
                    tool_states[key] = state
                    if is_custom:
                        placeholder_tc: dict[str, Any] = {
                            "id": item_id,
                            "type": "custom_tool_call",
                            "status": "in_progress",
                            "call_id": call_id,
                            "name": out_name,
                            "input": "",
                        }
                    else:
                        placeholder_tc = {
                            "id": item_id,
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": call_id,
                            "name": out_name,
                            "arguments": "",
                        }
                    if ns_name is not None:
                        placeholder_tc["namespace"] = ns_name
                    final_output.append(placeholder_tc)
                    yield {
                        "type": "response.output_item.added",
                        "sequence_number": next_seq(),
                        "output_index": out_idx,
                        "item": copy.deepcopy(placeholder_tc),
                    }
                else:
                    state = tool_states[key]
                    if isinstance(name_piece, str) and name_piece and not state["name"]:
                        state["name"] = name_piece
                        ns_name, out_name = _resolve_output_tool_identity(
                            name_piece, ctx
                        )
                        state["out_name"] = out_name
                        state["namespace"] = ns_name
                        state["is_custom"] = name_piece in ctx.custom_tool_names

                if isinstance(args_piece, str) and args_piece:
                    state["args_chunks"].append(args_piece)
                    if not state["is_custom"]:
                        yield {
                            "type": "response.function_call_arguments.delta",
                            "sequence_number": next_seq(),
                            "item_id": state["item_id"],
                            "output_index": state["output_index"],
                            "delta": args_piece,
                        }

    if (
        ctx.include_encrypted_reasoning
        and reasoning_opaque
        and reasoning_output_index is None
    ):
        rs_idx = len(final_output)
        rs_item = {
            "id": reasoning_item_id,
            "type": "reasoning",
            "summary": [],
            "encrypted_content": reasoning_opaque,
        }
        final_output.append(rs_item)
        yield {
            "type": "response.output_item.added",
            "sequence_number": next_seq(),
            "output_index": rs_idx,
            "item": copy.deepcopy(rs_item),
        }
        yield {
            "type": "response.output_item.done",
            "sequence_number": next_seq(),
            "output_index": rs_idx,
            "item": copy.deepcopy(rs_item),
        }

    for ev in close_reasoning_if_open():
        yield ev
    for ev in close_message_if_open():
        yield ev

    history_tool_calls: list[dict[str, Any]] = []
    for state in tool_states.values():
        raw_args = "".join(state["args_chunks"]) or "{}"
        history_tool_calls.append(
            {
                "id": state["call_id"],
                "type": "function",
                "function": {"name": state["name"], "arguments": raw_args},
            }
        )
        if state["is_custom"]:
            raw_input = _unwrap_custom_tool_input(raw_args)
            if raw_input:
                yield {
                    "type": "response.custom_tool_call_input.delta",
                    "sequence_number": next_seq(),
                    "item_id": state["item_id"],
                    "call_id": state["call_id"],
                    "output_index": state["output_index"],
                    "delta": raw_input,
                }
            yield {
                "type": "response.custom_tool_call_input.done",
                "sequence_number": next_seq(),
                "item_id": state["item_id"],
                "call_id": state["call_id"],
                "output_index": state["output_index"],
                "input": raw_input,
            }
            completed_tc: dict[str, Any] = {
                "id": state["item_id"],
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": state["call_id"],
                "name": state["out_name"],
                "input": raw_input,
            }
        else:
            yield {
                "type": "response.function_call_arguments.done",
                "sequence_number": next_seq(),
                "item_id": state["item_id"],
                "output_index": state["output_index"],
                "name": state["out_name"],
                "arguments": raw_args,
            }
            completed_tc = {
                "id": state["item_id"],
                "type": "function_call",
                "status": "completed",
                "call_id": state["call_id"],
                "name": state["out_name"],
                "arguments": raw_args,
            }
        if state["namespace"] is not None:
            completed_tc["namespace"] = state["namespace"]
        final_output[state["output_index"]] = completed_tc
        yield {
            "type": "response.output_item.done",
            "sequence_number": next_seq(),
            "output_index": state["output_index"],
            "item": copy.deepcopy(completed_tc),
        }

    # Ensure at least one assistant message item if neither text nor tool calls appeared
    has_content_or_tool = msg_output_index is not None or bool(tool_states)
    if not has_content_or_tool:
        empty_idx = len(final_output)
        empty_part = {"type": "output_text", "text": "", "annotations": []}
        empty_msg = {
            "id": msg_item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [empty_part],
        }
        final_output.append(empty_msg)
        yield {
            "type": "response.output_item.added",
            "sequence_number": next_seq(),
            "output_index": empty_idx,
            "item": {
                "id": msg_item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        }
        yield {
            "type": "response.content_part.added",
            "sequence_number": next_seq(),
            "item_id": msg_item_id,
            "output_index": empty_idx,
            "content_index": 0,
            "part": copy.deepcopy(empty_part),
        }
        yield {
            "type": "response.output_text.done",
            "sequence_number": next_seq(),
            "item_id": msg_item_id,
            "output_index": empty_idx,
            "content_index": 0,
            "text": "",
        }
        yield {
            "type": "response.content_part.done",
            "sequence_number": next_seq(),
            "item_id": msg_item_id,
            "output_index": empty_idx,
            "content_index": 0,
            "part": copy.deepcopy(empty_part),
        }
        yield {
            "type": "response.output_item.done",
            "sequence_number": next_seq(),
            "output_index": empty_idx,
            "item": copy.deepcopy(empty_msg),
        }

    full_text = "".join(text_chunks)
    _record_assistant_turn_in_history(
        response_id, ctx, full_text, history_tool_calls, reasoning_opaque
    )

    is_incomplete = finish_reason == "length"
    final_status = "incomplete" if is_incomplete else "completed"
    final_event_type = (
        "response.incomplete" if is_incomplete else "response.completed"
    )
    completed_resp = _build_response_envelope(
        response_id=response_id,
        created_at=created_at,
        model=ctx.model,
        status=final_status,
        output=final_output,
        usage=usage_obj,
        incomplete_details=(
            {"reason": "max_output_tokens"} if is_incomplete else None
        ),
    )
    yield {
        "type": final_event_type,
        "sequence_number": next_seq(),
        "response": completed_resp,
    }


def format_responses_sse_event(event: dict[str, Any]) -> bytes:
    """Serialize a Responses event dict as an SSE frame."""
    event_type = event.get("type", "message")
    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")
