"""
Bidirectional conversion between Anthropic Messages API and OpenAI Chat Completions API.

Used by the agentic web-tool loop because LiteLLM v1.82.3's `/v1/messages` endpoint
silently drops the `tools` array when proxying to `github_copilot/*` models. By
translating to OpenAI shape and calling `/chat/completions` instead, tool use works
correctly. The middleware externally still speaks the Anthropic Messages API.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def anthropic_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic tool definitions to OpenAI function tool definitions."""
    out: list[dict[str, Any]] = []
    for t in tools:
        if "input_schema" in t:
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
            )
        elif t.get("type") == "function" and "function" in t:
            out.append(t)
        else:
            logger.warning(f"Skipping unsupported tool definition: keys={list(t.keys())}")
    return out


def anthropic_tool_choice_to_openai(tc: Any) -> Any:
    if tc is None:
        return None
    if isinstance(tc, str):
        return tc
    if isinstance(tc, dict):
        kind = tc.get("type")
        if kind == "auto":
            return "auto"
        if kind == "any":
            return "required"
        if kind == "tool":
            return {"type": "function", "function": {"name": tc.get("name", "")}}
        if kind == "none":
            return "none"
    return None


def _system_to_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(
            b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _image_block_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    src = block.get("source") or {}
    if src.get("type") == "base64":
        media = src.get("media_type", "image/png")
        data = src.get("data", "")
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
    if src.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": src.get("url", "")}}
    return None


def _user_blocks_to_openai_messages(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a user-message content list into one or more OpenAI messages.

    Returns a list because tool_result blocks must become separate {role:tool} messages.
    Text and image blocks coalesce into a single {role:user} message; if only text,
    content is a string for maximum compatibility, otherwise a multi-part list.
    """
    parts: list[dict[str, Any]] = []
    text_only = True
    tool_msgs: list[dict[str, Any]] = []

    for b in content:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            parts.append({"type": "text", "text": b.get("text", "")})
        elif bt == "image":
            img = _image_block_to_openai(b)
            if img is not None:
                parts.append(img)
                text_only = False
            else:
                parts.append({"type": "text", "text": "[unsupported image]"})
        elif bt == "tool_result":
            tc_id = b.get("tool_use_id", "")
            inner = b.get("content", "")
            if isinstance(inner, list):
                inner = "\n".join(
                    x.get("text", "")
                    for x in inner
                    if isinstance(x, dict) and x.get("type") == "text"
                )
            elif not isinstance(inner, str):
                inner = json.dumps(inner)
            tool_msgs.append({"role": "tool", "tool_call_id": tc_id, "content": inner})
        else:
            logger.debug(f"Dropping unknown user content block type: {bt}")

    out: list[dict[str, Any]] = []
    if parts:
        if text_only:
            text = "\n".join(p["text"] for p in parts)
            out.append({"role": "user", "content": text})
        else:
            out.append({"role": "user", "content": parts})
    out.extend(tool_msgs)
    return out


def _assistant_blocks_to_openai_message(content: list[dict[str, Any]]) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            text_parts.append(b.get("text", ""))
        elif bt == "tool_use":
            tool_calls.append(
                {
                    "id": b.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": b.get("name", ""),
                        "arguments": json.dumps(b.get("input") or {}),
                    },
                }
            )
        # server_tool_use / web_search_tool_result are middleware-internal; they
        # should never appear in inbound assistant messages but are safe to drop.

    msg: dict[str, Any] = {"role": "assistant"}
    text = "\n".join(p for p in text_parts if p)
    msg["content"] = text if text else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def anthropic_messages_to_openai(
    messages: list[dict[str, Any]], system: Any = None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sys_text = _system_to_text(system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue
        if role == "assistant":
            out.append(_assistant_blocks_to_openai_message(content))
        elif role == "user":
            out.extend(_user_blocks_to_openai_messages(content))
        else:
            text = "\n".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            out.append({"role": role, "content": text})
    return out


def anthropic_request_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": body.get("model"),
        "messages": anthropic_messages_to_openai(body.get("messages", []), body.get("system")),
    }
    for k in ("max_tokens", "temperature", "top_p", "user", "stream", "n"):
        if k in body:
            out[k] = body[k]
    if "stop_sequences" in body:
        out["stop"] = body["stop_sequences"]
    tools = body.get("tools") or []
    if tools:
        oai_tools = anthropic_tools_to_openai(tools)
        if oai_tools:
            out["tools"] = oai_tools
    if "tool_choice" in body:
        oc = anthropic_tool_choice_to_openai(body["tool_choice"])
        if oc is not None:
            out["tool_choice"] = oc
    return out


def _pick_choice(choices: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick a single choice deterministically.

    GitHub Copilot sometimes returns multiple choices for one request; merging them
    would fabricate an assistant turn. Prefer the first choice with tool_calls;
    otherwise the first non-empty choice; otherwise the first.
    """
    if not choices:
        return {}
    for c in choices:
        msg = c.get("message") or {}
        if msg.get("tool_calls"):
            return c
    for c in choices:
        msg = c.get("message") or {}
        if msg.get("content"):
            return c
    return choices[0]


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_value": parsed}
        except json.JSONDecodeError:
            logger.warning(f"Tool arguments are not valid JSON: {raw[:200]!r}")
            return {"_raw": raw}
    return {"_value": raw}


def openai_response_to_anthropic(resp: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions response to an Anthropic Messages response."""
    choices = resp.get("choices") or []
    if len(choices) > 1:
        logger.info(f"OpenAI response had {len(choices)} choices; selecting one deterministically")
    choice = _pick_choice(choices)
    msg = choice.get("message") or {}

    content_blocks: list[dict[str, Any]] = []
    text = msg.get("content")
    if isinstance(text, list):
        # Multi-part content (rare on response side); flatten text parts
        text = "".join(
            p.get("text", "") for p in text if isinstance(p, dict) and p.get("type") == "text"
        )
    if text:
        content_blocks.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "input": _parse_tool_args(fn.get("arguments")),
            }
        )

    finish = _FINISH_REASON_MAP.get(choice.get("finish_reason") or "", "end_turn")
    usage = resp.get("usage") or {}
    # Anthropic clients (e.g. Claude Code HUD) read cache_* fields directly to
    # compute context usage; OpenAI-shape upstreams (Copilot) don't report
    # them, so we always include 0s. If the upstream did happen to surface
    # cache info under any of the known synonyms, prefer that.
    cache_creation = (
        usage.get("cache_creation_input_tokens")
        or usage.get("prompt_cache_miss_tokens")
        or 0
    )
    cache_read = (
        usage.get("cache_read_input_tokens")
        or usage.get("prompt_cache_hit_tokens")
        or 0
    )
    return {
        "id": resp.get("id", "msg_unknown"),
        "type": "message",
        "role": "assistant",
        "model": resp.get("model", "unknown"),
        "content": content_blocks,
        "stop_reason": finish,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        },
    }
