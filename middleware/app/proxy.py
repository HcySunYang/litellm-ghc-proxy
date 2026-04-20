"""
Core proxy logic.

Forwards requests to LiteLLM, handles the agentic tool-use loop
for web search and web fetch operations.
"""

import copy
import logging
import os
from typing import Any

import httpx

from . import search as search_module
from . import fetch as fetch_module
from .converters import (
    anthropic_request_to_openai,
    openai_response_to_anthropic,
)
from .tools import (
    extract_server_tools,
    inject_regular_tools,
    is_web_tool_call,
    convert_tool_use_to_server_format,
    build_search_result_block,
    build_fetch_result_block,
    convert_response_content,
)

logger = logging.getLogger(__name__)

LITELLM_URL = os.environ.get("LITELLM_URL", "http://ghc-proxy:4000")
MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", "10"))
PROXY_TIMEOUT = float(os.environ.get("PROXY_TIMEOUT", "300"))


async def forward_to_litellm(
    path: str,
    method: str,
    headers: dict[str, str],
    body: bytes | None = None,
    stream: bool = False,
) -> httpx.Response:
    """
    Forward a raw request to LiteLLM.

    For streaming, returns the response with stream open (caller must close).
    """
    url = f"{LITELLM_URL}{path}"

    # Filter out hop-by-hop headers and host
    forward_headers = {
        k: v
        for k, v in headers.items()
        if k.lower() not in ("host", "transfer-encoding", "connection", "content-length")
    }

    client = httpx.AsyncClient(timeout=PROXY_TIMEOUT)

    try:
        if stream:
            req = client.build_request(method, url, headers=forward_headers, content=body)
            resp = await client.send(req, stream=True)
            # Attach client to response so caller can close both
            resp._client = client  # type: ignore
            return resp
        else:
            resp = await client.request(
                method, url, headers=forward_headers, content=body
            )
            await client.aclose()
            return resp
    except Exception:
        await client.aclose()
        raise


async def proxy_messages_with_tools(
    request_body: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    """
    Handle a /v1/messages request that contains server-side tools.

    Runs the agentic loop:
    1. Replace server-side tools with regular tools
    2. Forward to LiteLLM
    3. If model calls web_search/web_fetch, execute and loop back
    4. Convert final response to server-side tool format

    Returns the final response dict in Anthropic format.

    NOTE: We drive the loop against LiteLLM's `/chat/completions` endpoint instead
    of `/v1/messages` because v1.82.3 silently drops the `tools` array when
    proxying to `github_copilot/*` models on the Messages surface. Externally we
    still speak the Anthropic Messages API; conversion happens here.
    """
    body = copy.deepcopy(request_body)

    # Step 1: Extract and replace server-side tools (still in Anthropic shape)
    tools = body.get("tools", [])
    remaining_tools, server_tools = extract_server_tools(tools)
    body["tools"] = inject_regular_tools(remaining_tools, server_tools)

    # Force non-streaming for the agentic loop
    body["stream"] = False

    # Translate the entire Anthropic request to OpenAI shape once.
    oai_body = anthropic_request_to_openai(body)
    oai_body["stream"] = False

    # Accumulate server_tool_use + web_search_tool_result blocks from
    # intermediate rounds so they appear in the final response.
    # Claude Code needs to see these to display "Did N searches".
    accumulated_blocks: list[dict[str, Any]] = []

    resp: dict[str, Any] | None = None

    for round_num in range(MAX_TOOL_ROUNDS):
        logger.info(f"Agentic loop round {round_num + 1}/{MAX_TOOL_ROUNDS}")

        # Forward to LiteLLM /chat/completions
        oai_resp = await _forward_chat_completions(oai_body, headers)

        if oai_resp is None:
            return _error_response("Failed to get response from LiteLLM")

        # If the upstream returned an OpenAI-shape error (no `choices`), surface it.
        if "choices" not in oai_resp:
            err_msg = (
                oai_resp.get("error", {}).get("message")
                if isinstance(oai_resp.get("error"), dict)
                else str(oai_resp)[:500]
            )
            logger.error(f"Upstream error from /chat/completions: {err_msg}")
            return _error_response(err_msg or "Upstream error")

        # Convert to Anthropic format so existing block-handling logic applies.
        resp = openai_response_to_anthropic(oai_resp)

        content = resp.get("content") or []
        tool_calls = [b for b in content if is_web_tool_call(b)]

        if not tool_calls:
            # No web tool calls — we're done.
            # Prepend accumulated server-side blocks before the final text.
            resp["content"] = accumulated_blocks + content
            return resp

        # Pick out the original OpenAI assistant message so we can echo it
        # verbatim back into the conversation (preserves tool_call IDs).
        oai_assistant_msg = _extract_assistant_message(oai_resp)
        oai_body["messages"].append(oai_assistant_msg)

        # Execute tool calls
        for tc in tool_calls:
            tool_id = tc["id"]
            tool_name = tc["name"]
            tool_input = tc.get("input", {})

            logger.info(f"Executing {tool_name} (id={tool_id}): {tool_input}")

            if tool_name == "web_search":
                query = tool_input.get("query", "")
                results = await search_module.search(query)
                result_block = build_search_result_block(tool_id, results)

                # Accumulate server_tool_use + result for final response
                accumulated_blocks.append(convert_tool_use_to_server_format(tc))
                accumulated_blocks.append(result_block)

                # Build OpenAI-style tool result message for the next round
                result_text = _format_search_results(results)
                oai_body["messages"].append(
                    {"role": "tool", "tool_call_id": tool_id, "content": result_text}
                )

            elif tool_name == "web_fetch":
                url = tool_input.get("url", "")
                prompt = tool_input.get("prompt", "")
                fetch_result = await fetch_module.fetch_url(url, prompt)
                result_block = build_fetch_result_block(
                    tool_id, url, fetch_result["content"], fetch_result["title"]
                )

                # Accumulate server_tool_use + result for final response
                accumulated_blocks.append(convert_tool_use_to_server_format(tc))
                accumulated_blocks.append(result_block)

                oai_body["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": fetch_result["content"][:50000],
                    }
                )

    # Max rounds exceeded
    logger.warning(f"Agentic loop exceeded {MAX_TOOL_ROUNDS} rounds")
    if resp is not None:
        resp["content"] = accumulated_blocks + (resp.get("content") or [])
        return resp

    return _error_response("Agentic loop exceeded maximum rounds")


def _extract_assistant_message(oai_resp: dict[str, Any]) -> dict[str, Any]:
    """Return the OpenAI assistant message we should echo back to continue the loop.

    Mirrors converters._pick_choice: prefer the choice with tool_calls so we keep
    matching tool_call_ids when posting tool results in the next round.
    """
    choices = oai_resp.get("choices") or []
    chosen = None
    for c in choices:
        msg = c.get("message") or {}
        if msg.get("tool_calls"):
            chosen = msg
            break
    if chosen is None:
        for c in choices:
            msg = c.get("message") or {}
            if msg.get("content"):
                chosen = msg
                break
    if chosen is None and choices:
        chosen = choices[0].get("message") or {}
    if chosen is None:
        return {"role": "assistant", "content": ""}
    # Strip non-standard provider fields that some upstreams add.
    out: dict[str, Any] = {"role": "assistant"}
    if chosen.get("content") is not None:
        out["content"] = chosen.get("content")
    else:
        out["content"] = None
    if chosen.get("tool_calls"):
        out["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "type": tc.get("type", "function"),
                "function": {
                    "name": (tc.get("function") or {}).get("name", ""),
                    "arguments": (tc.get("function") or {}).get("arguments", "{}"),
                },
            }
            for tc in chosen["tool_calls"]
        ]
    return out


async def _forward_chat_completions(
    body: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any] | None:
    """Forward a JSON request to LiteLLM /chat/completions and return parsed JSON."""
    url = f"{LITELLM_URL}/chat/completions"

    forward_headers = {
        k: v
        for k, v in headers.items()
        if k.lower()
        not in (
            "host",
            "transfer-encoding",
            "connection",
            "content-length",
            "anthropic-version",
            "anthropic-beta",
            "content-type",
        )
    }
    forward_headers["content-type"] = "application/json"

    try:
        async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
            resp = await client.post(url, json=body, headers=forward_headers)

            if resp.status_code != 200:
                logger.error(
                    f"LiteLLM /chat/completions returned {resp.status_code}: {resp.text[:500]}"
                )
                try:
                    return resp.json()
                except Exception:
                    return _error_response(
                        f"LiteLLM returned HTTP {resp.status_code}: {resp.text[:200]}"
                    )

            return resp.json()
    except Exception as e:
        logger.error(f"Failed to forward to LiteLLM /chat/completions: {e}")
        return None


async def _forward_json(
    body: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any] | None:
    """[Deprecated] Forward a JSON request to LiteLLM /v1/messages.

    Retained for any future passthrough callers; the agentic loop no longer uses
    this because /v1/messages drops `tools` when targeting `github_copilot/*`.
    """
    url = f"{LITELLM_URL}/v1/messages"

    forward_headers = {
        k: v
        for k, v in headers.items()
        if k.lower() not in ("host", "transfer-encoding", "connection", "content-length")
    }
    forward_headers["content-type"] = "application/json"

    try:
        async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
            resp = await client.post(url, json=body, headers=forward_headers)

            if resp.status_code != 200:
                logger.error(
                    f"LiteLLM returned {resp.status_code}: {resp.text[:500]}"
                )
                # Return the error as-is
                try:
                    return resp.json()
                except Exception:
                    return _error_response(
                        f"LiteLLM returned HTTP {resp.status_code}: {resp.text[:200]}"
                    )

            return resp.json()
    except Exception as e:
        logger.error(f"Failed to forward to LiteLLM: {e}")
        return None


def _format_search_results(results: list[dict[str, Any]]) -> str:
    """Format search results as readable text for the model."""
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        lines.append(f"[{i}] {title}")
        if url:
            lines.append(f"    URL: {url}")
        if snippet:
            lines.append(f"    {snippet}")
        lines.append("")
    return "\n".join(lines)


def _error_response(message: str) -> dict[str, Any]:
    """Build an error response in Anthropic format."""
    return {
        "id": "msg_error",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": f"Error: {message}"}],
        "model": "unknown",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
