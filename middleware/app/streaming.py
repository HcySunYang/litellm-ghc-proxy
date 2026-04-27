"""
SSE event synthesis for streaming responses.

When the agentic loop runs internally (non-streaming), we need to
synthesize Anthropic-format SSE events to stream back to Claude Code.
"""

import json
import logging
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)


async def synthesize_sse_events(
    response: dict[str, Any],
) -> AsyncGenerator[str, None]:
    """
    Convert a complete Anthropic response dict into a sequence of SSE events
    matching Anthropic's streaming format.

    Event types:
    - message_start
    - content_block_start / content_block_delta / content_block_stop
    - message_delta
    - message_stop
    """
    # 1. message_start
    # We have the real input_tokens here (the agentic loop runs upstream
    # non-streaming and waits for the final usage), so emit it now rather
    # than 0 — Claude Code's HUD reads message_start.usage to size the
    # context indicator.
    src_usage = response.get("usage", {}) or {}
    msg_start = {
        "type": "message_start",
        "message": {
            "id": response.get("id", "msg_unknown"),
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": response.get("model", "unknown"),
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": src_usage.get("input_tokens", 0),
                "output_tokens": 0,
                "cache_creation_input_tokens": src_usage.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": src_usage.get("cache_read_input_tokens", 0),
            },
        },
    }
    yield _sse_line(msg_start)

    # 2. Content blocks
    content = response.get("content", [])
    for idx, block in enumerate(content):
        block_type = block.get("type", "text")

        if block_type == "text":
            # content_block_start
            yield _sse_line({
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            })

            # content_block_delta — send the full text as one delta
            text = block.get("text", "")
            if text:
                yield _sse_line({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": text},
                })

            # content_block_stop
            yield _sse_line({
                "type": "content_block_stop",
                "index": idx,
            })

        elif block_type == "server_tool_use":
            # content_block_start for server_tool_use
            yield _sse_line({
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "server_tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": {},
                },
            })

            # Send input as a delta
            input_data = block.get("input", {})
            if input_data:
                yield _sse_line({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(input_data),
                    },
                })

            # content_block_stop
            yield _sse_line({
                "type": "content_block_stop",
                "index": idx,
            })

        elif block_type == "web_search_tool_result":
            # content_block_start
            yield _sse_line({
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "web_search_tool_result",
                    "tool_use_id": block.get("tool_use_id", ""),
                    "content": block.get("content", []),
                },
            })

            # content_block_stop (no delta for result blocks)
            yield _sse_line({
                "type": "content_block_stop",
                "index": idx,
            })

        elif block_type == "tool_use":
            # Regular tool_use (non-web tools)
            yield _sse_line({
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": {},
                },
            })

            input_data = block.get("input", {})
            if input_data:
                yield _sse_line({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(input_data),
                    },
                })

            yield _sse_line({
                "type": "content_block_stop",
                "index": idx,
            })

        else:
            # Unknown block type — pass through as-is
            yield _sse_line({
                "type": "content_block_start",
                "index": idx,
                "content_block": block,
            })
            yield _sse_line({
                "type": "content_block_stop",
                "index": idx,
            })

    # 3. message_delta
    yield _sse_line({
        "type": "message_delta",
        "delta": {
            "stop_reason": response.get("stop_reason", "end_turn"),
            "stop_sequence": response.get("stop_sequence"),
        },
        "usage": {
            "input_tokens": src_usage.get("input_tokens", 0),
            "output_tokens": src_usage.get("output_tokens", 0),
            "cache_creation_input_tokens": src_usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": src_usage.get("cache_read_input_tokens", 0),
        },
    })

    # 4. message_stop
    yield _sse_line({"type": "message_stop"})


def _sse_line(data: dict[str, Any]) -> str:
    """Format a dict as an SSE event line."""
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n"
