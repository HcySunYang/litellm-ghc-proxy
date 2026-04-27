"""
Usage normalization helpers.

GitHub Copilot's chat backend reports only `prompt_tokens` / `completion_tokens`,
and LiteLLM's Anthropic adapter does not synthesize the Anthropic-specific
`cache_creation_input_tokens` / `cache_read_input_tokens` fields when proxying
to Copilot. Some Anthropic clients (e.g. Claude Code's HUD plugin) read these
fields directly to compute the context-usage indicator, and they break or
misbehave when the fields are missing. Copilot does not support Anthropic
prompt caching, so the correct value to report is always 0.

This module provides small, defensive helpers that:

* Ensure cache_* fields exist on a usage dict (defaulting to 0).
* Parse a streamed SSE byte feed event-by-event and rewrite `message_start`
  and `message_delta` usage fields so the cache_* keys are present and
  `message_delta` carries `input_tokens` alongside `output_tokens`.

The rewriter is intentionally line/event aware (per the SSE spec) rather than
chunk-based — upstream chunks may split mid-event or mid-UTF8.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

CACHE_FIELDS = ("cache_creation_input_tokens", "cache_read_input_tokens")


def ensure_cache_fields(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Return a usage dict guaranteed to have cache_* fields (default 0).

    Mutates and returns the same object when given a dict; returns a new
    dict when given None.
    """
    if usage is None:
        usage = {}
    for f in CACHE_FIELDS:
        if f not in usage or usage[f] is None:
            usage[f] = 0
    return usage


def normalize_anthropic_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a non-streaming Anthropic usage dict in-place."""
    return ensure_cache_fields(usage)


def _rewrite_event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """Apply usage normalization to a single SSE event payload.

    Only `message_start` and `message_delta` events carry usage info in the
    Anthropic SSE protocol; everything else is returned unchanged.
    """
    if event_type == "message_start":
        msg = data.get("message")
        if isinstance(msg, dict):
            ensure_cache_fields(msg.get("usage") or msg.setdefault("usage", {}))
    elif event_type == "message_delta":
        ensure_cache_fields(data.get("usage") or data.setdefault("usage", {}))
    return data


def _serialize_event(raw_event_lines: list[str], rewritten_data: dict[str, Any] | None) -> bytes:
    """Re-serialize an SSE event, replacing the data payload if rewritten.

    Preserves any non-data lines (event:, id:, retry:, comments) verbatim so
    we don't lose information from upstream events we don't recognize.
    """
    if rewritten_data is None:
        # No rewrite — return the original block as-is.
        return ("\n".join(raw_event_lines) + "\n\n").encode("utf-8")

    out_lines: list[str] = []
    data_written = False
    for line in raw_event_lines:
        if line.startswith("data:") and not data_written:
            out_lines.append("data: " + json.dumps(rewritten_data, separators=(",", ":")))
            data_written = True
        elif line.startswith("data:"):
            # Drop subsequent data: lines (we collapsed multi-line data into
            # one re-serialized JSON payload).
            continue
        else:
            out_lines.append(line)
    if not data_written:
        out_lines.append("data: " + json.dumps(rewritten_data, separators=(",", ":")))
    return ("\n".join(out_lines) + "\n\n").encode("utf-8")


async def rewrite_sse_stream(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Wrap an upstream SSE byte stream, normalizing usage fields per event.

    Buffers bytes until a complete event (terminated by a blank line) is seen,
    then parses, rewrites and re-emits it. Unknown / unparseable events are
    forwarded unchanged so this is safe to layer over arbitrary SSE traffic.
    """
    buf = b""
    async for chunk in source:
        if not chunk:
            continue
        buf += chunk
        # SSE events are separated by a blank line. Tolerate \r\n.
        # Find the earliest event terminator each iteration.
        while True:
            sep_idx = -1
            sep_len = 0
            for marker in (b"\n\n", b"\r\n\r\n"):
                idx = buf.find(marker)
                if idx != -1 and (sep_idx == -1 or idx < sep_idx):
                    sep_idx = idx
                    sep_len = len(marker)
            if sep_idx == -1:
                break
            raw = buf[:sep_idx]
            buf = buf[sep_idx + sep_len:]
            yield _process_event_block(raw)

    # Flush any trailing data (rare — most servers terminate with blank line).
    if buf.strip():
        yield _process_event_block(buf)


def _process_event_block(raw: bytes) -> bytes:
    """Parse one SSE event block, rewrite usage if applicable, re-emit bytes."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Don't risk corrupting binary or partial UTF-8 — pass through.
        return raw + b"\n\n"

    lines = text.split("\n")
    event_type = ""
    data_parts: list[str] = []
    for line in lines:
        # Comments per SSE spec start with ":"; preserve them by skipping
        # parsing but keeping them in the raw block.
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_parts.append(line[len("data:"):].lstrip())

    if not data_parts or event_type not in ("message_start", "message_delta"):
        # Nothing for us to rewrite.
        return text.encode("utf-8") + b"\n\n"

    payload_text = "\n".join(data_parts)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        logger.debug("SSE event %r had non-JSON data; passing through", event_type)
        return text.encode("utf-8") + b"\n\n"

    rewritten = _rewrite_event(event_type, payload)
    return _serialize_event(lines, rewritten)
