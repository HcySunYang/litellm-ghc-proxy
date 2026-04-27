"""
FastAPI middleware application.

Sits between Claude Code and LiteLLM, intercepting Anthropic server-side
web_search/web_fetch tools and executing them via SearXNG.
"""

import json
import logging
import os
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response

from .tools import extract_server_tools, SERVER_TOOL_PATTERNS
from .proxy import forward_to_litellm, proxy_messages_with_tools, LITELLM_URL
from .streaming import synthesize_sse_events
from .usage import normalize_anthropic_usage, rewrite_sse_stream

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Parse the internal LiteLLM URL so we can rewrite it in proxied responses
_litellm_parsed = urlparse(LITELLM_URL)
_LITELLM_ORIGIN = f"{_litellm_parsed.scheme}://{_litellm_parsed.netloc}"


def _rewrite_location(location: str, request: Request) -> str:
    """Rewrite Location headers that point to the internal LiteLLM origin."""
    if _LITELLM_ORIGIN in location:
        external_origin = f"{request.url.scheme}://{request.headers.get('host', request.url.netloc)}"
        return location.replace(_LITELLM_ORIGIN, external_origin)
    return location


app = FastAPI(
    title="Web Search Middleware",
    description="Middleware that intercepts Anthropic server-side web tools and executes them via SearXNG",
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "web-search-middleware"}


@app.api_route("/v1/messages", methods=["POST"])
async def messages(request: Request):
    """
    Main endpoint: intercept /v1/messages requests.

    If the request contains server-side web tools, run the agentic loop.
    Otherwise, passthrough to LiteLLM unchanged.
    """
    body_bytes = await request.body()
    headers = dict(request.headers)

    try:
        body = json.loads(body_bytes)
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "invalid_request_error", "message": "Invalid JSON body"}},
        )

    # Check if request contains server-side tools
    tools = body.get("tools", [])
    has_server_tools = any(
        SERVER_TOOL_PATTERNS.match(t.get("type", "")) for t in tools
    )

    if not has_server_tools:
        # Pure passthrough — no server-side tools, forward as-is
        logger.debug("No server-side tools detected, passthrough mode")
        return await _passthrough(request, body_bytes, body, headers)

    # Server-side tools detected — handle with agentic loop
    is_streaming = body.get("stream", False)
    logger.info(
        f"Server-side tools detected (streaming={is_streaming}), "
        f"entering agentic loop"
    )

    # Run the agentic loop (always internally non-streaming)
    response_data = await proxy_messages_with_tools(body, headers)

    if is_streaming:
        # Synthesize SSE events from the completed response
        return StreamingResponse(
            _stream_events(response_data),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return JSONResponse(content=response_data)


async def _stream_events(response_data: dict):
    """Generate SSE events from response data."""
    async for event in synthesize_sse_events(response_data):
        yield event


async def _passthrough(request: Request, body_bytes: bytes, body: dict, headers: dict):
    """
    Passthrough request to LiteLLM unchanged.

    For streaming requests, streams the response back as-is.
    For non-streaming, returns the JSON response.
    """
    is_streaming = body.get("stream", False)

    if is_streaming:
        # Stream passthrough
        resp = await forward_to_litellm(
            "/v1/messages", "POST", headers, body_bytes, stream=True
        )

        async def _stream():
            try:
                async for chunk in rewrite_sse_stream(resp.aiter_bytes()):
                    yield chunk
            finally:
                await resp.aclose()
                if hasattr(resp, "_client"):
                    await resp._client.aclose()

        # Forward response headers
        response_headers = {}
        for key, value in resp.headers.items():
            if key.lower() not in ("transfer-encoding", "connection", "content-length"):
                response_headers[key] = value

        return StreamingResponse(
            _stream(),
            status_code=resp.status_code,
            headers=response_headers,
            media_type=resp.headers.get("content-type", "text/event-stream"),
        )
    else:
        # Non-streaming passthrough
        resp = await forward_to_litellm(
            "/v1/messages", "POST", headers, body_bytes, stream=False
        )
        content_type = resp.headers.get("content-type", "application/json")
        content = resp.content
        # Normalize usage so Anthropic-shape clients (e.g. Claude Code's HUD)
        # always see the cache_* fields. Skip if the upstream returned a
        # non-JSON error or an unexpected shape.
        if "json" in content_type.lower() and resp.status_code == 200 and content:
            try:
                data = json.loads(content)
                if isinstance(data, dict) and isinstance(data.get("usage"), dict):
                    normalize_anthropic_usage(data["usage"])
                    content = json.dumps(data).encode("utf-8")
            except (json.JSONDecodeError, ValueError):
                pass
        response_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "connection")
        }
        return Response(
            content=content,
            status_code=resp.status_code,
            headers=response_headers,
            media_type=content_type,
        )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all(request: Request, path: str):
    """
    Catch-all: forward any other request to LiteLLM unchanged.
    """
    body = await request.body()
    headers = dict(request.headers)
    method = request.method

    full_path = f"/{path}"
    if request.url.query:
        full_path += f"?{request.url.query}"

    logger.debug(f"Passthrough: {method} {full_path}")

    resp = await forward_to_litellm(full_path, method, headers, body, stream=False)

    # Forward response, rewriting Location headers that point to internal LiteLLM
    response_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("transfer-encoding", "connection", "content-encoding")
    }
    if "location" in response_headers:
        response_headers["location"] = _rewrite_location(response_headers["location"], request)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=response_headers,
        media_type=resp.headers.get("content-type"),
    )
