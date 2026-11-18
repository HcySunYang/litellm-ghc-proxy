"""
FastAPI middleware application.

Sits in front of LiteLLM, adding Codex model metadata and intercepting
Anthropic server-side web_search/web_fetch tools for execution via SearXNG.
"""

import asyncio
import hashlib
import json
import logging
import os
from urllib.parse import urlparse

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.websockets import WebSocketState
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from .codex_models import build_codex_models_response
from .tools import extract_server_tools, SERVER_TOOL_PATTERNS
from .proxy import (
    LITELLM_URL,
    PROXY_TIMEOUT,
    forward_to_litellm,
    proxy_messages_with_tools,
)
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
    title="LiteLLM Compatibility Middleware",
    description="Adds Codex model metadata and Anthropic server-side web tools.",
    docs_url="/middleware/docs",
    openapi_url="/middleware/openapi.json",
    redoc_url=None,
)


@app.get("/middleware/health")
async def middleware_health():
    """Health check endpoint."""
    return {"status": "ok", "service": "litellm-compatibility-middleware"}


@app.get("/health")
async def health(request: Request):
    """Keep the port-80 middleware probe while exposing LiteLLM health on 4000."""
    external_port = request.headers.get("x-forwarded-port")
    if external_port is None and request.url.port is not None:
        external_port = str(request.url.port)
    if external_port == "4000":
        return await _proxy_raw_request(request, "health")
    return await middleware_health()


def _copy_upstream_headers(
    response: Response,
    upstream,
    *,
    request: Request | None = None,
    preserve_content_encoding: bool = False,
    preserve_content_length: bool = False,
) -> Response:
    excluded = {"transfer-encoding", "connection"}
    if not preserve_content_encoding:
        excluded.add("content-encoding")
    if not preserve_content_length:
        excluded.add("content-length")
    elif "content-length" in response.headers:
        del response.headers["content-length"]

    for key, value in upstream.headers.multi_items():
        if key.lower() in excluded:
            continue
        if key.lower() == "location" and request is not None:
            value = _rewrite_location(value, request)
        response.headers.append(key, value)
    return response


def _request_is_streaming(body: bytes, headers: dict[str, str]) -> bool:
    if "text/event-stream" in headers.get("accept", "").lower():
        return True
    if not body or "json" not in headers.get("content-type", "").lower():
        return False
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("stream") is True


def _websocket_upstream_url(path: str, query: str) -> str:
    scheme = "wss" if _litellm_parsed.scheme == "https" else "ws"
    base_path = _litellm_parsed.path.rstrip("/")
    upstream_path = f"{base_path}/{path.lstrip('/')}"
    return _litellm_parsed._replace(
        scheme=scheme,
        path=upstream_path,
        params="",
        query=query,
        fragment="",
    ).geturl()


def _websocket_headers(websocket: WebSocket) -> list[tuple[str, str]]:
    excluded = {
        "connection",
        "host",
        "upgrade",
        "sec-websocket-accept",
        "sec-websocket-extensions",
        "sec-websocket-key",
        "sec-websocket-protocol",
        "sec-websocket-version",
    }
    return [
        (key, value)
        for key, value in websocket.headers.items()
        if key.lower() not in excluded
    ]


def _forwardable_websocket_close_code(code: int | None) -> int:
    standard_codes = {
        1000,
        1001,
        1002,
        1003,
        1007,
        1008,
        1009,
        1011,
        1012,
        1013,
        1014,
    }
    if code in standard_codes or (code is not None and 3000 <= code < 5000):
        return code
    return 1011


async def _deny_websocket(
    websocket: WebSocket, status_code: int, message: str
) -> None:
    await websocket.send_denial_response(
        JSONResponse(
            status_code=status_code,
            content={"error": {"type": "websocket_proxy_error", "message": message}},
        )
    )


@app.websocket("/responses")
@app.websocket("/v1/responses")
async def responses_websocket(websocket: WebSocket):
    """Proxy Responses API WebSocket traffic bidirectionally to LiteLLM."""
    upstream_url = _websocket_upstream_url(
        websocket.url.path,
        websocket.url.query,
    )
    requested_subprotocols = [
        protocol.strip()
        for protocol in websocket.headers.get(
            "sec-websocket-protocol", ""
        ).split(",")
        if protocol.strip()
    ]

    try:
        async with websocket_connect(
            upstream_url,
            additional_headers=_websocket_headers(websocket),
            subprotocols=requested_subprotocols or None,
            open_timeout=min(PROXY_TIMEOUT, 30),
            close_timeout=10,
            max_size=None,
        ) as upstream:
            await websocket.accept(subprotocol=upstream.subprotocol)

            async def client_to_upstream():
                try:
                    while True:
                        message = await websocket.receive()
                        if message["type"] == "websocket.disconnect":
                            await upstream.close(
                                code=_forwardable_websocket_close_code(
                                    message.get("code")
                                ),
                                reason=message.get("reason", ""),
                            )
                            return
                        if message.get("text") is not None:
                            await upstream.send(message["text"])
                        elif message.get("bytes") is not None:
                            await upstream.send(message["bytes"])
                except ConnectionClosed as exc:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.close(
                            code=_forwardable_websocket_close_code(exc.code),
                            reason=exc.reason or "",
                        )

            async def upstream_to_client():
                try:
                    async for message in upstream:
                        if isinstance(message, str):
                            await websocket.send_text(message)
                        else:
                            await websocket.send_bytes(message)
                except ConnectionClosed as exc:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.close(
                            code=_forwardable_websocket_close_code(exc.code),
                            reason=exc.reason or "",
                        )
                    return
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.close(
                        code=_forwardable_websocket_close_code(
                            upstream.close_code
                        ),
                        reason=upstream.close_reason or "",
                    )

            tasks = {
                asyncio.create_task(client_to_upstream()),
                asyncio.create_task(upstream_to_client()),
            }
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
    except InvalidStatus as exc:
        status_code = int(exc.response.status_code)
        if not 400 <= status_code < 600:
            status_code = 502
        logger.error(
            "Upstream WebSocket rejected %s with HTTP %d",
            upstream_url,
            status_code,
        )
        if websocket.application_state == WebSocketState.CONNECTING:
            await _deny_websocket(
                websocket,
                status_code,
                "Upstream rejected the WebSocket connection",
            )
        elif websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Upstream WebSocket rejection")
    except WebSocketDisconnect:
        return
    except (
        OSError,
        TimeoutError,
        RuntimeError,
        WebSocketException,
    ) as exc:
        logger.error("WebSocket proxy failed for %s: %s", upstream_url, exc)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Upstream WebSocket failure")
        elif websocket.application_state == WebSocketState.CONNECTING:
            await _deny_websocket(
                websocket,
                502,
                "Unable to connect to the upstream WebSocket",
            )


async def _proxy_raw_request(request: Request, path: str) -> Response:
    body = await request.body()
    headers = dict(request.headers)
    full_path = f"/{path}"
    if request.url.query:
        full_path += f"?{request.url.query}"

    is_streaming = _request_is_streaming(body, headers)
    logger.debug(
        "Passthrough: %s %s (streaming=%s)",
        request.method,
        full_path,
        is_streaming,
    )

    resp = await forward_to_litellm(
        full_path, request.method, headers, body, stream=is_streaming
    )

    if (
        resp.status_code == 413
        and request.method == "POST"
        and path in (
            "responses",
            "v1/responses",
            "responses/compact",
            "v1/responses/compact",
        )
    ):
        # Codex only recognizes context_length_exceeded on HTTP 400.
        logger.warning(
            "Upstream Responses request exceeded the payload limit: "
            "path=%s request_bytes=%d streaming=%s",
            path,
            len(body),
            is_streaming,
        )
        await resp.aclose()
        if hasattr(resp, "_client"):
            await resp._client.aclose()
        response = JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                    "param": "input",
                    "message": (
                        "The upstream Responses API rejected this request because "
                        "the serialized conversation is too large (HTTP 413). "
                        "Compact the conversation or reduce large tool outputs, "
                        "then retry."
                    ),
                }
            },
        )
        _copy_upstream_headers(response, resp, request=request)
        response.headers["content-type"] = "application/json"
        return response

    if is_streaming:

        async def _stream():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                if hasattr(resp, "_client"):
                    await resp._client.aclose()

        response = StreamingResponse(
            _stream(),
            status_code=resp.status_code,
        )
        return _copy_upstream_headers(
            response,
            resp,
            request=request,
            preserve_content_encoding=True,
        )

    response = Response(
        content=resp.content,
        status_code=resp.status_code,
    )
    return _copy_upstream_headers(
        response,
        resp,
        request=request,
        preserve_content_encoding=request.method == "HEAD",
        preserve_content_length=request.method == "HEAD",
    )


@app.get("/models")
@app.get("/v1/models")
async def models(request: Request):
    """
    Return Codex model metadata when Codex supplies its client version.

    Plain `/models` and `/v1/models` requests retain LiteLLM's standard shape.
    """
    client_version = request.query_params.get("client_version")
    if not client_version:
        return await _proxy_raw_request(request, request.url.path.lstrip("/"))

    upstream = await forward_to_litellm(
        "/v1/models", "GET", dict(request.headers), stream=False
    )
    if upstream.status_code != 200:
        response = Response(
            content=upstream.content,
            status_code=upstream.status_code,
        )
        return _copy_upstream_headers(
            response,
            upstream,
            request=request,
            preserve_content_encoding=False,
        )

    try:
        upstream_data = upstream.json()
        upstream_models = upstream_data["data"]
        if not isinstance(upstream_models, list):
            raise TypeError("data is not a list")
        model_ids = [
            model["id"]
            for model in upstream_models
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("Invalid LiteLLM /v1/models response: %s", exc)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "invalid_upstream_response",
                    "message": "LiteLLM returned an invalid model list",
                }
            },
        )

    catalog = build_codex_models_response(model_ids, client_version)
    content = json.dumps(
        catalog, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    etag = hashlib.sha256(content).hexdigest()
    logger.info(
        "Returning %d Codex model entries for client %s",
        len(catalog["models"]),
        client_version,
    )
    return Response(
        content=content,
        media_type="application/json",
        headers={"Cache-Control": "no-cache", "ETag": f'"{etag}"'},
    )


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

        response = StreamingResponse(
            _stream(),
            status_code=resp.status_code,
        )
        return _copy_upstream_headers(
            response,
            resp,
            request=request,
            preserve_content_encoding=False,
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
        response = Response(
            content=content,
            status_code=resp.status_code,
        )
        return _copy_upstream_headers(
            response,
            resp,
            request=request,
            preserve_content_encoding=False,
        )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all(request: Request, path: str):
    """
    Catch-all: forward any other request to LiteLLM unchanged.
    """
    return await _proxy_raw_request(request, path)
