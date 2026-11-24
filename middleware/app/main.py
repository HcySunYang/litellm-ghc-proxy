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
from .responses_bridge import (
    chat_completion_to_responses_object,
    format_responses_sse_event,
    is_chat_completions_responses_model,
    iter_responses_events_from_chat_sse,
    responses_request_to_chat_completions,
)
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


async def _handle_bridged_responses_websocket(
    websocket: WebSocket,
    default_model: str,
    requested_subprotocols: list[str],
) -> None:
    """Serve Responses API WebSocket turns via `/chat/completions` for bridged models."""
    await websocket.accept(
        subprotocol=requested_subprotocols[0] if requested_subprotocols else None
    )
    base_headers = dict(_websocket_headers(websocket))
    base_headers["content-type"] = "application/json"
    base_headers["accept"] = "text/event-stream"

    try:
        while True:
            ws_msg = await websocket.receive()
            if ws_msg["type"] == "websocket.disconnect":
                return
            raw_payload = ws_msg.get("text")
            if raw_payload is None and ws_msg.get("bytes") is not None:
                raw_payload = ws_msg["bytes"].decode("utf-8", errors="replace")
            if not raw_payload:
                continue

            try:
                frame = json.loads(raw_payload)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": "Invalid JSON WebSocket frame",
                            },
                        }
                    )
                )
                continue

            if not isinstance(frame, dict) or frame.get("type") != "response.create":
                continue

            req_body = {k: v for k, v in frame.items() if k not in ("type", "response")}
            if isinstance(frame.get("response"), dict):
                req_body.update(frame["response"])
            req_body.setdefault("model", default_model)

            chat_body, bridge_ctx = responses_request_to_chat_completions(
                req_body, stream=True
            )
            chat_bytes = json.dumps(chat_body).encode("utf-8")
            resp = await forward_to_litellm(
                "/chat/completions",
                "POST",
                base_headers,
                chat_bytes,
                stream=True,
            )
            try:
                if resp.status_code != 200:
                    raw_err = await resp.aread()
                    if resp.status_code == 413:
                        status, error_payload = _responses_size_error_payload(
                            raw_err,
                            len(raw_payload.encode("utf-8")),
                            len(chat_bytes),
                        )
                        logger.warning(
                            "Upstream WebSocket turn rejected request size: "
                            "request_bytes=%d forwarded_bytes=%d code=%s",
                            len(raw_payload.encode("utf-8")),
                            len(chat_bytes),
                            error_payload["error"]["code"],
                        )
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "error",
                                    "status": status,
                                    "error": error_payload["error"],
                                }
                            )
                        )
                        continue
                    try:
                        err_json = json.loads(raw_err)
                        err_obj = (
                            err_json.get("error")
                            if isinstance(err_json, dict)
                            and isinstance(err_json.get("error"), dict)
                            else {"message": raw_err.decode("utf-8", errors="replace")}
                        )
                    except json.JSONDecodeError:
                        err_obj = {
                            "message": raw_err.decode("utf-8", errors="replace")
                        }
                    await websocket.send_text(
                        json.dumps({"type": "error", "error": err_obj})
                    )
                    continue

                async for event in iter_responses_events_from_chat_sse(
                    resp.aiter_lines(), bridge_ctx
                ):
                    await websocket.send_text(
                        json.dumps(event, ensure_ascii=False)
                    )
            finally:
                await resp.aclose()
                if hasattr(resp, "_client"):
                    await resp._client.aclose()
    except WebSocketDisconnect:
        return


@app.websocket("/responses")
@app.websocket("/v1/responses")
async def responses_websocket(websocket: WebSocket):
    """Proxy Responses API WebSocket traffic bidirectionally to LiteLLM."""
    requested_subprotocols = [
        protocol.strip()
        for protocol in websocket.headers.get(
            "sec-websocket-protocol", ""
        ).split(",")
        if protocol.strip()
    ]
    query_model = websocket.query_params.get("model")
    if is_chat_completions_responses_model(query_model):
        await _handle_bridged_responses_websocket(
            websocket,
            str(query_model),
            requested_subprotocols,
        )
        return

    upstream_url = _websocket_upstream_url(
        websocket.url.path,
        websocket.url.query,
    )

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


def _responses_size_error_payload(
    raw_error: bytes, request_bytes: int, forwarded_bytes: int
) -> tuple[int, dict]:
    try:
        payload = json.loads(raw_error)
    except (ValueError, UnicodeDecodeError):
        payload = None
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("error"), dict)
        and payload["error"].get("code") == "context_length_exceeded"
    ):
        # Codex recognizes explicit token-limit errors on HTTP 400, not HTTP 413.
        return 400, payload
    return 413, {
        "error": {
            "type": "invalid_request_error",
            "code": "request_too_large",
            "param": "input",
            "message": (
                "The upstream API rejected the request size (HTTP 413): "
                f"client request {request_bytes} bytes, forwarded request "
                f"{forwarded_bytes} bytes. This limit is independent of the model's "
                "token context window. Reduce image file sizes, remove large "
                "attachments, or shorten tool outputs/history before retrying. "
                "Increasing model_context_window will not help."
            ),
        }
    }


async def _build_responses_size_error(
    request: Request,
    resp,
    path: str,
    request_bytes: int,
    is_streaming: bool,
    forwarded_bytes: int | None = None,
) -> JSONResponse:
    if forwarded_bytes is None:
        forwarded_bytes = request_bytes
    try:
        raw_error = await resp.aread()
    finally:
        if not resp.is_closed:
            await resp.aclose()
        if hasattr(resp, "_client"):
            await resp._client.aclose()
    status, payload = _responses_size_error_payload(
        raw_error, request_bytes, forwarded_bytes
    )
    logger.warning(
        "Upstream Responses request rejected request size: "
        "path=%s request_bytes=%d forwarded_bytes=%d streaming=%s code=%s",
        path,
        request_bytes,
        forwarded_bytes,
        is_streaming,
        payload["error"]["code"],
    )
    response = JSONResponse(
        status_code=status,
        content=payload,
    )
    _copy_upstream_headers(response, resp, request=request)
    response.headers["content-type"] = "application/json"
    return response


async def _proxy_bridged_responses_request(
    request: Request,
    path: str,
    raw_body: bytes,
    payload: dict,
    headers: dict[str, str],
) -> Response:
    is_streaming = payload.get("stream") is True or (
        payload.get("stream") is None
        and "text/event-stream" in headers.get("accept", "").lower()
    )
    chat_body, bridge_ctx = responses_request_to_chat_completions(
        payload, stream=is_streaming
    )
    forward_headers = dict(headers)
    forward_headers["content-type"] = "application/json"
    forward_headers["accept"] = (
        "text/event-stream" if is_streaming else "application/json"
    )

    logger.info(
        "Bridging POST /%s for model=%s via /chat/completions (streaming=%s, effort=%s)",
        path,
        bridge_ctx.model,
        is_streaming,
        chat_body.get("reasoning_effort"),
    )

    chat_bytes = json.dumps(chat_body).encode("utf-8")
    resp = await forward_to_litellm(
        "/chat/completions",
        "POST",
        forward_headers,
        chat_bytes,
        stream=is_streaming,
    )

    if resp.status_code == 413:
        return await _build_responses_size_error(
            request, resp, path, len(raw_body), is_streaming, len(chat_bytes)
        )

    if resp.status_code != 200:
        if is_streaming:
            err_content = await resp.aread()
            await resp.aclose()
            if hasattr(resp, "_client"):
                await resp._client.aclose()
        else:
            err_content = resp.content
        error_response = Response(
            content=err_content,
            status_code=resp.status_code,
        )
        return _copy_upstream_headers(error_response, resp, request=request)

    if is_streaming:

        async def _stream_responses_sse():
            try:
                async for event in iter_responses_events_from_chat_sse(
                    resp.aiter_lines(), bridge_ctx
                ):
                    yield format_responses_sse_event(event)
            finally:
                await resp.aclose()
                if hasattr(resp, "_client"):
                    await resp._client.aclose()

        stream_response = StreamingResponse(
            _stream_responses_sse(),
            status_code=200,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        for key, value in resp.headers.multi_items():
            if key.lower() in {
                "content-type",
                "content-length",
                "content-encoding",
                "transfer-encoding",
                "connection",
                "cache-control",
            }:
                continue
            stream_response.headers.append(key, value)
        return stream_response

    chat_data = resp.json()
    responses_data = chat_completion_to_responses_object(chat_data, bridge_ctx)
    json_response = JSONResponse(status_code=200, content=responses_data)
    for key, value in resp.headers.multi_items():
        if key.lower() in {
            "content-type",
            "content-length",
            "content-encoding",
            "transfer-encoding",
            "connection",
        }:
            continue
        json_response.headers.append(key, value)
    return json_response


async def _proxy_raw_request(request: Request, path: str) -> Response:
    body = await request.body()
    headers = dict(request.headers)

    if (
        request.method == "POST"
        and path in ("responses", "v1/responses")
        and body
    ):
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict) and is_chat_completions_responses_model(
            payload.get("model")
        ):
            return await _proxy_bridged_responses_request(
                request, path, body, payload, headers
            )

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
        return await _build_responses_size_error(
            request, resp, path, len(body), is_streaming
        )

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
