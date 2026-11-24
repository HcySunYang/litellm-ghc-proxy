import json
import unittest
from collections import OrderedDict
from itertools import product
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.responses_bridge import (
    chat_completion_to_responses_object,
    responses_request_to_chat_completions,
)


class CodexModelRoutesTests(unittest.TestCase):
    def test_response_size_errors_are_not_misreported_as_token_exhaustion(self):
        for path in (
            "/responses",
            "/v1/responses",
            "/responses/compact",
            "/v1/responses/compact",
        ):
            for streaming in (False, True):
                with self.subTest(path=path, streaming=streaming):
                    payload = {
                        "model": "gpt-6.1-sol",
                        "input": "Hello!",
                        "stream": streaming,
                    }
                    upstream = httpx.Response(
                        413,
                        stream=httpx.ByteStream(b"failed to parse request"),
                        headers={
                            "content-type": "text/plain",
                            "content-length": "22",
                            "x-request-id": "request-too-large",
                        },
                    )
                    if not streaming:
                        upstream.read()
                    upstream._client = AsyncMock(spec=httpx.AsyncClient)
                    with (
                        patch(
                            "app.main.forward_to_litellm",
                            new_callable=AsyncMock,
                            return_value=upstream,
                        ) as forward,
                        TestClient(app) as client,
                    ):
                        response = client.post(path, json=payload)

                    self.assertEqual(response.status_code, 413)
                    error = response.json()["error"]
                    self.assertEqual(error["code"], "request_too_large")
                    self.assertEqual(error["type"], "invalid_request_error")
                    self.assertEqual(error["param"], "input")
                    self.assertIn("HTTP 413", error["message"])
                    self.assertIn("independent of the model's token", error["message"])
                    self.assertIn(str(len(response.request.content)), error["message"])
                    self.assertEqual(
                        response.headers["content-type"], "application/json"
                    )
                    self.assertEqual(
                        response.headers["x-request-id"], "request-too-large"
                    )
                    self.assertEqual(
                        int(response.headers["content-length"]),
                        len(response.content),
                    )
                    self.assertEqual(
                        forward.await_args.args[3], response.request.content
                    )
                    self.assertTrue(upstream.is_closed)
                    upstream._client.aclose.assert_awaited_once()

    def test_explicit_token_limit_errors_still_enable_codex_context_recovery(self):
        payload = {
            "error": {
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "message": "Input tokens exceed the model context window.",
                "param": "input",
            }
        }
        for model, status, streaming in product(
            ("gpt-6.1-sol", "gemini-4-carbon"), (400, 413), (False, True)
        ):
            with self.subTest(model=model, status=status, streaming=streaming):
                upstream = httpx.Response(
                    status,
                    stream=httpx.ByteStream(json.dumps(payload).encode()),
                    headers={"content-type": "application/json"},
                )
                if not streaming:
                    upstream.read()
                upstream._client = AsyncMock(spec=httpx.AsyncClient)
                with (
                    patch(
                        "app.main.forward_to_litellm",
                        new_callable=AsyncMock,
                        return_value=upstream,
                    ),
                    TestClient(app) as client,
                ):
                    response = client.post(
                        "/responses",
                        json={"model": model, "input": "Hello", "stream": streaming},
                    )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json(), payload)

    def test_other_responses_errors_remain_unchanged(self):
        for status in (400, 401, 429, 500):
            for streaming in (False, True):
                with self.subTest(status=status, streaming=streaming):
                    payload = {
                        "error": {
                            "message": "failed to parse request",
                            "code": str(status),
                        }
                    }
                    upstream = httpx.Response(
                        status,
                        stream=httpx.ByteStream(json.dumps(payload).encode()),
                        headers={"content-type": "application/json"},
                    )
                    if not streaming:
                        upstream.read()
                    with (
                        patch(
                            "app.main.forward_to_litellm",
                            new_callable=AsyncMock,
                            return_value=upstream,
                        ),
                        TestClient(app) as client,
                    ):
                        response = client.post(
                            "/responses",
                            json={"model": "gpt-6.1-sol", "stream": streaming},
                        )

                    self.assertEqual(response.status_code, status)
                    self.assertEqual(response.json(), payload)

    def test_size_errors_outside_responses_posts_remain_unchanged(self):
        payload = {"error": {"message": "request too large", "code": "413"}}
        for method, path in (
            ("POST", "/chat/completions"),
            ("GET", "/responses"),
        ):
            with self.subTest(method=method, path=path):
                with (
                    patch(
                        "app.main.forward_to_litellm",
                        new_callable=AsyncMock,
                        return_value=httpx.Response(413, json=payload),
                    ),
                    TestClient(app) as client,
                ):
                    response = client.request(method, path)

                self.assertEqual(response.status_code, 413)
                self.assertEqual(response.json(), payload)

    def test_exposes_gpt_6_sol_in_codex_catalog(self):
        for slug, path in product(
            ("gpt-6.1-sol", "gpt-6.2-sol"), ("/models", "/v1/models")
        ):
            with self.subTest(slug=slug, path=path):
                upstream = httpx.Response(
                    200,
                    json={"object": "list", "data": [{"id": slug}]},
                )
                with patch(
                    "app.main.forward_to_litellm",
                    new_callable=AsyncMock,
                    return_value=upstream,
                ) as forward:
                    with TestClient(app) as client:
                        response = client.get(
                            path,
                            params={"client_version": "0.144.0"},
                            headers={"Authorization": "Bearer test-key"},
                        )

                self.assertEqual(response.status_code, 200)
                models = {
                    model["slug"]: model for model in response.json()["models"]
                }
                model = models[slug]
                self.assertEqual(model["visibility"], "list")
                self.assertTrue(model["supported_in_api"])
                self.assertEqual(model["context_window"], 1_050_000)
                self.assertEqual(model["auto_compact_token_limit"], 900_000)
                self.assertEqual(model["effective_context_window_percent"], 87)
                self.assertEqual(
                    [level["effort"] for level in model["supported_reasoning_levels"]],
                    ["low", "medium", "high", "xhigh", "max"],
                )
                self.assertEqual(response.headers["cache-control"], "no-cache")
                self.assertIn("etag", response.headers)
                forward.assert_awaited_once()
                args = forward.await_args.args
                self.assertEqual(args[:2], ("/v1/models", "GET"))
                self.assertEqual(args[2]["authorization"], "Bearer test-key")

    def test_preserves_gpt_6_sol_in_plain_openai_model_list(self):
        payload = {
            "object": "list",
            "data": [{"id": "gpt-6.1-sol"}, {"id": "gpt-6.2-sol"}],
        }
        upstream = httpx.Response(200, json=payload)
        for path in ("/models", "/v1/models"):
            with self.subTest(path=path):
                with patch(
                    "app.main.forward_to_litellm",
                    new_callable=AsyncMock,
                    return_value=upstream,
                ):
                    with TestClient(app) as client:
                        response = client.get(path)

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), payload)

    def test_forwards_gpt_6_sol_responses_and_all_reasoning_efforts_unchanged(self):
        for slug, path, effort in product(
            ("gpt-6.1-sol", "gpt-6.2-sol"),
            ("/responses", "/v1/responses"),
            ("low", "medium", "high", "xhigh", "max"),
        ):
            with self.subTest(slug=slug, path=path, effort=effort):
                payload = {
                    "model": slug,
                    "input": "Hello!",
                    "reasoning": {"effort": effort},
                }
                upstream_payload = {
                    "id": "resp_test",
                    "model": slug,
                    "status": "completed",
                    "output": [],
                }
                with patch(
                    "app.main.forward_to_litellm",
                    new_callable=AsyncMock,
                    return_value=httpx.Response(200, json=upstream_payload),
                ) as forward:
                    with TestClient(app) as client:
                        response = client.post(
                            path,
                            json=payload,
                            headers={"Authorization": "Bearer test-key"},
                        )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), upstream_payload)
                forward.assert_awaited_once()
                args = forward.await_args.args
                self.assertEqual(args[:2], (path, "POST"))
                self.assertEqual(args[2]["authorization"], "Bearer test-key")
                self.assertEqual(args[3], response.request.content)

    def test_exposes_astra_in_codex_catalog(self):
        upstream = httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "gpt-6-astra"}]},
        )
        for path in ("/models", "/v1/models"):
            with self.subTest(path=path):
                with patch(
                    "app.main.forward_to_litellm",
                    new_callable=AsyncMock,
                    return_value=upstream,
                ) as forward:
                    with TestClient(app) as client:
                        response = client.get(
                            path,
                            params={"client_version": "0.144.0"},
                            headers={"Authorization": "Bearer test-key"},
                        )

                self.assertEqual(response.status_code, 200)
                models = {
                    model["slug"]: model for model in response.json()["models"]
                }
                self.assertEqual(models["gpt-6-astra"]["visibility"], "list")
                self.assertTrue(models["gpt-6-astra"]["supported_in_api"])
                self.assertEqual(models["gpt-6-astra"]["context_window"], 1_000_000)
                self.assertEqual(response.headers["cache-control"], "no-cache")
                self.assertIn("etag", response.headers)
                forward.assert_awaited_once()
                args = forward.await_args.args
                self.assertEqual(args[:2], ("/v1/models", "GET"))
                self.assertEqual(args[2]["authorization"], "Bearer test-key")

    def test_preserves_astra_in_plain_openai_model_list(self):
        payload = {"object": "list", "data": [{"id": "gpt-6-astra"}]}
        upstream = httpx.Response(200, json=payload)
        for path in ("/models", "/v1/models"):
            with self.subTest(path=path):
                with patch(
                    "app.main.forward_to_litellm",
                    new_callable=AsyncMock,
                    return_value=upstream,
                ):
                    with TestClient(app) as client:
                        response = client.get(path)

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), payload)

    def test_exposes_gemini_models_in_codex_catalog(self):
        upstream = httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "gemini-4-carbon"}, {"id": "gemini-4-neon"}],
            },
        )
        for slug, path, client_version in product(
            ("gemini-4-carbon", "gemini-4-neon"),
            ("/models", "/v1/models"),
            ("0.125.0", "0.144.0", "0.144.6", "0.169.0-alpha.11.1"),
        ):
            with self.subTest(slug=slug, path=path, client_version=client_version):
                with patch(
                    "app.main.forward_to_litellm",
                    new_callable=AsyncMock,
                    return_value=upstream,
                ):
                    with TestClient(app) as client:
                        response = client.get(
                            path,
                            params={"client_version": client_version},
                            headers={"Authorization": "Bearer test-key"},
                        )

                self.assertEqual(response.status_code, 200)
                models = {
                    model["slug"]: model for model in response.json()["models"]
                }
                model = models[slug]
                self.assertEqual(model["visibility"], "list")
                self.assertTrue(model["supported_in_api"])
                self.assertEqual(
                    model["context_window"],
                    1_048_576 if slug == "gemini-4-neon" else 1_000_000,
                )
                self.assertEqual(model["apply_patch_tool_type"], "freeform")
                self.assertEqual(model["tool_mode"], "direct")
                self.assertFalse(model["supports_image_detail_original"])
                self.assertEqual(
                    [level["effort"] for level in model["supported_reasoning_levels"]],
                    ["low", "medium", "high", "xhigh"]
                    if slug == "gemini-4-neon"
                    else ["low", "medium", "high"],
                )

    def test_bridges_gemini_non_streaming_responses_and_efforts(self):
        for slug, path, requested_effort in product(
            ("gemini-4-carbon", "gemini-4-neon"),
            ("/responses", "/v1/responses"),
            ("low", "medium", "high", "xhigh", "max", "ultra", "minimal"),
        ):
            with self.subTest(slug=slug, path=path, effort=requested_effort):
                expected_effort = (
                    "low"
                    if requested_effort == "minimal"
                    else (
                        ("xhigh" if slug == "gemini-4-neon" else "high")
                        if requested_effort in ("xhigh", "max", "ultra")
                        else requested_effort
                    )
                )
                upstream_chat = {
                    "id": "chatcmpl_carbon",
                    "created": 1700000000,
                    "model": slug,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "CARBON_OK",
                                "reasoning_text": "**Thinking**\n\nChecked.",
                                "reasoning_opaque": "opaque-sig-123",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 25,
                        "completion_tokens": 12,
                        "total_tokens": 37,
                        "prompt_tokens_details": {"cached_tokens": 10},
                        "completion_tokens_details": {"reasoning_tokens": 8},
                    },
                }
                with patch(
                    "app.main.forward_to_litellm",
                    new_callable=AsyncMock,
                    return_value=httpx.Response(200, json=upstream_chat),
                ) as forward:
                    with TestClient(app) as client:
                        response = client.post(
                            path,
                            json={
                                "model": slug,
                                "instructions": "Be concise.",
                                "input": "Hello Carbon",
                                "reasoning": {"effort": requested_effort},
                                "include": ["reasoning.encrypted_content"],
                                "max_output_tokens": 512,
                            },
                            headers={"Authorization": "Bearer test-key"},
                        )

                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body["object"], "response")
                self.assertEqual(body["status"], "completed")
                self.assertEqual(body["model"], slug)
                self.assertEqual(body["output"][0]["type"], "reasoning")
                self.assertEqual(
                    body["output"][0]["encrypted_content"], "opaque-sig-123"
                )
                self.assertEqual(
                    body["output"][0]["summary"][0]["text"],
                    "**Thinking**\n\nChecked.",
                )
                self.assertEqual(
                    body["output"][1]["content"][0]["text"], "CARBON_OK"
                )
                self.assertEqual(
                    body["usage"],
                    {
                        "input_tokens": 25,
                        "input_tokens_details": {"cached_tokens": 10},
                        "output_tokens": 12,
                        "output_tokens_details": {"reasoning_tokens": 8},
                        "total_tokens": 37,
                    },
                )

                forward.assert_awaited_once()
                fwd_path, fwd_method, _, fwd_raw_body = forward.await_args.args
                self.assertEqual((fwd_path, fwd_method), ("/chat/completions", "POST"))
                fwd_json = json.loads(fwd_raw_body)
                self.assertEqual(fwd_json["model"], slug)
                self.assertEqual(fwd_json["reasoning_effort"], expected_effort)
                self.assertEqual(fwd_json["max_tokens"], 512)
                self.assertEqual(
                    fwd_json["messages"],
                    [
                        {"role": "developer", "content": "Be concise."},
                        {"role": "user", "content": "Hello Carbon"},
                    ],
                )

    def test_gemini_image_detail_conversion_preserves_images_and_request(self):
        image_url = "https://example.test/screenshot.png"
        for slug, detail, nested in product(
            ("gemini-4-carbon", "gemini-4-neon"),
            ("original", "high", "low", "auto", None), (False, True)
        ):
            with self.subTest(slug=slug, detail=detail, nested=nested):
                if nested:
                    image = {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                    if detail is not None:
                        image["image_url"]["detail"] = detail
                    image["detail"] = "original"
                else:
                    image = {"type": "input_image", "image_url": image_url}
                    if detail is not None:
                        image["detail"] = detail
                payload = {
                    "model": slug,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Read the image."},
                                image,
                            ],
                        }
                    ],
                }
                original_payload = json.dumps(payload)
                chat, ctx = responses_request_to_chat_completions(
                    payload, stream=False
                )
                expected_image = {"url": image_url}
                effective_detail = detail if detail is not None else (
                    "original" if nested else None
                )
                if effective_detail is not None:
                    expected_image["detail"] = (
                        "high" if effective_detail == "original" else effective_detail
                    )
                self.assertEqual(
                    chat["messages"],
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Read the image."},
                                {"type": "image_url", "image_url": expected_image},
                            ],
                        }
                    ],
                )
                self.assertEqual(ctx.input_messages, chat["messages"])
                self.assertEqual(json.dumps(payload), original_payload)

    def test_gemini_normalizes_original_image_detail_in_tool_results(self):
        image_url = "data:image/png;base64,test-image"
        for output_type, nested in product(
            ("function_call_output", "custom_tool_call_output"), (False, True)
        ):
            with self.subTest(output_type=output_type, nested=nested):
                call_type = output_type.removesuffix("_output")
                image = (
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "original"},
                    }
                    if nested
                    else {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "original",
                    }
                )
                call = {
                    "type": call_type,
                    "call_id": "call_image",
                    "name": "view_image",
                }
                if call_type == "custom_tool_call":
                    call["input"] = "screenshot.png"
                else:
                    call["arguments"] = '{"path":"screenshot.png"}'
                payload = {
                    "model": "gemini-4-carbon",
                    "input": [
                        {"role": "user", "content": "Inspect the screenshot."},
                        call,
                        {
                            "type": output_type,
                            "call_id": "call_image",
                            "output": [
                                {"type": "input_text", "text": "Screenshot attached."},
                                image,
                            ],
                        },
                    ],
                }
                original_payload = json.dumps(payload)
                chat, _ = responses_request_to_chat_completions(payload, stream=True)
                self.assertEqual(
                    chat["messages"][-2],
                    {
                        "role": "tool",
                        "tool_call_id": "call_image",
                        "content": "Screenshot attached.",
                    },
                )
                self.assertEqual(
                    chat["messages"][-1],
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url, "detail": "high"},
                            }
                        ],
                    },
                )
                self.assertEqual(json.dumps(payload), original_payload)

    def test_gemini_continuations_preserve_trailing_context(self):
        for role in ("assistant", "developer", "system"):
            with self.subTest(role=role):
                payload = {
                    "model": "gemini-4-carbon",
                    "input": [
                        {"role": "user", "content": "Finish the task."},
                        {
                            "type": "reasoning",
                            "encrypted_content": "opaque-signature",
                        },
                        {
                            "type": "message",
                            "role": role,
                            "phase": "commentary",
                            "content": [
                                {"type": "output_text", "text": "Pending context."}
                            ],
                        },
                    ],
                }
                chat, ctx = responses_request_to_chat_completions(
                    payload, stream=True
                )
                expected_tail = {
                    "role": "developer" if role == "system" else role,
                    "content": "Pending context.",
                }
                if role == "assistant":
                    expected_tail["reasoning_opaque"] = "opaque-signature"
                self.assertEqual(chat["messages"][:-1], ctx.input_messages)
                self.assertEqual(ctx.input_messages[-1], expected_tail)
                self.assertEqual(chat["messages"][-1]["role"], "user")
                self.assertTrue(chat["messages"][-1]["content"])
                self.assertEqual(payload["input"][-1]["role"], role)

    def test_gemini_keeps_valid_user_and_tool_endings_unchanged(self):
        for output_type in ("function_call_output", "custom_tool_call_output"):
            with self.subTest(output_type=output_type):
                call_type = output_type.removesuffix("_output")
                call = {
                    "type": call_type,
                    "call_id": "call_tail",
                    "name": "example",
                }
                call["input" if call_type == "custom_tool_call" else "arguments"] = "{}"
                payload = {
                    "model": "gemini-4-carbon",
                    "input": [
                        {"role": "user", "content": "Use the tool."},
                        call,
                        {
                            "type": output_type,
                            "call_id": "call_tail",
                            "output": "Tool completed.",
                        },
                    ],
                }
                for next_user in (False, True):
                    with self.subTest(next_user=next_user):
                        if next_user:
                            payload["input"].append(
                                {"role": "user", "content": "Now reply."}
                            )
                        chat, ctx = responses_request_to_chat_completions(
                            payload, stream=False
                        )
                        self.assertEqual(chat["messages"], ctx.input_messages)
                        self.assertEqual(
                            chat["messages"][-1],
                            (
                                {"role": "user", "content": "Now reply."}
                                if next_user
                                else {
                                    "role": "tool",
                                    "tool_call_id": "call_tail",
                                    "content": "Tool completed.",
                                }
                            ),
                        )

    def test_gemini_cached_continuations_do_not_accumulate_synthetic_turns(self):
        with patch(
            "app.responses_bridge._RESPONSE_HISTORY_CACHE", OrderedDict()
        ):
            payload = {
                "model": "gemini-4-carbon",
                "input": [
                    {"role": "user", "content": "Finish the task."},
                    {"role": "assistant", "content": "Planning."},
                ],
            }
            for turn in range(3):
                chat, ctx = responses_request_to_chat_completions(
                    payload, stream=False
                )
                self.assertEqual(chat["messages"][:-1], ctx.input_messages)
                self.assertEqual(chat["messages"][-1]["role"], "user")
                self.assertEqual(len(ctx.input_messages), 2 + turn)
                self.assertEqual(
                    [m for m in ctx.input_messages if m["role"] == "user"],
                    [{"role": "user", "content": "Finish the task."}],
                )
                response = chat_completion_to_responses_object(
                    {"choices": [{"message": {"content": "Continuing."}}]}, ctx
                )
                payload = {
                    "model": "gemini-4-carbon",
                    "previous_response_id": response["id"],
                    "input": [],
                }

    def test_gemini_assistant_tail_is_normalized_for_http_sse_and_websocket(self):
        async def upstream(path, method, headers, body, *, stream):
            chat = json.loads(body)
            if chat["messages"][-1]["role"] not in ("user", "tool"):
                return httpx.Response(
                    400,
                    json={"error": {"message": "Gemini rejects assistant tails."}},
                )
            message = {"role": "assistant", "content": "TAIL_OK"}
            if not stream:
                return httpx.Response(
                    200,
                    json={"choices": [{"message": message, "finish_reason": "stop"}]},
                )
            chunk = {
                "choices": [
                    {"index": 0, "delta": message, "finish_reason": "stop"}
                ]
            }
            response = httpx.Response(
                200,
                stream=httpx.ByteStream(
                    f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
                ),
                headers={"content-type": "text/event-stream"},
            )
            response._client = AsyncMock(spec=httpx.AsyncClient)
            return response

        for slug, path, transport in product(
            ("gemini-4-carbon", "gemini-4-neon"),
            ("/responses", "/v1/responses"), ("http", "sse", "websocket")
        ):
            with self.subTest(slug=slug, path=path, transport=transport):
                payload = {
                    "model": slug,
                    "input": [
                        {"role": "user", "content": "Reply with TAIL_OK."},
                        {"role": "assistant", "content": "I will respond now."},
                    ],
                }
                with (
                    patch(
                        "app.main.forward_to_litellm",
                        new_callable=AsyncMock,
                        side_effect=upstream,
                    ) as forward,
                    TestClient(app) as client,
                ):
                    if transport == "websocket":
                        with client.websocket_connect(
                            f"{path}?model={slug}"
                        ) as ws:
                            ws.send_json({"type": "response.create", **payload})
                            while True:
                                event = ws.receive_json()
                                if event["type"] in ("response.completed", "error"):
                                    break
                        self.assertEqual(event["type"], "response.completed")
                        result = event["response"]
                    else:
                        payload["stream"] = transport == "sse"
                        response = client.post(path, json=payload)
                        self.assertEqual(response.status_code, 200)
                        if transport == "sse":
                            events = [
                                json.loads(line[5:].strip())
                                for line in response.text.splitlines()
                                if line.startswith("data:")
                            ]
                            result = events[-1]["response"]
                        else:
                            result = response.json()
                self.assertEqual(
                    result["output"][0]["content"][0]["text"], "TAIL_OK"
                )
                forwarded = json.loads(forward.await_args.args[3])["messages"]
                self.assertEqual(forwarded[1], payload["input"][1])
                self.assertEqual(forwarded[-1]["role"], "user")

    def test_bridges_gemini_4_carbon_streaming_tool_calls_and_multiturn_history(self):
        sse_chunks = (
            b'data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_text":"Planning tool call."}}]}\n\n'
            b'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_42","type":"function","function":{"name":"exec_command","arguments":"{\\"cmd\\":"}}]}}]}\n\n'
            b'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"pwd\\"}"}}],"reasoning_opaque":"sig_turn2"},"finish_reason":"tool_calls"}]}\n\n'
            b'data: {"id":"c1","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":30,"total_tokens":130,"completion_tokens_details":{"reasoning_tokens":18}}}\n\n'
            b"data: [DONE]\n\n"
        )
        upstream = httpx.Response(
            200,
            stream=httpx.ByteStream(sse_chunks),
            headers={"content-type": "text/event-stream"},
        )
        upstream._client = AsyncMock(spec=httpx.AsyncClient)

        payload = {
            "model": "gemini-4-carbon",
            "instructions": "Run commands carefully.",
            "stream": True,
            "reasoning": {"effort": "high"},
            "include": ["reasoning.encrypted_content"],
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Execute shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
                {"type": "web_search"},
            ],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Check cwd"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_prev",
                    "summary": [],
                    "encrypted_content": "sig_turn1",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"ls"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "README.md\n",
                },
            ],
        }

        with patch(
            "app.main.forward_to_litellm",
            new_callable=AsyncMock,
            return_value=upstream,
        ) as forward:
            with TestClient(app) as client:
                response = client.post("/responses", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])

        fwd_json = json.loads(forward.await_args.args[3])
        self.assertEqual(fwd_json["reasoning_effort"], "high")
        self.assertEqual(len(fwd_json["tools"]), 1)
        self.assertEqual(fwd_json["tools"][0]["function"]["name"], "exec_command")
        self.assertEqual(
            fwd_json["messages"],
            [
                {"role": "developer", "content": "Run commands carefully."},
                {"role": "user", "content": "Check cwd"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_opaque": "sig_turn1",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "exec_command",
                                "arguments": '{"cmd":"ls"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "README.md\n",
                },
            ],
        )

        events = [
            json.loads(line[5:].strip())
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        event_types = [ev["type"] for ev in events]
        self.assertEqual(event_types[0], "response.created")
        self.assertIn("response.reasoning_summary_text.delta", event_types)
        self.assertIn("response.function_call_arguments.delta", event_types)
        self.assertIn("response.function_call_arguments.done", event_types)
        self.assertEqual(event_types[-1], "response.completed")

        completed = events[-1]["response"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["output"][0]["type"], "reasoning")
        self.assertEqual(completed["output"][0]["encrypted_content"], "sig_turn2")
        self.assertEqual(completed["output"][1]["type"], "function_call")
        self.assertEqual(completed["output"][1]["call_id"], "call_42")
        self.assertEqual(completed["output"][1]["name"], "exec_command")
        self.assertEqual(completed["output"][1]["arguments"], '{"cmd":"pwd"}')
        self.assertEqual(
            completed["usage"]["output_tokens_details"]["reasoning_tokens"], 18
        )

    def test_bridges_gemini_4_carbon_websocket_responses(self):
        sse_chunks = (
            b'data: {"id":"c_ws","choices":[{"index":0,"delta":{"role":"assistant","content":"WS_CARBON_OK"},"finish_reason":"stop"}]}\n\n'
            b'data: {"id":"c_ws","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}\n\n'
            b"data: [DONE]\n\n"
        )
        upstream = httpx.Response(
            200,
            stream=httpx.ByteStream(sse_chunks),
            headers={"content-type": "text/event-stream"},
        )
        upstream._client = AsyncMock(spec=httpx.AsyncClient)

        with patch(
            "app.main.forward_to_litellm",
            new_callable=AsyncMock,
            return_value=upstream,
        ) as forward:
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/responses?model=gemini-4-carbon"
                ) as ws:
                    ws.send_text(
                        json.dumps(
                            {
                                "type": "response.create",
                                "input": "Ping over WS",
                                "reasoning": {"effort": "medium"},
                            }
                        )
                    )
                    received = []
                    while True:
                        msg = json.loads(ws.receive_text())
                        received.append(msg)
                        if msg["type"] == "response.completed":
                            break

        self.assertEqual(received[0]["type"], "response.created")
        self.assertEqual(
            received[-1]["response"]["output"][0]["content"][0]["text"],
            "WS_CARBON_OK",
        )
        fwd_json = json.loads(forward.await_args.args[3])
        self.assertEqual(fwd_json["model"], "gemini-4-carbon")
        self.assertEqual(fwd_json["reasoning_effort"], "medium")

    def test_bridged_gemini_4_carbon_413_reports_request_bytes_not_tokens(self):
        upstream = httpx.Response(
            413,
            stream=httpx.ByteStream(b"failed to parse request"),
            headers={"content-type": "text/plain", "x-request-id": "req-413"},
        )
        upstream._client = AsyncMock(spec=httpx.AsyncClient)
        with (
            patch(
                "app.main.forward_to_litellm",
                new_callable=AsyncMock,
                return_value=upstream,
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/responses",
                json={
                    "model": "gemini-4-carbon",
                    "input": "Oversized payload",
                    "stream": True,
                },
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(
            response.json()["error"]["code"], "request_too_large"
        )
        self.assertIn(
            str(len(response.request.content)), response.json()["error"]["message"]
        )

    def test_gemini_websocket_distinguishes_body_size_from_token_limits(self):
        for code, expected_status in (("413", 413), ("context_length_exceeded", 400)):
            with self.subTest(code=code):
                upstream = httpx.Response(
                    413,
                    stream=httpx.ByteStream(
                        json.dumps({"error": {"code": code, "message": "Too large"}}).encode()
                    ),
                    headers={"content-type": "application/json"},
                )
                upstream._client = AsyncMock(spec=httpx.AsyncClient)
                with (
                    patch(
                        "app.main.forward_to_litellm",
                        new_callable=AsyncMock,
                        return_value=upstream,
                    ),
                    TestClient(app) as client,
                ):
                    with client.websocket_connect(
                        "/responses?model=gemini-4-carbon"
                    ) as ws:
                        ws.send_json({"type": "response.create", "input": "Hello"})
                        event = ws.receive_json()
                self.assertEqual(event["type"], "error")
                self.assertEqual(event["status"], expected_status)
                self.assertEqual(
                    event["error"]["code"],
                    "request_too_large" if expected_status == 413 else code,
                )

    def test_bridges_freeform_apply_patch_and_namespaced_tools_for_gemini_4_carbon(self):
        patch_text = (
            "*** Begin Patch\n"
            "*** Update File: hello.txt\n"
            "@@\n"
            "-before\n"
            "+after\n"
            "*** End Patch\n"
        )
        escaped_args = json.dumps({"input": patch_text})
        sse_chunks = (
            b'data: {"id":"c_patch","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":['
            b'{"index":0,"id":"call_patch_1","type":"function","function":{"name":"apply_patch","arguments":'
            + json.dumps(escaped_args).encode()
            + b'}},{"index":1,"id":"call_ns_1","type":"function","function":{"name":"multi_agent__wait_agent","arguments":"{\\"agent_id\\":\\"a1\\"}"}}'
            b']},"finish_reason":"tool_calls"}]}\n\n'
            b'data: {"id":"c_patch","choices":[],"usage":{"prompt_tokens":50,"completion_tokens":20,"total_tokens":70}}\n\n'
            b"data: [DONE]\n\n"
        )
        upstream = httpx.Response(
            200,
            stream=httpx.ByteStream(sse_chunks),
            headers={"content-type": "text/event-stream"},
        )
        upstream._client = AsyncMock(spec=httpx.AsyncClient)

        payload = {
            "model": "gemini-4-carbon",
            "stream": True,
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": (
                        "The `apply_patch` tool can be used to edit files. "
                        "This is a FREEFORM tool, so do not wrap the patch in JSON."
                    ),
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": "start: begin_patch hunk+ end_patch",
                    },
                },
                {
                    "type": "namespace",
                    "name": "multi_agent",
                    "description": "Tools in the multi_agent namespace.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "wait_agent",
                            "description": "Wait for background agent.",
                            "parameters": {
                                "type": "object",
                                "properties": {"agent_id": {"type": "string"}},
                                "required": ["agent_id"],
                            },
                        }
                    ],
                },
            ],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "Update hello.txt",
                },
                {
                    "type": "custom_tool_call",
                    "call_id": "call_prev_patch",
                    "name": "apply_patch",
                    "input": patch_text,
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_prev_patch",
                    "output": "Success. Updated hello.txt",
                },
                {
                    "type": "function_call",
                    "call_id": "call_prev_ns",
                    "namespace": "multi_agent",
                    "name": "wait_agent",
                    "arguments": '{"agent_id":"a0"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_prev_ns",
                    "output": "done",
                },
            ],
        }

        with patch(
            "app.main.forward_to_litellm",
            new_callable=AsyncMock,
            return_value=upstream,
        ) as forward:
            with TestClient(app) as client:
                response = client.post("/responses", json=payload)

        self.assertEqual(response.status_code, 200)
        fwd_json = json.loads(forward.await_args.args[3])
        self.assertEqual(len(fwd_json["tools"]), 2)
        patch_fn = fwd_json["tools"][0]["function"]
        self.assertEqual(patch_fn["name"], "apply_patch")
        self.assertNotIn("do not wrap", patch_fn["description"])
        self.assertIn("*** Begin Patch", patch_fn["description"])
        ns_fn = fwd_json["tools"][1]["function"]
        self.assertEqual(ns_fn["name"], "multi_agent__wait_agent")

        # Assistant history maps custom_tool_call & namespaced function_call accurately
        first_assistant_calls = fwd_json["messages"][1]["tool_calls"]
        self.assertEqual(first_assistant_calls[0]["function"]["name"], "apply_patch")
        self.assertEqual(
            json.loads(first_assistant_calls[0]["function"]["arguments"]),
            {"input": patch_text},
        )
        second_assistant_calls = fwd_json["messages"][3]["tool_calls"]
        self.assertEqual(
            second_assistant_calls[0]["function"]["name"], "multi_agent__wait_agent"
        )

        events = [
            json.loads(line[5:].strip())
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        delta_events = [
            ev
            for ev in events
            if ev["type"] == "response.custom_tool_call_input.delta"
        ]
        self.assertEqual(len(delta_events), 1)
        self.assertEqual(delta_events[0]["call_id"], "call_patch_1")
        self.assertEqual(delta_events[0]["delta"], patch_text)

        completed_outputs = events[-1]["response"]["output"]
        self.assertEqual(
            completed_outputs[0]["type"], "custom_tool_call"
        )
        self.assertEqual(completed_outputs[0]["name"], "apply_patch")
        self.assertEqual(completed_outputs[0]["input"], patch_text)
        self.assertNotIn("namespace", completed_outputs[0])

        self.assertEqual(completed_outputs[1]["type"], "function_call")
        self.assertEqual(completed_outputs[1]["namespace"], "multi_agent")
        self.assertEqual(completed_outputs[1]["name"], "wait_agent")
        self.assertEqual(
            completed_outputs[1]["arguments"], '{"agent_id":"a1"}'
        )


if __name__ == "__main__":
    unittest.main()
