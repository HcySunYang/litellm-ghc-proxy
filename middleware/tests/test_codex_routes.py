import json
import unittest
from itertools import product
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from app.main import app


class CodexModelRoutesTests(unittest.TestCase):
    def test_response_size_errors_enable_codex_context_recovery(self):
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
                        patch.object(
                            upstream, "aclose", wraps=upstream.aclose
                        ) as close,
                        TestClient(app) as client,
                    ):
                        response = client.post(path, json=payload)

                    self.assertEqual(response.status_code, 400)
                    error = response.json()["error"]
                    self.assertEqual(error["code"], "context_length_exceeded")
                    self.assertEqual(error["type"], "invalid_request_error")
                    self.assertEqual(error["param"], "input")
                    self.assertIn("Compact", error["message"])
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
                    close.assert_awaited_once()
                    upstream._client.aclose.assert_awaited_once()

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


if __name__ == "__main__":
    unittest.main()
