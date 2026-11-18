import gzip
import unittest

import httpx
from fastapi.responses import Response

from app.main import (
    _copy_upstream_headers,
    _forwardable_websocket_close_code,
    _request_is_streaming,
)


class ResponseHeadersTests(unittest.TestCase):
    def test_invalid_utf8_json_is_not_treated_as_streaming(self):
        self.assertFalse(
            _request_is_streaming(
                b"\xff",
                {"content-type": "application/json"},
            )
        )

    def test_filters_reserved_websocket_close_codes(self):
        self.assertEqual(_forwardable_websocket_close_code(1008), 1008)
        self.assertEqual(_forwardable_websocket_close_code(4001), 4001)
        self.assertEqual(_forwardable_websocket_close_code(1006), 1011)

    def test_preserves_repeated_set_cookie_headers(self):
        upstream = httpx.Response(
            200,
            headers=[
                ("set-cookie", "session=one; Path=/"),
                ("set-cookie", "csrf=two; Path=/"),
                ("content-type", "application/json"),
            ],
            content=b"{}",
        )
        response = Response(content=b"{}")

        _copy_upstream_headers(
            response,
            upstream,
        )

        self.assertEqual(
            response.headers.getlist("set-cookie"),
            ["session=one; Path=/", "csrf=two; Path=/"],
        )

    def test_drops_content_encoding_for_decompressed_bodies(self):
        upstream = httpx.Response(
            200,
            headers=[
                ("content-encoding", "gzip"),
                ("content-type", "application/json"),
            ],
            content=gzip.compress(b"{}"),
        )
        response = Response(content=b"{}")

        _copy_upstream_headers(
            response,
            upstream,
        )

        self.assertNotIn("content-encoding", response.headers)

    def test_preserves_representation_length_for_head_responses(self):
        upstream = httpx.Response(
            200,
            headers=[
                ("content-length", "26494"),
                ("content-type", "text/html"),
            ],
        )
        response = Response(content=b"")

        _copy_upstream_headers(
            response,
            upstream,
            preserve_content_encoding=True,
            preserve_content_length=True,
        )

        self.assertEqual(response.headers["content-length"], "26494")

    def test_drops_synthesized_length_when_head_upstream_has_none(self):
        upstream = httpx.Response(
            200,
            headers=[("content-type", "text/html")],
        )
        response = Response(content=b"")

        _copy_upstream_headers(
            response,
            upstream,
            preserve_content_encoding=True,
            preserve_content_length=True,
        )

        self.assertNotIn("content-length", response.headers)


if __name__ == "__main__":
    unittest.main()
