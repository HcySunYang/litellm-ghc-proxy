"""
Web page fetcher.

Fetches URLs and converts HTML content to markdown using html2text.
"""

import logging
import os
from typing import Any

import html2text
import httpx

logger = logging.getLogger(__name__)

FETCH_TIMEOUT = float(os.environ.get("FETCH_TIMEOUT", "30"))
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", "100000"))  # ~100KB of text

# Configure html2text
_h2t = html2text.HTML2Text()
_h2t.ignore_links = False
_h2t.ignore_images = True
_h2t.ignore_emphasis = False
_h2t.body_width = 0  # Don't wrap lines
_h2t.skip_internal_links = True


async def fetch_url(url: str, prompt: str = "") -> dict[str, Any]:
    """
    Fetch a URL and convert its content to markdown.

    Args:
        url: The URL to fetch
        prompt: Optional prompt describing what to extract (included as context)

    Returns:
        Dict with keys: url, title, content
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")

            if "text/html" in content_type or "application/xhtml" in content_type:
                html_content = resp.text
                markdown = _h2t.handle(html_content)

                # Extract title from HTML
                title = _extract_title(html_content)
            elif "application/json" in content_type:
                markdown = f"```json\n{resp.text[:MAX_CONTENT_LENGTH]}\n```"
                title = url
            elif "text/" in content_type:
                markdown = resp.text[:MAX_CONTENT_LENGTH]
                title = url
            else:
                markdown = f"[Binary content: {content_type}, {len(resp.content)} bytes]"
                title = url

            # Truncate if too long
            if len(markdown) > MAX_CONTENT_LENGTH:
                markdown = markdown[:MAX_CONTENT_LENGTH] + "\n\n[Content truncated...]"

    except httpx.TimeoutException:
        logger.error(f"Fetch timed out for URL: {url}")
        return {
            "url": url,
            "title": "Fetch Error",
            "content": f"Request timed out when fetching {url}",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"Fetch HTTP error: {e.response.status_code} for URL: {url}")
        return {
            "url": url,
            "title": "Fetch Error",
            "content": f"HTTP {e.response.status_code} when fetching {url}",
        }
    except Exception as e:
        logger.error(f"Fetch failed for URL {url}: {e}")
        return {
            "url": url,
            "title": "Fetch Error",
            "content": f"Failed to fetch {url}: {str(e)}",
        }

    logger.info(f"Fetched {url}: {len(markdown)} chars")
    return {
        "url": url,
        "title": title,
        "content": markdown,
    }


def _extract_title(html: str) -> str:
    """Extract the <title> from HTML content."""
    import re

    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        title = match.group(1).strip()
        # Clean up whitespace
        title = re.sub(r"\s+", " ", title)
        return title[:200]  # Limit title length
    return ""
