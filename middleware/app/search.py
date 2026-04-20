"""
SearXNG search client.

Executes web searches via a self-hosted SearXNG instance
and returns structured results.
"""

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")
SEARCH_TIMEOUT = float(os.environ.get("SEARCH_TIMEOUT", "15"))
MAX_RESULTS = int(os.environ.get("MAX_SEARCH_RESULTS", "10"))
# Comma-separated SearXNG categories. Including "news" surfaces fresh content
# for time-sensitive queries (stock prices, weather, breaking news) which the
# "general" category alone often misses.
SEARCH_CATEGORIES = os.environ.get("SEARCH_CATEGORIES", "general,news")


async def search(query: str) -> list[dict[str, Any]]:
    """
    Execute a web search via SearXNG.

    Args:
        query: The search query string

    Returns:
        List of result dicts with keys: url, title, snippet, page_age
    """
    params = {
        "q": query,
        "format": "json",
        "categories": SEARCH_CATEGORIES,
    }

    try:
        async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
            resp = await client.get(f"{SEARXNG_URL}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        logger.error(f"SearXNG search timed out for query: {query}")
        return [{"url": "", "title": "Search Error", "snippet": "Search request timed out. Please try again."}]
    except httpx.HTTPStatusError as e:
        logger.error(f"SearXNG HTTP error: {e.response.status_code} for query: {query}")
        return [{"url": "", "title": "Search Error", "snippet": f"Search returned HTTP {e.response.status_code}"}]
    except Exception as e:
        logger.error(f"SearXNG search failed: {e}")
        return [{"url": "", "title": "Search Error", "snippet": f"Search failed: {str(e)}"}]

    results = []
    for item in data.get("results", [])[:MAX_RESULTS]:
        result = {
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "snippet": item.get("content", ""),
        }
        # SearXNG sometimes provides publishedDate
        if item.get("publishedDate"):
            result["page_age"] = item["publishedDate"]
        results.append(result)

    if not results:
        results = [{
            "url": "",
            "title": "No Results",
            "snippet": f"No search results found for: {query}",
        }]

    logger.info(f"Search for '{query}' returned {len(results)} results")
    return results
