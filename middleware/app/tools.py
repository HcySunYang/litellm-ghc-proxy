"""
Tool interception and conversion for Anthropic server-side tools.

Handles:
- Extracting web_search_* and web_fetch_* server-side tools from requests
- Injecting equivalent regular function tools
- Converting response format back to Anthropic's server-side tool format
"""

import base64
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Server-side tool type patterns
SERVER_TOOL_PATTERNS = re.compile(r"^web_search_\d+$|^web_fetch_\d+$")

# Regular tool definitions to inject as replacements
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current information using a search engine. "
        "Returns a list of search results with titles, URLs, and snippets. "
        "Use this when you need up-to-date information that may not be in your training data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to execute",
            }
        },
        "required": ["query"],
    },
}

WEB_FETCH_TOOL = {
    "name": "web_fetch",
    "description": (
        "Fetch and read the contents of a web page at a given URL. "
        "Returns the page content converted to markdown format. "
        "Use this to read specific web pages, documentation, articles, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the web page to fetch",
            },
            "prompt": {
                "type": "string",
                "description": "A prompt describing what information to extract from the page",
            },
        },
        "required": ["url"],
    },
}


def extract_server_tools(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict]]:
    """
    Extract server-side tools from the tools list.

    Returns:
        Tuple of:
        - remaining_tools: tools list with server-side tools removed
        - server_tools: dict mapping tool type to its config
          e.g. {"web_search_20250305": {"type": "web_search_20250305", "name": "web_search", ...}}
    """
    remaining = []
    server_tools = {}

    for tool in tools:
        tool_type = tool.get("type", "")
        if SERVER_TOOL_PATTERNS.match(tool_type):
            server_tools[tool_type] = tool
            logger.info(f"Extracted server-side tool: {tool_type}")
        else:
            remaining.append(tool)

    return remaining, server_tools


def inject_regular_tools(
    tools: list[dict[str, Any]], server_tools: dict[str, dict]
) -> list[dict[str, Any]]:
    """
    Inject regular function tools to replace extracted server-side tools.
    """
    result = list(tools)
    has_search = False
    has_fetch = False

    for tool_type in server_tools:
        if "web_search" in tool_type:
            has_search = True
        if "web_fetch" in tool_type:
            has_fetch = True

    if has_search:
        result.append(WEB_SEARCH_TOOL)
        logger.info("Injected regular web_search tool")

    if has_fetch:
        result.append(WEB_FETCH_TOOL)
        logger.info("Injected regular web_fetch tool")

    return result


def is_web_tool_call(block: dict[str, Any]) -> bool:
    """Check if a content block is a tool_use for our web tools."""
    return (
        block.get("type") == "tool_use"
        and block.get("name") in ("web_search", "web_fetch")
    )


def convert_tool_use_to_server_format(block: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a regular tool_use block to server_tool_use format.

    Input:  {"type": "tool_use", "id": "toolu_xxx", "name": "web_search", "input": {"query": "..."}}
    Output: {"type": "server_tool_use", "id": "srvtoolu_xxx", "name": "web_search", "input": {"query": "..."}}
    """
    original_id = block.get("id", "")
    # Replace toolu_ prefix with srvtoolu_ prefix
    if original_id.startswith("toolu_"):
        server_id = "srvtoolu_" + original_id[6:]
    else:
        server_id = "srvtoolu_" + original_id

    return {
        "type": "server_tool_use",
        "id": server_id,
        "name": block["name"],
        "input": block.get("input", {}),
    }


def build_search_result_block(
    tool_use_id: str, results: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Build a web_search_tool_result block from search results.

    Each result should have: url, title, snippet, page_age (optional)
    """
    # Convert tool_use_id to server format if needed
    if tool_use_id.startswith("toolu_"):
        server_id = "srvtoolu_" + tool_use_id[6:]
    else:
        server_id = tool_use_id

    content = []
    for r in results:
        snippet = r.get("snippet", r.get("content", ""))
        try:
            encoded = base64.b64encode(snippet.encode("utf-8", errors="replace")).decode("utf-8")
        except Exception:
            encoded = base64.b64encode(b"[Content encoding error]").decode("utf-8")
        entry = {
            "type": "web_search_result",
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "encrypted_content": encoded,
        }
        if r.get("page_age"):
            entry["page_age"] = r["page_age"]
        content.append(entry)

    return {
        "type": "web_search_tool_result",
        "tool_use_id": server_id,
        "content": content,
    }


def build_fetch_result_block(
    tool_use_id: str, url: str, content: str, title: str = ""
) -> dict[str, Any]:
    """
    Build a web_fetch_tool_result block from fetched content.
    We reuse the web_search_tool_result format with a single result entry.
    """
    if tool_use_id.startswith("toolu_"):
        server_id = "srvtoolu_" + tool_use_id[6:]
    else:
        server_id = tool_use_id

    try:
        encoded = base64.b64encode(content.encode("utf-8", errors="replace")).decode("utf-8")
    except Exception:
        encoded = base64.b64encode(b"[Content encoding error]").decode("utf-8")

    return {
        "type": "web_search_tool_result",
        "tool_use_id": server_id,
        "content": [
            {
                "type": "web_search_result",
                "url": url,
                "title": title or url,
                "encrypted_content": encoded,
            }
        ],
    }


def convert_response_content(content: list[dict[str, Any]], tool_results: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert response content blocks:
    - tool_use blocks for web tools → server_tool_use + web_search_tool_result
    - other blocks pass through unchanged

    Args:
        content: The response content blocks
        tool_results: Dict mapping tool_use_id to result block
    """
    converted = []
    for block in content:
        if is_web_tool_call(block):
            # Convert to server_tool_use
            server_block = convert_tool_use_to_server_format(block)
            converted.append(server_block)

            # Append the corresponding result block
            tool_id = block.get("id", "")
            if tool_id in tool_results:
                converted.append(tool_results[tool_id])
        else:
            converted.append(block)
    return converted
