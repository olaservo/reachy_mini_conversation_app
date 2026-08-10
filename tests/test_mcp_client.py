from __future__ import annotations

import pytest


pytest.importorskip("mcp_types")

from mcp_types import Tool, TextContent, CallToolResult, ListToolsResult

from reachy_mini_conversation_app.mcp_client import (
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
    RemoteToolCallResponse,
    validate_http_mcp_url,
    build_namespaced_tool_name,
)


def test_server_config_refuses_credentials_over_plain_http_to_lan_host() -> None:
    """The config is the last line of defense: no credential header may ride plain HTTP off-machine without an explicit opt-in."""
    with pytest.raises(ValueError, match="credentials over plain HTTP"):
        RemoteMcpServerConfig(
            alias="example",
            url="http://192.168.1.50:8000/mcp",
            headers={"Authorization": "Bearer secret"},
        )


def test_server_config_allows_credentials_over_plain_http_with_opt_in_or_loopback() -> None:
    """The explicit opt-in and loopback hosts keep credentialed plain-HTTP configs buildable."""
    RemoteMcpServerConfig(
        alias="example",
        url="http://192.168.1.50:8000/mcp",
        headers={"Authorization": "Bearer secret"},
        allow_insecure_http=True,
    )
    RemoteMcpServerConfig(
        alias="example",
        url="http://127.0.0.1:8000/mcp",
        headers={"Authorization": "Bearer secret"},
    )
    RemoteMcpServerConfig(alias="example", url="http://192.168.1.50:8000/mcp")


def test_server_config_requires_opt_in_for_credentials_on_localhost_subdomains() -> None:
    """RFC 6761 loopback-pinning of '*.localhost' is only a SHOULD, so credentials over plain HTTP to a subdomain need the explicit opt-in."""
    with pytest.raises(ValueError, match="credentials over plain HTTP"):
        RemoteMcpServerConfig(
            alias="example",
            url="http://dev.localhost:8000/mcp",
            headers={"Authorization": "Bearer secret"},
        )
    RemoteMcpServerConfig(
        alias="example",
        url="http://dev.localhost:8000/mcp",
        headers={"Authorization": "Bearer secret"},
        allow_insecure_http=True,
    )


def test_alias_rejects_namespace_ambiguous_forms() -> None:
    """An alias containing '__' or ending in '_' blurs the namespace separator, letting one server's removal strip a sibling's tools."""
    for bad_alias in ("ha_", "h__a"):
        with pytest.raises(ValueError, match="Aliases cannot"):
            build_namespaced_tool_name(bad_alias, "light")
        with pytest.raises(ValueError, match="Aliases cannot"):
            RemoteMcpServerConfig(alias=bad_alias, url="http://127.0.0.1:8000/mcp")


def test_validate_http_mcp_url_rejects_non_http_scheme() -> None:
    """Only HTTP(S) MCP endpoints are supported."""
    with pytest.raises(ValueError, match="Unsupported MCP URL scheme"):
        validate_http_mcp_url("stdio://local-server")


def test_validate_http_mcp_url_rejects_non_local_plain_http() -> None:
    """Remote servers must use HTTPS unless they are local development endpoints."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_http_mcp_url("http://example.com/mcp")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/mcp",
        "http://localhost:8000/mcp",
        "http://dev.localhost:8000/mcp",
        "http://[::1]:8000/mcp",
        "http://192.168.1.50:8000/mcp",
        "http://169.254.10.10/mcp",
        "http://my-mcp-server.local:8000/mcp",
        "https://example.com/mcp",
    ],
)
def test_validate_http_mcp_url_accepts_local_network_endpoints(url: str) -> None:
    """Plain HTTP is fine on the local network (loopback, private, link-local, mDNS); HTTPS always is."""
    assert validate_http_mcp_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8/mcp",
        "http://my-server.example.org:8000/mcp",
        "http://intranet-host:8000/mcp",
    ],
)
def test_validate_http_mcp_url_rejects_public_plain_http(url: str) -> None:
    """Plain HTTP to public IPs or unresolvable-name hosts is rejected (the plain public-name case is covered above)."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_http_mcp_url(url)


def test_validate_http_mcp_url_classifies_ipv4_mapped_ipv6_by_embedded_address() -> None:
    """A mapped public address must not slip through is_private (CVE-2024-4032, is_private true for all of ::ffff:0:0/96 before 3.11.10/3.12.4)."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_http_mcp_url("http://[::ffff:8.8.8.8]/mcp")
    assert validate_http_mcp_url("http://[::ffff:192.168.1.50]/mcp")


def test_build_namespaced_tool_name_normalizes_tool_segment() -> None:
    """Remote tool names are normalized into app-safe tool IDs."""
    assert build_namespaced_tool_name("gradio_docs", "search-docs") == "gradio_docs__search_docs"


def test_remote_tool_spec_translates_to_function_spec() -> None:
    """Discovered MCP tools should translate into app function specs."""
    tool = Tool(
        name="search-docs",
        description="Search the docs",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    spec = RemoteToolSpec.from_mcp_tool("gradio_docs", tool)

    assert spec.remote_name == "search-docs"
    assert spec.namespaced_name == "gradio_docs__search_docs"
    assert spec.to_function_spec() == {
        "type": "function",
        "name": "gradio_docs__search_docs",
        "description": "Search the docs",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }


def test_remote_tool_error_result_maps_to_app_payload() -> None:
    """Remote tool errors should remain visible after response mapping."""
    result = CallToolResult(
        content=[TextContent(type="text", text="Search backend unavailable")],
        structured_content=None,
        is_error=True,
    )

    payload = RemoteToolCallResponse.from_call_tool_result(
        server_alias="gradio_docs",
        remote_tool_name="search-docs",
        result=result,
    ).to_tool_result()

    assert payload["status"] == "error"
    assert payload["namespaced_tool_name"] == "gradio_docs__search_docs"
    assert payload["text"] == "Search backend unavailable"


def test_remote_tool_spec_reads_fields_the_sdk_actually_exposes() -> None:
    """Pins the v1→v2 rename: the wire names are pydantic *aliases*, not attributes.

    Reading `tool.inputSchema` returns nothing on v2, which silently degrades every
    remote tool to an empty parameters schema instead of raising anywhere.
    """
    tool = Tool(
        name="search-docs",
        description="Search the docs",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )
    assert not hasattr(tool, "inputSchema")

    spec = RemoteToolSpec.from_mcp_tool("gradio_docs", tool)
    assert spec.parameters_schema["properties"] == {"query": {"type": "string"}}
    assert spec.parameters_schema["required"] == ["query"]


def test_remote_tool_call_response_reads_fields_the_sdk_actually_exposes() -> None:
    """Same rename on results: `isError`/`structuredContent` are aliases, so an error would read as success."""
    result = CallToolResult(
        content=[TextContent(type="text", text="done")],
        structured_content={"rows": 2},
        is_error=True,
    )
    assert not hasattr(result, "isError")
    assert not hasattr(result, "structuredContent")

    response = RemoteToolCallResponse.from_call_tool_result(
        server_alias="gradio_docs",
        remote_tool_name="search-docs",
        result=result,
    )
    assert response.status == "error"
    assert response.structured_content == {"rows": 2}


@pytest.mark.asyncio
async def test_list_all_tools_follows_the_pagination_cursor() -> None:
    """Pins the v1→v2 rename on the *paging* field, which fails silently.

    `nextCursor` is a pydantic alias on v2, so a client still reading it gets
    None on a page that genuinely has a successor: discovery stops after page
    one and every tool beyond it simply never exists, with no error raised
    anywhere. Servers with a small tool count hide this, which is why it
    survives casual testing.
    """
    first = ListToolsResult(
        tools=[Tool(name="alpha", description="first page", input_schema={"type": "object"})],
        nextCursor="page-2",
    )
    last = ListToolsResult(
        tools=[Tool(name="omega", description="second page", input_schema={"type": "object"})],
    )
    assert not hasattr(first, "nextCursor")

    pages = [first, last]
    seen_cursors: list[str | None] = []

    class _PagingClient:
        async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult:
            seen_cursors.append(cursor)
            return pages[len(seen_cursors) - 1]

    client = RemoteMcpToolClient(RemoteMcpServerConfig(alias="gradio_docs", url="https://example.invalid/mcp"))
    tools = await client._list_all_tools(_PagingClient())

    assert [tool.name for tool in tools] == ["alpha", "omega"]
    assert seen_cursors == [None, "page-2"]
