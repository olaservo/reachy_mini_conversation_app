"""Async MCP events client for the experimental Triggers & Events primitive.

The app's ``mcp_client.py`` is tools-only (``list_tool_specs`` / ``call_tool`` via the
official MCP SDK) and there is no Python SDK for the alpha *events* primitive, so this
module hand-rolls the client side: JSON-RPC over the MCP Streamable-HTTP transport plus
SSE parsing of ``notifications/events/event``.

It is an async ``httpx`` port of the validated standalone spike
(``ha-events-bridge/scripts/spike-client.py``), so it integrates as an asyncio task on
the conversation handler's event loop with no threads. Used by ``events_loop.py`` to
subscribe to ``ha.state_changed`` on the ha-events-bridge and feed fires into the
realtime injection path.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _load_httpx() -> Any:
    """Import httpx lazily so importing this module never hard-fails."""
    import httpx

    return httpx


class EventsClientError(RuntimeError):
    """Base error for the events client."""


class EventsTransportError(EventsClientError):
    """Raised when an HTTP/transport-level failure occurs."""


class EventsProtocolError(EventsClientError):
    """Raised when the server returns a JSON-RPC error for a request."""


def _iter_sse_data(buffer: str) -> tuple[list[dict[str, Any]], str]:
    """Drain complete SSE frames from ``buffer``, returning (messages, remainder).

    Frames are separated by a blank line (``\\n\\n``); each ``data:`` line is collected
    and the joined payload is parsed as JSON. Non-JSON priming frames are skipped.
    """
    messages: list[dict[str, Any]] = []
    while True:
        sep = buffer.find("\n\n")
        if sep == -1:
            break
        raw_event = buffer[:sep]
        buffer = buffer[sep + 2:]
        data_lines = [
            line[5:].lstrip()
            for line in raw_event.split("\n")
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        try:
            messages.append(json.loads("\n".join(data_lines)))
        except json.JSONDecodeError:
            continue  # priming / non-JSON event
    return messages, buffer


class EventsClient:
    """Minimal JSON-RPC-over-Streamable-HTTP client for the MCP events primitive.

    One instance == one MCP session against one bridge URL. Construct, ``initialize()``,
    optionally ``list_events()``, then iterate ``stream(...)`` for pushed occurrences.
    Reuses a single ``httpx.AsyncClient`` for the session lifetime.
    """

    def __init__(self, url: str, *, request_timeout_s: float = 15.0) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported events bridge URL scheme {parsed.scheme!r}; use http(s)://")
        self.url = url
        self.session_id: str | None = None
        self.protocol_version = "2025-06-18"
        self._next_id = 1
        self._request_timeout_s = request_timeout_s
        self._httpx = _load_httpx()
        # No read timeout on the client: events/stream is a long-lived SSE response.
        self._client = self._httpx.AsyncClient(
            timeout=self._httpx.Timeout(request_timeout_s, read=None),
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "EventsClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        if self.protocol_version:
            headers["mcp-protocol-version"] = self.protocol_version
        return headers

    async def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and return the matching response message.

        Handles both a plain JSON response and an SSE response that carries the reply
        among other frames (per the MCP Streamable-HTTP transport).
        """
        req_id = self._next_id
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}

        try:
            async with self._client.stream(
                "POST", self.url, json=payload, headers=self._headers()
            ) as response:
                sid = response.headers.get("mcp-session-id")
                if sid:
                    self.session_id = sid
                if response.status_code not in (200, 202):
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise EventsTransportError(f"{method} -> HTTP {response.status_code}: {body}")

                ctype = response.headers.get("content-type", "")
                if "text/event-stream" in ctype:
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        messages, buffer = _iter_sse_data(buffer)
                        for msg in messages:
                            if msg.get("id") == req_id:
                                return msg
                    raise EventsTransportError(f"{method}: stream closed before a response with id {req_id}")

                return json.loads((await response.aread()).decode("utf-8"))
        except self._httpx.HTTPError as exc:
            raise EventsTransportError(f"{method}: transport error: {exc}") from exc

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        try:
            response = await self._client.post(self.url, json=payload, headers=self._headers())
        except self._httpx.HTTPError as exc:
            raise EventsTransportError(f"{method} (notification): transport error: {exc}") from exc
        if response.status_code not in (200, 202):
            raise EventsTransportError(
                f"{method} (notification) -> HTTP {response.status_code}: {response.text}"
            )

    async def initialize(self, *, client_name: str = "reachy-events-client", client_version: str = "0.1.0") -> dict[str, Any]:
        """Run the MCP handshake: ``initialize`` then ``notifications/initialized``."""
        result = await self._rpc(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": client_version},
            },
        )
        if result.get("error"):
            raise EventsProtocolError(f"initialize error: {json.dumps(result['error'])}")
        self.protocol_version = (result.get("result") or {}).get("protocolVersion", self.protocol_version)
        await self._notify("notifications/initialized", {})
        return result

    async def list_events(self) -> list[dict[str, Any]]:
        """Return the event types advertised by the bridge (``events/list``)."""
        result = await self._rpc("events/list", {})
        if result.get("error"):
            raise EventsProtocolError(f"events/list error: {json.dumps(result['error'])}")
        return (result.get("result") or {}).get("events", [])

    async def stream(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        cursor: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Open ``events/stream`` and yield each pushed JSON-RPC message.

        Yields raw messages so the caller decides which notification methods matter
        (``notifications/events/event`` and the ``.../active`` / ``.../heartbeat`` /
        ``.../error`` lifecycle signals). ``cursor=None`` means no replay. The async
        generator runs until the SSE closes or the caller stops iterating.
        """
        stream_id = self._next_id
        self._next_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": stream_id,
            "method": "events/stream",
            "params": {"name": name, "params": params or {}, "cursor": cursor},
        }

        async with self._client.stream(
            "POST", self.url, json=payload, headers=self._headers()
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", "replace")
                raise EventsTransportError(f"events/stream -> HTTP {response.status_code}: {body}")
            ctype = response.headers.get("content-type", "")
            if "text/event-stream" not in ctype:
                raise EventsTransportError(f"events/stream did not return an SSE stream (content-type: {ctype})")

            logger.debug("events/stream opened for %r (session=%s)", name, self.session_id)
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                messages, buffer = _iter_sse_data(buffer)
                for msg in messages:
                    yield msg
