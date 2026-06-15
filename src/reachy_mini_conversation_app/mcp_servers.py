"""Manage generic remote MCP server tool sources for the conversation app.

Unlike ``tool_spaces`` (which is specific to public Hugging Face Gradio Spaces),
this module lets the app talk to any HTTP(S) MCP server given a URL and an
optional auth token. The token value is never persisted: only the *name* of the
environment variable that holds it is stored in the manifest, and the value is
read from the environment at runtime.
"""

from __future__ import annotations
import os
import json
import asyncio
import logging
import argparse
from typing import Any
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Sequence

from reachy_mini_conversation_app.mcp_client import (
    McpClientError,
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
    _require_name_segment,
    validate_http_mcp_url,
)
from reachy_mini_conversation_app.tool_spaces import (
    TERMINAL_EXTERNAL_CONTENT_DIRECTORY,
    InstalledToolSpaceTool,
    _append_tools_to_profile,
    read_installed_tool_spaces,
)


logger = logging.getLogger(__name__)

MCP_SERVERS_FILENAME = "mcp_servers.json"
BEARER_AUTH_TYPE = "bearer"
_SUPPORTED_AUTH_TYPES = {BEARER_AUTH_TYPE}


@dataclass(frozen=True)
class McpServerAuth:
    """Auth descriptor for an MCP server. Stores the env-var name, never the secret."""

    type: str
    token_env: str

    def __post_init__(self) -> None:
        """Validate the auth descriptor."""
        auth_type = self.type.strip().lower()
        if auth_type not in _SUPPORTED_AUTH_TYPES:
            raise ValueError(f"Unsupported MCP auth type '{self.type}'. Expected one of: {sorted(_SUPPORTED_AUTH_TYPES)}.")
        object.__setattr__(self, "type", auth_type)
        token_env = self.token_env.strip()
        if not token_env:
            raise ValueError("MCP auth 'token_env' (the environment variable name) cannot be empty.")
        object.__setattr__(self, "token_env", token_env)


@dataclass(frozen=True)
class InstalledMcpServer:
    """Persisted record for one configured generic MCP server."""

    alias: str
    url: str
    auth: McpServerAuth | None = None
    request_timeout_s: float = 10.0
    tool_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        """Validate alias, URL and timeouts once the dataclass is created."""
        object.__setattr__(self, "alias", _require_name_segment("server alias", self.alias))
        object.__setattr__(self, "url", validate_http_mcp_url(self.url))
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be greater than zero.")
        if self.tool_timeout_s <= 0:
            raise ValueError("tool_timeout_s must be greater than zero.")


@dataclass(frozen=True)
class InstalledMcpServersManifest:
    """Persisted manifest of configured generic MCP servers."""

    version: int = 1
    servers: list[InstalledMcpServer] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedMcpServer:
    """Runtime description of a configured generic MCP server."""

    alias: str
    url: str
    tools: list[InstalledToolSpaceTool]
    client: RemoteMcpToolClient


def get_mcp_servers_path(instance_path: str | Path | None) -> Path:
    """Return the MCP servers manifest path for the current mode."""
    if instance_path is not None:
        return Path(instance_path) / MCP_SERVERS_FILENAME
    return TERMINAL_EXTERNAL_CONTENT_DIRECTORY / MCP_SERVERS_FILENAME


def _parse_auth(raw_auth: Any, alias: str, manifest_path: Path) -> McpServerAuth | None:
    if raw_auth is None:
        return None
    if not isinstance(raw_auth, dict):
        raise RuntimeError(f"Invalid 'auth' for MCP server '{alias}' in {manifest_path}: expected an object.")
    try:
        return McpServerAuth(type=str(raw_auth.get("type", "")), token_env=str(raw_auth.get("token_env", "")))
    except ValueError as exc:
        raise RuntimeError(f"Invalid 'auth' for MCP server '{alias}' in {manifest_path}: {exc}") from exc


def read_mcp_servers(instance_path: str | Path | None) -> InstalledMcpServersManifest:
    """Read the configured MCP servers manifest if present."""
    manifest_path = get_mcp_servers_path(instance_path)
    if not manifest_path.exists():
        return InstalledMcpServersManifest()

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read MCP servers from {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid MCP servers payload in {manifest_path}: expected a JSON object.")

    raw_servers = payload.get("servers", [])
    if not isinstance(raw_servers, list):
        raise RuntimeError(f"Invalid MCP servers payload in {manifest_path}: 'servers' must be a list.")

    servers: list[InstalledMcpServer] = []
    seen_aliases: set[str] = set()
    for raw_server in raw_servers:
        if not isinstance(raw_server, dict):
            raise RuntimeError(f"Invalid MCP servers entry in {manifest_path}: expected an object.")

        alias = str(raw_server.get("alias", ""))
        auth = _parse_auth(raw_server.get("auth"), alias, manifest_path)
        try:
            server = InstalledMcpServer(
                alias=alias,
                url=str(raw_server.get("url", "")),
                auth=auth,
                request_timeout_s=float(raw_server.get("request_timeout_s", 10.0)),
                tool_timeout_s=float(raw_server.get("tool_timeout_s", 30.0)),
            )
        except ValueError as exc:
            raise RuntimeError(f"Invalid MCP server entry in {manifest_path}: {exc}") from exc

        if server.alias in seen_aliases:
            raise RuntimeError(f"Duplicate MCP server alias '{server.alias}' found in {manifest_path}.")
        seen_aliases.add(server.alias)
        servers.append(server)

    version = payload.get("version", 1)
    if not isinstance(version, int):
        raise RuntimeError(f"Invalid MCP servers payload in {manifest_path}: 'version' must be an int.")
    return InstalledMcpServersManifest(version=version, servers=servers)


def write_mcp_servers(instance_path: str | Path | None, manifest: InstalledMcpServersManifest) -> Path:
    """Persist the MCP servers manifest. The token value is never stored, only token_env."""
    manifest_path = get_mcp_servers_path(instance_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    servers_payload: list[dict[str, Any]] = []
    for server in manifest.servers:
        entry: dict[str, Any] = {
            "alias": server.alias,
            "url": server.url,
            "request_timeout_s": server.request_timeout_s,
            "tool_timeout_s": server.tool_timeout_s,
        }
        if server.auth is not None:
            entry["auth"] = {"type": server.auth.type, "token_env": server.auth.token_env}
        servers_payload.append(entry)

    payload = {"version": manifest.version, "servers": servers_payload}
    manifest_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return manifest_path


def _resolve_auth_headers(server: InstalledMcpServer) -> dict[str, str]:
    """Build request headers for a server, reading the secret from the environment."""
    if server.auth is None:
        return {}
    if server.auth.type == BEARER_AUTH_TYPE:
        token = (os.environ.get(server.auth.token_env) or "").strip()
        if not token:
            raise RuntimeError(
                f"Env var '{server.auth.token_env}' for MCP server '{server.alias}' is not set or empty."
            )
        return {"Authorization": f"Bearer {token}"}
    # Unreachable: McpServerAuth validates the type, but keep this defensive.
    raise RuntimeError(f"Unsupported MCP auth type '{server.auth.type}' for server '{server.alias}'.")


def build_server_config(server: InstalledMcpServer) -> RemoteMcpServerConfig:
    """Build a transport config for a server, resolving auth headers from the environment."""
    return RemoteMcpServerConfig(
        alias=server.alias,
        url=server.url,
        headers=_resolve_auth_headers(server),
        request_timeout_s=server.request_timeout_s,
        tool_timeout_s=server.tool_timeout_s,
    )


def _build_generic_server_tools(remote_specs: Sequence[RemoteToolSpec]) -> list[InstalledToolSpaceTool]:
    """Map discovered remote specs to app-facing tools without HF-specific name cleaning."""
    return [
        InstalledToolSpaceTool(
            local_name=spec.namespaced_name,
            client_tool_name=spec.namespaced_name,
            remote_name=spec.remote_name,
            description=spec.description,
            parameters_schema=dict(spec.parameters_schema),
        )
        for spec in remote_specs
    ]


async def resolve_mcp_server(server: InstalledMcpServer) -> ResolvedMcpServer:
    """Connect to a configured MCP server and discover its tools."""
    client = RemoteMcpToolClient(build_server_config(server))
    try:
        remote_specs = await client.list_tool_specs()
    except McpClientError as exc:
        raise RuntimeError(f"Failed to discover MCP tools for '{server.alias}': {exc}") from exc

    return ResolvedMcpServer(
        alias=server.alias,
        url=server.url,
        tools=_build_generic_server_tools(remote_specs),
        client=client,
    )


def resolve_mcp_server_sync(server: InstalledMcpServer) -> ResolvedMcpServer:
    """Resolve one configured MCP server synchronously."""
    return asyncio.run(resolve_mcp_server(server))


def format_mcp_server_listing(server: ResolvedMcpServer) -> str:
    """Format one resolved MCP server for terminal output (no secrets)."""
    lines = [
        f"{server.alias}",
        f"  MCP endpoint: {server.url}",
    ]
    if server.tools:
        lines.append("  Tools:")
        lines.extend([f"    - {tool.local_name}" for tool in server.tools])
    else:
        lines.append("  Tools: none discovered")
    return "\n".join(lines)


def handle_mcp_servers_command(args: argparse.Namespace, *, instance_path: str | Path | None = None) -> int:
    """Handle mcp-servers subcommands from the main CLI."""
    # Importing config loads the .env file, so an auth token placed there (e.g.
    # HA_ACCESS_TOKEN) is available when resolving servers from the standalone CLI.
    import reachy_mini_conversation_app.config  # noqa: F401

    command = getattr(args, "mcp_servers_command", None)
    if command == "add":
        auth = None
        token_env = (getattr(args, "token_env", None) or "").strip()
        if token_env:
            auth = McpServerAuth(type=BEARER_AUTH_TYPE, token_env=token_env)

        try:
            server = InstalledMcpServer(
                alias=args.alias,
                url=args.url,
                auth=auth,
                request_timeout_s=args.request_timeout,
                tool_timeout_s=args.tool_timeout,
            )
        except ValueError as exc:
            logger.error("Invalid MCP server configuration: %s", exc)
            return 1

        manifest = read_mcp_servers(instance_path)
        if any(existing.alias == server.alias for existing in manifest.servers):
            logger.error("MCP server alias '%s' is already configured. Remove it first to reconfigure.", server.alias)
            return 1
        try:
            space_aliases = {space.alias for space in read_installed_tool_spaces(instance_path).spaces}
        except Exception:
            space_aliases = set()
        if server.alias in space_aliases:
            logger.error(
                "Cannot add MCP server '%s': its alias collides with an installed tool space. "
                "Choose a different alias.",
                server.alias,
            )
            return 1

        # Resolve first so we fail fast on bad URL, unreachable server, or missing token,
        # before persisting anything.
        try:
            resolved = resolve_mcp_server_sync(server)
        except Exception as exc:
            logger.error("Could not connect to MCP server '%s' at %s: %s", server.alias, server.url, exc)
            return 1

        updated_servers = sorted([*manifest.servers, server], key=lambda s: s.alias)
        manifest_path = write_mcp_servers(
            instance_path,
            InstalledMcpServersManifest(version=manifest.version, servers=updated_servers),
        )
        logger.info("Configured MCP server: %s", server.alias)
        logger.info("Manifest: %s", manifest_path)
        logger.info("%s", format_mcp_server_listing(resolved))

        if args.install_only:
            logger.info("Server configured. Add tool IDs to a profile's tools.txt to enable them.")
            return 0

        target_profile = args.profile
        if target_profile is None:
            from reachy_mini_conversation_app.config import config

            target_profile = config.REACHY_MINI_CUSTOM_PROFILE or "default"

        tool_ids = [tool.local_name for tool in resolved.tools]
        try:
            added = _append_tools_to_profile(target_profile, tool_ids)
        except RuntimeError as exc:
            logger.error("Cannot enable tools: %s", exc)
            return 1
        if added:
            logger.info("Enabled in profile '%s': %s", target_profile, added)
        else:
            logger.info("All tool IDs already present in profile '%s'.", target_profile)
        return 0

    if command == "remove":
        alias = _require_name_segment("server alias", args.alias)
        manifest = read_mcp_servers(instance_path)
        remaining = [server for server in manifest.servers if server.alias != alias]
        if len(remaining) == len(manifest.servers):
            logger.warning("MCP server not configured: %s", alias)
            return 1
        write_mcp_servers(instance_path, InstalledMcpServersManifest(version=manifest.version, servers=remaining))
        logger.info("Removed MCP server: %s", alias)
        return 0

    if command == "list":
        manifest = read_mcp_servers(instance_path)
        manifest_path = get_mcp_servers_path(instance_path)
        logger.info("Manifest: %s", manifest_path)
        if not manifest.servers:
            logger.info("No configured MCP servers.")
            return 0
        for server in manifest.servers:
            try:
                resolved = resolve_mcp_server_sync(server)
            except Exception as exc:
                logger.warning("MCP server '%s' is unavailable: %s", server.alias, exc)
                continue
            logger.info("%s", format_mcp_server_listing(resolved))
        return 0

    raise RuntimeError(f"Unknown mcp-servers command: {command}")
