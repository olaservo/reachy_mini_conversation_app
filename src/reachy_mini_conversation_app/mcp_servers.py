"""Manage generic remote MCP server tool sources for the conversation app.

Unlike ``tool_spaces`` (which is specific to Hugging Face Gradio Spaces), this
module lets the app talk to any HTTP(S) MCP server given a URL and an optional
auth token. The token value is never persisted: only the *name* of the
environment variable that holds it is stored in the manifest, and the value is
read from the environment at runtime.

Tools discovered at ``mcp-servers add`` time are cached in the manifest, so
startup builds clients from the cache without any network discovery (matching
the installed tool-spaces behavior). Unlike Space tools, generic server tools
keep their raw namespaced name ``{alias}__{tool}`` with no redundant-prefix
cleaning: arbitrary servers have no naming convention to strip.
"""

from __future__ import annotations
import os
import re
import json
import asyncio
import logging
import argparse
from typing import Any
from pathlib import Path
from dataclasses import field, asdict, dataclass
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
    append_tools_to_profile,
    installed_space_aliases,
    disable_alias_tools_in_profiles,
)


logger = logging.getLogger(__name__)

MCP_SERVERS_FILENAME = "mcp_servers.json"
MCP_SERVERS_VERSION = 1
BEARER_AUTH_TYPE = "bearer"
# POSIX-style environment variable name: leading letter/underscore, then
# letters/digits/underscores. Names outside this set can't round-trip through the
# instance `.env` (e.g. a name with a space writes a line python-dotenv won't parse).
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
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
            raise ValueError(
                f"Unsupported MCP auth type '{self.type}'. Expected one of: {sorted(_SUPPORTED_AUTH_TYPES)}."
            )
        object.__setattr__(self, "type", auth_type)
        token_env = self.token_env.strip()
        if not token_env:
            raise ValueError("MCP auth 'token_env' (the environment variable name) cannot be empty.")
        if not _ENV_VAR_NAME_RE.match(token_env):
            raise ValueError(
                f"Invalid MCP auth 'token_env' name '{token_env}'. Use a valid environment variable "
                "name: a letter or underscore followed by letters, digits, or underscores "
                "(e.g. MCP_SERVER_TOKEN)."
            )
        object.__setattr__(self, "token_env", token_env)


@dataclass(frozen=True)
class InstalledMcpServer:
    """Persisted record for one configured MCP server and the tools discovered at add time."""

    alias: str
    url: str
    auth: McpServerAuth | None = None
    request_timeout_s: float = 10.0
    tool_timeout_s: float = 30.0
    tools: list[InstalledToolSpaceTool] = field(default_factory=list)

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
    """Persisted manifest of configured MCP servers."""

    version: int = MCP_SERVERS_VERSION
    servers: list[InstalledMcpServer] = field(default_factory=list)


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
        cached_tools = [
            InstalledToolSpaceTool(
                local_name=str(tool["local_name"]),
                client_tool_name=str(tool["client_tool_name"]),
                remote_name=str(tool.get("remote_name", "")),
                description=str(tool.get("description", "")),
                parameters_schema=dict(tool.get("parameters_schema") or {}),
            )
            for tool in raw_server.get("tools", [])
            if isinstance(tool, dict) and tool.get("local_name") and tool.get("client_tool_name")
        ]
        try:
            server = InstalledMcpServer(
                alias=alias,
                url=str(raw_server.get("url", "")),
                auth=auth,
                request_timeout_s=float(raw_server.get("request_timeout_s", 10.0)),
                tool_timeout_s=float(raw_server.get("tool_timeout_s", 30.0)),
                tools=cached_tools,
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


def configured_server_aliases(instance_path: str | Path | None) -> set[str]:
    """Aliases claimed by configured MCP servers, for cross-source collision checks. Empty on read failure."""
    try:
        return {server.alias for server in read_mcp_servers(instance_path).servers}
    except Exception:
        return set()


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
            "tools": [asdict(tool) for tool in server.tools],
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
        if server.url.lower().startswith("http://"):
            logger.warning(
                "MCP server '%s' sends its bearer token over plain HTTP (%s); the token is "
                "visible to anyone on the local network. Prefer HTTPS.",
                server.alias,
                server.url,
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


def build_generic_remote_client(server: InstalledMcpServer) -> RemoteMcpToolClient:
    """Build an MCP client for a configured server from its cached tools.

    A missing auth token is not fatal here: startup must still register the
    cached tools so the settings UI can report the unset token, and calls fail
    with an auth error until the token is supplied.
    """
    try:
        headers = _resolve_auth_headers(server)
    except RuntimeError as exc:
        logger.warning("%s Calls to '%s' will fail until it is set.", exc, server.alias)
        headers = {}
    return RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias=server.alias,
            url=server.url,
            headers=headers,
            request_timeout_s=server.request_timeout_s,
            tool_timeout_s=server.tool_timeout_s,
        ),
        known_tools=[
            RemoteToolSpec(
                server_alias=server.alias,
                remote_name=tool.remote_name,
                namespaced_name=tool.client_tool_name,
                description=tool.description,
                parameters_schema=tool.parameters_schema,
            )
            for tool in server.tools
            if tool.remote_name
        ],
    )


@dataclass(frozen=True)
class McpTokenRequirement:
    """One configured MCP server's auth-token requirement, for the settings UI."""

    alias: str
    token_env: str
    token_set: bool


def list_token_requirements(instance_path: str | Path | None) -> list[McpTokenRequirement]:
    """Return the env-var token requirements for configured MCP servers.

    Used by the headless settings UI to let users supply a server's token without
    editing the instance ``.env`` by hand. ``token_set`` reflects whether the named
    environment variable currently holds a non-empty value.
    """
    requirements: list[McpTokenRequirement] = []
    for server in read_mcp_servers(instance_path).servers:
        if server.auth is not None and server.auth.type == BEARER_AUTH_TYPE:
            token_set = bool((os.environ.get(server.auth.token_env) or "").strip())
            requirements.append(
                McpTokenRequirement(alias=server.alias, token_env=server.auth.token_env, token_set=token_set)
            )
    return requirements


def find_server_token_env(instance_path: str | Path | None, alias: str) -> str | None:
    """Return the token env-var name for a configured MCP server alias, or None."""
    for server in read_mcp_servers(instance_path).servers:
        if server.alias == alias and server.auth is not None and server.auth.type == BEARER_AUTH_TYPE:
            return server.auth.token_env
    return None


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


async def resolve_mcp_server(server: InstalledMcpServer) -> InstalledMcpServer:
    """Connect to a configured MCP server and return it with freshly discovered tools."""
    client = RemoteMcpToolClient(build_server_config(server))
    try:
        remote_specs = await client.list_tool_specs()
    except McpClientError as exc:
        raise RuntimeError(f"Failed to discover MCP tools for '{server.alias}': {exc}") from exc

    return InstalledMcpServer(
        alias=server.alias,
        url=server.url,
        auth=server.auth,
        request_timeout_s=server.request_timeout_s,
        tool_timeout_s=server.tool_timeout_s,
        tools=_build_generic_server_tools(remote_specs),
    )


def resolve_mcp_server_sync(server: InstalledMcpServer) -> InstalledMcpServer:
    """Resolve one configured MCP server synchronously."""
    return asyncio.run(resolve_mcp_server(server))


def format_mcp_server_listing(server: InstalledMcpServer) -> str:
    """Format one configured MCP server for terminal output (no secrets)."""
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
    # Importing config loads the .env file, so an auth token placed there is
    # available when resolving servers from the standalone CLI.
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
        existing = next((entry for entry in manifest.servers if entry.alias == server.alias), None)
        if existing is not None and existing.url != server.url:
            logger.error(
                "MCP server alias '%s' is already configured for %s. Remove it first to point it elsewhere.",
                server.alias,
                existing.url,
            )
            return 1
        if existing is None and server.alias in installed_space_aliases(instance_path):
            logger.error(
                "Cannot add MCP server '%s': its alias collides with an installed tool space. "
                "Choose a different alias.",
                server.alias,
            )
            return 1

        # Resolve first so we fail fast on bad URL, unreachable server, or missing token,
        # before persisting anything. Discovered tools are cached in the manifest so
        # startup needs no network; re-running add refreshes the cache.
        try:
            resolved = resolve_mcp_server_sync(server)
        except Exception as exc:
            logger.error("Could not connect to MCP server '%s' at %s: %s", server.alias, server.url, exc)
            return 1

        other_servers = [entry for entry in manifest.servers if entry.alias != server.alias]
        updated_servers = sorted([*other_servers, resolved], key=lambda s: s.alias)
        manifest_path = write_mcp_servers(
            instance_path,
            InstalledMcpServersManifest(version=manifest.version, servers=updated_servers),
        )
        logger.info("%s MCP server: %s", "Refreshed" if existing is not None else "Configured", server.alias)
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
            added = append_tools_to_profile(target_profile, tool_ids)
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
        for profile_name, disabled_tool_ids in disable_alias_tools_in_profiles(alias):
            logger.info("Disabled in profile '%s': %s", profile_name, disabled_tool_ids)
        return 0

    if command == "list":
        manifest = read_mcp_servers(instance_path)
        manifest_path = get_mcp_servers_path(instance_path)
        logger.info("Manifest: %s", manifest_path)
        if not manifest.servers:
            logger.info("No configured MCP servers.")
            return 0
        for server in manifest.servers:
            logger.info("%s", format_mcp_server_listing(server))
        return 0

    raise RuntimeError(f"Unknown mcp-servers command: {command}")
