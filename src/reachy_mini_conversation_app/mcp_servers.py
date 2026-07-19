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

import os
import re
import asyncio
import logging
import argparse
from typing import Any
from pathlib import Path
from dataclasses import field, asdict, replace, dataclass
from collections.abc import Sequence

# Importing config loads the .env file, so an auth token placed there is
# available when resolving servers from the standalone CLI.
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.mcp_client import (
    McpClientError,
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
    _require_alias_segment,
    validate_http_mcp_url,
    is_plaintext_remote_url,
)
from reachy_mini_conversation_app.tool_spaces import installed_space_aliases
from reachy_mini_conversation_app.remote_tool_sources import (
    MCP_SERVERS_FILENAME,
    CachedRemoteTool,
    manifest_path,
    parse_cached_tools,
    read_manifest_envelope,
    write_manifest_payload,
    append_tools_to_profile,
    build_cached_tools_client,
    disable_alias_tools_in_profiles,
)


logger = logging.getLogger(__name__)

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
    # Explicit opt-in to sending the token over plain HTTP to a non-loopback host.
    allow_insecure_http: bool = False

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
    tools: list[CachedRemoteTool] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate alias, URL and timeouts once the dataclass is created."""
        object.__setattr__(self, "alias", _require_alias_segment("server alias", self.alias))
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
    # Entries the reader dropped as invalid. A rewrite from this manifest would
    # permanently delete them, so the CLI refuses add/remove while it is non-zero.
    skipped_entries: int = 0


def get_mcp_servers_path(instance_path: str | Path | None) -> Path:
    """Return the MCP servers manifest path for the current mode."""
    return manifest_path(instance_path, MCP_SERVERS_FILENAME)


def _parse_auth(raw_auth: object, alias: str, path: Path) -> McpServerAuth | None:
    if raw_auth is None:
        return None
    if not isinstance(raw_auth, dict):
        raise RuntimeError(f"Invalid 'auth' for MCP server '{alias}' in {path}: expected an object.")
    allow_insecure_http = raw_auth.get("allow_insecure_http", False)
    if not isinstance(allow_insecure_http, bool):
        # Refuse truthy strings like "false": a mis-typed value must never grant the insecure opt-in.
        raise RuntimeError(
            f"Invalid 'auth' for MCP server '{alias}' in {path}: 'allow_insecure_http' must be true or false."
        )
    try:
        return McpServerAuth(
            type=str(raw_auth.get("type", "")),
            token_env=str(raw_auth.get("token_env", "")),
            allow_insecure_http=allow_insecure_http,
        )
    except ValueError as exc:
        raise RuntimeError(f"Invalid 'auth' for MCP server '{alias}' in {path}: {exc}") from exc


def _parse_server_entry(raw_server: object, path: Path) -> InstalledMcpServer:
    if not isinstance(raw_server, dict):
        raise RuntimeError(f"Invalid MCP servers entry in {path}: expected an object.")
    alias = str(raw_server.get("alias", ""))
    auth = _parse_auth(raw_server.get("auth"), alias, path)
    try:
        return InstalledMcpServer(
            alias=alias,
            url=str(raw_server.get("url", "")),
            auth=auth,
            request_timeout_s=float(raw_server.get("request_timeout_s", 10.0)),
            tool_timeout_s=float(raw_server.get("tool_timeout_s", 30.0)),
            tools=parse_cached_tools(raw_server.get("tools", [])),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid MCP server entry in {path}: {exc}") from exc


def read_mcp_servers(instance_path: str | Path | None) -> InstalledMcpServersManifest:
    """Read the configured MCP servers manifest if present, skipping invalid entries so one bad server cannot disable the rest."""
    path = get_mcp_servers_path(instance_path)
    envelope = read_manifest_envelope(path, "servers")
    if envelope is None:
        return InstalledMcpServersManifest()
    raw_servers, version = envelope

    servers: list[InstalledMcpServer] = []
    seen_aliases: set[str] = set()
    skipped = 0
    for raw_server in raw_servers:
        try:
            server = _parse_server_entry(raw_server, path)
            if server.alias in seen_aliases:
                raise RuntimeError(f"Duplicate MCP server alias '{server.alias}' found in {path}.")
        except RuntimeError as exc:
            logger.warning("Skipping invalid MCP server entry: %s", exc)
            skipped += 1
            continue
        seen_aliases.add(server.alias)
        servers.append(server)

    return InstalledMcpServersManifest(version=version, servers=servers, skipped_entries=skipped)


def write_mcp_servers(instance_path: str | Path | None, manifest: InstalledMcpServersManifest) -> Path:
    """Persist the MCP servers manifest. The token value is never stored, only token_env."""
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
            if server.auth.allow_insecure_http:
                entry["auth"]["allow_insecure_http"] = True
        servers_payload.append(entry)

    payload = {"version": manifest.version, "servers": servers_payload}
    return write_manifest_payload(get_mcp_servers_path(instance_path), payload)


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
        if is_plaintext_remote_url(server.url):
            if not server.auth.allow_insecure_http:
                raise RuntimeError(
                    f"MCP server '{server.alias}' would send its bearer token over plain HTTP ({server.url}), "
                    "exposing it to anyone on the network. Use HTTPS, a loopback address, or opt in "
                    "explicitly with 'mcp-servers add ... --allow-insecure-token'."
                )
            logger.warning(
                "MCP server '%s' sends its bearer token over plain HTTP (%s) per --allow-insecure-token; "
                "the token is visible to anyone on the local network.",
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
        allow_insecure_http=server.auth.allow_insecure_http if server.auth is not None else False,
    )


def build_generic_remote_client(server: InstalledMcpServer) -> RemoteMcpToolClient:
    """Build an MCP client from cached tools, raising RuntimeError when auth cannot be resolved."""
    return build_cached_tools_client(build_server_config(server), server.tools)


@dataclass(frozen=True)
class McpTokenRequirement:
    """One configured MCP server's auth-token requirement, for the settings UI."""

    alias: str
    token_env: str
    token_set: bool


def list_token_requirements(instance_path: str | Path | None) -> list[McpTokenRequirement]:
    """Return the configured servers' auth-token requirements for the settings UI."""
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
    return next((req.token_env for req in list_token_requirements(instance_path) if req.alias == alias), None)


def _build_generic_server_tools(remote_specs: Sequence[RemoteToolSpec]) -> list[CachedRemoteTool]:
    """Map discovered remote specs to app-facing tools without HF-specific name cleaning."""
    return [
        CachedRemoteTool(
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

    return replace(server, tools=_build_generic_server_tools(remote_specs))


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
    command = getattr(args, "mcp_servers_command", None)
    if command == "add":
        manifest = read_mcp_servers(instance_path)
        if manifest.skipped_entries:
            logger.error(
                "The MCP servers manifest contains %d invalid entry(ies) that a rewrite would permanently "
                "delete. Fix or remove them in %s first (see the warnings above).",
                manifest.skipped_entries,
                get_mcp_servers_path(instance_path),
            )
            return 1
        existing = next((entry for entry in manifest.servers if entry.alias == args.alias.strip()), None)

        # Re-running add is the documented cache-refresh flow, so flags that are
        # not repeated keep their stored values instead of resetting to defaults.
        # That includes the insecure-HTTP opt-in, which sticks until 'remove'.
        auth = existing.auth if existing is not None else None
        token_env = (getattr(args, "token_env", None) or "").strip()
        stored_allow_insecure = (
            existing is not None and existing.auth is not None and existing.auth.allow_insecure_http
        )
        allow_insecure_token = bool(getattr(args, "allow_insecure_token", False)) or stored_allow_insecure
        if stored_allow_insecure and not getattr(args, "allow_insecure_token", False) and token_env:
            logger.info("Keeping the stored --allow-insecure-token opt-in for '%s'.", args.alias.strip())
        try:
            if token_env:
                auth = McpServerAuth(
                    type=BEARER_AUTH_TYPE,
                    token_env=token_env,
                    allow_insecure_http=allow_insecure_token,
                )
            elif allow_insecure_token:
                logger.warning("--allow-insecure-token has no effect without --token-env.")

            # All optional flags are read with getattr so the documented programmatic
            # call with a minimal Namespace gets the same defaults as the CLI.
            request_timeout = getattr(args, "request_timeout", None)
            tool_timeout = getattr(args, "tool_timeout", None)
            server = InstalledMcpServer(
                alias=args.alias,
                url=args.url,
                auth=auth,
                request_timeout_s=(
                    request_timeout
                    if request_timeout is not None
                    else existing.request_timeout_s
                    if existing is not None
                    else 10.0
                ),
                tool_timeout_s=(
                    tool_timeout
                    if tool_timeout is not None
                    else existing.tool_timeout_s
                    if existing is not None
                    else 30.0
                ),
            )
        except ValueError as exc:
            logger.error("Invalid MCP server configuration: %s", exc)
            return 1

        if existing is not None and existing.url != server.url:
            logger.error(
                "MCP server alias '%s' is already configured for %s. Remove it first to point it elsewhere.",
                server.alias,
                existing.url,
            )
            return 1
        if existing is not None and not token_env and existing.auth is not None:
            logger.info("Keeping stored auth for '%s' (token env '%s').", server.alias, existing.auth.token_env)
        if existing is None:
            # Fail closed: an unnoticed collision crashes tool registration at boot,
            # so refuse the add when the other manifest cannot be checked.
            try:
                space_aliases = installed_space_aliases(instance_path)
            except RuntimeError as exc:
                logger.error(
                    "Cannot add MCP server '%s': the installed tool-spaces manifest is unreadable, "
                    "so the alias cannot be checked for collisions. Fix it first: %s",
                    server.alias,
                    exc,
                )
                return 1
            if server.alias in space_aliases:
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
            logger.error("Could not resolve MCP server '%s' at %s: %s", server.alias, server.url, exc)
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

        if getattr(args, "install_only", False):
            logger.info("Server configured. Add tool IDs to a profile's tools.txt to enable them.")
            return 0

        target_profile = getattr(args, "profile", None)
        if target_profile is None:
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
        alias = _require_alias_segment("server alias", args.alias)
        manifest = read_mcp_servers(instance_path)
        if manifest.skipped_entries:
            logger.error(
                "The MCP servers manifest contains %d invalid entry(ies) that a rewrite would permanently "
                "delete. Fix or remove them in %s first (see the warnings above).",
                manifest.skipped_entries,
                get_mcp_servers_path(instance_path),
            )
            return 1
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
