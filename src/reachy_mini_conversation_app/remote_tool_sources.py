"""Source-agnostic building blocks shared by the remote tool-source manifests.

Both installed Hugging Face Spaces (``tool_spaces``) and generic MCP servers
(``mcp_servers``) persist the tools discovered at install time as
``CachedRemoteTool`` records, rebuild clients from that cache at startup, and
wire their tool IDs into profiles. This module holds those shared pieces so the
generic MCP-server code does not depend on Space-specific machinery.
"""

import json
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.mcp_client import (
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
)


# Where terminal mode (no managed app instance) keeps manifests and downloads.
TERMINAL_EXTERNAL_CONTENT_DIRECTORY = Path("external_content")


def manifest_path(instance_path: str | Path | None, filename: str) -> Path:
    """Return a source manifest's path: the app instance dir, or external_content/ in terminal mode."""
    if instance_path is not None:
        return Path(instance_path) / filename
    return TERMINAL_EXTERNAL_CONTENT_DIRECTORY / filename


def read_manifest_envelope(path: Path, entries_key: str) -> tuple[list[object], int] | None:
    """Read and validate a manifest's shared JSON envelope, returning (raw entries, version) or None when absent."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid payload in {path}: expected a JSON object.")
    entries = payload.get(entries_key, [])
    if not isinstance(entries, list):
        raise RuntimeError(f"Invalid payload in {path}: '{entries_key}' must be a list.")
    version = payload.get("version", 1)
    if not isinstance(version, int):
        raise RuntimeError(f"Invalid payload in {path}: 'version' must be an int.")
    return entries, version


def write_manifest_payload(path: Path, payload: dict[str, Any]) -> Path:
    """Persist a manifest payload with the shared on-disk format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class CachedRemoteTool:
    """App-facing metadata for one remote tool cached in a source's manifest."""

    local_name: str
    client_tool_name: str
    remote_name: str
    description: str
    parameters_schema: dict[str, Any]


def parse_cached_tools(raw_tools: object) -> list[CachedRemoteTool]:
    """Parse a manifest entry's cached-tools list, skipping malformed items."""
    if not isinstance(raw_tools, list):
        return []
    return [
        CachedRemoteTool(
            local_name=str(tool["local_name"]),
            client_tool_name=str(tool["client_tool_name"]),
            remote_name=str(tool.get("remote_name", "")),
            description=str(tool.get("description", "")),
            parameters_schema=dict(tool.get("parameters_schema") or {}),
        )
        for tool in raw_tools
        if isinstance(tool, dict) and tool.get("local_name") and tool.get("client_tool_name")
    ]


def build_cached_tools_client(
    server_config: RemoteMcpServerConfig,
    cached_tools: Sequence[CachedRemoteTool],
) -> RemoteMcpToolClient:
    """Build an MCP client from a transport config and manifest-cached tool records."""
    return RemoteMcpToolClient(
        server_config,
        known_tools=[
            RemoteToolSpec(
                server_alias=server_config.alias,
                remote_name=tool.remote_name,
                namespaced_name=tool.client_tool_name,
                description=tool.description,
                parameters_schema=tool.parameters_schema,
            )
            for tool in cached_tools
            if tool.remote_name
        ],
    )


def append_tools_to_profile(profile: str, tool_ids: list[str]) -> list[str]:
    """Append tool IDs to a profile's tools.txt. Returns the IDs that were added."""
    tools_txt = config.resolve_profile_dir(profile) / "tools.txt"
    if not tools_txt.parent.is_dir():
        raise RuntimeError(
            f"Profile '{profile}' not found at {tools_txt.parent}. Use --install-only to skip profile wiring."
        )

    existing_content = tools_txt.read_text(encoding="utf-8") if tools_txt.exists() else ""
    existing: set[str] = set()
    for line in existing_content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            existing.add(stripped)

    to_add = [tid for tid in tool_ids if tid not in existing]
    if to_add:
        with tools_txt.open("a", encoding="utf-8") as f:
            if existing_content and not existing_content.endswith("\n"):
                f.write("\n")
            for tid in to_add:
                f.write(f"{tid}\n")
    return to_add


def disable_alias_tools_in_profiles(alias: str) -> list[tuple[str, list[str]]]:
    """Strip an alias's tool IDs from every profile's tools.txt. Returns (profile, removed IDs) per profile touched."""
    prefix = f"{alias}__"
    removed_by_profile: list[tuple[str, list[str]]] = []
    seen: set[Path] = set()
    for root in (config.PROFILES_DIRECTORY, config.user_personalities_root()):
        for tools_txt in sorted(root.glob("*/tools.txt")):
            resolved = tools_txt.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            lines = tools_txt.read_text(encoding="utf-8").splitlines()
            removed = [line.strip() for line in lines if line.strip().startswith(prefix)]
            if not removed:
                continue
            kept = [line for line in lines if not line.strip().startswith(prefix)]
            tools_txt.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
            removed_by_profile.append((tools_txt.parent.name, removed))
    return removed_by_profile
