"""Tests for the custom MCP server management methods."""

import os
import dataclasses
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.mcp_servers import (
    McpServerAuth,
    InstalledMcpServer,
    InstalledMcpServersManifest,
    read_mcp_servers,
    write_mcp_servers,
)
from reachy_mini_conversation_app.personality import available_tool_catalog
from reachy_mini_conversation_app.tool_spaces import InstalledToolSpacesManifest, write_installed_tool_spaces
from reachy_mini_conversation_app.profile_store import write_profile
from reachy_mini_conversation_app.profile_toolsets import read_profile_tool_names
from reachy_mini_conversation_app.mcp_server_routes import register_mcp_server_methods
from reachy_mini_conversation_app.remote_tool_sources import CachedRemoteTool


SERVER_ALIAS = "acme"
SERVER_URL = "https://mcp.example.com/mcp"
TOOL_NAME = f"{SERVER_ALIAS}__ping"
TOKEN_ENV = "ACME_MCP_TOKEN"


def _resolved_server(auth: McpServerAuth | None = None) -> InstalledMcpServer:
    return InstalledMcpServer(
        alias=SERVER_ALIAS,
        url=SERVER_URL,
        auth=auth,
        tools=[
            CachedRemoteTool(
                local_name=TOOL_NAME,
                client_tool_name=TOOL_NAME,
                remote_name="ping",
                description="Ping the server",
                parameters_schema={"type": "object", "properties": {}, "required": []},
            )
        ],
    )


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    instance_path = tmp_path / "instance"
    profiles_root = tmp_path / "profiles"
    write_profile("default", profiles_root / "default", "Default profile.", ["dance"])
    write_installed_tool_spaces(instance_path, InstalledToolSpacesManifest(spaces=[]))
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles_root)
    monkeypatch.setattr("reachy_mini_conversation_app.profile_store.DEFAULT_PROFILES_DIRECTORY", profiles_root)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.mcp_servers.resolve_mcp_server_sync",
        lambda server: dataclasses.replace(server, tools=_resolved_server().tools),
    )
    return instance_path


def _rpc_call(client: TestClient, method: str, params: dict[str, object] | None = None) -> dict[str, Any]:
    with client.websocket_connect("/rpc") as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        response: dict[str, Any] = websocket.receive_json()
        return response


def _mount_rpc(
    instance_path: Path | None,
    apply_env_values: Any = None,
    get_loop: MagicMock | None = None,
    restart_conversation: AsyncMock | None = None,
) -> TestClient:
    app = FastAPI()
    rpc = JsonRpcServer()
    register_mcp_server_methods(
        rpc,
        get_loop or MagicMock(return_value=None),
        restart_conversation or AsyncMock(),
        apply_env_values or (lambda updates: None),
        instance_path=instance_path,
    )
    rpc.mount(app)
    return TestClient(app)


def test_configured_server_tools_appear_in_the_tool_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool access lists a configured server's tools."""
    instance_path = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "INSTANCE_PATH", instance_path)
    write_mcp_servers(instance_path, InstalledMcpServersManifest(servers=[_resolved_server()]))

    catalog = {tool["id"]: tool for tool in available_tool_catalog()}

    assert catalog[TOOL_NAME] == {
        "id": TOOL_NAME,
        "kind": "mcp_server",
        "source": SERVER_ALIAS,
        "description": "Ping the server",
    }


def test_add_list_remove_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding publishes the server to the inventory without enabling its tools."""
    instance_path = _configure(tmp_path, monkeypatch)
    client = _mount_rpc(instance_path)

    added = _rpc_call(client, "mcp_servers.add", {"alias": SERVER_ALIAS, "url": SERVER_URL})["result"]
    assert added["servers"] == [
        {
            "alias": SERVER_ALIAS,
            "url": SERVER_URL,
            "tool_count": 1,
            "token_env": None,
            "token_set": False,
        }
    ]
    assert "ready to assign to personalities" in added["message"]
    assert TOOL_NAME not in read_profile_tool_names("default", instance_path)
    assert _rpc_call(client, "mcp_servers.list")["result"]["servers"] == added["servers"]

    removed = _rpc_call(client, "mcp_servers.remove", {"alias": SERVER_ALIAS})["result"]
    assert removed["servers"] == []
    assert read_mcp_servers(instance_path).servers == []


def test_add_reports_token_requirement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A server with a token variable reports whether it is set."""
    instance_path = _configure(tmp_path, monkeypatch)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    client = _mount_rpc(instance_path)

    added = _rpc_call(
        client,
        "mcp_servers.add",
        {"alias": SERVER_ALIAS, "url": SERVER_URL, "token_env": TOKEN_ENV},
    )["result"]
    assert added["servers"][0]["token_env"] == TOKEN_ENV
    assert added["servers"][0]["token_set"] is False

    monkeypatch.setenv(TOKEN_ENV, "a-real-token")
    assert _rpc_call(client, "mcp_servers.list")["result"]["servers"][0]["token_set"] is True


def test_save_token_persists_and_rebuilds_the_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving a token sets the variable and reloads tools, since a client reads its token when built."""
    instance_path = _configure(tmp_path, monkeypatch)
    saved: dict[str, str] = {}

    def _apply(updates: dict[str, str]) -> None:
        saved.update(updates)
        monkeypatch.setenv(TOKEN_ENV, updates[TOKEN_ENV])

    initialized: list[bool] = []
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_settings.initialize_tools",
        lambda instance_path=None, force=False: initialized.append(force),
    )
    client = _mount_rpc(instance_path, apply_env_values=_apply)
    write_mcp_servers(
        instance_path,
        InstalledMcpServersManifest(
            servers=[_resolved_server(auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV))]
        ),
    )

    response = _rpc_call(client, "mcp_servers.save_token", {"alias": SERVER_ALIAS, "token": "s3cr#t"})["result"]
    assert saved == {TOKEN_ENV: "s3cr#t"}
    assert response["servers"][0]["token_set"] is True
    assert initialized == [True]


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        pytest.param({"alias": SERVER_ALIAS, "token": "   "}, "empty_token", id="blank-token"),
        pytest.param({"alias": SERVER_ALIAS}, "empty_token", id="missing-token"),
        pytest.param({"alias": "", "token": "x"}, "invalid_mcp_alias", id="blank-alias"),
        pytest.param({"alias": "nope", "token": "x"}, "unknown_mcp_server", id="unconfigured-server"),
    ],
)
def test_save_token_rejects_bad_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    params: dict[str, object],
    reason: str,
) -> None:
    """Bad token saves fail with a stable reason and write nothing."""
    instance_path = _configure(tmp_path, monkeypatch)
    saved: dict[str, str] = {}
    client = _mount_rpc(instance_path, apply_env_values=saved.update)
    write_mcp_servers(
        instance_path,
        InstalledMcpServersManifest(
            servers=[_resolved_server(auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV))]
        ),
    )

    assert _rpc_call(client, "mcp_servers.save_token", params)["error"]["data"]["reason"] == reason
    assert saved == {}


def test_save_token_reply_names_the_variable_to_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reply says the token lasts for the session and names the variable to set."""
    instance_path = _configure(tmp_path, monkeypatch)
    client = _mount_rpc(instance_path)
    write_mcp_servers(
        instance_path,
        InstalledMcpServersManifest(
            servers=[_resolved_server(auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV))]
        ),
    )

    message = _rpc_call(client, "mcp_servers.save_token", {"alias": SERVER_ALIAS, "token": "x"})["result"]["message"]
    assert "this session" in message
    assert TOKEN_ENV in message


def test_add_with_a_token_applies_it_before_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A token given with the add is in the environment before discovery runs."""
    instance_path = _configure(tmp_path, monkeypatch)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    seen_at_discovery: list[str | None] = []

    def _resolve(server: InstalledMcpServer) -> InstalledMcpServer:
        seen_at_discovery.append(os.environ.get(TOKEN_ENV))
        return dataclasses.replace(server, tools=_resolved_server().tools)

    monkeypatch.setattr("reachy_mini_conversation_app.mcp_servers.resolve_mcp_server_sync", _resolve)
    client = _mount_rpc(
        instance_path,
        apply_env_values=lambda updates: [monkeypatch.setenv(name, value) for name, value in updates.items()],
    )

    added = _rpc_call(
        client,
        "mcp_servers.add",
        {"alias": SERVER_ALIAS, "url": SERVER_URL, "token_env": TOKEN_ENV, "token": "s3cr#t"},
    )["result"]

    assert seen_at_discovery == ["s3cr#t"]
    assert added["servers"][0]["token_set"] is True
    assert "this session" in added["message"]
    assert TOKEN_ENV in added["message"]
    assert read_mcp_servers(instance_path).servers[0].auth == McpServerAuth(type="bearer", token_env=TOKEN_ENV)


def test_add_rejects_a_token_without_a_variable_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A token value without a variable name is refused and nothing is applied."""
    instance_path = _configure(tmp_path, monkeypatch)
    applied: dict[str, str] = {}
    client = _mount_rpc(instance_path, apply_env_values=applied.update)

    error = _rpc_call(client, "mcp_servers.add", {"alias": SERVER_ALIAS, "url": SERVER_URL, "token": "x"})["error"]

    assert error["data"]["reason"] == "invalid_mcp_token_env"
    assert applied == {}
    assert read_mcp_servers(instance_path).servers == []


def test_list_reports_an_unreadable_manifest_instead_of_an_empty_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A damaged manifest surfaces as an error, not as an empty list."""
    instance_path = _configure(tmp_path, monkeypatch)
    instance_path.mkdir(parents=True, exist_ok=True)
    (instance_path / "mcp_servers.json").write_text("{not valid json", encoding="utf-8")
    client = _mount_rpc(instance_path)

    error = _rpc_call(client, "mcp_servers.list")["error"]
    assert error["data"]["reason"] == "mcp_servers_unavailable"


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        pytest.param({"url": SERVER_URL}, "invalid_mcp_alias", id="missing-alias"),
        pytest.param({"alias": SERVER_ALIAS}, "invalid_mcp_url", id="missing-url"),
        pytest.param({"alias": "bad__alias", "url": SERVER_URL}, "invalid_mcp_server", id="ambiguous-alias"),
        pytest.param(
            {"alias": SERVER_ALIAS, "url": "http://example.com/mcp"},
            "invalid_mcp_server",
            id="public-plain-http",
        ),
        pytest.param(
            {"alias": SERVER_ALIAS, "url": SERVER_URL, "token_env": "BAD NAME"},
            "invalid_mcp_token_env",
            id="invalid-token-env",
        ),
        pytest.param(
            {"alias": SERVER_ALIAS, "url": SERVER_URL, "request_timeout_s": 0},
            "invalid_mcp_timeout",
            id="non-positive-timeout",
        ),
    ],
)
def test_add_rejects_invalid_input_without_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    params: dict[str, object],
    reason: str,
) -> None:
    """Invalid configuration is refused with a stable reason and nothing is written."""
    instance_path = _configure(tmp_path, monkeypatch)
    client = _mount_rpc(instance_path)

    assert _rpc_call(client, "mcp_servers.add", params)["error"]["data"]["reason"] == reason
    assert read_mcp_servers(instance_path).servers == []


def test_add_rejects_an_alias_already_claimed_by_a_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cross-source alias collision is refused."""
    instance_path = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.mcp_servers._space_aliases",
        lambda path: {SERVER_ALIAS},
    )
    client = _mount_rpc(instance_path)

    error = _rpc_call(client, "mcp_servers.add", {"alias": SERVER_ALIAS, "url": SERVER_URL})["error"]
    assert error["data"]["reason"] == "mcp_server_alias_conflict"
    assert read_mcp_servers(instance_path).servers == []


def test_remove_rejects_a_server_that_is_not_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing an unknown alias reports it."""
    instance_path = _configure(tmp_path, monkeypatch)
    client = _mount_rpc(instance_path)

    error = _rpc_call(client, "mcp_servers.remove", {"alias": "ghost"})["error"]
    assert error["data"]["reason"] == "mcp_server_not_configured"


def test_locked_mode_exposes_the_inventory_but_rejects_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A locked profile may read the server list but may not change it."""
    instance_path = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr("reachy_mini_conversation_app.mcp_server_routes.LOCKED_PROFILE", "kiosk")
    client = _mount_rpc(instance_path)

    assert _rpc_call(client, "mcp_servers.list")["result"]["editable"] is False
    for method, params in (
        ("mcp_servers.add", {"alias": SERVER_ALIAS, "url": SERVER_URL}),
        ("mcp_servers.remove", {"alias": SERVER_ALIAS}),
        ("mcp_servers.save_token", {"alias": SERVER_ALIAS, "token": "x"}),
    ):
        assert _rpc_call(client, method, params)["error"]["data"]["reason"] == "profile_locked"
