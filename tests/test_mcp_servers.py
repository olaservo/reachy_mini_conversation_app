from __future__ import annotations
import sys
import json
import importlib
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

import reachy_mini_conversation_app.config as config_mod
import reachy_mini_conversation_app.mcp_servers as mcp_servers_mod
from reachy_mini_conversation_app.main import main
from reachy_mini_conversation_app.mcp_client import RemoteToolSpec
from reachy_mini_conversation_app.mcp_servers import (
    McpServerAuth,
    ResolvedMcpServer,
    InstalledMcpServer,
    InstalledToolSpaceTool,
    InstalledMcpServersManifest,
    read_mcp_servers,
    write_mcp_servers,
    find_server_token_env,
    list_token_requirements,
)
from reachy_mini_conversation_app.tool_spaces import (
    InstalledToolSpace,
    InstalledToolSpacesManifest,
    write_installed_tool_spaces,
)


HA_URL = "http://10.0.0.136:8123/api/mcp"
HA_LOCAL_URL = "http://homeassistant.local:8123/api/mcp"
TOKEN_ENV = "HA_ACCESS_TOKEN"
TOKEN_VALUE = "super-secret-long-lived-token"
HASS_TURN_ON = "hass__HassTurnOn"


async def _mock_list_tool_specs(self: object) -> list[RemoteToolSpec]:
    """Return fake HA tools namespaced to whatever alias the client was configured with."""
    alias = self.server.alias  # type: ignore[attr-defined]
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": []}
    return [
        RemoteToolSpec(
            server_alias=alias,
            remote_name="HassTurnOn",
            namespaced_name=f"{alias}__HassTurnOn",
            description="Turn a device on",
            parameters_schema=schema,
        ),
        RemoteToolSpec(
            server_alias=alias,
            remote_name="HassTurnOff",
            namespaced_name=f"{alias}__HassTurnOff",
            description="Turn a device off",
            parameters_schema=schema,
        ),
    ]


def _patch_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "reachy_mini_conversation_app.mcp_client.RemoteMcpToolClient.list_tool_specs",
        _mock_list_tool_specs,
    )


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        main()
    return int(exc.value.code)


def _setup_profile(tmp_path: Path, profile: str) -> Path:
    profile_dir = tmp_path / profile
    profile_dir.mkdir(parents=True)
    tools_txt = profile_dir / "tools.txt"
    tools_txt.write_text("", encoding="utf-8")
    return tools_txt


def test_mcp_servers_add_list_remove_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI should configure, list, and remove a generic MCP server cleanly."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, TOKEN_VALUE)
    _patch_discovery(monkeypatch)

    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", "hass", HA_URL, "--token-env", TOKEN_ENV, "--install-only"],
        )
        == 0
    )

    manifest_path = tmp_path / "external_content" / "mcp_servers.json"
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "servers": [
            {
                "alias": "hass",
                "auth": {"token_env": TOKEN_ENV, "type": "bearer"},
                "request_timeout_s": 10.0,
                "tool_timeout_s": 30.0,
                "url": HA_URL,
            }
        ],
    }

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "list"]) == 0

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "remove", "hass"]) == 0
    assert read_mcp_servers(None).servers == []


def test_mcp_servers_add_never_persists_token_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The secret token value must never be written to the manifest."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, TOKEN_VALUE)
    _patch_discovery(monkeypatch)

    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", "hass", HA_URL, "--token-env", TOKEN_ENV, "--install-only"],
        )
        == 0
    )

    manifest_text = (tmp_path / "external_content" / "mcp_servers.json").read_text(encoding="utf-8")
    assert TOKEN_VALUE not in manifest_text
    assert TOKEN_ENV in manifest_text


def test_mcp_servers_add_fails_when_token_env_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding a server whose token env var is unset should fail before persisting."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _patch_discovery(monkeypatch)

    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", "hass", HA_URL, "--token-env", TOKEN_ENV, "--install-only"],
        )
        == 1
    )
    assert not (tmp_path / "external_content" / "mcp_servers.json").exists()


def test_mcp_servers_add_rejects_public_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A public plain-HTTP URL must be rejected before persisting."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, TOKEN_VALUE)
    _patch_discovery(monkeypatch)

    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", "hass", "http://example.com/api/mcp", "--token-env", TOKEN_ENV],
        )
        == 1
    )
    assert not (tmp_path / "external_content" / "mcp_servers.json").exists()


@pytest.mark.parametrize("url", [HA_URL, HA_LOCAL_URL])
def test_mcp_servers_add_accepts_local_network_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """Private LAN IPs and *.local mDNS names over plain HTTP are accepted."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, TOKEN_VALUE)
    _patch_discovery(monkeypatch)

    assert (
        _run_cli(monkeypatch, ["app", "mcp-servers", "add", "hass", url, "--token-env", TOKEN_ENV, "--install-only"])
        == 0
    )
    assert read_mcp_servers(None).servers[0].url == url


def test_mcp_servers_add_enables_in_named_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--profile should enable the discovered tools in the named profile's tools.txt."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, TOKEN_VALUE)
    _patch_discovery(monkeypatch)
    tools_txt = _setup_profile(tmp_path, "smart_home")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)

    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", "hass", HA_URL, "--token-env", TOKEN_ENV, "--profile", "smart_home"],
        )
        == 0
    )

    enabled = tools_txt.read_text(encoding="utf-8")
    assert HASS_TURN_ON in enabled
    assert "hass__HassTurnOff" in enabled


def test_mcp_servers_add_rejects_alias_collision_with_tool_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An alias already used by an installed tool space must be rejected."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, TOKEN_VALUE)
    _patch_discovery(monkeypatch)

    # Pre-seed a tool space; its alias is derived from the slug as 'owner_hass'.
    write_installed_tool_spaces(
        None,
        InstalledToolSpacesManifest(spaces=[InstalledToolSpace(slug="owner/hass", alias="owner_hass")]),
    )

    # Configuring an MCP server with the same alias must be rejected.
    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", "owner_hass", HA_URL, "--token-env", TOKEN_ENV, "--install-only"],
        )
        == 1
    )
    assert read_mcp_servers(None).servers == []


def test_read_mcp_servers_rejects_duplicate_alias(tmp_path: Path) -> None:
    """A manifest with two servers sharing an alias must be rejected on read."""
    payload = {
        "version": 1,
        "servers": [
            {"alias": "hass", "url": HA_URL},
            {"alias": "hass", "url": HA_LOCAL_URL},
        ],
    }
    (tmp_path / "mcp_servers.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Duplicate MCP server alias"):
        read_mcp_servers(tmp_path)


def test_write_mcp_servers_uses_instance_path_when_provided(tmp_path: Path) -> None:
    """Managed instance paths store the manifest beside other instance-local state."""
    manifest = InstalledMcpServersManifest(servers=[InstalledMcpServer(alias="hass", url=HA_URL)])
    path = write_mcp_servers(tmp_path, manifest)
    assert path == tmp_path / "mcp_servers.json"
    assert path.is_file()
    assert not (tmp_path / "external_content" / "mcp_servers.json").exists()


def test_list_token_requirements_reflects_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Token requirements should list bearer-auth servers and whether their env var is set."""
    write_mcp_servers(
        tmp_path,
        InstalledMcpServersManifest(
            servers=[
                InstalledMcpServer(alias="hass", url=HA_URL, auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV)),
                InstalledMcpServer(alias="noauth", url=HA_LOCAL_URL),  # no auth -> excluded
            ]
        ),
    )

    monkeypatch.delenv(TOKEN_ENV, raising=False)
    reqs = list_token_requirements(tmp_path)
    assert [(r.alias, r.token_env, r.token_set) for r in reqs] == [("hass", TOKEN_ENV, False)]

    monkeypatch.setenv(TOKEN_ENV, TOKEN_VALUE)
    reqs = list_token_requirements(tmp_path)
    assert reqs[0].token_set is True


def test_find_server_token_env(tmp_path: Path) -> None:
    """find_server_token_env resolves a known alias and returns None otherwise."""
    write_mcp_servers(
        tmp_path,
        InstalledMcpServersManifest(
            servers=[InstalledMcpServer(alias="hass", url=HA_URL, auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV))]
        ),
    )
    assert find_server_token_env(tmp_path, "hass") == TOKEN_ENV
    assert find_server_token_env(tmp_path, "nope") is None


def _reload_core_tools() -> ModuleType:
    """Reload core_tools after config/mcp_servers have been patched."""
    for module_name in list(sys.modules):
        if module_name.startswith("reachy_mini_conversation_app.tools."):
            sys.modules.pop(module_name, None)
    sys.modules.pop("reachy_mini_conversation_app.tools.core_tools", None)
    core_tools_mod = importlib.import_module("reachy_mini_conversation_app.tools.core_tools")
    core_tools_mod.initialize_tools(force=True)
    return core_tools_mod


def test_generic_mcp_tool_registers_in_active_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hass__* tool listed in tools.txt should be registered from the configured MCP server."""
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "smart_home"
    profile_dir.mkdir(parents=True)
    (profile_dir / "instructions.txt").write_text("hi\n", encoding="utf-8")
    (profile_dir / "tools.txt").write_text(f"{HASS_TURN_ON}\n", encoding="utf-8")

    server = InstalledMcpServer(alias="hass", url=HA_URL)
    monkeypatch.setattr(
        mcp_servers_mod,
        "read_mcp_servers",
        lambda instance_path: InstalledMcpServersManifest(servers=[server]),
    )
    monkeypatch.setattr(
        mcp_servers_mod,
        "resolve_mcp_server_sync",
        lambda srv: ResolvedMcpServer(
            alias=srv.alias,
            url=srv.url,
            tools=[
                InstalledToolSpaceTool(
                    local_name=HASS_TURN_ON,
                    client_tool_name=HASS_TURN_ON,
                    remote_name="HassTurnOn",
                    description="Turn a device on",
                    parameters_schema={"type": "object", "properties": {}, "required": []},
                )
            ],
            client=SimpleNamespace(),
        ),
    )

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "smart_home")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    core_tools_mod = _reload_core_tools()

    assert HASS_TURN_ON in core_tools_mod.ALL_TOOLS
    assert HASS_TURN_ON in {spec["name"] for spec in core_tools_mod.get_tool_specs()}
