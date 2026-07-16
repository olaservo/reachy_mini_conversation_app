from __future__ import annotations
import sys
import json
from types import SimpleNamespace
from pathlib import Path
from argparse import Namespace

import pytest

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app.main import main
from reachy_mini_conversation_app.mcp_client import RemoteToolSpec
from reachy_mini_conversation_app.mcp_servers import (
    McpServerAuth,
    InstalledMcpServer,
    InstalledMcpServersManifest,
    read_mcp_servers,
    write_mcp_servers,
    _resolve_auth_headers,
    find_server_token_env,
    list_token_requirements,
    handle_mcp_servers_command,
)
from reachy_mini_conversation_app.tool_spaces import (
    InstalledToolSpace,
    InstalledToolSpaceTool,
    InstalledToolSpacesManifest,
    write_installed_tool_spaces,
)


SERVER_ALIAS = "example"
SERVER_URL = "http://192.168.1.50:8000/mcp"
OTHER_SERVER_URL = "http://192.168.1.51:8000/mcp"
TOOL_ID = f"{SERVER_ALIAS}__do_thing"
TOKEN_ENV = "MCP_SERVER_TOKEN_EXAMPLE"

# Alias derived from this slug collides with the MCP server alias used in cross-source tests.
SPACE_SLUG = "example/search-tool"
SPACE_ALIAS = "example_search_tool"


def _remote_spec(alias: str, remote_name: str = "do_thing") -> RemoteToolSpec:
    return RemoteToolSpec(
        server_alias=alias,
        remote_name=remote_name,
        namespaced_name=f"{alias}__{remote_name}",
        description=f"Remote tool {remote_name}",
        parameters_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    )


def _mock_discovery(monkeypatch: pytest.MonkeyPatch, remote_names: list[str] | None = None) -> None:
    names = remote_names or ["do_thing"]

    async def _mock_list_tool_specs(self: object) -> list[RemoteToolSpec]:
        alias = self.server.alias  # type: ignore[attr-defined]
        return [_remote_spec(alias, remote_name) for remote_name in names]

    monkeypatch.setattr(
        "reachy_mini_conversation_app.mcp_client.RemoteMcpToolClient.list_tool_specs",
        _mock_list_tool_specs,
    )


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        main()
    return int(exc.value.code)


def test_mcp_servers_add_list_remove_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI should configure, list, and remove a generic MCP server cleanly."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)

    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--install-only"],
        )
        == 0
    )

    manifest_path = tmp_path / "external_content" / "mcp_servers.json"
    assert manifest_path.is_file()
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["version"] == 1
    assert written["servers"] == [
        {
            "alias": SERVER_ALIAS,
            "url": SERVER_URL,
            "request_timeout_s": 10.0,
            "tool_timeout_s": 30.0,
            "tools": [
                {
                    "local_name": TOOL_ID,
                    "client_tool_name": TOOL_ID,
                    "remote_name": "do_thing",
                    "description": "Remote tool do_thing",
                    "parameters_schema": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                }
            ],
        }
    ]

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "list"]) == 0

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "remove", SERVER_ALIAS]) == 0
    assert read_mcp_servers(None).servers == []


def test_mcp_servers_list_reads_from_cache_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Listing must not reconnect to the configured servers."""
    monkeypatch.chdir(tmp_path)

    async def _fail_discovery(self: object) -> list[RemoteToolSpec]:
        raise AssertionError("list must not trigger network discovery")

    write_mcp_servers(
        None,
        InstalledMcpServersManifest(
            servers=[
                InstalledMcpServer(
                    alias=SERVER_ALIAS,
                    url=SERVER_URL,
                    tools=[
                        InstalledToolSpaceTool(
                            local_name=TOOL_ID,
                            client_tool_name=TOOL_ID,
                            remote_name="do_thing",
                            description="Remote tool do_thing",
                            parameters_schema={},
                        )
                    ],
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.mcp_client.RemoteMcpToolClient.list_tool_specs",
        _fail_discovery,
    )

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "list"]) == 0


def test_mcp_servers_add_never_persists_token_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the env-var name may reach the manifest, never the secret."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, "super-secret-value")
    _mock_discovery(monkeypatch)

    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--token-env", TOKEN_ENV, "--install-only"],
        )
        == 0
    )

    manifest_text = (tmp_path / "external_content" / "mcp_servers.json").read_text(encoding="utf-8")
    assert "super-secret-value" not in manifest_text
    entry = json.loads(manifest_text)["servers"][0]
    assert entry["auth"] == {"type": "bearer", "token_env": TOKEN_ENV}


def test_mcp_servers_add_fails_fast_when_token_env_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing token must fail the add before anything is persisted."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _mock_discovery(monkeypatch)

    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--token-env", TOKEN_ENV, "--install-only"],
        )
        == 1
    )
    assert not (tmp_path / "external_content" / "mcp_servers.json").exists()


def test_mcp_servers_add_rejects_public_plain_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain HTTP is only allowed for local-network hosts."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)

    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", SERVER_ALIAS, "http://example.com/mcp", "--install-only"],
        )
        == 1
    )
    assert not (tmp_path / "external_content" / "mcp_servers.json").exists()


def test_mcp_servers_add_rejects_invalid_token_env_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An env-var name that can't round-trip through .env must be rejected."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)

    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--token-env", "BAD NAME", "--install-only"],
        )
        == 1
    )
    assert not (tmp_path / "external_content" / "mcp_servers.json").exists()


def test_mcp_servers_add_refreshes_cached_tools_for_same_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running add for the same alias and URL re-discovers and rewrites the cache."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch, ["do_thing"])
    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--install-only"]) == 0
    assert [tool.local_name for tool in read_mcp_servers(None).servers[0].tools] == [TOOL_ID]

    _mock_discovery(monkeypatch, ["do_thing", "do_other_thing"])
    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--install-only"]) == 0

    manifest = read_mcp_servers(None)
    assert len(manifest.servers) == 1
    assert [tool.local_name for tool in manifest.servers[0].tools] == [
        TOOL_ID,
        f"{SERVER_ALIAS}__do_other_thing",
    ]


def test_mcp_servers_add_rejects_same_alias_different_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alias can't silently switch to another endpoint."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)
    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--install-only"]) == 0

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SERVER_ALIAS, OTHER_SERVER_URL, "--install-only"]) == 1
    assert read_mcp_servers(None).servers[0].url == SERVER_URL


def test_mcp_servers_add_rejects_installed_space_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server alias that collides with an installed Space alias is rejected."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)
    write_installed_tool_spaces(
        None,
        InstalledToolSpacesManifest(
            spaces=[
                InstalledToolSpace(
                    slug=SPACE_SLUG,
                    alias=SPACE_ALIAS,
                    mcp_url="https://example-search-tool.hf.space/gradio_api/mcp/",
                    private=False,
                )
            ]
        ),
    )

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SPACE_ALIAS, SERVER_URL, "--install-only"]) == 1
    assert not (tmp_path / "external_content" / "mcp_servers.json").exists()


def test_tool_spaces_add_rejects_configured_server_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installing a Space whose alias collides with a configured MCP server is rejected."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)
    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SPACE_ALIAS, SERVER_URL, "--install-only"]) == 0

    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.HfApi.space_info",
        lambda self, slug, **kwargs: SimpleNamespace(
            id=slug,
            private=False,
            disabled=False,
            sdk="gradio",
            host=None,
            subdomain=slug.replace("/", "-"),
            tags=[],
        ),
    )

    assert _run_cli(monkeypatch, ["app", "tool-spaces", "add", SPACE_SLUG, "--install-only"]) == 1


def test_read_mcp_servers_raises_on_duplicate_alias(tmp_path: Path) -> None:
    """A manifest with two servers sharing an alias must be rejected on read."""
    payload = {
        "version": 1,
        "servers": [
            {"alias": SERVER_ALIAS, "url": SERVER_URL, "request_timeout_s": 10.0, "tool_timeout_s": 30.0},
            {"alias": SERVER_ALIAS, "url": OTHER_SERVER_URL, "request_timeout_s": 10.0, "tool_timeout_s": 30.0},
        ],
    }
    (tmp_path / "mcp_servers.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Duplicate MCP server alias"):
        read_mcp_servers(tmp_path)


def test_mcp_servers_manifest_uses_instance_path_when_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed instance paths should store the manifest beside other instance-local state."""
    _mock_discovery(monkeypatch)

    args = Namespace(
        mcp_servers_command="add",
        alias=SERVER_ALIAS,
        url=SERVER_URL,
        token_env=None,
        request_timeout=10.0,
        tool_timeout=30.0,
        install_only=True,
        profile=None,
    )
    assert handle_mcp_servers_command(args, instance_path=tmp_path) == 0
    assert (tmp_path / "mcp_servers.json").is_file()
    assert not (tmp_path / "external_content" / "mcp_servers.json").exists()


def _setup_profile(tmp_path: Path, profile: str) -> Path:
    profile_dir = tmp_path / profile
    profile_dir.mkdir(parents=True)
    tools_txt = profile_dir / "tools.txt"
    tools_txt.write_text("", encoding="utf-8")
    return tools_txt


def test_mcp_servers_add_enables_in_active_profile_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Add without flags should enable the discovered tools in the active profile."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)
    tools_txt = _setup_profile(tmp_path, "default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", None)

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL]) == 0

    assert TOOL_ID in tools_txt.read_text(encoding="utf-8")


def test_mcp_servers_remove_disables_tools_in_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing a server strips its tool IDs from the profile they were enabled in."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)
    tools_txt = _setup_profile(tmp_path, "default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", None)

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL]) == 0
    assert TOOL_ID in tools_txt.read_text(encoding="utf-8")

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "remove", SERVER_ALIAS]) == 0
    assert TOOL_ID not in tools_txt.read_text(encoding="utf-8")
    assert read_mcp_servers(None).servers == []


def test_list_token_requirements_reflects_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """token_set must track whether the named env var currently holds a value."""
    write_mcp_servers(
        tmp_path,
        InstalledMcpServersManifest(
            servers=[
                InstalledMcpServer(
                    alias=SERVER_ALIAS,
                    url=SERVER_URL,
                    auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV),
                ),
                InstalledMcpServer(alias="no_auth", url=OTHER_SERVER_URL),
            ]
        ),
    )

    monkeypatch.delenv(TOKEN_ENV, raising=False)
    requirements = list_token_requirements(tmp_path)
    assert [(req.alias, req.token_env, req.token_set) for req in requirements] == [(SERVER_ALIAS, TOKEN_ENV, False)]

    monkeypatch.setenv(TOKEN_ENV, "some-token")
    assert list_token_requirements(tmp_path)[0].token_set is True

    assert find_server_token_env(tmp_path, SERVER_ALIAS) == TOKEN_ENV
    assert find_server_token_env(tmp_path, "no_auth") is None
    assert find_server_token_env(tmp_path, "missing") is None


def test_resolve_auth_headers_warns_on_plain_http_bearer(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sending a bearer token over plain HTTP is allowed locally but warned about."""
    monkeypatch.setenv(TOKEN_ENV, "some-token")
    server = InstalledMcpServer(
        alias=SERVER_ALIAS,
        url=SERVER_URL,
        auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV),
    )

    with caplog.at_level("WARNING"):
        headers = _resolve_auth_headers(server)

    assert headers == {"Authorization": "Bearer some-token"}
    assert any("plain HTTP" in record.message for record in caplog.records)
