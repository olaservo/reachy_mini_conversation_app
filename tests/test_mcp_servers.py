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
    InstalledToolSpacesManifest,
    write_installed_tool_spaces,
)
from reachy_mini_conversation_app.profile_store import write_profile
from reachy_mini_conversation_app.profile_toolsets import read_profile_tool_names
from reachy_mini_conversation_app.remote_tool_sources import CachedRemoteTool


SERVER_ALIAS = "example"
SERVER_URL = "http://192.168.1.50:8000/mcp"
OTHER_SERVER_URL = "http://192.168.1.51:8000/mcp"
LOOPBACK_SERVER_URL = "http://127.0.0.1:8000/mcp"
TOOL_ID = f"{SERVER_ALIAS}__do_thing"
TOKEN_ENV = "MCP_SERVER_TOKEN_EXAMPLE"

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
    """The CLI configures, lists and removes a server."""
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
    """Listing does not reconnect to the servers."""
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
                        CachedRemoteTool(
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
    """Only the variable name reaches the manifest."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, "super-secret-value")
    _mock_discovery(monkeypatch)

    assert (
        _run_cli(
            monkeypatch,
            [
                "app",
                "mcp-servers",
                "add",
                SERVER_ALIAS,
                LOOPBACK_SERVER_URL,
                "--token-env",
                TOKEN_ENV,
                "--install-only",
            ],
        )
        == 0
    )

    manifest_text = (tmp_path / "external_content" / "mcp_servers.json").read_text(encoding="utf-8")
    assert "super-secret-value" not in manifest_text
    entry = json.loads(manifest_text)["servers"][0]
    assert entry["auth"] == {"type": "bearer", "token_env": TOKEN_ENV}


@pytest.mark.parametrize(
    "add_args",
    [
        pytest.param([SERVER_ALIAS, SERVER_URL, "--token-env", TOKEN_ENV], id="token-env-unset"),
        pytest.param([SERVER_ALIAS, "http://example.com/mcp"], id="public-plain-http"),
        pytest.param([SERVER_ALIAS, SERVER_URL, "--token-env", "BAD NAME"], id="invalid-token-env-name"),
    ],
)
def test_mcp_servers_add_rejects_invalid_config_before_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    add_args: list[str],
) -> None:
    """A bad add fails before anything is written."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    _mock_discovery(monkeypatch)

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", *add_args, "--install-only"]) == 1
    assert not (tmp_path / "external_content" / "mcp_servers.json").exists()


def test_mcp_servers_add_refresh_keeps_stored_auth_and_timeouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running add without flags refreshes the cache and keeps auth, timeouts and the opt-in."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, "secret")
    _mock_discovery(monkeypatch, ["do_thing"])
    assert (
        _run_cli(
            monkeypatch,
            [
                "app",
                "mcp-servers",
                "add",
                SERVER_ALIAS,
                SERVER_URL,
                "--token-env",
                TOKEN_ENV,
                "--allow-insecure-http",
                "--request-timeout",
                "5",
                "--tool-timeout",
                "60",
                "--install-only",
            ],
        )
        == 0
    )

    _mock_discovery(monkeypatch, ["do_thing", "do_other_thing"])
    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--install-only"]) == 0

    server = read_mcp_servers(None).servers[0]
    assert server.auth is not None
    assert server.auth.token_env == TOKEN_ENV
    assert server.auth.allow_insecure_http is True
    assert server.request_timeout_s == 5.0
    assert server.tool_timeout_s == 60.0
    assert [tool.local_name for tool in server.tools] == [TOOL_ID, f"{SERVER_ALIAS}__do_other_thing"]


def test_mcp_servers_add_token_rotation_keeps_insecure_http_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-adding with a new --token-env keeps the stored opt-in."""
    other_token_env = f"{TOKEN_ENV}_V2"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, "secret")
    monkeypatch.setenv(other_token_env, "secret-v2")
    _mock_discovery(monkeypatch)
    assert (
        _run_cli(
            monkeypatch,
            [
                "app",
                "mcp-servers",
                "add",
                SERVER_ALIAS,
                SERVER_URL,
                "--token-env",
                TOKEN_ENV,
                "--allow-insecure-http",
                "--install-only",
            ],
        )
        == 0
    )

    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--token-env", other_token_env, "--install-only"],
        )
        == 0
    )
    server = read_mcp_servers(None).servers[0]
    assert server.auth is not None
    assert server.auth.token_env == other_token_env
    assert server.auth.allow_insecure_http is True


def test_mcp_servers_add_fails_closed_when_spaces_manifest_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable tool-spaces manifest blocks the add, since the alias cannot be checked."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)
    (tmp_path / "external_content").mkdir()
    (tmp_path / "external_content" / "installed_tool_spaces.json").write_text("{not json", encoding="utf-8")

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--install-only"]) == 1
    assert read_mcp_servers(None).servers == []


def test_mcp_servers_add_rejects_same_alias_different_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alias cannot be re-pointed at another URL."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)
    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--install-only"]) == 0

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SERVER_ALIAS, OTHER_SERVER_URL, "--install-only"]) == 1
    assert read_mcp_servers(None).servers[0].url == SERVER_URL


def test_mcp_servers_add_rejects_installed_space_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alias already used by an installed Space is refused."""
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
    """Installing a Space whose alias is a configured server is refused."""
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


def test_read_mcp_servers_skips_duplicate_alias(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A duplicate alias keeps the first entry and skips the later one."""
    payload = {
        "version": 1,
        "servers": [
            {"alias": SERVER_ALIAS, "url": SERVER_URL, "request_timeout_s": 10.0, "tool_timeout_s": 30.0},
            {"alias": SERVER_ALIAS, "url": OTHER_SERVER_URL, "request_timeout_s": 10.0, "tool_timeout_s": 30.0},
        ],
    }
    (tmp_path / "mcp_servers.json").write_text(json.dumps(payload), encoding="utf-8")

    with caplog.at_level("WARNING"):
        manifest = read_mcp_servers(tmp_path)

    assert [server.url for server in manifest.servers] == [SERVER_URL]
    assert "Duplicate MCP server alias" in caplog.text


def test_mcp_servers_add_and_remove_wire_tools_into_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Add enables the tools in the active profile; remove strips them."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)
    write_profile("default", tmp_path / "default", "hello", [])
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", None)

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL]) == 0
    assert TOOL_ID in read_profile_tool_names("default", None)

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "remove", SERVER_ALIAS]) == 0
    assert TOOL_ID not in read_profile_tool_names("default", None)
    assert read_mcp_servers(None).servers == []


def test_list_token_requirements_reflects_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """token_set tracks whether the variable currently holds a value."""
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


def test_resolve_auth_headers_plain_http_bearer_policy(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bearer over plain HTTP: refused on the LAN, allowed on loopback, allowed with a warning when opted in."""
    monkeypatch.setenv(TOKEN_ENV, "some-token")

    def _server(url: str, allow_insecure_http: bool = False) -> InstalledMcpServer:
        return InstalledMcpServer(
            alias=SERVER_ALIAS,
            url=url,
            auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV, allow_insecure_http=allow_insecure_http),
        )

    with pytest.raises(RuntimeError, match="plain HTTP"):
        _resolve_auth_headers(_server(SERVER_URL))

    assert _resolve_auth_headers(_server(LOOPBACK_SERVER_URL)) == {"Authorization": "Bearer some-token"}

    with caplog.at_level("WARNING"):
        assert _resolve_auth_headers(_server(SERVER_URL, allow_insecure_http=True)) == {
            "Authorization": "Bearer some-token"
        }
    assert any("plain HTTP" in record.message for record in caplog.records)


def test_mcp_servers_add_allows_token_over_lan_plain_http_with_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--allow-insecure-http lets a LAN plain-HTTP server carry a token, and the opt-in is persisted."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, "some-token")
    _mock_discovery(monkeypatch)

    assert (
        _run_cli(
            monkeypatch,
            [
                "app",
                "mcp-servers",
                "add",
                SERVER_ALIAS,
                SERVER_URL,
                "--token-env",
                TOKEN_ENV,
                "--allow-insecure-http",
                "--install-only",
            ],
        )
        == 0
    )

    manifest_text = (tmp_path / "external_content" / "mcp_servers.json").read_text(encoding="utf-8")
    assert "some-token" not in manifest_text
    entry = json.loads(manifest_text)["servers"][0]
    assert entry["auth"] == {"type": "bearer", "token_env": TOKEN_ENV, "allow_insecure_http": True}
    assert read_mcp_servers(None).servers[0].auth == McpServerAuth(
        type="bearer", token_env=TOKEN_ENV, allow_insecure_http=True
    )


def test_read_mcp_servers_skips_corrupt_entries_and_keeps_the_rest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Invalid entries are skipped with a warning; valid ones still load."""
    corrupt_entries = [
        {"alias": "corrupt", "url": SERVER_URL, "request_timeout_s": None},
        "not-an-object",
        # A truthy string must not grant the opt-in.
        {
            "alias": "corrupt",
            "url": SERVER_URL,
            "auth": {"type": "bearer", "token_env": TOKEN_ENV, "allow_insecure_http": "false"},
        },
    ]
    for corrupt_entry in corrupt_entries:
        payload = {
            "version": 1,
            "servers": [
                corrupt_entry,
                {"alias": SERVER_ALIAS, "url": SERVER_URL, "request_timeout_s": 10.0, "tool_timeout_s": 30.0},
            ],
        }
        (tmp_path / "mcp_servers.json").write_text(json.dumps(payload), encoding="utf-8")

        caplog.clear()
        with caplog.at_level("WARNING"):
            manifest = read_mcp_servers(tmp_path)

        assert [server.alias for server in manifest.servers] == [SERVER_ALIAS]
        assert "Skipping invalid MCP server entry" in caplog.text


def test_mcp_servers_add_and_remove_refuse_rewrite_when_entries_were_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Add and remove refuse to rewrite a manifest that holds entries the reader skipped."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)
    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--install-only"]) == 0

    manifest_path = tmp_path / "external_content" / "mcp_servers.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["servers"].append(
        {"alias": "broken", "url": SERVER_URL, "auth": {"type": "bearer", "token_env": "BAD NAME"}}
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    before = manifest_path.read_text(encoding="utf-8")

    assert _run_cli(monkeypatch, ["app", "mcp-servers", "add", "other", OTHER_SERVER_URL, "--install-only"]) == 1
    assert _run_cli(monkeypatch, ["app", "mcp-servers", "remove", SERVER_ALIAS]) == 1
    assert manifest_path.read_text(encoding="utf-8") == before


def test_handle_mcp_servers_command_accepts_minimal_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The programmatic call works with only the required fields."""
    _mock_discovery(monkeypatch)
    profiles_dir = tmp_path / "profiles"
    write_profile("default", profiles_dir / "default", "hello", [])
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", profiles_dir)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", None)

    args = Namespace(mcp_servers_command="add", alias=SERVER_ALIAS, url=SERVER_URL)
    assert handle_mcp_servers_command(args, instance_path=tmp_path) == 0

    server = read_mcp_servers(tmp_path).servers[0]
    assert server.auth is None
    assert server.request_timeout_s == 10.0
    assert server.tool_timeout_s == 30.0
    assert TOOL_ID in read_profile_tool_names("default", tmp_path)


def test_cli_rejects_unrecognized_mcp_servers_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mistyped flag fails the command instead of being dropped by parse_known_args."""
    monkeypatch.chdir(tmp_path)
    _mock_discovery(monkeypatch)

    assert (
        _run_cli(
            monkeypatch,
            ["app", "mcp-servers", "add", SERVER_ALIAS, SERVER_URL, "--token_env", TOKEN_ENV, "--install-only"],
        )
        == 2
    )
    assert not (tmp_path / "external_content" / "mcp_servers.json").exists()


def test_read_mcp_servers_drops_tools_with_non_mapping_schema(tmp_path: Path) -> None:
    """A cached tool whose schema is not an object is dropped; the rest load."""
    payload = {
        "version": 1,
        "servers": [
            {
                "alias": SERVER_ALIAS,
                "url": SERVER_URL,
                "tools": [
                    {"local_name": "bad", "client_tool_name": "bad", "parameters_schema": ["oops"]},
                    {"local_name": "good", "client_tool_name": "good", "parameters_schema": {"type": "object"}},
                ],
            }
        ],
    }
    (tmp_path / "mcp_servers.json").write_text(json.dumps(payload), encoding="utf-8")

    manifest = read_mcp_servers(tmp_path)

    assert [server.alias for server in manifest.servers] == [SERVER_ALIAS]
    assert [tool.local_name for tool in manifest.servers[0].tools] == ["good"]


def test_discovery_failure_names_the_token_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed discovery on a tokened server names the variable to check."""
    import asyncio

    from reachy_mini_conversation_app.mcp_client import McpTransportError, RemoteMcpToolClient
    from reachy_mini_conversation_app.mcp_servers import McpServerAuth, InstalledMcpServer, resolve_mcp_server

    async def _fail(self: RemoteMcpToolClient) -> list[object]:
        raise McpTransportError("Server returned an error response")

    monkeypatch.setattr(RemoteMcpToolClient, "list_tool_specs", _fail)
    server = InstalledMcpServer(
        alias="hf",
        url="https://huggingface.co/mcp",
        auth=McpServerAuth(type="bearer", token_env="HF_TOKEN"),
    )
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")

    with pytest.raises(RuntimeError, match="check the token in HF_TOKEN"):
        asyncio.run(resolve_mcp_server(server))
