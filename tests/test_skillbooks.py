"""Tests for the SEP-2640 skillbooks host (registry scan, install, security, storage, tool)."""

from __future__ import annotations
import io
import json
import types
import hashlib
import tarfile
from pathlib import Path

import pytest

from reachy_mini_conversation_app import skillbooks as sb


def _sha256(data: bytes) -> str:
    """Return the ``sha256:<hex>`` digest of ``data`` (the SEP digest format)."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


class _Caps:
    """Minimal ServerCapabilities stand-in exposing ``model_extra`` like the pydantic model."""

    def __init__(self, *, skills: bool = True, directory_read: bool = False) -> None:
        """Build capabilities optionally advertising the skills extension / directoryRead."""
        self.model_extra: dict = {}
        if skills:
            self.model_extra = {"extensions": {sb.SKILLS_EXTENSION: ({"directoryRead": True} if directory_read else {})}}


class FakeClient:
    """In-memory RemoteMcpToolClient stand-in: serves a fixed map of uri -> bytes/text."""

    def __init__(self, resources: dict[str, bytes | str], *, caps: _Caps | None = None) -> None:
        """Store the resource map and the capabilities to report."""
        self._resources = resources
        self._caps = caps if caps is not None else _Caps()
        self.server = types.SimpleNamespace(alias="src", url="http://127.0.0.1:9/mcp")

    async def get_server_capabilities(self):  # noqa: ANN201
        """Return the configured capabilities."""
        return self._caps

    async def read_resource(self, uri: str):  # noqa: ANN201
        """Return a ReadResourceResult for ``uri`` (text or base64 blob)."""
        import base64

        from pydantic import AnyUrl
        from mcp.types import ReadResourceResult, BlobResourceContents, TextResourceContents

        if uri not in self._resources:
            raise ValueError(f"resource not found: {uri}")
        payload = self._resources[uri]
        if isinstance(payload, str):
            content = TextResourceContents(uri=AnyUrl(uri), text=payload, mimeType="text/markdown")
        else:
            content = BlobResourceContents(uri=AnyUrl(uri), blob=base64.b64encode(payload).decode(), mimeType="application/gzip")
        return ReadResourceResult(contents=[content])

    async def read_directory(self, uri: str, cursor: str | None = None):  # noqa: ANN201
        """Return an empty directory listing (supporting-file walk not exercised here)."""
        from mcp.types import ListResourcesResult

        return ListResourcesResult(resources=[])


def test_parse_manifest_valid() -> None:
    """A well-formed SKILL.md parses into name/description/body."""
    m, err = sb.parse_manifest_text("---\nname: fallout-gm\ndescription: Run a game.\n---\nBody.")
    assert err is None and m is not None
    assert m.name == "fallout-gm" and m.description == "Run a game." and m.body == "Body."


@pytest.mark.parametrize(
    "text",
    [
        "no frontmatter",
        "---\nname: Bad Name\ndescription: x\n---\n",  # uppercase/space name
        "---\nname: ok\n---\n",  # missing description
        "---\nname: ok\ndescription: x\n",  # unclosed frontmatter
    ],
)
def test_parse_manifest_invalid(text: str) -> None:
    """Malformed SKILL.md frontmatter is rejected with an error string."""
    m, err = sb.parse_manifest_text(text)
    assert m is None and err


def test_capability_detection() -> None:
    """The skills extension + directoryRead flag are detected from capabilities."""
    assert sb.server_supports_mcp_skills(_Caps()) is True
    assert sb.server_supports_mcp_skills(_Caps(skills=False)) is False
    assert sb.server_supports_mcp_skills(None) is False
    assert sb.server_supports_directory_read(_Caps(directory_read=True)) is True
    assert sb.server_supports_directory_read(_Caps(directory_read=False)) is False


def test_digest_verify() -> None:
    """SHA-256 verification accepts a match and rejects mismatch / malformed digests."""
    sb._verify_artifact_digest(b"hello", _sha256(b"hello"))
    with pytest.raises(ValueError):
        sb._verify_artifact_digest(b"hello", _sha256(b"other"))
    with pytest.raises(ValueError):
        sb._verify_artifact_digest(b"hello", "not-a-digest")


@pytest.mark.parametrize("name", ["../escape", "/abs", "a/../b", "C:\\x", "a\\b"])
def test_validate_archive_name_rejects_unsafe(name: str) -> None:
    """Path-traversal / absolute / drive-letter / backslash archive members are rejected."""
    with pytest.raises(ValueError):
        sb._validate_archive_name(name)


def test_extract_tar_rejects_traversal(tmp_path: Path) -> None:
    """A tar containing a traversal member is rejected before extraction."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("../evil")
        data = b"x"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError):
        sb._extract_tar_safely(buf.getvalue(), tmp_path)


def test_index_parse_prefers_archive_and_rejects_bad() -> None:
    """Index parsing prefers archives, and drops file:// urls and frontmatter-less entries."""
    skill_md = b"---\nname: a\ndescription: d\n---\n"
    index = {
        "skills": [
            {"frontmatter": {"name": "a", "description": "d"}, "url": "skill://a/SKILL.md", "digest": _sha256(skill_md),
             "archives": [{"url": "skill://a.tar.gz", "mimeType": "application/gzip", "digest": _sha256(b"arc")}]},
            {"frontmatter": {"name": "b", "description": "d"}, "url": "file:///etc/passwd", "digest": _sha256(b"x")},
            {"url": "skill://c/SKILL.md", "digest": _sha256(b"x")},
        ]
    }
    from pydantic import AnyUrl
    from mcp.types import ReadResourceResult, TextResourceContents

    result = ReadResourceResult(
        contents=[TextResourceContents(uri=AnyUrl(sb.INDEX_URI), text=json.dumps(index), mimeType="application/json")]
    )
    entries = sb._parse_index(result, "src")
    built = [s for e in entries if (s := sb._build_registry_skill(e, source="src")) is not None]
    assert [s.name for s in built] == ["a"]
    assert built[0].artifact_type == "archive" and built[0].source_url == "skill://a.tar.gz"


@pytest.mark.asyncio
async def test_scan_and_install_skill_md(tmp_path: Path) -> None:
    """Scan an index then install a SKILL.md skill: content + provenance sidecar written."""
    skill_md = "---\nname: fallout-gm\ndescription: Be the GM.\n---\nGM body.\n"
    skill_bytes = skill_md.encode()
    index = {"skills": [{"frontmatter": {"name": "fallout-gm", "description": "Be the GM."},
                          "url": "skill://fallout-gm/SKILL.md", "digest": _sha256(skill_bytes)}]}
    client = FakeClient({sb.INDEX_URI: json.dumps(index), "skill://fallout-gm/SKILL.md": skill_md})

    registry = await sb.scan_skill_registry(client)
    assert registry is not None and [s.name for s in registry] == ["fallout-gm"]

    install_dir = await sb.install_skill(client, registry[0], destination_root=tmp_path)
    assert (install_dir / "SKILL.md").read_text() == skill_md
    sidecar = json.loads((install_dir / ".skill-source.json").read_text())
    assert sidecar["installed_via"] == "mcp" and sidecar["artifact_digest"] == _sha256(skill_bytes)
    assert sidecar["content_fingerprint"].startswith("sha256:")


@pytest.mark.asyncio
async def test_install_digest_mismatch_rolls_back(tmp_path: Path) -> None:
    """A digest mismatch raises and leaves no partial install dir behind."""
    skill_md = "---\nname: x\ndescription: d\n---\nbody"
    client = FakeClient({"skill://x/SKILL.md": skill_md})
    skill = sb.RegistrySkill(name="x", description="d", source_url="skill://x/SKILL.md", digest=_sha256(b"WRONG"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        await sb.install_skill(client, skill, destination_root=tmp_path)
    assert not (tmp_path / "x").exists()


@pytest.mark.asyncio
async def test_scan_returns_none_without_extension() -> None:
    """A server that doesn't advertise the skills extension yields ``None``."""
    client = FakeClient({}, caps=_Caps(skills=False))
    assert await sb.scan_skill_registry(client) is None


def test_installed_skillbooks_roundtrip(tmp_path: Path) -> None:
    """The installed-skillbooks manifest round-trips through write/read."""
    manifest = sb.InstalledSkillbooksManifest(
        skillbooks=[sb.InstalledSkillbook(name="fallout-helper", source_url="http://x/mcp", server_alias="src", skills=["fallout-gm"])]
    )
    sb.write_installed_skillbooks(tmp_path, manifest)
    loaded = sb.read_installed_skillbooks(tmp_path)
    assert loaded.skillbooks[0].name == "fallout-helper" and loaded.skillbooks[0].skills == ["fallout-gm"]


def test_load_manifests_and_format_block(tmp_path: Path) -> None:
    """Installed skill manifests load from disk and render into the prompt block."""
    skill_dir = tmp_path / "skillbooks" / "fallout-helper" / "fallout-gm"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: fallout-gm\ndescription: Be the GM.\n---\nbody")
    manifests = sb.load_installed_skill_manifests(tmp_path)
    assert [m.name for m in manifests] == ["fallout-gm"]
    block = sb.format_skillbooks_block(manifests)
    assert "fallout-gm" in block and "read_skill" in block
    assert sb.format_skillbooks_block([]) == ""


@pytest.mark.asyncio
async def test_read_skill_tool(tmp_path: Path) -> None:
    """read_skill returns SKILL.md + supporting files and blocks traversal / unknown names."""
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.tools.read_skill import ReadSkill

    skill_dir = tmp_path / "skillbooks" / "book" / "fallout-gm"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: fallout-gm\ndescription: d\n---\nGM body")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "GUIDE.md").write_text("guide text")

    deps = ToolDependencies(reachy_mini=None, movement_manager=None, instance_path=str(tmp_path))
    tool = ReadSkill()

    out = await tool(deps, name="fallout-gm")
    assert "GM body" in out["content"] and out["supporting_files"] == ["references/GUIDE.md"]
    sub = await tool(deps, name="fallout-gm", path="references/GUIDE.md")
    assert sub["content"] == "guide text"
    assert "escapes" in (await tool(deps, name="fallout-gm", path="../../x")).get("error", "")
    assert "no installed skill" in (await tool(deps, name="missing")).get("error", "")
