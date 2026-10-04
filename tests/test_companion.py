"""Synthetic-only contracts for the local companion workspace."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import zipfile
from xml.sax.saxutils import escape

import pytest

from meeting import companion as core


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Companion core must not use networking")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def pptx(path, texts, *, order=None, extra=None):
    order = list(range(1, len(texts) + 1)) if order is None else order
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ppt/presentation.xml", '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst>' +
                         ''.join(f'<p:sldId id="{255+i}" r:id="r{i}"/>' for i in order) + '</p:sldIdLst></p:presentation>')
        archive.writestr("ppt/_rels/presentation.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
                         ''.join(f'<Relationship Id="r{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>' for i in range(1, len(texts) + 1)) +
                         '<Relationship Id="web" Type="hyperlink" Target="https://never.invalid/" TargetMode="External"/></Relationships>')
        for i, text in enumerate(texts, 1):
            archive.writestr(f"ppt/slides/slide{i}.xml", '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:p><a:r><a:t>' + escape(text) + '</a:t></a:r></a:p></p:sld>')
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
    return path


def material(tmp_path, text="Chosen facts", name="facts.md"):
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return core.load_material(path)


def test_modes_are_independent_and_survive_atomic_reload(tmp_path):
    repository = core.CompanionRepository(tmp_path / "data/companion.json")
    state = repository.load()
    assert state == core.default_state()
    state["mode"] = "presentation"
    state["profiles"]["presentation"].update(topic="结果", answer="My exact answer\n第二行", notes="No decision yet.")
    state["profiles"]["presentation"]["materials"].append(material(tmp_path))
    state["profiles"]["interview"]["notes"] = "Private interview notes"
    assert state["profiles"]["meeting"]["materials"] == []
    repository.save(state)
    reloaded = core.CompanionRepository(repository.path).load()
    assert reloaded == state
    reloaded["profiles"]["presentation"]["materials"].clear()
    assert len(state["profiles"]["presentation"]["materials"]) == 1
    assert not list(repository.path.parent.glob(".companion-*.tmp"))


@pytest.mark.parametrize("content", [b"{broken", b'{"mode":"meeting","profiles":{}}', b"\xff\xfe"])
def test_corrupt_workspace_never_overwritten(tmp_path, content):
    path = tmp_path / "companion.json"
    path.write_bytes(content)
    repository = core.CompanionRepository(path)
    with pytest.raises(ValueError, match="not overwritten"):
        repository.load()
    with pytest.raises(ValueError, match="not overwritten"):
        repository.save(core.default_state())
    assert path.read_bytes() == content


def test_atomic_replace_failure_preserves_prior_state(tmp_path, monkeypatch):
    repository = core.CompanionRepository(tmp_path / "companion.json")
    repository.save(core.default_state())
    before = repository.path.read_bytes()
    changed = repository.load()
    changed["profiles"]["meeting"]["notes"] = "New local note"
    def fail(*args):
        raise OSError("simulated failed atomic replace")
    monkeypatch.setattr(core.os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        repository.save(changed)
    assert repository.path.read_bytes() == before
    assert not list(tmp_path.glob(".companion-*.tmp"))


@pytest.mark.parametrize("path", [r"\\server\share\private.txt", "//server/share/private.txt", r"\\?\C:\file.txt", r"\Device\Mup\file.txt", "https://never.invalid/file.txt"])
def test_network_names_rejected_before_filesystem_resolution(path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("resolve must not run for network paths")
    monkeypatch.setattr(Path, "resolve", forbidden)
    with pytest.raises(ValueError, match="local path"):
        core.load_material(path)


def test_text_sections_lossless_stable_and_explicit(tmp_path):
    text = "# Heading\n" + "中文 text\n" * 650
    imported = material(tmp_path, text)
    assert imported == core.load_material(imported["path"])
    assert "".join(section["text"] for section in imported["sections"]) == text
    assert all(len(section["text"]) <= 4000 for section in imported["sections"])
    assert all(section["id"].startswith(imported["sha256"] + ":") for section in imported["sections"])
    profile = core.default_state()["profiles"]["meeting"]
    profile["materials"] = [imported]
    assert core.review_sources(profile) == ()
    profile["selected_ids"] = [imported["sections"][-1]["id"]]
    selected = core.review_sources(profile)
    assert selected[0].id == "F1" and selected[0].kind == "file"
    assert selected[0].text == imported["sections"][-1]["text"]


@pytest.mark.parametrize("name,content,error", [("large.txt", b"x" * (128*1024+1), "128 KiB"), ("text.md", b"x"*60001, "60000"),
                                              ("binary.txt", b"abc\x00def", "NUL"), ("bad.json", b"{oops", "Expecting"),
                                              ("scan.pdf", b"%PDF", "paste text"), ("bad.txt", b"\xff", "local text")],
                         ids=["byte_limit", "character_limit", "binary", "invalid_json", "pdf_guidance", "invalid_utf8"])
def test_invalid_materials_fail_clearly(tmp_path, name, content, error):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(ValueError, match=error):
        core.load_material(path)


def test_pptx_slide_order_text_only_and_no_external_content(tmp_path):
    path = pptx(tmp_path / "slides.pptx", ["First XML slide", "先展示的页"], order=[2, 1], extra={
        "ppt/notesSlides/notesSlide1.xml": "Secret speaker notes",
        "ppt/embeddings/program.bin": b"never execute",
        "ppt/media/image.png": b"not a real image"})
    imported = core.load_material(path)
    assert [section["text"] for section in imported["sections"]] == ["先展示的页", "First XML slide"]
    assert imported["sections"][0]["title"].startswith("Slide 1")
    assert "speaker notes" in imported["notice"]
    assert "Secret" not in json.dumps(imported)


@pytest.mark.parametrize("kind", ["too_many_slides", "oversized_xml", "expanded_archive", "doctype_utf16", "traversal", "duplicate", "malformed"])
def test_bounded_pptx_rejects_unsafe_or_excessive_parts(tmp_path, kind):
    if kind == "too_many_slides":
        path = pptx(tmp_path / "bad.pptx", ["x"] * 101)
    elif kind == "oversized_xml":
        path = pptx(tmp_path / "bad.pptx", ["x" * (core.MAX_XML_BYTES + 1)])
    elif kind == "expanded_archive":
        path = pptx(tmp_path / "bad.pptx", ["small"], extra={"ppt/media/big.bin": b"0" * (core.MAX_UNZIPPED_BYTES + 1)})
    elif kind == "doctype_utf16":
        path = pptx(tmp_path / "bad.pptx", ["small"])
        with zipfile.ZipFile(path) as archive:
            parts = {name: archive.read(name) for name in archive.namelist()}
        parts["ppt/slides/slide1.xml"] = '<?xml version="1.0" encoding="utf-16"?><!DOCTYPE x [<!ENTITY secret "injected">]><x>&secret;</x>'.encode("utf-16")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in parts.items():
                archive.writestr(name, content)
    elif kind == "traversal":
        path = pptx(tmp_path / "bad.pptx", ["small"], extra={"../outside.xml": "never extract"})
    elif kind == "duplicate":
        path = pptx(tmp_path / "bad.pptx", ["small"])
        with pytest.warns(UserWarning):
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("ppt/slides/slide1.xml", "duplicate")
    else:
        path = tmp_path / "bad.pptx"
        path.write_bytes(b"not a zip")
    with pytest.raises(ValueError):
        core.load_material(path)


def test_selection_requires_known_unique_ids_and_bounded_total(tmp_path):
    profile = core.default_state()["profiles"]["presentation"]
    imported = material(tmp_path, "a" * 6000)
    decoy = material(tmp_path, "Do not include me", "unselected.txt")
    profile["materials"] = [imported, decoy]
    profile["selected_ids"] = [imported["sections"][1]["id"], imported["sections"][0]["id"]]
    with pytest.raises(ValueError, match="5000"):
        core.review_sources(profile)
    profile["selected_ids"] = [imported["sections"][1]["id"]]
    assert [source.text for source in core.review_sources(profile)] == ["a" * 2000]
    for selected in (["stale"], [profile["selected_ids"][0]] * 2):
        profile["selected_ids"] = selected
        with pytest.raises(ValueError):
            core.review_sources(profile)


def test_nested_pptx_paragraphs_visit_each_text_node_once(tmp_path):
    path = pptx(tmp_path / "nested.pptx", ["placeholder"])
    with zipfile.ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    text = "Only once " + "x" * 3990
    parts["ppt/slides/slide1.xml"] = ('<a:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">' +
                                     '<a:p>' * 500 + '<a:t>' + text + '</a:t>' + '</a:p>' * 500 + '</a:sld>').encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    result = "".join(section["text"] for section in core.load_material(path)["sections"])
    assert result == "\n" * 499 + text


@pytest.mark.parametrize("mutation", ["answer", "notes", "history", "materials", "mode"])
def test_repository_bounds_validation_does_not_create_file(tmp_path, mutation):
    state = core.default_state()
    profile = state["profiles"]["meeting"]
    if mutation in ("answer", "notes"):
        profile[mutation] = "x" * (core.TEXT_LIMITS[mutation] + 1)
    elif mutation == "history":
        profile["practice_log"] = [{"question": "q", "answer": "a", "notes": "", "created_at": ""}] * 51
    elif mutation == "materials":
        profile["materials"] = [{}] * 6
    else:
        state["mode"] = "unrecognized"
    repository = core.CompanionRepository(tmp_path / "new.json")
    with pytest.raises(ValueError):
        repository.save(state)
    assert not repository.path.exists()


def test_question_banks_and_notes_preserve_user_content_only(tmp_path):
    for mode in core.MODES:
        prompts = core.question_bank(mode, "My topic")
        assert len(prompts) == len(set(prompts)) == 8
        assert all("Rule-based practice" in prompt and "My topic" in prompt for prompt in prompts)
    profile = core.default_state()["profiles"]["interview"]
    profile.update(answer="User answer\nwith exact spacing  .", notes="Owner is unresolved; do not assign one.")
    imported = material(tmp_path)
    profile["materials"] = [imported]
    profile["selected_ids"] = [imported["sections"][0]["id"]]
    profile["practice_log"] = [{"question": "My question?", "answer": "My historical answer", "notes": "My feedback", "created_at": "2026-10-04"}]
    before = deepcopy(profile)
    exported = core.notes_markdown("interview", profile)
    assert profile == before
    assert profile["answer"] in exported and profile["notes"] in exported
    assert imported["sha256"] in exported and imported["path"] in exported
    assert all(value in exported for value in profile["practice_log"][0].values())
