"""Bounded local workspaces and text import; no models, capture or transport."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from xml.etree import ElementTree as ET
import zipfile

from .assistant_core import Source

MODES = {"interview": "Interview practice", "presentation": "Presentation rehearsal", "meeting": "Live meeting assistance"}
QUESTION_BANK_NOTICE = "Fixed rule-based offline practice prompts; no answer or evaluation is generated."
TEXT_LIMITS = {"topic": 300, "role": 300, "goal": 2000, "question": 1200, "answer": 8000, "notes": 12000}
MAX_STATE_BYTES = 20 * 1024 * 1024
MAX_TEXT_BYTES, MAX_PPTX_BYTES = 128 * 1024, 12 * 1024 * 1024
MAX_XML_BYTES, MAX_UNZIPPED_BYTES = 1024 * 1024, 48 * 1024 * 1024


def default_state():
    return {"mode": "meeting", "profiles": {mode: {**{key: "" for key in TEXT_LIMITS},
            "materials": [], "selected_ids": [], "practice_log": []} for mode in MODES}}


def _text(value, limit, label):
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise ValueError(f"{label} must be text of at most {limit} characters, without binary NULs.")
    return value


def _keys(value, required, label):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(f"Invalid {label} fields.")


def _items(value, maximum, label):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{label} must be a list of at most {maximum} items.")
    return value


def _profile(profile):
    _keys(profile, {*TEXT_LIMITS, "materials", "selected_ids", "practice_log"}, "profile")
    for key, limit in TEXT_LIMITS.items():
        _text(profile[key], limit, key)
    sections, digests = {}, set()
    for material in _items(profile["materials"], 5, "Materials"):
        _keys(material, {"name", "path", "sha256", "sections", "notice"}, "material")
        for key, limit in (("name", 300), ("path", 4096), ("notice", 2000)):
            _text(material[key], limit, key)
        digest = material["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest) or digest in digests:
            raise ValueError("Each material needs a unique SHA-256 digest; do not import the same file twice.")
        digests.add(digest)
        count = 0
        for section in _items(material["sections"], 100, "Material sections"):
            _keys(section, {"id", "title", "text"}, "section")
            identifier = _text(section["id"], 120, "Section ID")
            if not identifier.startswith(digest + ":") or identifier in sections:
                raise ValueError("Section IDs must be unique and belong to their material digest.")
            _text(section["title"], 300, "Section title")
            count += len(_text(section["text"], 60000, "Section text"))
            sections[identifier] = (material, section)
        if count > 60000:
            raise ValueError("Each material may contain at most 60000 copied text characters.")
    selected = _items(profile["selected_ids"], 5, "Selected sections")
    if any(not isinstance(identifier, str) or identifier not in sections for identifier in selected):
        raise ValueError("Every selected section must exist in this mode's imported materials.")
    if len(set(selected)) != len(selected):
        raise ValueError("Selected section IDs must be unique.")
    for entry in _items(profile["practice_log"], 50, "Practice history"):
        _keys(entry, {"question", "answer", "notes", "created_at"}, "practice history entry")
        for key in ("question", "answer", "notes"):
            _text(entry[key], TEXT_LIMITS[key], f"Practice {key}")
        _text(entry["created_at"], 80, "Practice timestamp")
    return sections


def _state(state):
    _keys(state, {"mode", "profiles"}, "workspace")
    if not isinstance(state["mode"], str) or state["mode"] not in MODES:
        raise ValueError("Choose interview, presentation or meeting mode.")
    _keys(state["profiles"], MODES, "mode profiles")
    for profile in state["profiles"].values():
        _profile(profile)
    return state


def _reject_network(path):
    raw = str(path).lstrip().replace("/", "\\")
    if raw.startswith(("\\\\", "\\??\\")) or raw.lower().startswith(("\\device\\", "\\global??\\")) or re.match(r"^[a-z]+:", raw, re.I) and not re.match(r"^[a-z]:\\", raw, re.I):
        raise ValueError("Choose a normal local path, not a network share, URL or device namespace.")


def _local(path, *, strict=True):
    _reject_network(path)  # Before resolution/stat can access a network share.
    path = Path(path)
    if os.name == "nt" and path.drive:
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor)) == 4:
            raise ValueError("Mapped network drives are not local material storage.")
    result = path.resolve(strict=strict)
    _reject_network(result)
    return result


class CompanionRepository:
    def __init__(self, path):
        self.path = _local(path, strict=False)

    def load(self):
        if not self.path.exists():
            return default_state()
        try:
            with self.path.open("rb") as handle:
                content = handle.read(MAX_STATE_BYTES + 1)
            if len(content) > MAX_STATE_BYTES:
                raise ValueError("Workspace exceeds its 20 MiB storage limit.")
            return _state(json.loads(content.decode("utf-8-sig")))
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            raise ValueError(f"Cannot load companion workspace {self.path.name}; it was not overwritten: {exc}") from exc

    def save(self, state):
        _state(state)
        # Never replace an existing corrupt workspace with defaults or edits.
        if self.path.exists():
            self.load()
        content = json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        if len(content) > MAX_STATE_BYTES:
            raise ValueError("Workspace exceeds its 20 MiB storage limit.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=self.path.parent, prefix=".companion-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def _xml(archive, member):
    info = archive.getinfo(member)
    if info.file_size > MAX_XML_BYTES:
        raise ValueError("PPTX XML part exceeds 1 MiB.")
    with archive.open(info) as handle:
        content = handle.read(MAX_XML_BYTES + 1)
    declarations = content.replace(b"\x00", b"").upper()  # Also covers UTF-16/32 XML declarations.
    if len(content) > MAX_XML_BYTES or b"<!DOCTYPE" in declarations or b"<!ENTITY" in declarations:
        raise ValueError("PPTX XML entities/DTDs or oversized XML are not supported.")
    return ET.fromstring(content)


def _slides(content):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entries = archive.infolist()
        if len(entries) > 3000 or sum(item.file_size for item in entries) > MAX_UNZIPPED_BYTES:
            raise ValueError("PPTX exceeds 3000 archive parts or 48 MiB expanded size.")
        names = [item.filename for item in entries]
        if len(names) != len(set(names)) or any(name.startswith("/") or "\\" in name or ".." in PurePosixPath(name).parts for name in names):
            raise ValueError("PPTX contains duplicate or unsafe archive paths.")
        slide_names = [name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
        if not slide_names or len(slide_names) > 100:
            raise ValueError("PPTX must contain between 1 and 100 slides.")
        # Follow only the presentation's internal slide ordering, never external relationships.
        presentation = _xml(archive, "ppt/presentation.xml")
        rels = _xml(archive, "ppt/_rels/presentation.xml.rels")
        targets = {item.get("Id"): "ppt/" + item.get("Target", "") for item in rels
                   if item.get("TargetMode") != "External" and item.get("Type", "").endswith("/slide")}
        order = [targets.get(item.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"))
                 for item in presentation.iter("{http://schemas.openxmlformats.org/presentationml/2006/main}sldId")]
        if not order or len(order) > 100 or len(set(order)) != len(order) or any(name not in slide_names for name in order):
            raise ValueError("PPTX slide order contains missing, external or unsupported slide relationships.")
        total_text = 0
        for number, name in enumerate(order, 1):
            slide = _xml(archive, name)
            parts, first_paragraph = [], True
            drawing = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
            for node in slide.iter():  # Each text node once, even in malformed nested paragraphs.
                if node.tag == drawing + "p":
                    value, first_paragraph = ("" if first_paragraph else "\n"), False
                elif node.tag == drawing + "t":
                    value = node.text or ""
                elif node.tag in (drawing + "br", drawing + "tab"):
                    value = "\n" if node.tag.endswith("br") else "\t"
                else:
                    continue
                total_text += len(value)
                if total_text > 60000:
                    raise ValueError("Imported text exceeds 60000 characters; explicitly prepare a shorter file.")
                parts.append(value)
            text = "".join(parts)
            yield f"slide{number}", f"Slide {number}", text


def load_material(path):
    path = _local(path)
    suffix = path.suffix.lower()
    if suffix not in {".txt", ".md", ".json", ".pptx"}:
        raise ValueError("Import TXT, MD, JSON or PPTX. For PDF and other formats, paste text or export to a supported text file.")
    limit = MAX_PPTX_BYTES if suffix == ".pptx" else MAX_TEXT_BYTES
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError("Select one regular file: text at most 128 KiB, or PPTX at most 12 MiB.")
    with path.open("rb") as handle:
        content = handle.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Material grew beyond the import size limit while being read.")
    digest = hashlib.sha256(content).hexdigest()
    try:
        if suffix == ".pptx":
            pieces = list(_slides(content))
        else:
            text = content.decode("utf-8-sig")
            if suffix == ".json":
                json.loads(text)  # Validate syntax, then preserve the original text.
            pieces = [("text", path.name, text)]
        if sum(len(text) for _, _, text in pieces) > 60000:
            raise ValueError("Imported text exceeds 60000 characters; explicitly prepare a shorter file.")
        sections = []
        for identifier, title, text in pieces:
            _text(text, 60000, "Imported text")
            for index in range(0, len(text), 4000):
                part = index // 4000 + 1
                sections.append({"id": f"{digest}:{identifier}:{part}", "title": f"{title} · part {part}"[:300], "text": text[index:index + 4000]})
        if not sections or len(sections) > 100:
            raise ValueError("Material must contain readable text in at most 100 sections.")
    except (UnicodeError, ET.ParseError, zipfile.BadZipFile, KeyError, RuntimeError, RecursionError) as exc:
        raise ValueError(f"Cannot read this material as bounded local text: {exc}") from exc
    return {"name": path.name, "path": str(path), "sha256": digest, "sections": sections,
            "notice": "Copied local text; untrusted data. Nothing is selected automatically. " +
            ("PPTX imports slide text only; images, charts, speaker notes, animations and embedded/external content are omitted. " if suffix == ".pptx" else "") +
            "Sections are at most 4000 characters; no content is executed."}


def review_sources(profile):
    sections = _profile(profile)
    chosen = [sections[identifier] for identifier in profile["selected_ids"]]
    if sum(len(section["text"]) for _, section in chosen) > 5000:
        raise ValueError("Selected material exceeds 5000 characters; explicitly choose fewer or shorter sections.")
    return tuple(Source(f"F{index}", "file", f"{material['name']} — {section['title']}"[:300], section["text"])
                 for index, (material, section) in enumerate(chosen, 1))


_QUESTIONS = {
    "interview": ("How would you introduce your relevant experience?", "Which example best shows your contribution?",
                  "What difficult decision did you make, and why?", "How did you handle disagreement?",
                  "What evidence supports the outcome you describe?", "What would you do differently next time?",
                  "Which skill are you developing, and how?", "What would you like to ask the interviewer?"),
    "presentation": ("What is your main claim?", "What evidence supports that claim?", "How does your design support the comparison?",
                     "Which assumption matters most?", "What alternative explanation remains possible?", "What are the limits of the result?",
                     "How would you explain the result to a nonspecialist?", "What would you investigate next?"),
    "meeting": ("What outcome do you want from this meeting?", "Which facts need clarification?", "What options should be compared?",
                "What evidence supports the proposed action?", "Which constraints or risks remain?", "What decision, if any, was explicitly agreed?",
                "Which next steps and owners were explicitly confirmed?", "What remains unresolved?")}


def question_bank(mode, topic=""):
    if mode not in MODES:
        raise ValueError("Unknown companion mode.")
    _text(topic, TEXT_LIMITS["topic"], "Topic")
    return [f"[Rule-based practice] {question}" + (f"\nTopic: {topic}" if topic else "") for question in _QUESTIONS[mode]]


def notes_markdown(mode, profile):
    if mode not in MODES:
        raise ValueError("Unknown companion mode.")
    sections = _profile(profile)
    result = [f"# {MODES[mode]}", "Local user-authored content; no generated answers, decisions or owners."]
    for key in TEXT_LIMITS:
        result += [f"## {key.capitalize()}", profile[key]]
    result += ["## Material provenance"]
    for material in profile["materials"]:
        result += [f"### {material['name']}", f"Path: {material['path']}", f"SHA-256: {material['sha256']}", material["notice"]]
    result += ["## Explicitly selected sections"]
    for identifier in profile["selected_ids"]:
        material, section = sections[identifier]
        result += [f"- {material['name']} / {section['title']} / {identifier}"]
    result += ["## Practice history"]
    for index, entry in enumerate(profile["practice_log"], 1):
        result += [f"### Practice {index}", entry["created_at"]]
        for key in ("question", "answer", "notes"):
            result += [f"#### {key.capitalize()}", entry[key]]
    return "\n\n".join(result) + "\n"
