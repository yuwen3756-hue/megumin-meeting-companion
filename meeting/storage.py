"""Durable, local-only meeting history and portable transcript exports."""
from __future__ import annotations

import csv
import io
import json
import math
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp(seconds: float, *, srt: bool = False) -> str:
    millis = max(0, round(float(seconds) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{',' if srt else '.'}{millis:03d}"


class Store:
    """One connection guarded across capture/UI threads; every write commits."""

    def __init__(self, path: Path | str, *, recover: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=20)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        with self.db:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
                    ended_at TEXT, status TEXT NOT NULL DEFAULT 'loading',
                    language TEXT NOT NULL DEFAULT 'auto', simulated INTEGER NOT NULL DEFAULT 0,
                    retain_audio INTEGER NOT NULL DEFAULT 0, notes TEXT NOT NULL DEFAULT '',
                    profile TEXT NOT NULL DEFAULT '', duration REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    start REAL NOT NULL, end REAL NOT NULL, source TEXT NOT NULL,
                    speaker TEXT NOT NULL, original_speaker TEXT NOT NULL,
                    text TEXT NOT NULL, original_text TEXT NOT NULL,
                    confidence REAL, warning TEXT, latency_ms REAL, edited_at TEXT,
                    caption_delay_ms REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS segments_session ON segments(session_id, start, id);
                CREATE TABLE IF NOT EXISTS speaker_names (
                    session_id TEXT NOT NULL, speaker TEXT NOT NULL, name TEXT NOT NULL,
                    PRIMARY KEY(session_id, speaker),
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL, payload TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
            """)
            columns = {row[1] for row in self.db.execute("PRAGMA table_info(segments)")}
            if "caption_delay_ms" not in columns:
                self.db.execute("ALTER TABLE segments ADD COLUMN caption_delay_ms REAL")
            if "speech_end_delay_ms" not in columns:
                self.db.execute("ALTER TABLE segments ADD COLUMN speech_end_delay_ms REAL")
        self.recovered_count = self.recover_interrupted() if recover else 0

    def recover_interrupted(self) -> int:
        with self._lock, self.db:
            cursor = self.db.execute(
                "UPDATE sessions SET status='interrupted', ended_at=? "
                "WHERE status IN ('loading','recording','running','stopping')", (now_iso(),))
            return cursor.rowcount

    def create_session(self, title: str = "Untitled meeting", *, language: str = "auto",
                       simulated: bool = False, retain_audio: bool = False,
                       profile: str = "") -> str:
        session_id = uuid.uuid4().hex
        with self._lock, self.db:
            self.db.execute(
                "INSERT INTO sessions(id,title,created_at,language,simulated,retain_audio,profile) "
                "VALUES(?,?,?,?,?,?,?)",
                (session_id, title.strip() or "Untitled meeting", now_iso(), language,
                 int(simulated), int(retain_audio), profile))
        return session_id

    def get_session(self, session_id: str) -> dict:
        with self._lock:
            row = self.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise KeyError(session_id)
            return dict(row)

    def list_sessions(self, limit: int = 200) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM segments t WHERE t.session_id=s.id) "
                "AS segment_count FROM sessions s ORDER BY created_at DESC,rowid DESC LIMIT ?",
                (limit,))]

    def update_session(self, session_id: str, **fields) -> None:
        allowed = {"title", "status", "notes", "duration", "ended_at", "profile"}
        if not fields or not set(fields).issubset(allowed):
            raise ValueError("Unsupported or empty session update")
        with self._lock, self.db:
            self.db.execute("UPDATE sessions SET " + ",".join(f"{key}=?" for key in fields)
                            + " WHERE id=?", (*fields.values(), session_id))

    def finish_session(self, session_id: str, *, status: str = "completed",
                       duration: float | None = None) -> None:
        fields = {"status": status, "ended_at": now_iso()}
        if duration is not None:
            fields["duration"] = max(0, duration)
        self.update_session(session_id, **fields)

    def append_event(self, session_id: str, event: dict) -> int | None:
        """Persist before displaying; returns segment ID for segment events."""
        with self._lock, self.db:
            self.db.execute("INSERT INTO events(session_id,created_at,payload) VALUES(?,?,?)",
                            (session_id, now_iso(), json.dumps(event, ensure_ascii=False, default=str)))
            if event.get("type") != "segment":
                return None
            start = max(0.0, float(event.get("start", 0)))
            end = max(start, float(event.get("end", start)))
            speaker = str(event.get("speaker") or "Unknown")
            text = str(event.get("text") or "")
            cursor = self.db.execute(
                "INSERT INTO segments(session_id,start,end,source,speaker,original_speaker,"
                "text,original_text,confidence,warning,latency_ms,caption_delay_ms,speech_end_delay_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (session_id, start, end, str(event.get("source") or "unknown"), speaker,
                 speaker, text, text, event.get("confidence"), event.get("warning"),
                 event.get("latency_ms"), event.get("caption_delay_ms"),
                 event.get("caption_delay_from_speech_end_ms") if event.get("caption_delay_valid") else None))
            self.db.execute("UPDATE sessions SET duration=MAX(duration,?) WHERE id=?", (end, session_id))
            return int(cursor.lastrowid)

    def segments(self, session_id: str, *, search: str = "", limit: int | None = None,
                 offset: int = 0, newest: bool = False) -> list[dict]:
        query = ("SELECT t.*, COALESCE(n.name,t.speaker) AS speaker_name FROM segments t "
                 "LEFT JOIN speaker_names n ON n.session_id=t.session_id AND n.speaker=t.speaker "
                 "WHERE t.session_id=?")
        args: list = [session_id]
        if search:
            query += " AND (t.text LIKE ? ESCAPE '\\' OR COALESCE(n.name,t.speaker) LIKE ? ESCAPE '\\')"
            pattern = "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            args.extend([pattern, pattern])
        direction = " DESC" if newest else " ASC"
        query += " ORDER BY t.start" + direction + ",t.id" + direction
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            args.extend([max(0, limit), max(0, offset)])
        with self._lock:
            return [dict(row) for row in self.db.execute(query, args)]

    def preceding_segments(self, session_id: str, before: float, *, seconds: float = 60.0,
                           limit: int = 6) -> list[dict]:
        """Read a bounded local context window even for an older question."""
        before, seconds = float(before), float(seconds)
        if not math.isfinite(before) or before < 0 or not math.isfinite(seconds) or not 0 <= seconds <= 60:
            raise ValueError("Context times must be finite; the window cannot exceed 60 seconds.")
        if not isinstance(limit, int) or not 0 <= limit <= 6:
            raise ValueError("Context can contain at most six preceding segments.")
        query = ("SELECT t.*, COALESCE(n.name,t.speaker) AS speaker_name FROM segments t "
                 "LEFT JOIN speaker_names n ON n.session_id=t.session_id AND n.speaker=t.speaker "
                 "WHERE t.session_id=? AND t.start>=? AND t.end<=? AND t.end>=t.start "
                 "AND length(trim(t.text))>0 ORDER BY t.end DESC,t.start DESC,t.id DESC LIMIT ?")
        with self._lock:
            return [dict(row) for row in self.db.execute(query, (session_id, max(0, before - seconds), before, limit))]

    def segment_count(self, session_id: str, search: str = "") -> int:
        if search:
            # Same escaped matching semantics as the paginated display.
            with self._lock:
                pattern = "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                return self.db.execute(
                    "SELECT COUNT(*) FROM segments t LEFT JOIN speaker_names n ON "
                    "n.session_id=t.session_id AND n.speaker=t.speaker WHERE t.session_id=? "
                    "AND (t.text LIKE ? ESCAPE '\\' OR COALESCE(n.name,t.speaker) LIKE ? ESCAPE '\\')",
                    (session_id, pattern, pattern)).fetchone()[0]
        with self._lock:
            return self.db.execute("SELECT COUNT(*) FROM segments WHERE session_id=?",
                                   (session_id,)).fetchone()[0]

    def get_segment(self, segment_id: int) -> dict:
        with self._lock:
            row = self.db.execute(
                "SELECT t.*,COALESCE(n.name,t.speaker) AS speaker_name FROM segments t "
                "LEFT JOIN speaker_names n ON n.session_id=t.session_id AND n.speaker=t.speaker "
                "WHERE t.id=?", (segment_id,)).fetchone()
            if row is None:
                raise KeyError(segment_id)
            return dict(row)

    def correct_segment(self, segment_id: int, text: str, *, speaker: str | None = None) -> None:
        with self._lock, self.db:
            fields = {"text": text, "edited_at": now_iso()}
            if speaker is not None:
                fields["speaker"] = speaker.strip() or "Unknown"
            self.db.execute("UPDATE segments SET " + ",".join(f"{key}=?" for key in fields)
                            + " WHERE id=?", (*fields.values(), segment_id))

    def rename_speaker(self, session_id: str, speaker: str, name: str) -> None:
        with self._lock, self.db:
            if name.strip():
                self.db.execute("INSERT INTO speaker_names(session_id,speaker,name) VALUES(?,?,?) "
                                "ON CONFLICT(session_id,speaker) DO UPDATE SET name=excluded.name",
                                (session_id, speaker, name.strip()))
            else:
                self.db.execute("DELETE FROM speaker_names WHERE session_id=? AND speaker=?",
                                (session_id, speaker))

    def export_session(self, session_id: str, directory: Path | str,
                       formats: Iterable[str] = ("txt", "md", "json", "csv", "srt")) -> list[Path]:
        # Snapshot under one lock so a concurrent capture cannot split formats.
        with self._lock:
            session = self.get_session(session_id)
            segments = self.segments(session_id)
            events = [json.loads(r[0]) for r in self.db.execute(
                "SELECT payload FROM events WHERE session_id=? ORDER BY id", (session_id,))]
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        safe_title = re.sub(r'[^\w\- ]', "", session["title"], flags=re.UNICODE).strip()[:70] or "Meeting"
        stem = f"{session['created_at'][:10]} {safe_title} {session_id[:8]}"
        disclosure = "SIMULATED DEMO — no audio captured" if session["simulated"] else "Local meeting transcript"
        warning = "Automatic transcript and speaker labels may be inaccurate. Names are user-assigned."
        lines = [f"[{timestamp(s['start'])}] [{s['source']}] {s['speaker_name']}: {s['text']}" for s in segments]
        outputs = {
            "txt": "\n".join([session["title"], disclosure, warning, "", *lines, "", "NOTES", session["notes"]]) + "\n",
            "md": "\n".join(["# " + session["title"].replace("\n", " "), "", disclosure, "", warning, "",
                              "## Transcript", "", *["> " + line.replace("\n", "\n> ") + "\n" for line in lines],
                              "## Notes", "", session["notes"]]) + "\n",
            "json": json.dumps({"schema_version": 1, "session": session, "segments": segments,
                                "events": events, "notice": warning}, ensure_ascii=False, indent=2) + "\n",
            "srt": "\n\n".join(f"{i}\n{timestamp(s['start'], srt=True)} --> "
                                  f"{timestamp(max(s['end'], s['start'] + .001), srt=True)}\n"
                                  f"{'[SIMULATED] ' if session['simulated'] else ''}[{s['source']}] {s['speaker_name']}: {s['text']}"
                                  for i, s in enumerate(segments, 1)) + "\n",
        }
        stream = io.StringIO(newline="")
        columns = ["session_id", "session_title", "session_simulated", "start", "end", "source", "speaker", "speaker_name", "text", "original_text",
                   "original_speaker", "confidence", "warning", "latency_ms", "caption_delay_ms", "speech_end_delay_ms", "edited_at"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for segment in segments:
            segment = {**segment, "session_title": session["title"], "session_simulated": session["simulated"]}
            # Keep spreadsheet programs from treating transcript text as formulas.
            writer.writerow({key: ("'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
                                   else value) for key, value in segment.items()})
        outputs["csv"] = stream.getvalue()
        paths = []
        for format_name in formats:
            format_name = format_name.lower().lstrip(".")
            if format_name not in outputs:
                raise ValueError(f"Unsupported export format: {format_name}")
            path = directory / f"{stem}.{format_name}"
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(outputs[format_name], encoding="utf-8-sig" if format_name == "csv" else "utf-8", newline="")
            temporary.replace(path)
            paths.append(path)
        return paths

    def close(self) -> None:
        with self._lock:
            self.db.close()


SessionStore = Store
