import csv
import json
from pathlib import Path

import pytest

from meeting.storage import Store, timestamp


def segment(text='你好, "world"\nSecond line', **extra):
    return {"type": "segment", "start": 1.234, "end": 3.9996, "source": "remote",
            "speaker": "Speaker 1", "text": text, "confidence": .8,
            "warning": "Review this", "latency_ms": 800, **extra}


def test_persistence_corrections_and_speaker_names(tmp_path):
    path = tmp_path / "local" / "meetings.sqlite3"
    store = Store(path)
    sid = store.create_session("中文 meeting", retain_audio=False)
    ident = store.append_event(sid, segment())
    store.correct_segment(ident, "User corrected 中文", speaker="Speaker 2")
    store.rename_speaker(sid, "Speaker 2", "张伟")
    store.update_session(sid, notes="My local notes\n第二行")
    store.finish_session(sid)
    store.close()
    restored = Store(path)
    result = restored.get_segment(ident)
    assert result["text"] == "User corrected 中文"
    assert result["original_text"] == '你好, "world"\nSecond line'
    assert result["speaker"] == "Speaker 2"
    assert result["original_speaker"] == "Speaker 1"
    assert result["speaker_name"] == "张伟"
    assert result["edited_at"]
    assert restored.get_session(sid)["notes"] == "My local notes\n第二行"
    assert restored.get_session(sid)["retain_audio"] == 0
    restored.rename_speaker(sid, "Speaker 2", "")
    assert restored.get_segment(ident)["speaker_name"] == "Speaker 2"
    restored.close()


def test_recovery_marks_only_interrupted_sessions(tmp_path):
    path = tmp_path / "meetings.sqlite3"
    store = Store(path)
    active = store.create_session("Interrupted")
    ended = store.create_session("Finished")
    store.append_event(active, segment("Persist before crash"))
    store.update_session(active, status="recording")
    store.finish_session(ended)
    store.close()
    store = Store(path)
    assert store.recovered_count == 1
    assert store.get_session(active)["status"] == "interrupted"
    assert store.get_session(active)["ended_at"]
    assert store.get_session(ended)["status"] == "completed"
    assert store.segments(active)[0]["text"] == "Persist before crash"
    store.close()


def test_speech_end_latency_is_separate_from_legacy_asr_timestamp(tmp_path):
    with_store = Store(tmp_path / "timing.sqlite3")
    sid = with_store.create_session("Timing")
    ident = with_store.append_event(sid, segment(caption_delay_ms=1900,
        caption_delay_from_speech_end_ms=850, caption_delay_valid=True))
    row = with_store.get_segment(ident)
    assert row["caption_delay_ms"] == 1900
    assert row["speech_end_delay_ms"] == 850
    invalid = with_store.append_event(sid, segment(caption_delay_from_speech_end_ms=2,
        caption_delay_valid=False))
    assert with_store.get_segment(invalid)["speech_end_delay_ms"] is None
    with_store.close()


def test_exports_preserve_unicode_multiline_and_audit_fields(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    sid = store.create_session('Demo: /路径?', simulated=True)
    ident = store.append_event(sid, segment())
    store.correct_segment(ident, 'Corrected "你好",\nline 2')
    store.rename_speaker(sid, "Speaker 1", "李")
    store.update_session(sid, notes="Local notes 中文")
    paths = store.export_session(sid, tmp_path / "export")
    by_type = {path.suffix: path for path in paths}
    assert set(by_type) == {".txt", ".md", ".json", ".csv", ".srt"}
    assert all(path.parent == tmp_path / "export" for path in paths)
    data = json.loads(by_type[".json"].read_text(encoding="utf-8"))
    assert data["segments"][0]["original_text"] == '你好, "world"\nSecond line'
    assert data["segments"][0]["text"] == 'Corrected "你好",\nline 2'
    assert data["session"]["simulated"] == 1
    assert data["events"][0]["text"] == '你好, "world"\nSecond line'
    with by_type[".csv"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["text"] == 'Corrected "你好",\nline 2'
    assert rows[0]["speaker_name"] == "李"
    srt = by_type[".srt"].read_text(encoding="utf-8")
    assert rows[0]["session_simulated"] == "1"
    assert srt.startswith("1\n00:00:01,234 --> 00:00:04,000\n[SIMULATED] [remote] 李:")
    for suffix in (".txt", ".md"):
        exported = by_type[suffix].read_text(encoding="utf-8")
        assert "SIMULATED DEMO" in exported
        assert "Local notes 中文" in exported
    store.close()


def test_search_pagination_literal_like_and_ordering(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    sid = store.create_session()
    for i, text in enumerate(["100% ready", "1000 ready", "a_b", "acb", "你好"]):
        store.append_event(sid, segment(text, start=i, end=i + .9))
    assert len(store.segments(sid, search="%")) == 1
    assert len(store.segments(sid, search="_")) == 1
    assert store.segment_count(sid, "你好") == 1
    assert [s["start"] for s in store.segments(sid, limit=2, offset=1, newest=True)] == [3, 2]
    store.rename_speaker(sid, "Speaker 1", "Alice")
    assert store.segment_count(sid, "Alice") == 5
    assert len(store.segments(sid, search="Alice", limit=2)) == 2
    store.close()


def test_csv_formula_escaping_and_timestamp_boundaries(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    sid = store.create_session()
    store.append_event(sid, segment("=SUM(1,2)"))
    path = store.export_session(sid, tmp_path / "exports", ["csv"])[0]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        assert next(csv.DictReader(stream))["text"].startswith("'=")
    assert timestamp(-4, srt=True) == "00:00:00,000"
    assert timestamp(3599.9996, srt=True) == "01:00:00,000"
    with pytest.raises(ValueError):
        store.update_session(sid, unrecognized="bad")
    store.close()
