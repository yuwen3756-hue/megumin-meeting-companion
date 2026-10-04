import subprocess
import sys
import pytest
from meeting.notes import suggest_actions, suggest_notes
from meeting.instance import InstanceLock


def test_extractive_candidates_preserve_negation_and_unknown_ownership():
    text = "I will not promise delivery tomorrow."
    rows = [dict(id=21, start=1, end=4, text=text, speaker="Unknown", source="remote"),
            dict(id=22, start=5, end=8, text="我会在明天发送测试报告。", speaker="Speaker 1", source="remote")]
    actions = suggest_actions(rows)
    assert len(actions) == 2
    assert all(x["owner"] is None and x["deadline"] is None and x["needs_review"] for x in actions)
    assert actions[0]["quote"] == text and actions[0]["segment_id"] == 21
    notes = suggest_notes(rows)
    assert text in notes and rows[1]["text"] in notes
    assert "negations" in notes and "unverified" in notes


def test_empty_notes_cannot_invent_action():
    assert suggest_actions([]) == []
    assert "No transcript text" in suggest_notes([])


def test_second_process_cannot_recover_live_sessions(tmp_path):
    lock = InstanceLock(tmp_path)
    code = "from meeting.instance import InstanceLock; from pathlib import Path; lock=InstanceLock(Path(__import__('sys').argv[1])); lock.close()"
    first = subprocess.run([sys.executable,"-c",code,str(tmp_path)], capture_output=True,text=True)
    assert first.returncode != 0 and "already open" in first.stderr
    lock.close()
    second = subprocess.run([sys.executable,"-c",code,str(tmp_path)],capture_output=True,text=True)
    assert second.returncode == 0, second.stderr
