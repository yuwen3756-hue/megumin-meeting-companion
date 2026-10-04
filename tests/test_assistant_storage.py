"""Context retrieval must stay anchored to the reviewed question/session."""
import pytest

from meeting.storage import Store


def test_older_question_context_is_bounded_and_excludes_other_sessions(tmp_path):
    store = Store(tmp_path / "context.sqlite3")
    sid = store.create_session("Selected synthetic meeting")
    other = store.create_session("Unselected synthetic meeting")
    for session in (sid, other):
        for i in range(300):
            store.append_event(session, {"type": "segment", "start": i * 4, "end": i * 4 + 3,
                                         "source": "remote", "speaker": "Speaker 1", "text": f"Synthetic fact {i}"})
    store.rename_speaker(sid, "Speaker 1", "User-assigned example label")
    rows = store.preceding_segments(sid, before=120)
    assert len(rows) == 6
    assert [row["text"] for row in rows] == [f"Synthetic fact {i}" for i in range(29, 23, -1)]
    assert all(row["session_id"] == sid and 60 <= row["start"] <= row["end"] <= 120 for row in rows)
    assert all(row["speaker_name"] == "User-assigned example label" for row in rows)
    assert store.preceding_segments(sid, before=0) == []
    store.close()


@pytest.mark.parametrize("arguments", [{"before": float("nan")}, {"before": -1},
                                       {"before": 20, "seconds": 61}, {"before": 20, "limit": 7}])
def test_invalid_context_window_rejected(tmp_path, arguments):
    store = Store(tmp_path / "context.sqlite3")
    with pytest.raises(ValueError):
        store.preceding_segments("unused", **arguments)
    store.close()
