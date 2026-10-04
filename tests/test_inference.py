import numpy as np
import pytest
from meeting.inference import Recognizer, SpeakerTracker


def test_missing_model_never_falls_back_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        Recognizer(str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        Recognizer(str(tmp_path))


def test_missing_embedding_and_short_speech_unknown():
    tracker = SpeakerTracker()
    assert tracker.classify(np.ones(40000))["speaker"] == "Unknown"
    tracker.extractor = object()
    assert tracker.classify(np.ones(1000))["speaker"] == "Unknown"


def test_actual_clustering_logic_stable_ambiguous_and_reset():
    tracker = SpeakerTracker()
    tracker.extractor = object()
    vector = np.array([1., 0., 0.], np.float32)
    tracker._embedding = lambda x: vector
    audio = np.ones(48000, np.float32)
    assert tracker.classify(audio)["speaker"] == "Speaker 1"
    assert tracker.classify(audio)["speaker"] == "Speaker 1"
    vector = np.array([0., 1., 0.], np.float32)
    assert tracker.classify(audio)["speaker"] == "Speaker 2"
    vector = np.array([.7071068, .7071068, 0.], np.float32)
    assert tracker.classify(audio)["speaker"] == "Unknown"
    tracker.reset()
    assert len(tracker.centroids) == 0


def test_voice_changes_inside_utterance_unassigned():
    tracker = SpeakerTracker()
    tracker.extractor = object()
    calls = iter([np.array([1., 0.]), np.array([0., 1.]), np.array([1., 0.])])
    tracker._embedding = lambda x: next(calls)
    result = tracker.classify(np.ones(80000))
    assert result["speaker"] == "Unknown"
    assert "multiple speakers" in result["warning"]


def test_embedding_failure_keeps_transcription_running():
    tracker = SpeakerTracker()
    tracker.extractor = object()
    def fail(x):
        raise RuntimeError("model unavailable")
    tracker._embedding = fail
    assert tracker.classify(np.ones(48000))["speaker"] == "Unknown"


def test_offline_guard_in_separate_process():
    import subprocess, sys
    code = "from meeting.offline import enable; enable(); import socket; socket.create_connection(('example.com',443))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "meeting mode is offline" in result.stderr
