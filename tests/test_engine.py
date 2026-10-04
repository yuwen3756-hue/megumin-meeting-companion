"""Synthetic capture/engine contracts: no test opens a real audio device."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
import threading
import time
import types
import wave

import numpy as np
import pytest


SAMPLE_RATE = 16_000


def voice(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Deterministic voiced harmonics above the energy gate; no speech asset."""
    t = np.arange(round(seconds * sample_rate), dtype=np.float64) / sample_rate
    envelope = 0.75 + 0.25 * np.sin(2 * np.pi * 3 * t)
    samples = envelope * (
        0.10 * np.sin(2 * np.pi * 173 * t)
        + 0.05 * np.sin(2 * np.pi * 346 * t)
        + 0.025 * np.sin(2 * np.pi * 519 * t)
    )
    return samples.astype(np.float32)


class RecognizerStub:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def transcribe(self, audio: np.ndarray) -> list[dict]:
        self.calls.append(np.array(audio, copy=True))
        return [{"start": 0.0, "end": len(audio) / SAMPLE_RATE, "text": "fixture speech"}]


class TrackerStub:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def classify(self, audio: np.ndarray) -> dict:
        self.calls.append(np.array(audio, copy=True))
        return {"speaker": "Remote 1", "confidence": 0.9, "warning": ""}


class BlockingRecognizer(RecognizerStub):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def transcribe(self, audio: np.ndarray) -> list[dict]:
        self.entered.set()
        if not self.release.wait(timeout=10):
            raise RuntimeError("test did not release the blocked recognizer")
        return super().transcribe(audio)


@pytest.fixture(autouse=True)
def forbid_live_audio(monkeypatch):
    """Fail before device construction if fixture mode accidentally captures."""
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic tests must never initialize live audio")

    for module_name in ("pyaudiowpatch", "pyaudio"):
        fake_module = types.ModuleType(module_name)
        fake_module.PyAudio = forbidden
        fake_module.paFloat32 = 1
        fake_module.paInt16 = 8
        fake_module.paContinue = 0
        monkeypatch.setitem(sys.modules, module_name, fake_module)


@pytest.fixture
def audio_module():
    return importlib.import_module("meeting.audio")


@pytest.fixture
def engine_factory(tmp_path):
    engine_type = importlib.import_module("meeting.engine").Engine
    engines = []

    def make(recognizer=None, tracker=None, on_event=None):
        recognizer = recognizer if recognizer is not None else RecognizerStub()
        tracker = tracker if tracker is not None else TrackerStub()
        events: list[dict] = []

        def receive_event(event):
            events.append(event)
            if on_event is not None:
                on_event(event)

        instance = engine_type(recognizer, tracker, receive_event, tmp_path)
        engines.append((instance, recognizer))
        return instance, recognizer, tracker, events

    yield make

    for instance, recognizer in engines:
        if isinstance(recognizer, BlockingRecognizer):
            recognizer.release.set()
        instance.stop(timeout=3)


def segments(events: list[dict]) -> list[dict]:
    return [event for event in events if event.get("type") == "segment"]


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("synthetic worker condition was not reached before timeout")
        time.sleep(0.005)


def test_audio_import_does_not_load_device_backend():
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'pyaudiowpatch', 'pyaudio'}:
        raise AssertionError('audio backend must be imported lazily')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import meeting.audio
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=15,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_device_enumeration_uses_metadata_without_opening_streams(
    audio_module, monkeypatch
):
    class MetadataBackend:
        terminated = False
        devices = [
            {"hostApi": 2, "name": "Fixture microphone", "maxInputChannels": 1,
             "defaultSampleRate": 48_000, "isLoopbackDevice": False},
            {"hostApi": 2, "name": "Fixture loopback", "maxInputChannels": 2,
             "defaultSampleRate": 44_100, "isLoopbackDevice": True},
            {"hostApi": 1, "name": "Other API", "maxInputChannels": 1,
             "defaultSampleRate": 16_000},
            {"hostApi": 2, "name": "Output only", "maxInputChannels": 0,
             "defaultSampleRate": 48_000},
        ]

        def get_host_api_info_by_type(self, api):
            assert api == 13
            return {"index": 2}

        def get_device_count(self):
            return len(self.devices)

        def get_device_info_by_index(self, index):
            return self.devices[index]

        def open(self, *args, **kwargs):
            raise AssertionError("enumeration must never open a recording stream")

        def terminate(self):
            self.terminated = True

    backend = MetadataBackend()
    module = sys.modules["pyaudiowpatch"]
    monkeypatch.setattr(module, "PyAudio", lambda: backend)
    monkeypatch.setattr(module, "paWASAPI", 13, raising=False)

    assert audio_module.list_devices() == [
        {"id": 0, "name": "Fixture microphone", "kind": "mic",
         "sample_rate": 48_000, "channels": 1},
        {"id": 1, "name": "Fixture loopback", "kind": "loopback",
         "sample_rate": 44_100, "channels": 2},
    ]
    assert backend.terminated


@pytest.mark.parametrize("input_rate", [16_000, 44_100, 48_000])
def test_resampling_preserves_native_block_boundaries(audio_module, input_rate):
    source = voice(1.137, sample_rate=input_rate)
    whole = audio_module.StreamingResampler(input_rate)
    expected = np.concatenate([whole.process(source), whole.flush()])

    streaming = audio_module.StreamingResampler(input_rate)
    parts = []
    offset = 0
    sizes = (1, 17, 1024, 379, 2048, 53)
    index = 0
    while offset < len(source):
        size = sizes[index % len(sizes)]
        parts.append(streaming.process(source[offset : offset + size]))
        offset += size
        index += 1
    parts.append(streaming.flush())
    actual = np.concatenate(parts)

    assert actual.dtype == np.float32
    assert actual.ndim == 1
    assert len(actual) == len(source) * SAMPLE_RATE // input_rate
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_silence_does_not_reach_recognizer(engine_factory):
    engine, recognizer, tracker, events = engine_factory()
    engine.feed_fixture(np.zeros(SAMPLE_RATE * 3, dtype=np.float32))
    status = engine.stop(timeout=3)

    assert isinstance(status, dict)
    assert isinstance(engine.status, dict)
    assert not engine.running
    assert recognizer.calls == []
    assert tracker.calls == []
    assert segments(events) == []
    assert status["oldest_audio_queue_age_ms"] == 0
    assert status["oldest_segment_queue_age_ms"] == 0


def test_remote_speech_is_transcribed_and_attributed(engine_factory):
    engine, recognizer, tracker, events = engine_factory()
    engine.feed_fixture(voice(2), source="remote")
    engine.stop(timeout=3)

    observed = segments(events)
    assert recognizer.calls
    assert tracker.calls
    assert observed
    assert all(event["text"] == "fixture speech" for event in observed)
    assert all(event["source"] == "remote" for event in observed)
    assert all(event["speaker"] == "Remote 1" for event in observed)
    assert all(event["confidence"] == pytest.approx(0.9) for event in observed)
    assert all(0 <= event["start"] < event["end"] for event in observed)
    assert all(event["latency_ms"] >= 0 for event in observed)
    assert all(event["caption_delay_ms"] is None for event in observed)
    assert all(event["caption_delay_valid"] is False for event in observed)
    assert all(event["caption_delay_from_speech_end_ms"] is None for event in observed)
    assert all(event["caption_delay_from_utterance_start_ms"] is None for event in observed)
    assert all(audio.dtype == np.float32 and audio.ndim == 1 for audio in recognizer.calls)


def test_paced_fixture_reports_caption_delay_including_silence_wait(engine_factory):
    engine, _, _, events = engine_factory()
    audio = np.concatenate([
        voice(0.4), np.zeros(round(0.6 * SAMPLE_RATE), dtype=np.float32)
    ])
    engine.feed_fixture(audio, source="local", realtime=True)
    engine.stop(timeout=3)

    observed = segments(events)
    assert observed
    assert all(event["caption_delay_valid"] is True for event in observed)
    assert all(np.isfinite(event["caption_delay_ms"]) for event in observed)
    assert all(event["caption_delay_ms"] >= event["latency_ms"] >= 0 for event in observed)


def test_local_speech_bypasses_speaker_tracker(engine_factory):
    engine, recognizer, tracker, events = engine_factory()
    engine.feed_fixture(voice(2), source="local")
    engine.stop(timeout=3)

    observed = segments(events)
    assert recognizer.calls
    assert observed
    assert tracker.calls == []
    assert all(event["source"] == "local" for event in observed)
    assert all(event["speaker"] == "Local" for event in observed)


def test_short_remote_speech_is_unknown_without_embedding(engine_factory):
    engine, recognizer, tracker, events = engine_factory()
    engine.feed_fixture(voice(0.4), source="remote")
    engine.stop(timeout=3)

    observed = segments(events)
    assert recognizer.calls
    assert observed
    assert tracker.calls == []
    assert all(event["speaker"] == "Unknown" for event in observed)


def test_interleaved_sources_keep_independent_speaker_labels(engine_factory):
    engine, recognizer, tracker, events = engine_factory()
    audio = voice(2)
    for start in range(0, len(audio), 1600):
        block = audio[start : start + 1600]
        engine.feed_fixture(block, source="local")
        engine.feed_fixture(block * 0.8, source="remote")
    engine.stop(timeout=3)

    observed = segments(events)
    assert len(recognizer.calls) == 2
    assert len(tracker.calls) == 1
    assert {(event["source"], event["speaker"]) for event in observed} == {
        ("local", "Local"), ("remote", "Remote 1")
    }


def test_speaker_failure_preserves_transcript_with_unknown_label(engine_factory):
    class FailingTracker(TrackerStub):
        def classify(self, audio):
            raise RuntimeError("synthetic embedding failure")

    engine, _, _, events = engine_factory(tracker=FailingTracker())
    engine.feed_fixture(voice(2))
    engine.stop(timeout=3)

    observed = segments(events)
    assert observed
    assert all(event["speaker"] == "Unknown" for event in observed)
    assert all(event["text"] == "fixture speech" for event in observed)
    assert any(event.get("warning") for event in observed)


def test_stop_flushes_final_phrase_and_restart_resets_session(engine_factory):
    engine, recognizer, tracker, events = engine_factory()
    for run in range(2):
        engine.start()
        assert engine.running
        before = len(segments(events))
        engine.feed_fixture(voice(1.2), source="local")
        engine.stop(timeout=3)
        assert not engine.running
        assert len(segments(events)) > before
        assert len(recognizer.calls) == run + 1
    assert tracker.resets >= 2


def test_unavailable_selected_device_reports_error_and_allows_restart(
    engine_factory, monkeypatch
):
    class UnavailableCapture:
        def __init__(self, device_id, source, *args):
            self.device_id, self.source = device_id, source
            self.closed = threading.Event()
            self.closed.set()

        def start(self):
            raise OSError("synthetic device unavailable")

        def stop(self, timeout=2):
            self.closed.set()
            return True

    module = importlib.import_module("meeting.engine")
    monkeypatch.setattr(module, "AudioCapture", UnavailableCapture)
    engine, _, _, events = engine_factory()
    status = engine.start(mic_id=123)

    assert status["state"] == "error"
    assert not engine.running
    assert any(event.get("type") == "warning" for event in events)
    engine.start()
    engine.feed_fixture(voice(0.4), source="local")
    engine.stop(timeout=3)
    assert segments(events)
    assert not engine.running


def test_mid_session_disconnect_closes_all_sources_and_recordings(
    engine_factory, monkeypatch, tmp_path
):
    captures = []

    class FakeCapture:
        def __init__(self, device_id, source, started_at, on_audio, on_warning, on_error):
            self.device_id, self.source = device_id, source
            self.on_audio, self.on_error = on_audio, on_error
            self.active = False
            self.closed = threading.Event()
            self.closed.set()
            captures.append(self)

        def start(self):
            self.closed.clear()
            self.active = True

        def stop(self, timeout=2):
            self.active = False
            self.closed.set()
            return True

    shutdown_seen = threading.Event()

    def receive_event(event):
        if event.get("type") == "status" and event.get("state") == "error":
            shutdown_seen.set()

    module = importlib.import_module("meeting.engine")
    monkeypatch.setattr(module, "AudioCapture", FakeCapture)
    engine, _, _, events = engine_factory(on_event=receive_event)
    engine.start(mic_id=123, loopback_id=456, retain_audio=True)
    assert engine.running
    assert len(captures) == 2
    audio = voice(0.4)
    for capture in captures:
        assert capture.active
        capture.on_audio(capture.source, audio, 0.0)

    captures[0].on_error("synthetic microphone disconnect")
    assert shutdown_seen.wait(timeout=3), "device failure did not finish shutdown"
    assert not engine.running
    assert engine.status["state"] == "error"
    assert all(capture.closed.is_set() and not capture.active for capture in captures)
    assert any(event.get("type") == "warning" for event in events)

    paths = list(tmp_path.rglob("*.wav"))
    assert {path.name for path in paths} == {"local.wav", "remote.wav"}
    for path in paths:
        with wave.open(str(path), "rb") as recording:
            assert recording.getnchannels() == 1
            assert recording.getsampwidth() == 2
            assert recording.getframerate() == SAMPLE_RATE
            assert recording.getnframes() == len(audio)
            assert len(recording.readframes(len(audio))) == len(audio) * 2
        renamed = path.with_suffix(".closed-check")
        path.rename(renamed)
        renamed.rename(path)

    engine.start()
    assert engine.running
    engine.stop(timeout=3)
    assert not engine.running


def test_long_speech_is_split_into_bounded_chunks(engine_factory):
    engine, recognizer, _, events = engine_factory()
    audio = voice(13)
    engine.feed_fixture(audio, source="local")
    engine.stop(timeout=3)

    assert len(recognizer.calls) >= 3
    assert all(len(chunk) <= round(6.02 * SAMPLE_RATE) for chunk in recognizer.calls)
    assert sum(len(chunk) for chunk in recognizer.calls) == len(audio)
    observed = segments(events)
    assert len(observed) == len(recognizer.calls)
    assert all(first["end"] <= second["start"] for first, second in zip(observed, observed[1:]))
    assert len(observed) == 3
    assert len({event["utterance_id"] for event in observed}) == 1
    assert [event["chunk_index"] for event in observed] == [0, 1, 2]
    assert [event["first_in_utterance"] for event in observed] == [True, False, False]
    assert [event["final_in_utterance"] for event in observed] == [False, False, True]
    assert [event["boundary_reason"] for event in observed] == ["max_duration", "max_duration", "stop"]
    assert all(event["utterance_start"] == pytest.approx(0, abs=0.021) for event in observed)
    assert all(event["caption_kind"] == "stable_chunk" and event["is_partial"] is False for event in observed)


def test_inference_failure_does_not_strand_later_speech(engine_factory):
    class FailOnceRecognizer(RecognizerStub):
        attempts = 0

        def transcribe(self, audio):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("synthetic ASR failure")
            return super().transcribe(audio)

    recognizer = FailOnceRecognizer()
    engine, _, _, events = engine_factory(recognizer=recognizer)
    audio = np.concatenate([
        voice(1.2), np.zeros(SAMPLE_RATE, dtype=np.float32), voice(1.2)
    ])
    engine.feed_fixture(audio, source="local")
    engine.stop(timeout=3)

    assert recognizer.attempts == 2
    assert len(segments(events)) == 1
    assert segments(events)[0]["text"] == "fixture speech"
    assert any(event.get("type") == "warning" for event in events)
    assert not engine.running


def test_stop_is_bounded_and_restart_waits_for_blocked_inference(engine_factory):
    recognizer = BlockingRecognizer()
    engine, _, _, _ = engine_factory(recognizer=recognizer)
    phrase = np.concatenate([voice(1.2), np.zeros(SAMPLE_RATE, dtype=np.float32)])
    engine.feed_fixture(phrase)
    assert recognizer.entered.wait(timeout=3), "synthetic phrase did not reach ASR"

    started = time.monotonic()
    status = engine.stop(timeout=0.05)
    elapsed = time.monotonic() - started
    assert isinstance(status, dict)
    assert elapsed < 1.0, f"bounded stop blocked for {elapsed:.3f}s"
    with pytest.raises(RuntimeError):
        engine.start()

    recognizer.release.set()
    engine.stop(timeout=3)
    engine.start()
    assert engine.running
    engine.stop(timeout=3)
    assert not engine.running


def test_retention_is_opt_in_and_waves_are_closed_after_stop(engine_factory, tmp_path):
    engine, _, _, _ = engine_factory()
    engine.feed_fixture(voice(0.4), source="local")
    engine.stop(timeout=3)
    assert list(tmp_path.rglob("*.wav")) == []

    engine.start(retain_audio=True)
    audio = voice(1.2)
    engine.feed_fixture(audio, source="local")
    engine.stop(timeout=3)
    paths = list(tmp_path.rglob("*.wav"))
    assert paths, "retention must produce at least one WAV"
    frame_count = 0
    for path in paths:
        with wave.open(str(path), "rb") as recorded:
            assert recorded.getnchannels() == 1
            assert recorded.getsampwidth() == 2
            assert recorded.getframerate() == SAMPLE_RATE
            frames = recorded.getnframes()
            assert len(recorded.readframes(frames)) == frames * 2
            frame_count += frames
        # On Windows, a still-open writer prevents this rename.
        renamed = path.with_suffix(".closed-check")
        path.rename(renamed)
        renamed.rename(path)
    assert frame_count == len(audio)
    snapshots = {path: path.read_bytes() for path in paths}

    engine.start(retain_audio=False)
    engine.feed_fixture(voice(0.4), source="local")
    engine.stop(timeout=3)
    assert set(tmp_path.rglob("*.wav")) == set(paths)
    assert all(path.read_bytes() == snapshot for path, snapshot in snapshots.items())


def test_backpressure_is_reported_without_blocking_fixture_producer(
    engine_factory, monkeypatch
):
    engine_type = importlib.import_module("meeting.engine").Engine
    monkeypatch.setattr(engine_type, "SEGMENT_QUEUE_SIZE", 1)
    monkeypatch.setattr(engine_type, "MAX_SEGMENT_SECONDS", 0.25)
    recognizer = BlockingRecognizer()
    warning_seen = threading.Event()

    def receive_event(event):
        message = event.get("message", "").lower()
        if event.get("type") == "warning" and any(
            word in message for word in ("queue", "drop", "backpressure")
        ):
            warning_seen.set()

    engine, _, _, _ = engine_factory(recognizer=recognizer, on_event=receive_event)
    engine.feed_fixture(voice(0.5))
    assert recognizer.entered.wait(timeout=3)

    started = time.monotonic()
    engine.feed_fixture(voice(3))
    assert time.monotonic() - started < 1.0
    assert warning_seen.wait(timeout=3), "queue saturation was not reported"
    engine.stop(timeout=0.05)
    recognizer.release.set()
    engine.stop(timeout=3)


def test_paced_speech_delay_uses_energy_end_independently_of_asr_timestamps(engine_factory):
    class ShortTimestampRecognizer(RecognizerStub):
        def transcribe(self, audio):
            self.calls.append(np.array(audio, copy=True))
            return [{"start": 0.0, "end": 0.01, "text": "fixture speech"}]

    engine, _, _, events = engine_factory(recognizer=ShortTimestampRecognizer())
    audio = np.concatenate([voice(0.4), np.zeros(round(0.6 * SAMPLE_RATE), dtype=np.float32)])
    engine.feed_fixture(audio, source="local", realtime=True)
    engine.stop(timeout=3)

    assert len(segments(events)) == 1
    event = segments(events)[0]
    assert event["caption_delay_valid"] is True
    assert event["end"] == pytest.approx(0.01, abs=0.001)
    assert event["utterance_start"] == pytest.approx(0.0, abs=0.001)
    assert event["speech_last_voiced_end"] == pytest.approx(0.4, abs=0.001)
    assert event["clip_start"] == pytest.approx(0.0, abs=0.001)
    assert event["clip_end"] > event["speech_last_voiced_end"]
    # The silence gate is 500 ms. Waiting before enqueue avoids a 50 ms head start.
    assert event["clip_submitted_at"] - event["speech_last_voiced_end"] >= 0.49
    assert event["emitted_at"] >= event["worker_started_at"] >= event["clip_submitted_at"]
    speech_delay = event["caption_delay_from_speech_end_ms"]
    utterance_delay = event["caption_delay_from_utterance_start_ms"]
    assert speech_delay == pytest.approx(
        (event["emitted_at"] - event["speech_last_voiced_end"]) * 1000, abs=2
    )
    assert utterance_delay == pytest.approx(
        (event["emitted_at"] - event["utterance_start"]) * 1000, abs=2
    )
    assert event["caption_delay_ms"] - speech_delay == pytest.approx(390, abs=2)
    assert utterance_delay - speech_delay == pytest.approx(400, abs=2)
    assert event["boundary_reason"] == "silence"


def test_asr_speaker_and_worker_durations_are_separate(engine_factory, monkeypatch):
    # Windows/Python 3.12 monotonic uses GetTickCount64 (~15.625 ms ticks).
    # A real 15 ms sleep can therefore measure zero. Exercise the measurement
    # contract with an engine-local clock rather than relying on timer phase or
    # OS scheduling; queue waits and other tests keep their real system clocks.
    class ControlledClock:
        def __init__(self):
            self.value = 100.0

        def monotonic(self):
            return self.value

        def advance(self, seconds):
            self.value += seconds

    clock = ControlledClock()
    module = importlib.import_module("meeting.engine")
    monkeypatch.setattr(module, "time", types.SimpleNamespace(monotonic=clock.monotonic))

    class SlowRecognizer(RecognizerStub):
        def transcribe(self, audio):
            clock.advance(0.025)
            return super().transcribe(audio)

    class SlowTracker(TrackerStub):
        def classify(self, audio):
            clock.advance(0.015)
            return super().classify(audio)

    engine, _, _, events = engine_factory(recognizer=SlowRecognizer(), tracker=SlowTracker())
    engine.feed_fixture(voice(1.2), source="remote")
    engine.stop(timeout=3)

    assert len(segments(events)) == 1
    event = segments(events)[0]
    assert event["asr_ms"] == pytest.approx(25, abs=0.001)
    assert event["speaker_ms"] == pytest.approx(15, abs=0.001)
    assert event["worker_ms"] == pytest.approx(40, abs=0.001)
    assert event["worker_ms"] >= event["asr_ms"] + event["speaker_ms"] - 0.5
    assert event["queue_wait_ms"] >= 0
    assert event["latency_ms"] == pytest.approx(event["queue_wait_ms"] + event["worker_ms"], abs=0.5)
    assert event["queue_wait_ms"] == pytest.approx(
        (event["worker_started_at"] - event["clip_submitted_at"]) * 1000, abs=2
    )
    assert event["worker_ms"] == pytest.approx(
        (event["emitted_at"] - event["worker_started_at"]) * 1000, abs=2
    )
    assert event["caption_delay_from_speech_end_ms"] is None
    assert event["caption_delay_from_utterance_start_ms"] is None


def test_silence_starts_a_new_utterance_with_a_new_onset(engine_factory):
    engine, _, _, events = engine_factory()
    audio = np.concatenate([voice(0.4), np.zeros(round(0.7 * SAMPLE_RATE), dtype=np.float32), voice(0.4)])
    engine.feed_fixture(audio, source="local")
    engine.stop(timeout=3)

    observed = segments(events)
    assert len(observed) == 2
    first, second = observed
    assert first["utterance_id"] != second["utterance_id"]
    assert first["utterance_start"] == pytest.approx(0, abs=0.001)
    assert second["utterance_start"] == pytest.approx(1.1, abs=0.001)
    assert first["speech_last_voiced_end"] == pytest.approx(0.4, abs=0.001)
    assert second["speech_last_voiced_end"] == pytest.approx(1.5, abs=0.001)
    assert [event["boundary_reason"] for event in observed] == ["silence", "stop"]
    assert all(event["chunk_index"] == 0 for event in observed)
    assert all(event["first_in_utterance"] and event["final_in_utterance"] for event in observed)


def test_first_caption_tracks_emission_when_an_earlier_chunk_has_no_text(engine_factory, monkeypatch):
    class EmptyFirstRecognizer(RecognizerStub):
        def transcribe(self, audio):
            result = super().transcribe(audio)
            return [] if len(self.calls) == 1 else result

    engine_type = importlib.import_module("meeting.engine").Engine
    monkeypatch.setattr(engine_type, "MAX_SEGMENT_SECONDS", 0.4)
    engine, _, _, events = engine_factory(recognizer=EmptyFirstRecognizer())
    engine.feed_fixture(voice(1.05), source="local")
    engine.stop(timeout=3)

    observed = segments(events)
    assert len(observed) == 2
    assert [event["chunk_index"] for event in observed] == [1, 2]
    assert [event["first_in_utterance"] for event in observed] == [True, False]
    assert [event["final_in_utterance"] for event in observed] == [False, True]
    assert len({event["utterance_id"] for event in observed}) == 1


def test_only_last_asr_result_finishes_an_utterance(engine_factory):
    class MultipleResultsRecognizer(RecognizerStub):
        def transcribe(self, audio):
            return [
                {"start": 0.0, "end": 0.2, "text": "first"},
                {"start": 0.2, "end": 0.4, "text": "second"},
                {"start": 0.4, "end": 0.4, "text": "   "},
            ]

    engine, _, _, events = engine_factory(recognizer=MultipleResultsRecognizer())
    engine.feed_fixture(voice(0.4), source="local")
    engine.stop(timeout=3)

    observed = segments(events)
    assert len(observed) == 2
    assert [event["first_in_utterance"] for event in observed] == [True, False]
    assert [event["final_in_utterance"] for event in observed] == [False, True]
    assert len({event["utterance_id"] for event in observed}) == 1
    assert all(event["chunk_index"] == 0 for event in observed)


def test_gap_boundary_is_incomplete_and_resets_utterance_tracking():
    segmenter_type = importlib.import_module("meeting.engine")._Segmenter
    clips = []
    segmenter = segmenter_type("local", clips.append)
    segmenter.push(voice(0.4), 10.0)
    segmenter.push(voice(0.4), 11.0)
    segmenter.flush()

    assert len(clips) == 2
    assert [clip.boundary_reason for clip in clips] == ["gap", "stop"]
    assert [clip.final_in_utterance for clip in clips] == [False, True]
    assert clips[0].utterance_id != clips[1].utterance_id
    assert clips[0].utterance_start == pytest.approx(10.0)
    assert clips[1].utterance_start == pytest.approx(11.0)


def test_segment_queue_age_and_queue_wait_reflect_blocked_asr(engine_factory):
    recognizer = BlockingRecognizer()
    engine, _, _, events = engine_factory(recognizer=recognizer)
    silence = np.zeros(round(0.6 * SAMPLE_RATE), dtype=np.float32)
    engine.feed_fixture(np.concatenate([voice(1.2), silence]), source="local")
    assert recognizer.entered.wait(timeout=3)
    engine.feed_fixture(np.concatenate([voice(0.4), silence]), source="local")
    wait_until(lambda: engine.status["queued_segments"] >= 1 and engine.status["queued_audio"] == 0)
    before = engine.status["oldest_segment_queue_age_ms"]
    time.sleep(0.035)
    assert engine.status["oldest_segment_queue_age_ms"] >= before + 20
    assert engine.status["oldest_audio_queue_age_ms"] == 0

    recognizer.release.set()
    status = engine.stop(timeout=3)
    observed = segments(events)
    assert len(observed) == 2
    assert observed[1]["queue_wait_ms"] >= 20
    assert observed[1]["clip_submitted_at"] < observed[0]["emitted_at"]
    assert observed[1]["worker_started_at"] >= observed[0]["emitted_at"]
    assert status["oldest_audio_queue_age_ms"] == status["oldest_segment_queue_age_ms"] == 0


def test_audio_queue_age_reflects_waiting_frames(engine_factory, monkeypatch):
    module = importlib.import_module("meeting.engine")
    original_push = module._Segmenter.push
    entered, release = threading.Event(), threading.Event()

    def held_push(self, samples, start):
        if not entered.is_set():
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("synthetic segmenter was not released")
        return original_push(self, samples, start)

    monkeypatch.setattr(module._Segmenter, "push", held_push)
    engine, _, _, _ = engine_factory()
    try:
        engine.feed_fixture(voice(0.3), source="local")
        assert entered.wait(timeout=3)
        assert engine.status["queued_audio"] >= 1
        before = engine.status["oldest_audio_queue_age_ms"]
        time.sleep(0.035)
        assert engine.status["oldest_audio_queue_age_ms"] >= before + 20
        assert engine.status["oldest_segment_queue_age_ms"] == 0
    finally:
        release.set()
        status = engine.stop(timeout=3)
    assert status["oldest_audio_queue_age_ms"] == status["oldest_segment_queue_age_ms"] == 0
