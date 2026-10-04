"""An accelerated synthetic hour; this is not a live-capture or model benchmark."""
from __future__ import annotations

import gc
import json
import time
import tracemalloc

import numpy as np

from meeting.engine import Engine


class _CountingRecognizer:
    """Keep scalar counters only, so the fixture cannot create a false leak."""

    def __init__(self):
        self.calls = 0
        self.nonzero_samples = 0
        self.max_samples = 0

    def transcribe(self, audio):
        self.calls += 1
        self.nonzero_samples += int(np.count_nonzero(audio))
        self.max_samples = max(self.max_samples, len(audio))
        return [{"start": 0.0, "end": len(audio) / 16000, "text": "synthetic phrase"}]


class _CountingTracker:
    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls = 0

    def classify(self, audio):
        self.calls += 1
        return {"speaker": "Remote 1", "confidence": 0.9, "warning": ""}


def _bounded_drain(engine, timeout=5.0):
    """Wait for audio before inference so a pending VAD clip is not missed."""
    deadline = time.monotonic() + timeout
    for work_queue in (engine._audio_queue, engine._segment_queue):
        # queue.join() has no timeout. Its condition lets test failures terminate
        # promptly instead of leaving pytest hung on an interrupted worker.
        with work_queue.all_tasks_done:
            complete = work_queue.all_tasks_done.wait_for(
                lambda: work_queue.unfinished_tasks == 0,
                timeout=max(0.0, deadline - time.monotonic()),
            )
        assert complete, f"Synthetic pipeline did not drain: {engine.status}"


def test_hour_equivalent_stream_is_bounded_and_conserves_voiced_samples(
    tmp_path, monkeypatch, record_property
):
    import meeting.audio as audio_module
    import meeting.engine as engine_module

    def forbid_capture(*args, **kwargs):
        raise AssertionError("Long-session validation must never open an audio device")

    monkeypatch.setattr(audio_module, "_pyaudio", forbid_capture)
    monkeypatch.setattr(engine_module, "AudioCapture", forbid_capture)

    # Reuse this 128 KB period for the whole hour. Never allocate an hour-sized
    # waveform, transcript list, retained recognition input, or recording file.
    rate = 16000
    repeats = 1800
    period_seconds = 2.0
    voice_seconds = 1.2
    times = np.arange(round(rate * voice_seconds), dtype=np.float64) / rate
    voiced = (0.12 * np.sin(2 * np.pi * 173 * times)).astype(np.float32)
    period = np.concatenate((voiced, np.zeros(round(rate * 0.8), dtype=np.float32)))
    expected_nonzero = int(np.count_nonzero(period)) * repeats
    del times, voiced

    recognizer, tracker = _CountingRecognizer(), _CountingTracker()
    counts = {"segments": 0, "warnings": 0, "last_end": 0.0, "unexpected_events": 0}

    def receive(event):
        if event["type"] == "segment":
            counts["segments"] += 1
            counts["last_end"] = event["end"]
            if event["source"] != "remote" or event["caption_delay_valid"]:
                counts["unexpected_events"] += 1
        elif event["type"] == "warning":
            counts["warnings"] += 1

    engine = Engine(recognizer, tracker, receive, tmp_path)
    high_water = {"audio": 0, "segments": 0}
    started = time.perf_counter()
    tracing_started_here = not tracemalloc.is_tracing()
    if tracing_started_here:
        tracemalloc.start()
    baseline = halfway = final_memory = peak_memory = None
    try:
        engine.start()
        for index in range(repeats):
            engine.feed_fixture(period, source="remote", realtime=False)
            status = engine.status
            high_water["audio"] = max(high_water["audio"], status["queued_audio"])
            high_water["segments"] = max(high_water["segments"], status["queued_segments"])
            assert status["queued_audio"] <= engine.AUDIO_QUEUE_SIZE
            assert status["queued_segments"] <= engine.SEGMENT_QUEUE_SIZE

            # Model the producer waiting for work in eight-utterance batches.
            # This is accelerated backpressure pacing, not real-time capture.
            # It allows meaningful conservation checks without intentionally
            # overwhelming a queue with an entire hour in one producer burst.
            if (index + 1) % 8 == 0 or index + 1 == repeats:
                _bounded_drain(engine)
                assert counts["segments"] == index + 1
                assert recognizer.nonzero_samples == int(np.count_nonzero(period)) * (index + 1)
                assert engine.status["dropped_audio_blocks"] == 0
                assert engine.status["dropped_segments"] == 0
                if index + 1 >= 24 and baseline is None:
                    gc.collect()
                    baseline = tracemalloc.get_traced_memory()[0]
                if index + 1 >= repeats // 2 and halfway is None:
                    gc.collect()
                    halfway = tracemalloc.get_traced_memory()[0]

        # Observe memory while the session is still running, so stop() cannot
        # conceal an accumulating live-session buffer by freeing it afterward.
        _bounded_drain(engine)
        gc.collect()
        final_memory, peak_memory = tracemalloc.get_traced_memory()
        final_status = engine.stop(timeout=5)
    finally:
        engine.stop(timeout=5)
        if tracing_started_here:
            tracemalloc.stop()

    wall_seconds = time.perf_counter() - started
    assert not engine.running
    assert final_status["state"] == "stopped"
    assert final_status["queued_audio"] == final_status["queued_segments"] == 0
    assert final_status["dropped_audio_blocks"] == final_status["dropped_segments"] == 0
    assert counts["warnings"] == counts["unexpected_events"] == 0
    assert recognizer.calls == tracker.calls == counts["segments"] == repeats
    assert recognizer.nonzero_samples == expected_nonzero
    assert recognizer.max_samples <= round(engine.MAX_SEGMENT_SECONDS * rate)
    assert engine._fixture_positions["remote"] == repeats * len(period)
    assert engine._audio_queue.maxsize == engine.AUDIO_QUEUE_SIZE
    assert engine._segment_queue.maxsize == engine.SEGMENT_QUEUE_SIZE
    assert 3599.0 < counts["last_end"] < 3600.0
    assert not list(tmp_path.rglob("*.wav")), "Retention must remain disabled"

    # This covers Python/NumPy pipeline allocations. It makes no claim about
    # native audio drivers, actual ASR models, or GPU/native allocator memory.
    mib = 1024 * 1024
    assert final_memory - baseline < 4 * mib
    assert final_memory - halfway < 2 * mib
    assert peak_memory - baseline < 30 * mib
    metrics = {
        "simulated_seconds": repeats * period_seconds,
        "wall_seconds": round(wall_seconds, 3),
        "utterances": counts["segments"],
        "input_samples": repeats * len(period),
        "voiced_samples_preserved": recognizer.nonzero_samples,
        "max_audio_queue_observed": high_water["audio"],
        "max_segment_queue_observed": high_water["segments"],
        "traced_growth_mib": round((final_memory - baseline) / mib, 3),
        "traced_second_half_growth_mib": round((final_memory - halfway) / mib, 3),
        "traced_peak_above_warmup_mib": round((peak_memory - baseline) / mib, 3),
        "live_capture_or_model_inference": False,
    }
    for name, value in metrics.items():
        record_property(name, value)
    print("LONG_SESSION_METRICS " + json.dumps(metrics, sort_keys=True))
