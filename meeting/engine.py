"""Bounded offline meeting pipeline with independent local and remote channels."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import queue
import threading
import time
import uuid
import wave

import numpy as np

from .audio import AudioCapture, list_devices

SAMPLE_RATE = 16000


@dataclass
class _Clip:
    source: str
    start: float
    audio: np.ndarray
    submitted_at: float
    warning: str = ""
    speech_last_voiced_end: float | None = None
    utterance_start: float | None = None
    utterance_id: int = 0
    chunk_index: int = 0
    final_in_utterance: bool = True
    boundary_reason: str = "stop"

    @property
    def end(self):
        return self.start + len(self.audio) / SAMPLE_RATE


class _Segmenter:
    """Energy VAD with onset, adaptive floor, preroll, hangover and bounded clips."""

    def __init__(self, source, emit, max_seconds=6.0, silence_seconds=0.5):
        self.source, self.emit = source, emit
        self.max_samples = int(max_seconds * SAMPLE_RATE)
        self.silence_samples = int(silence_seconds * SAMPLE_RATE)
        self.preroll = deque(maxlen=10)
        self.frames = []
        self.start = 0.0
        self.total = self.voiced = self.quiet = self.onset = 0
        self.noise = 0.0015
        self.expected = None
        self.speech_last_voiced_end = None
        self.utterance_start = None
        self.utterance_id = 0
        self.chunk_index = 0
        self._utterance_counter = 0
        self._onset_start = None
        self._continuation_quiet = 0

    def _end_utterance(self):
        self.utterance_start = None
        self.utterance_id = 0
        self.chunk_index = 0
        self._continuation_quiet = 0

    def push(self, samples, start):
        # Gaps are never concatenated into a fictitiously continuous utterance.
        if self.expected is not None and abs(start - self.expected) > 0.025:
            self.flush("gap")
            self.preroll.clear()
            self.onset = 0
        for index in range(0, len(samples), 320):
            block = samples[index:index + 320]
            at = start + index / SAMPLE_RATE
            rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) if len(block) else 0.0
            threshold = max(0.006 if not self.frames else 0.004, min(0.025, self.noise * 3.0))
            voiced = rms >= threshold
            if voiced:
                self.speech_last_voiced_end = at + len(block) / SAMPLE_RATE
            if not voiced and not self.frames:
                self.noise = 0.98 * self.noise + 0.02 * min(rms, 0.008)
            if not self.frames:
                # A forced chunk can be followed by silence without creating
                # another ASR clip. End its metadata continuity once the same
                # silence boundary is reached; keep the existing VAD behavior.
                if self.utterance_start is not None:
                    self._continuation_quiet = 0 if voiced else self._continuation_quiet + len(block)
                    if self._continuation_quiet >= self.silence_samples:
                        self._end_utterance()
                self.preroll.append((at, block.copy()))
                if voiced and self.onset == 0:
                    self._onset_start = at
                elif not voiced:
                    self._onset_start = None
                self.onset = self.onset + len(block) if voiced else 0
                if self.onset < int(0.06 * SAMPLE_RATE):
                    continue
                self.start = self.preroll[0][0]
                if self.utterance_start is None:
                    self._utterance_counter += 1
                    self.utterance_id = self._utterance_counter
                    self.utterance_start = self._onset_start if self._onset_start is not None else at
                    self.chunk_index = 0
                self.frames = [part for _, part in self.preroll]
                self.total = sum(len(part) for part in self.frames)
                self.voiced = self.onset
                self.quiet = 0
                self.preroll.clear()
                continue
            self.frames.append(block.copy())
            self.total += len(block)
            self.voiced += len(block) if voiced else 0
            self.quiet = 0 if voiced else self.quiet + len(block)
            if self.quiet >= self.silence_samples:
                self.flush("silence")
            elif self.total >= self.max_samples:
                self.flush("max_duration")
        self.expected = start + len(samples) / SAMPLE_RATE

    def flush(self, boundary_reason="stop"):
        if self.frames and self.voiced >= int(0.18 * SAMPLE_RATE):
            audio = np.concatenate(self.frames)
            trim = max(0, self.quiet - int(0.16 * SAMPLE_RATE))
            if trim:
                audio = audio[:-trim]
            self.emit(_Clip(
                self.source, self.start, audio, time.monotonic(),
                speech_last_voiced_end=self.speech_last_voiced_end,
                utterance_start=self.utterance_start, utterance_id=self.utterance_id,
                chunk_index=self.chunk_index,
                final_in_utterance=boundary_reason in {"silence", "stop"},
                boundary_reason=boundary_reason,
            ))
            self.chunk_index += 1
        if boundary_reason == "max_duration":
            self._continuation_quiet = self.quiet
        else:
            self._end_utterance()
        self.frames = []
        self.total = self.voiced = self.quiet = self.onset = 0
        self.speech_last_voiced_end = None
        self._onset_start = None


class Engine:
    AUDIO_QUEUE_SIZE = 240
    SEGMENT_QUEUE_SIZE = 12
    MAX_SEGMENT_SECONDS = 6.0
    SILENCE_SECONDS = 0.5

    def __init__(self, recognizer, speaker_tracker, on_event, data_dir: Path):
        self.recognizer, self.speaker_tracker = recognizer, speaker_tracker
        self.on_event, self.data_dir = on_event, Path(data_dir)
        self._lock = threading.RLock()
        self._lifecycle = threading.RLock()
        self._stats = {"state": "idle", "message": "Ready; audio capture is off.", "running": False}
        self._threads = []
        self._captures = []
        self._audio_queue = queue.Queue(self.AUDIO_QUEUE_SIZE)
        self._segment_queue = queue.Queue(self.SEGMENT_QUEUE_SIZE)
        self._accepting = False
        self._started_at = None
        self._fixture_positions = {"local": 0, "remote": 0}
        self._fixture_paced = True
        self._last_warning = {}
        self._audio_done = threading.Event()
        self._segment_done = threading.Event()
        self._inference_done = threading.Event()
        self._cancel_pending = threading.Event()
        self._recordings = {}

    @property
    def running(self):
        with self._lock:
            return bool(self._stats["running"])

    @property
    def status(self):
        with self._lock:
            result = dict(self._stats)
            result["queued_audio"] = self._audio_queue.qsize()
            result["queued_segments"] = self._segment_queue.qsize()
            result["oldest_audio_queue_age_ms"] = self._oldest_queue_age(self._audio_queue, is_audio=True)
            result["oldest_segment_queue_age_ms"] = self._oldest_queue_age(self._segment_queue, is_audio=False)
            result["elapsed_seconds"] = round(time.monotonic() - self._started_at, 2) if self.running and self._started_at else self._stats.get("elapsed_seconds", 0)
            return result

    @staticmethod
    def _oldest_queue_age(work_queue, is_audio):
        # Peek under Queue's own mutex without removing or copying the buffer.
        # This measures pending work only; in-flight inference is not queued.
        with work_queue.mutex:
            if not work_queue.queue:
                return 0.0
            oldest = work_queue.queue[0]
            submitted_at = oldest[3] if is_audio else oldest.submitted_at
        return round(max(0.0, time.monotonic() - submitted_at) * 1000, 3)

    def _event(self, event):
        try:
            self.on_event(event)
        except Exception:
            # A UI callback must not terminate acquisition or strand open WAVs.
            with self._lock:
                self._stats["event_callback_errors"] = self._stats.get("event_callback_errors", 0) + 1

    def _state(self, state, message):
        with self._lock:
            self._stats.update(state=state, message=message)
        self._event({"type": "status", "state": state, "message": message})

    def _warning(self, message, key=None):
        now = time.monotonic()
        with self._lock:
            if key and now - self._last_warning.get(key, -float("inf")) < 2:
                return
            if key:
                self._last_warning[key] = now
            self._stats["last_warning"] = message
        self._event({"type": "warning", "message": message})

    def _capture_error(self, message):
        self._warning(message)
        if self.running and self._accepting:
            # Stop every source after a mid-session failure: a seemingly healthy
            # second source must not hide that half of the meeting is missing.
            with self._lock:
                self._stats["device_error"] = message
            self._accepting = False
            self._audio_done.set()
            self._state("stopping", message + " All capture is stopping; refresh devices and restart.")

    def start(self, mic_id=None, loopback_id=None, retain_audio=False):
        """Begin only on explicit call. Both IDs None creates a fixture-only run."""
        with self._lifecycle:
            if any(t.is_alive() for t in self._threads) or any(not c.closed.is_set() for c in self._captures):
                raise RuntimeError("The current session is still running or stopping; wait for it to finish before restarting.")
            self._audio_queue = queue.Queue(self.AUDIO_QUEUE_SIZE)
            self._segment_queue = queue.Queue(self.SEGMENT_QUEUE_SIZE)
            self._audio_done.clear()
            self._segment_done.clear()
            self._inference_done.clear()
            self._cancel_pending.clear()
            self._captures, self._recordings = [], {}
            self._fixture_positions = {"local": 0, "remote": 0}
            self._fixture_paced = True
            self._last_warning = {}
            self._started_at = time.monotonic()
            self._stats = {
                "state": "starting", "message": "Starting offline pipeline.", "running": True,
                "retain_audio": bool(retain_audio), "recording_paths": {},
                "dropped_audio_blocks": 0, "dropped_audio_seconds": 0.0,
                "dropped_segments": 0, "processed_segments": 0, "event_callback_errors": 0,
                "fixture_mode": mic_id is None and loopback_id is None,
                "inference_busy": False,
            }
            try:
                self.speaker_tracker.reset()
                if retain_audio:
                    directory = self.data_dir / "recordings" / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
                    directory.mkdir(parents=True, exist_ok=False)
                    for source in ("local", "remote"):
                        path = directory / f"{source}.wav"
                        writer = wave.open(str(path), "wb")
                        writer.setnchannels(1)
                        writer.setsampwidth(2)
                        writer.setframerate(SAMPLE_RATE)
                        self._recordings[source] = (writer, 0)
                        self._stats["recording_paths"][source] = str(path)
            except Exception:
                self._close_recordings()
                self._stats["running"] = False
                self._state("error", "Could not initialize session; audio capture is off.")
                raise
            self._accepting = True
            self._threads = [
                threading.Thread(target=self._segment_worker, name="meeting-segmentation", daemon=True),
                threading.Thread(target=self._inference_worker, name="meeting-inference", daemon=True),
            ]
            for thread in self._threads:
                thread.start()
            failures = []
            for source, device_id in (("local", mic_id), ("remote", loopback_id)):
                if device_id is None:
                    continue
                capture = AudioCapture(device_id, source, self._started_at, self._enqueue_audio, self._warning, self._capture_error)
                self._captures.append(capture)
                try:
                    capture.start()
                except Exception as exc:
                    failures.append(f"{source}: cannot open selected audio device ({exc}). Refresh devices and try again.")
                    self._warning(failures[-1])
            requested = sum(item is not None for item in (mic_id, loopback_id))
            if requested and len(failures) == requested:
                self.stop()
                self._state("error", "No selected audio device could be opened. Capture is off; refresh devices and restart.")
                return self.status
            if self._captures:
                monitor = threading.Thread(target=self._capture_monitor, name="meeting-device-monitor", daemon=True)
                self._threads.append(monitor)
                monitor.start()
            if self._accepting:
                self._state("degraded" if failures else "running", "Offline session running." if requested else "Synthetic fixture session running; audio devices are off.")
            return self.status

    def _capture_monitor(self):
        reported = set()
        while not self._audio_done.wait(0.25):
            for capture in self._captures:
                if capture not in reported and not capture.closed.is_set() and not capture.active:
                    reported.add(capture)
                    self._capture_error(f"{capture.source}: audio device stopped responding. Capture from this source is unavailable; refresh devices and restart.")
                    capture.stop(timeout=0)
            if self._captures and all(c.closed.is_set() for c in self._captures):
                self._accepting = False
                self._audio_done.set()
                self._warning("All selected audio devices are unavailable. Capture has stopped; refresh devices and restart.")
        # Finish even if a driver needed longer than stop()'s bounded wait. This
        # monitor also closes native streams after an unexpected segmenter error.
        for capture in self._captures:
            capture.stop(timeout=0)
        for capture in self._captures:
            while not capture.closed.wait(0.25):
                pass
        self._finish_if_ready()

    def _enqueue_audio(self, source, audio, start):
        if not self._accepting:
            return False
        block = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not len(block):
            return True
        if not np.isfinite(block).all():
            block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
            self._warning(f"{source}: invalid audio samples were replaced by silence.", "invalid")
        try:
            self._audio_queue.put_nowait((source, block.copy(), float(start), time.monotonic()))
            return True
        except queue.Full:
            with self._lock:
                self._stats["dropped_audio_blocks"] += 1
                self._stats["dropped_audio_seconds"] += len(block) / SAMPLE_RATE
            self._warning("Audio input queue is full: audio was dropped. Timestamps preserve the gap; the transcript and any recording are incomplete.", "audio-backpressure")
            return False

    def feed_fixture(self, audio16k, source="remote", realtime=False):
        """Feed synthetic 16 kHz mono audio; never enumerates or opens devices.

        Fast mode waits for queue space, so benchmark fixtures are lossless.
        Call stop() after the last fixture to flush and drain the utterances.
        """
        if source not in ("local", "remote"):
            raise ValueError("source must be 'local' or 'remote'")
        if not self.running:
            self.start()
        if not self._stats.get("fixture_mode"):
            raise RuntimeError("Fixture audio cannot be injected into a live capture session")
        if not self._accepting:
            raise RuntimeError("Session is stopping")
        audio = np.asarray(audio16k, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Fixture must be finite mono float audio at 16000 Hz")
        self._fixture_paced = self._fixture_paced and bool(realtime)
        began = time.monotonic()
        for index in range(0, len(audio), 800):
            if not self._accepting:
                break
            part = audio[index:index + 800]
            at = self._fixture_positions[source] / SAMPLE_RATE
            if realtime:
                # A device delivers a block after its samples have elapsed.
                # Waiting here avoids giving replay benchmarks 50 ms of future
                # audio before the corresponding speech-end timestamp.
                delay = began + (index + len(part)) / SAMPLE_RATE - time.monotonic()
                if delay > 0 and self._audio_done.wait(delay):
                    break
                self._enqueue_audio(source, part, at)
            else:
                while self._accepting:
                    try:
                        self._audio_queue.put((source, part.copy(), at, time.monotonic()), timeout=0.05)
                        break
                    except queue.Full:
                        continue
            self._fixture_positions[source] += len(part)

    def _record(self, source, block, start):
        item = self._recordings.get(source)
        if item is None:
            return
        writer, position = item
        wanted = max(0, round(start * SAMPLE_RATE))
        if wanted > position:
            # Bounded writes even after a long device gap; preserve the time axis.
            missing = wanted - position
            while missing:
                count = min(missing, SAMPLE_RATE)
                writer.writeframesraw(bytes(count * 2))
                missing -= count
        skip = max(0, position - wanted)
        data = (np.clip(block[skip:], -1, 1) * 32767).astype("<i2")
        writer.writeframesraw(data.tobytes())
        self._recordings[source] = (writer, max(position, wanted + len(block)))

    def _close_recordings(self):
        for writer, _ in list(self._recordings.values()):
            try:
                writer.close()
            except Exception as exc:
                self._warning(f"Audio recording could not be finalized: {exc}")
        self._recordings.clear()

    def _segment_worker(self):
        recent = {"local": deque(maxlen=20), "remote": deque(maxlen=20)}

        def submit(clip):
            if self._cancel_pending.is_set():
                with self._lock:
                    self._stats["dropped_segments"] += 1
                return
            other = "local" if clip.source == "remote" else "remote"
            overlap = any(min(old.end, clip.end) - max(old.start, clip.start) > 0.20 for old in recent[other])
            if overlap:
                clip.warning = "Microphone and remote audio overlap in time; this cross-channel timing check cannot identify overlapping voices inside the remote mix."
                self._warning(clip.warning, "overlap")
            recent[clip.source].append(clip)
            try:
                self._segment_queue.put_nowait(clip)
            except queue.Full:
                with self._lock:
                    self._stats["dropped_segments"] += 1
                self._warning("Transcription queue is full: a speech segment was dropped. The transcript is incomplete; use a faster model or device.", "segment-backpressure")

        segmenters = {source: _Segmenter(source, submit, self.MAX_SEGMENT_SECONDS, self.SILENCE_SECONDS) for source in ("local", "remote")}
        try:
            while not self._audio_done.is_set() or not self._audio_queue.empty():
                try:
                    source, audio, at, _queued_at = self._audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if self._cancel_pending.is_set():
                        with self._lock:
                            self._stats["dropped_audio_blocks"] += 1
                            self._stats["dropped_audio_seconds"] += len(audio) / SAMPLE_RATE
                        continue
                    try:
                        self._record(source, audio, at)
                    except Exception as exc:
                        self._warning(f"Audio retention failed ({exc}); recording files are being closed. Transcription continues.")
                        self._close_recordings()
                    segmenters[source].push(audio, at)
                finally:
                    self._audio_queue.task_done()
            for segmenter in segmenters.values():
                segmenter.flush()
        except Exception as exc:
            self._warning(f"Audio segmentation failed: {exc}. Stop and restart the session.")
            self._accepting = False
            self._audio_done.set()
            self._state("error", "Segmentation failed; audio input has been disabled.")
        finally:
            self._close_recordings()
            self._segment_done.set()
            self._finish_if_ready()

    def _speaker(self, clip):
        if clip.source == "local":
            return {"speaker": "Local", "confidence": None, "warning": ""}
        if len(clip.audio) < SAMPLE_RATE:
            return {"speaker": "Unknown", "confidence": None, "warning": "Remote speech is too short for reliable speaker attribution."}
        try:
            result = dict(self.speaker_tracker.classify(clip.audio))
            confidence = result.get("confidence")
            warning = str(result.get("warning") or "")
            if confidence is not None and float(confidence) < 0.50:
                result.update(speaker="Unknown", warning=warning or "Remote speaker attribution is uncertain.")
            if any(word in warning.lower() for word in ("unstable", "mixed", "overlap")):
                result["speaker"] = "Unknown"
            return result
        except Exception as exc:
            return {"speaker": "Unknown", "confidence": None, "warning": f"Speaker attribution unavailable: {exc}"}

    def _inference_worker(self):
        last_caption_utterance = {"local": None, "remote": None}
        try:
            while not self._segment_done.is_set() or not self._segment_queue.empty():
                try:
                    clip = self._segment_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                worker_started_at = time.monotonic()
                try:
                    if self._cancel_pending.is_set():
                        with self._lock:
                            self._stats["dropped_segments"] += 1
                        continue
                    with self._lock:
                        self._stats["inference_busy"] = True
                    asr_started_at = time.monotonic()
                    results = self.recognizer.transcribe(clip.audio)
                    asr_finished_at = time.monotonic()
                    if not results:
                        continue
                    speaker_started_at = time.monotonic()
                    speaker = self._speaker(clip)
                    speaker_finished_at = time.monotonic()
                    queue_wait_ms = (worker_started_at - clip.submitted_at) * 1000
                    asr_ms = (asr_finished_at - asr_started_at) * 1000
                    speaker_ms = (speaker_finished_at - speaker_started_at) * 1000
                    visible_results = [item for item in results if str(item.get("text", "")).strip()]
                    for result_index, item in enumerate(visible_results):
                        text = str(item.get("text", "")).strip()
                        duration = len(clip.audio) / SAMPLE_RATE
                        start = max(0.0, min(duration, float(item.get("start", 0))))
                        end = max(start, min(duration, float(item.get("end", duration))))
                        warning = " ".join(value for value in (clip.warning, str(speaker.get("warning") or "")) if value)
                        caption_delay_valid = not self._stats.get("fixture_mode") or self._fixture_paced
                        emitted_at = time.monotonic()
                        emitted_offset = emitted_at - self._started_at
                        worker_ms = (emitted_at - worker_started_at) * 1000
                        latency = round((emitted_at - clip.submitted_at) * 1000, 3)
                        # Legacy delay uses ASR-generated end timestamps. Keep
                        # it for existing consumers, but do not use it as the
                        # speech-end latency benchmark: model timestamps may be
                        # inconsistent with the actual captured voiced frames.
                        caption_delay = max(0.0, (emitted_offset - clip.start - end) * 1000) if caption_delay_valid else None
                        speech_end_delay = (
                            max(0.0, (emitted_offset - clip.speech_last_voiced_end) * 1000)
                            if caption_delay_valid and clip.speech_last_voiced_end is not None else None
                        )
                        onset_delay = (
                            max(0.0, (emitted_offset - clip.utterance_start) * 1000)
                            if caption_delay_valid and clip.utterance_start is not None else None
                        )
                        first_caption = last_caption_utterance[clip.source] != clip.utterance_id
                        last_caption_utterance[clip.source] = clip.utterance_id
                        timings = {
                            "queue_wait_ms": round(max(0.0, queue_wait_ms), 3),
                            "asr_ms": round(max(0.0, asr_ms), 3),
                            "speaker_ms": round(max(0.0, speaker_ms), 3),
                            "worker_ms": round(max(0.0, worker_ms), 3),
                            "caption_delay_ms": round(caption_delay, 3) if caption_delay is not None else None,
                            "caption_delay_from_speech_end_ms": round(speech_end_delay, 3) if speech_end_delay is not None else None,
                            "caption_delay_from_utterance_start_ms": round(onset_delay, 3) if onset_delay is not None else None,
                        }
                        self._event({
                            "type": "segment", "start": round(clip.start + start, 3),
                            "end": round(clip.start + end, 3), "text": text,
                            "source": clip.source, "speaker": speaker.get("speaker") or "Unknown",
                            "confidence": speaker.get("confidence"), "warning": warning, "latency_ms": latency,
                            "caption_delay_valid": caption_delay_valid,
                            # All exposed times below are seconds relative to
                            # this session, never absolute monotonic timestamps.
                            "clip_start": round(clip.start, 6),
                            "clip_end": round(clip.end, 6),
                            "clip_submitted_at": round(clip.submitted_at - self._started_at, 6),
                            "worker_started_at": round(worker_started_at - self._started_at, 6),
                            "emitted_at": round(emitted_offset, 6),
                            "speech_last_voiced_end": round(clip.speech_last_voiced_end, 6) if clip.speech_last_voiced_end is not None else None,
                            "utterance_start": round(clip.utterance_start, 6) if clip.utterance_start is not None else None,
                            "utterance_id": clip.utterance_id,
                            "chunk_index": clip.chunk_index,
                            "first_in_utterance": first_caption,
                            "final_in_utterance": clip.final_in_utterance and result_index == len(visible_results) - 1,
                            "clip_final_in_utterance": clip.final_in_utterance,
                            "boundary_reason": clip.boundary_reason,
                            "caption_kind": "stable_chunk", "is_partial": False,
                            "asr_segment_index": result_index, "asr_segment_count": len(visible_results),
                            **timings,
                        })
                        with self._lock:
                            self._stats["processed_segments"] += 1
                            self._stats["latency_ms"] = latency
                            self._stats.update(timings)
                except Exception as exc:
                    self._warning(f"Offline inference failed for {clip.source} at {clip.start:.1f}s: {exc}")
                    with self._lock:
                        self._stats["failed_segments"] = self._stats.get("failed_segments", 0) + 1
                finally:
                    with self._lock:
                        self._stats["inference_busy"] = False
                    self._segment_queue.task_done()
        finally:
            self._inference_done.set()
            self._finish_if_ready()

    def _finish_if_ready(self):
        if self._segment_done.is_set() and self._inference_done.is_set() and all(c.closed.is_set() for c in self._captures):
            with self._lock:
                if not self._stats.get("running") and self._stats.get("state") not in ("stopping", "running", "degraded"):
                    return
                self._stats["running"] = False
                self._stats["elapsed_seconds"] = round(time.monotonic() - self._started_at, 2)
            drops = self._stats.get("dropped_audio_blocks", 0) + self._stats.get("dropped_segments", 0)
            if self._stats.get("device_error"):
                self._state("error", "Capture stopped after an audio device error. Refresh devices and restart; the transcript may be incomplete.")
            else:
                self._state("stopped", "Session stopped; pending audio drained." if not drops else "Session stopped with dropped audio or speech segments; transcript is incomplete.")

    def stop(self, timeout=12.0):
        """Stop capture and drain within a bounded wait; never kill model threads.

        On timeout pending work is discarded with an explicit warning. An active
        inference call may finish later; restarting is blocked until it exits.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._lifecycle:
            if not self.running and not any(t.is_alive() for t in self._threads):
                return self.status
            self._state("stopping", "Capture stopping; draining pending offline transcription.")
            for capture in self._captures:
                capture.stop(timeout=min(2.0, max(0.0, deadline - time.monotonic())))
            self._accepting = False
            self._audio_done.set()
            for thread in self._threads:
                if thread is not threading.current_thread():
                    thread.join(max(0.0, deadline - time.monotonic()))
            self._finish_if_ready()
            if self.running:
                self._cancel_pending.set()
                self._warning("Stop drain timed out. Pending audio and speech segments will be discarded; in-flight inference may still finish. Restart is disabled until shutdown completes.", "stop-timeout")
                self._state("stopping", "Capture is off; waiting for in-flight work to finish. Restart is temporarily disabled.")
                # The worker may finish between the running check and the
                # timeout notification. Keep the final state authoritative.
                self._finish_if_ready()
            return self.status
