"""Paced, real-model fixture replay through Engine and optionally the native App.

This script never enumerates or opens recording devices. It is a measurement
harness, not a certification of transcription quality or safe GPU headroom.
Only the standard library is imported when its metric helpers are imported.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import traceback
import unicodedata
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def normalized_tokens(text: str, language: str = "en") -> list[str]:
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("’", "'").replace("‘", "'")
    if language == "zh":
        return [character for character in text if character.isalnum()]
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text, flags=re.UNICODE)


def edit_distance(reference: list, hypothesis: list) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def quality_score(expected: str, observed: str, language: str) -> dict:
    reference, hypothesis = normalized_tokens(expected, language), normalized_tokens(observed, language)
    errors = edit_distance(reference, hypothesis)
    return {"metric": "CER" if language == "zh" else "WER", "errors": errors,
            "reference_units": len(reference),
            "error_rate": errors / len(reference) if reference else None, "empty_output": not hypothesis}


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values: list[float], q: float) -> float | None:
    if not 0 <= q <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    values = sorted(float(value) for value in values if _finite(value))
    if not values:
        return None
    index = (len(values) - 1) * q / 100
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def distribution(values) -> dict:
    valid = [float(value) for value in values if _finite(value)]
    return {"count": len(valid), "p50": percentile(valid, 50), "p95": percentile(valid, 95),
            "max": max(valid) if valid else None}


def summarize_latency(events: list[dict]) -> dict:
    result = {}
    for key in ("caption_delay_ms", "latency_ms", "processing_ms", "queue_wait_ms", "asr_ms",
                "speaker_ms", "worker_ms", "caption_delay_from_speech_end_ms",
                "caption_delay_from_utterance_start_ms", "first_caption_delay_ms", "final_caption_delay_ms",
                "ui_delivery_ms"):
        values = [event[key] for event in events if _finite(event.get(key))
                  and not (key.startswith("caption_delay") and event.get("caption_delay_valid") is False)]
        if values:
            result[key] = distribution(values)
    return result


class Journal:
    """Flush each JSONL entry so native crashes leave inspectable evidence."""
    def __init__(self, path: Path):
        self.stream = path.open("a", encoding="utf-8", buffering=1)
        self.lock = threading.RLock()

    def write(self, event: dict):
        with self.lock:
            self.stream.write(json.dumps(event, ensure_ascii=False, default=str, allow_nan=False) + "\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())

    def close(self):
        with self.lock:
            self.stream.close()


def write_report(path: Path, report: dict):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def network_guard(journal: Journal) -> list[dict]:
    """Block Python IP connects, datagrams and DNS; native access is sampled."""
    attempts = []
    lock = threading.Lock()

    def deny(operation, original=None):
        def blocked(*args, **kwargs):
            if original and args and getattr(args[0], "family", None) not in (socket.AF_INET, socket.AF_INET6):
                return original(*args, **kwargs)
            record = {"type": "network_block", "operation": operation, "monotonic": time.monotonic()}
            with lock:
                attempts.append(record)
            journal.write(record)
            raise OSError("Offline replay benchmark blocked a Python IP or DNS operation")
        return blocked

    for name in ("connect", "connect_ex", "sendto"):
        setattr(socket.socket, name, deny(name, getattr(socket.socket, name)))
    for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        setattr(socket, name, deny(name))
    return attempts


class Resources:
    def __init__(self, journal: Journal, phase, engine_status):
        self.journal, self.phase, self.engine_status = journal, phase, engine_status
        self.samples = []
        self.errors = []
        self.done = threading.Event()
        import psutil
        self.process = psutil.Process()
        try:
            from scripts.gpu_memory import GpuMemory
            self.gpu = GpuMemory()
        except Exception as exc:
            self.gpu = None
            self.errors.append(f"GPU counters unavailable: {exc}")
        self.started = time.monotonic()
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="benchmark-resource-sampler")
        self.thread.start()

    def _sample(self):
        sample = {"type": "resources", "monotonic": time.monotonic(), "phase": self.phase()}
        try:
            sample["rss_bytes"] = self.process.memory_info().rss
            connections = self.process.net_connections(kind="inet")
            sample["inet_connections"] = [
                {"status": connection.status, "family": int(connection.family),
                 "remote": list(connection.raddr) if connection.raddr else None,
                 "local_port": connection.laddr.port if connection.laddr else None}
                for connection in connections]
        except Exception as exc:
            sample["process_sampling_error"] = str(exc)
        if self.gpu:
            try:
                sample["gpu"] = self.gpu.sample()
            except Exception as exc:
                sample["gpu_sampling_error"] = str(exc)
        sample["engine"] = self.engine_status()
        self.samples.append(sample)
        self.journal.write(sample)

    def _loop(self):
        last_progress = time.monotonic()
        while not self.done.wait(1):
            try:
                self._sample()
            except Exception as exc:
                self.errors.append(str(exc))
            if time.monotonic() - last_progress >= 15:
                print(f"[{self.phase()}] elapsed {time.monotonic() - self.started:.0f}s; "
                      f"engine={self.engine_status()}", flush=True)
                last_progress = time.monotonic()

    def finish(self) -> dict:
        self.done.set()
        self.thread.join(timeout=3)
        try:
            self._sample()
        except Exception as exc:
            self.errors.append(str(exc))
        if self.gpu:
            self.gpu.close()
        gpu_samples = [sample["gpu"] for sample in self.samples if sample.get("gpu")]
        gpu_report = {"source": "Windows PDH dedicated GPU memory", "sample_count": len(gpu_samples),
                      "baseline": gpu_samples[0] if gpu_samples else {},
                      "scope": "Adapter values sum all GPU adapters; process values sum this PID across adapters."}
        for prefix in ("adapter", "process"):
            values = [sample[f"{prefix}_dedicated_bytes"] for sample in gpu_samples
                      if _finite(sample.get(f"{prefix}_dedicated_bytes"))]
            gpu_report[f"{prefix}_peak_bytes"] = max(values) if values else None
            gpu_report[f"{prefix}_peak_minus_baseline_bytes"] = max(values) - values[0] if values else None
        observed = [sample.get("inet_connections", []) for sample in self.samples]
        return {"sample_count": len(self.samples),
                "baseline_process_rss_bytes": self.samples[0].get("rss_bytes") if self.samples else None,
                "peak_process_rss_bytes": max((s["rss_bytes"] for s in self.samples if "rss_bytes" in s), default=None),
                "gpu": gpu_report, "errors": self.errors,
                "inet_observation": {"interval_seconds": 1, "samples_with_remote_endpoints": sum(
                    any(connection["remote"] for connection in sample) for sample in observed),
                    "sampling_failures": sum("process_sampling_error" in sample for sample in self.samples),
                    "limitation": "Coarse process socket snapshots can miss short native connections; Python blocking is not an OS firewall."}}


def read_fixture(path: Path):
    import numpy as np
    from scipy.signal import resample_poly
    with wave.open(str(path), "rb") as reader:
        if reader.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM fixture: {path}")
        audio = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2").astype(np.float32) / 32768
        audio = audio.reshape(-1, reader.getnchannels()).mean(axis=1)
        rate = reader.getframerate()
    divisor = math.gcd(rate, 16000)
    return resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)


def speech_bounds(audio) -> tuple[float, float]:
    """Fixture-derived 10 ms RMS activity bounds, not human transcript alignment."""
    import numpy as np
    rms = [float(np.sqrt(np.mean(audio[index:index + 160].astype(np.float64) ** 2)))
           for index in range(0, len(audio), 160)]
    threshold = max(.002, max(rms, default=0) * .02)
    voiced = [index for index, value in enumerate(rms) if value >= threshold]
    if not voiced:
        raise ValueError("Fixture has no energy above the activity threshold")
    return voiced[0] * .01, min(len(audio) / 16000, (voiced[-1] + 1) * .01)


class Benchmark:
    def __init__(self, args, output: Path, journal: Journal, profile, gpus):
        self.args, self.output, self.journal, self.profile = args, output, journal, profile
        self.phase = "initializing"
        self.engine = None
        self.app = None
        self.store = None
        self.session_id = None
        self.abort = threading.Event()
        self.done = threading.Event()
        self.events = []
        self.events_lock = threading.RLock()
        self.segment_events = {}
        self.occurrences = []
        self.ui_observations = []
        self.ui_errors = []
        self.notes_samples = []
        self.epoch = None
        self.exit_code = 0
        self.report = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                       "status": "initializing", "synthetic_fixture_replay": True, "no_live_capture": True,
                       "real_recognizer": True, "profile": asdict(profile), "detected_gpus": gpus,
                       "language": args.language, "repeat": args.repeat, "dual_source": args.dual_source,
                       "ui_requested": args.ui, "output_directory": str(output),
                       "cold_replay": args.cold_replay,
                       "measurement_notes": [
                           "Source waveforms are generated speech, not real meeting recordings.",
                           "One recognizer is loaded once. Warmup is excluded from paced metrics unless --cold-replay skips warmup entirely.",
                           "Language stays fixed for the complete replay; auto is used for bilingual fixtures by default.",
                           "Engine speech-end latency uses energy boundaries, independent of ASR end timestamps.",
                           "Fixture first/final caption metrics use independently calculated 10 ms RMS activity bounds; these are not human word alignments.",
                           "JSONL fsync, SQLite persistence, notes generation, resource sampling and optional native UI all contribute to observed load.",
                           "Two-source replay duplicates the same generated corpus concurrently; it tests queue pressure, not diarization accuracy.",
                           "Results describe only this fixture, machine and run. Empty output is a failed observation, never a quality success."]}
        self.checkpoint("initializing")

    def checkpoint(self, phase: str):
        self.phase = phase
        self.report["status"] = phase
        write_report(self.output / "report.json", self.report)
        self.journal.write({"type": "phase", "phase": phase, "monotonic": time.monotonic()})
        print(f"[{phase}] {self.output}", flush=True)

    def status(self):
        return self.engine.status if self.engine else {}

    def on_event(self, original: dict):
        event = dict(original)
        event.update(phase="replay", received_monotonic=time.monotonic(), session_id=self.session_id)
        event["received_elapsed_seconds"] = event["received_monotonic"] - self.epoch if self.epoch else None
        self.journal.write(event)
        segment_id = self.store.append_event(self.session_id, event)
        if segment_id is not None:
            event["segment_id"] = segment_id
            self.segment_events[str(segment_id)] = event
        with self.events_lock:
            self.events.append(event)
        if self.app:
            self.app.events.put(event)

    def _notes_loop(self, stopped):
        from meeting.notes import suggest_notes
        while not stopped.wait(5):
            mark = time.perf_counter()
            try:
                segments = self.store.segments(self.session_id)
                notes = suggest_notes(segments)
                sample = {"type": "notes_measurement", "seconds": time.perf_counter() - mark,
                          "segments": len(segments), "output_characters": len(notes), "model_used": False}
            except Exception as exc:
                sample = {"type": "notes_error", "message": str(exc)}
            self.notes_samples.append(sample)
            self.journal.write(sample)

    def _feed(self, fixtures, source, barrier):
        import numpy as np
        timeline = 0.0
        barrier.wait()
        for repeat in range(self.args.repeat):
            for fixture in fixtures:
                if self.abort.is_set() or not self.engine.running:
                    return
                duration = len(fixture["audio"]) / 16000
                occurrence = {"id": f"{source}-{repeat + 1}-{fixture['file']}", "source": source,
                              "repeat": repeat + 1, "fixture": fixture["file"], "language": fixture["language"],
                              "voice": fixture.get("voice"), "expected": fixture["text"],
                              "timeline_start": timeline, "audio_seconds": duration,
                              "speech_onset": timeline + fixture["speech_onset"],
                              "speech_end": timeline + fixture["speech_end"],
                              "injection_started_elapsed": time.monotonic() - self.epoch}
                occurrence["speech_onset_elapsed"] = occurrence["injection_started_elapsed"] + fixture["speech_onset"]
                occurrence["speech_end_elapsed"] = occurrence["injection_started_elapsed"] + fixture["speech_end"]
                with self.events_lock:
                    self.occurrences.append(occurrence)
                self.journal.write({"type": "fixture_start", **occurrence})
                print(f"[replay] {source} repetition {repeat + 1}/{self.args.repeat}: {fixture['file']} ({duration:.1f}s)", flush=True)
                padded = np.concatenate((fixture["audio"], np.zeros(12800, dtype=np.float32)))
                self.engine.feed_fixture(padded, source=source, realtime=True)
                occurrence["injection_finished_elapsed"] = time.monotonic() - self.epoch
                self.journal.write({"type": "fixture_finished", "id": occurrence["id"],
                                    "elapsed_seconds": occurrence["injection_finished_elapsed"]})
                timeline += len(padded) / 16000

    def run(self):
        notes_stopped = threading.Event()
        notes_thread = None
        try:
            self.checkpoint("importing_runtime")
            mark = time.perf_counter()
            from meeting.inference import Recognizer, SpeakerTracker
            from meeting.engine import Engine
            self.report["runtime_import_seconds"] = time.perf_counter() - mark
            sources = json.loads((ROOT / "artifacts/fixtures/sources.json").read_text(encoding="utf-8-sig"))
            fixtures = []
            for source in sources:
                audio = read_fixture(ROOT / "artifacts/fixtures" / source["file"])
                onset, end = speech_bounds(audio)
                fixtures.append({**source, "audio": audio, "speech_onset": onset, "speech_end": end})
            if not fixtures:
                raise ValueError("No generated speech fixtures were found")
            self.report["fixtures"] = [{key: value for key, value in fixture.items() if key != "audio"} for fixture in fixtures]
            self.checkpoint("loading_model")
            mark = time.perf_counter()
            recognizer = Recognizer(str(ROOT / "models" / self.profile.model_dir), language=self.args.language,
                                    **self.profile.recognizer_options())
            self.report["cold_model_construct_seconds"] = time.perf_counter() - mark
            self.report["recognizer_load_seconds"] = getattr(recognizer, "load_seconds", None)
            speaker_path = ROOT / "models/speaker.onnx"
            if not speaker_path.is_file():
                raise FileNotFoundError("The actual local speaker model is required for this benchmark")
            mark = time.perf_counter()
            tracker = SpeakerTracker(str(speaker_path))
            self.report["speaker_model_construct_seconds"] = time.perf_counter() - mark
            self.checkpoint("cold_replay_ready" if self.args.cold_replay else "warming_up")
            self.report["warmup"] = []
            # First EN and first Mandarin samples warm both scripts without changing model language.
            warmed_languages = set()
            for fixture in ([] if self.args.cold_replay else fixtures):
                if fixture["language"] in warmed_languages:
                    continue
                warmed_languages.add(fixture["language"])
                clip = fixture["audio"][:6 * 16000]
                mark = time.perf_counter()
                recognized = recognizer.transcribe(clip)
                asr_seconds = time.perf_counter() - mark
                mark = time.perf_counter()
                speaker = tracker.classify(clip)
                warmup = {"fixture": fixture["file"], "audio_seconds": len(clip) / 16000,
                          "cold_first_call": len(self.report["warmup"]) == 0, "asr_seconds": asr_seconds,
                          "speaker_seconds": time.perf_counter() - mark, "segments": recognized, "speaker": speaker}
                self.report["warmup"].append(warmup)
                self.journal.write({"type": "warmup", **warmup})
                print(f"[warmup] {fixture['language']}: actual ASR {asr_seconds:.2f}s; {len(recognized)} segments", flush=True)
                write_report(self.output / "report.json", self.report)
            self.checkpoint("replaying")
            self.engine = Engine(recognizer, tracker, self.on_event, self.output / "fixture_audio_unused")
            if self.app:
                self.app.engine = self.engine
            self.epoch = time.monotonic()
            self.engine.start()  # Both IDs omitted: fixture-only. No capture device opens.
            self.epoch = self.engine._started_at  # Exact clock origin used by Engine event timing.
            if self.app:
                self.app.recording_since = self.epoch
            if not self.engine.status.get("fixture_mode"):
                raise RuntimeError("Replay refused: engine is not in fixture-only mode")
            self.store.update_session(self.session_id, status="running")
            notes_thread = threading.Thread(target=self._notes_loop, args=(notes_stopped,), daemon=True,
                                            name="benchmark-concurrent-notes")
            notes_thread.start()
            sources_to_feed = ["remote", "local"] if self.args.dual_source else ["remote"]
            barrier = threading.Barrier(len(sources_to_feed))
            feed_errors = []

            def feed(source):
                try:
                    self._feed(fixtures, source, barrier)
                except Exception as exc:
                    feed_errors.append({"source": source, "error": repr(exc)})
                    self.journal.write({"type": "feed_error", "source": source, "error": repr(exc)})
                    self.abort.set()

            feeders = [threading.Thread(target=feed, args=(source,), name=f"fixture-{source}", daemon=True)
                       for source in sources_to_feed]
            for feeder in feeders:
                feeder.start()
            while any(feeder.is_alive() for feeder in feeders):
                for feeder in feeders:
                    feeder.join(timeout=.25)
                if self.abort.is_set():
                    self.engine.stop(timeout=2)
            self.report["feed_errors"] = feed_errors
            self.checkpoint("draining")
            self.engine.stop(timeout=60)
            deadline = time.monotonic() + 120
            while self.engine.running and time.monotonic() < deadline:
                self.done.wait(.25)
            self.report["replay_wall_seconds"] = time.monotonic() - self.epoch
            self.report["final_engine_status"] = self.engine.status
            self.report["cancelled"] = self.abort.is_set()
            self._summarize()
            stats = self.engine.status
            incomplete = self.abort.is_set() or self.engine.running or bool(feed_errors) or stats.get("state") == "error" or any(
                stats.get(key, 0) for key in ("dropped_audio_blocks", "dropped_segments", "failed_segments", "event_callback_errors"))
            status = "incomplete" if incomplete else "no_recognized_text" if not self.report["recognized_segment_count"] else "completed"
            self.exit_code = 0 if status == "completed" else 2
            self.store.finish_session(self.session_id, status=status, duration=self.report["replay_wall_seconds"])
            self.checkpoint(status)
        except Exception as exc:
            self.exit_code = 1
            self.report.update(error=repr(exc), traceback=traceback.format_exc())
            self.journal.write({"type": "benchmark_error", "error": repr(exc), "traceback": traceback.format_exc()})
            if self.engine and self.engine.running:
                self.engine.stop(timeout=5)
            try:
                self.store.finish_session(self.session_id, status="error")
            except Exception:
                pass
            self.checkpoint("error")
            print(f"[error] {exc}", flush=True)
        finally:
            notes_stopped.set()
            if notes_thread:
                notes_thread.join(timeout=2)
            self.report["concurrent_notes"] = self.notes_samples
            self.done.set()

    def _summarize(self):
        with self.events_lock:
            events = [dict(event) for event in self.events]
            occurrences = [dict(occurrence) for occurrence in self.occurrences]
        segments = [event for event in events if event.get("type") == "segment" and str(event.get("text", "")).strip()]
        assignments = {occurrence["id"]: [] for occurrence in occurrences}
        unmatched = []
        for segment in segments:
            candidates = [occurrence for occurrence in occurrences if occurrence["source"] == segment.get("source")]
            def overlap(occurrence):
                return max(0, min(segment.get("end", 0), occurrence["timeline_start"] + occurrence["audio_seconds"])
                           - max(segment.get("start", 0), occurrence["timeline_start"]))
            best = max(candidates, key=overlap, default=None)
            if best is None or overlap(best) == 0:
                unmatched.append(segment)
            else:
                assignments[best["id"]].append(segment)
        cases = []
        for occurrence in occurrences:
            assigned = sorted(assignments[occurrence["id"]], key=lambda event: (event.get("start", 0), event.get("received_monotonic", 0)))
            observed = " ".join(event["text"] for event in assigned)
            received = [event["received_elapsed_seconds"] for event in assigned if _finite(event.get("received_elapsed_seconds"))]
            case = {**occurrence, "observed": observed, "quality": quality_score(occurrence["expected"], observed, occurrence["language"]),
                    "segments": assigned, "first_caption_delay_ms": (min(received) - occurrence["speech_onset_elapsed"]) * 1000 if received else None,
                    "final_caption_delay_ms": (max(received) - occurrence["speech_end_elapsed"]) * 1000 if received else None}
            cases.append(case)
        aggregates = []
        for source in ("remote", "local"):
            for language in ("en", "zh"):
                selected = [case for case in cases if case["source"] == source and case["language"] == language]
                if selected:
                    expected, observed = " ".join(case["expected"] for case in selected), " ".join(case["observed"] for case in selected)
                    aggregates.append({"source": source, "language": language, "expected": expected,
                                       "observed": observed, **quality_score(expected, observed, language)})
        first = [event for event in segments if event.get("first_in_utterance")]
        final = [event for event in segments if event.get("final_in_utterance")]
        # One clip can emit several ASR segments. Count its repeated queue/ASR/
        # speaker cost once; the last event includes preceding callback overhead.
        clips = {}
        for event in segments:
            if "utterance_id" in event and "chunk_index" in event:
                clips[(event.get("source"), event["utterance_id"], event["chunk_index"])] = event
        self.report.update(cases=cases, quality_by_source_language=aggregates, recognized_segment_count=len(segments),
                           empty_fixture_occurrences=sum(case["quality"]["empty_output"] for case in cases),
                           unmatched_segments=unmatched, stable_chunk_latency_ms=summarize_latency(segments),
                           fixture_caption_latency_ms=summarize_latency(cases),
                           unique_emitted_clip_count=len(clips),
                           clip_processing_latency_ms=summarize_latency([
                               {key: event.get(key) for key in ("queue_wait_ms", "asr_ms", "speaker_ms", "worker_ms", "latency_ms")}
                               for event in clips.values()]),
                           utterance_first_stable_caption_ms=distribution([event.get("caption_delay_from_utterance_start_ms") for event in first]),
                           utterance_final_caption_after_speech_end_ms=distribution([event.get("caption_delay_from_speech_end_ms") for event in final]),
                           warnings=[event for event in events if event.get("type") == "warning"],
                           timing_metadata_available=any("caption_delay_from_speech_end_ms" in event for event in segments))


def run_ui(benchmark: Benchmark):
    import ctypes
    import tkinter as tk
    from meeting.ui import App, PURPLE
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass

    class ReplayApp(App):
        def _discover_devices(self):
            pass  # Scheduled App startup callback is disabled: no device enumeration.

        def _start(self):
            self._notice("Replay benchmark only. Live capture is disabled.")

        def _demo(self):
            self._notice("This replay uses generated audio and the real local recognizer.")

        def _new_view(self):
            self._notice("The current replay session is controlled by the benchmark.")

        def _set_capture_controls(self, active):
            super()._set_capture_controls(active)
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")

        def _finish_current(self, session_id, status, engine_status=None):
            # Benchmark owns durable completion; App still owns transcript rendering.
            if self.engine and self.engine.running:
                return
            self.active_id = None
            self.engine = None
            self.recording_since = None
            self._set_capture_controls(False)
            self._refresh_history()
            self._refresh_transcript()

        def _poll_once(self):
            super()._poll_once()
            self.indicator.configure(text="◈  REPLAY" if not benchmark.done.is_set() else "◈  REPLAY DONE",
                                     bg="#EDE9FC", fg=PURPLE)

        def _refresh_transcript(self):
            super()._refresh_transcript()
            for ident in self.tree.get_children():
                event = benchmark.segment_events.get(ident)
                if event and "ui_applied_elapsed_seconds" not in event and benchmark.epoch:
                    elapsed = time.monotonic() - benchmark.epoch
                    event["ui_applied_elapsed_seconds"] = elapsed
                    observation = {"type": "ui_applied", "segment_id": ident, "elapsed_seconds": elapsed,
                                   "ui_delivery_ms": (elapsed - event["received_elapsed_seconds"]) * 1000}
                    benchmark.ui_observations.append(observation)
                    benchmark.journal.write(observation)

        def _close(self):
            if not benchmark.done.is_set():
                benchmark.abort.set()
                self._notice("Stopping synthetic replay. Waiting for current model work to finish…")
                if benchmark.engine:
                    threading.Thread(target=lambda: benchmark.engine.stop(timeout=2), daemon=True).start()
            else:
                self._destroy()

    root = tk.Tk()
    root.title("Megumin Meet · Real-model synthetic replay benchmark · No capture")
    app = ReplayApp(root, benchmark.output)
    benchmark.app, benchmark.store = app, app.store
    benchmark.session_id = app.store.create_session("Real-model replay · generated speech", language=benchmark.args.language,
                                                   simulated=True, profile=benchmark.profile.label)
    app.store.update_session(benchmark.session_id, notes="SYNTHETIC AUDIO REPLAY\nNo microphone or system audio capture.\n\nReal local recognizer and speaker model.\nGenerated English and Mandarin fixtures.\nNotes quote extraction runs concurrently every five seconds.\n\nThis run measures one candidate; it does not certify real meeting accuracy.")
    app.active_id = benchmark.session_id
    app._load_session(benchmark.session_id)
    app._set_capture_controls(True)
    app.indicator.configure(text="◈  REPLAY LOADING", bg="#EDE9FC", fg=PURPLE)
    app.subtitle_var.set("SYNTHETIC AUDIO · Actual local inference · Capture disabled")
    root.geometry("1360x920+20+20")
    root.report_callback_exception = lambda typ, exc, trace: benchmark.ui_errors.append(repr(exc))
    worker = threading.Thread(target=benchmark.run, daemon=True, name="real-model-replay")
    app.worker = worker
    worker.start()

    def finish():
        if not benchmark.done.is_set() or not app.events.empty():
            root.after(200, finish)
            return
        try:
            app._refresh_transcript()
            app.indicator.configure(text="◈  REPLAY DONE", bg="#EDE9FC", fg=PURPLE)
            app._notice(f"Replay {benchmark.report['status']} · No live capture · Results saved in the benchmark folder")
            root.update_idletasks()
            from PIL import ImageGrab
            root.lift()
            root.attributes("-topmost", True)
            root.update()
            box = (root.winfo_rootx(), root.winfo_rooty(), root.winfo_rootx() + root.winfo_width(),
                   root.winfo_rooty() + root.winfo_height())
            screenshot = benchmark.output / "replay-ui.png"
            ImageGrab.grab(bbox=box).save(screenshot)
            benchmark.report["ui_screenshot"] = str(screenshot)
        except Exception as exc:
            benchmark.ui_errors.append(f"Screenshot/display finalization: {exc}")
        finally:
            benchmark.report["ui_widget_application_latency_ms"] = summarize_latency(benchmark.ui_observations)
            benchmark.report["ui_callback_errors"] = benchmark.ui_errors
            benchmark.report["ui_latency_scope"] = "Tk widget insertion, not a measurement of physical display refresh."
            app._destroy()

    root.after(200, finish)
    root.mainloop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("turbo", "small"), default="turbo")
    parser.add_argument("--target", choices=("arc", "nvidia"), default="arc")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dual-source", action="store_true")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--cold-replay", action="store_true", help="Skip model warmup; include first native generation in paced caption timing")
    parser.add_argument("--language", choices=("auto", "en", "zh"), default="auto")
    parser.add_argument("--output", help="Local artifact folder tag; letters, digits, hyphens and underscores only")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    tag = args.output or f"{args.target}-{args.model}-{'dual' if args.dual_source else 'single'}-{datetime.now():%Y%m%d-%H%M%S}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        parser.error("--output must be a simple folder tag")
    output = ROOT / "artifacts" / f"replay-{tag}"
    if output.exists():
        output = output.with_name(output.name + f"-{datetime.now():%Y%m%d-%H%M%S-%f}")
    output.mkdir(parents=True)
    journal = Journal(output / "events.jsonl")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        os.environ[key] = "1"
    from meeting.offline import enable
    enable()  # Includes the project's telemetry suppression before inference imports.
    attempts = network_guard(journal)
    from meeting.profiles import get_profile, detect_gpus
    from meeting.storage import Store
    gpus = detect_gpus()
    profile = get_profile(args.target, args.model, gpus=gpus)
    benchmark = Benchmark(args, output, journal, profile, gpus)
    resources = None
    try:
        resources = Resources(journal, lambda: benchmark.phase, benchmark.status)
        if args.ui:
            run_ui(benchmark)
        else:
            benchmark.store = Store(output / "meetings.sqlite3")
            benchmark.session_id = benchmark.store.create_session("Real-model replay · generated speech", simulated=True,
                                                                  language=args.language, profile=profile.label)
            benchmark.run()
    except Exception as exc:
        benchmark.exit_code = 1
        benchmark.report.update(status="error", error=repr(exc), traceback=traceback.format_exc())
        journal.write({"type": "harness_error", "error": repr(exc), "traceback": traceback.format_exc()})
        print(f"[error] {exc}", flush=True)
    finally:
        if resources:
            benchmark.report["resources"] = resources.finish()
        benchmark.report["network"] = {"python_guard": "IP connect/connect_ex/sendto and DNS blocked before model initialization",
                                       "blocked_attempts": attempts, "os_firewall_changed": False,
                                       "limitation": "Native library network access is only coarsely observed by process socket sampling."}
        benchmark.report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        benchmark.report["concurrent_notes"] = benchmark.notes_samples
        write_report(output / "report.json", benchmark.report)
        if benchmark.store and not args.ui:
            benchmark.store.close()
        journal.close()
    print(f"Report: {output / 'report.json'}", flush=True)
    print(f"Status: {benchmark.report['status']}; recognized segments: {benchmark.report.get('recognized_segment_count', 0)}", flush=True)
    for score in benchmark.report.get("quality_by_source_language", []):
        print(f"{score['source']} {score['language']} {score['metric']}: {score['error_rate']}; empty output={score['empty_output']}", flush=True)
    print("Primary speech-end caption latency (ms):", benchmark.report.get("stable_chunk_latency_ms", {}).get("caption_delay_from_speech_end_ms"), flush=True)
    return benchmark.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
