"""Two synthetic questions through real local ASR, SQLite, Tk, and detection.

No recording-device access, cloud traffic, or mock answer generation. Each
fixture is one question. Audio end is the last active 10 ms RMS frame; timestamps
are based on actual paced injection, not ASR timestamp estimates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting.offline import enable
enable()
from scripts.replay_benchmark import read_fixture, speech_bounds


def main(output):
    import ctypes
    import numpy as np
    import tkinter as tk
    from meeting.ui import App
    from meeting.inference import Recognizer, SpeakerTracker
    from meeting.engine import Engine
    from meeting.profiles import resolve_profile
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    output.mkdir(parents=True, exist_ok=False)
    fixtures_dir = ROOT / "artifacts/assistant-fixtures"
    fixtures = json.loads((fixtures_dir / "sources.json").read_text(encoding="utf-8-sig"))
    for fixture in fixtures:
        fixture["audio"] = read_fixture(fixtures_dir / fixture["file"])
        fixture["onset"], fixture["speech_end"] = speech_bounds(fixture["audio"])
    report = {"kind": "Real-model paced synthetic question replay; no capture or cloud",
              "cases": [], "candidates": [], "errors": [], "warmup": True,
              "cloud_submitted_at": None, "cloud_first_token_at": None, "cloud_completed_at": None}
    root = tk.Tk()
    root.report_callback_exception = lambda kind, error, traceback: report["errors"].append(str(error))
    class ReplayApp(App):
        def _discover_devices(self):
            pass
        def _start(self):
            self._notice("Synthetic harness only; capture is disabled.")
    app = ReplayApp(root, output)
    sid = app.store.create_session("Synthetic question replay — no capture", simulated=True)
    app._load_session(sid)
    app._open_assistant()
    panel = app.assistant
    panel.detect_var.set(True)
    panel._toggle_detection()
    panel.mode_var.set("REAL LOCAL ASR • SYNTHETIC AUDIO • NO CAPTURE OR CLOUD")
    original_feed = panel.feed_segment
    def observe(event):
        before = set(panel.candidates)
        original_feed(event)
        for candidate_id in set(panel.candidates) - before:
            candidate = panel.candidates[candidate_id]
            panel.window.update_idletasks()
            displayed = time.perf_counter()
            matches = [case for case in report["cases"] if case["timeline_start"] <= candidate.start < case["timeline_end"]]
            case = matches[-1] if matches else None
            report["candidates"].append({
                "text": candidate.text, "source": candidate.source, "segment_id": candidate.segment_id,
                "fixture": case["file"] if case else None, "detected_at": candidate.detected_at,
                "ui_updated_at": displayed, "local_detection_ms": candidate.detection_ms,
                "speech_end_to_candidate_ms": (candidate.detected_at - case["speech_end_at"]) * 1000 if case else None,
                "speech_end_to_ui_ms": (displayed - case["speech_end_at"]) * 1000 if case else None,
                "onset_to_candidate_ms": (candidate.detected_at - case["speech_onset_at"]) * 1000 if case else None})
    panel.feed_segment = observe
    done = threading.Event()
    def work():
        engine = None
        try:
            profile, _ = resolve_profile(ROOT)
            report["profile"] = profile.label
            mark = time.perf_counter()
            recognizer = Recognizer(str(ROOT / "models" / profile.model_dir), language="auto", **profile.recognizer_options())
            tracker = SpeakerTracker(str(ROOT / "models/speaker.onnx"))
            report["model_construct_seconds"] = time.perf_counter() - mark
            for fixture in fixtures:
                recognizer.transcribe(fixture["audio"])
            def on_event(event):
                if event.get("type") == "segment":
                    event = dict(event)
                    event["segment_id"] = app.store.append_event(sid, event)
                    event["session_id"] = sid
                    app.events.put(event)
            engine = Engine(recognizer, tracker, on_event, output / "unused_audio")
            engine.start()
            assert engine.status["fixture_mode"]
            timeline = 0.0
            for fixture in fixtures:
                audio = fixture["audio"]
                padded = np.concatenate((audio, np.zeros(16000, dtype=np.float32)))
                started = time.perf_counter()
                report["cases"].append({"file": fixture["file"], "text": fixture["text"], "language": fixture["language"],
                                       "timeline_start": timeline, "timeline_end": timeline + len(padded) / 16000,
                                       "speech_onset_at": started + fixture["onset"],
                                       "speech_end_at": started + fixture["speech_end"]})
                engine.feed_fixture(padded, source="remote", realtime=True)
                timeline += len(padded) / 16000
            engine.stop(timeout=30)
            report["engine_status"] = engine.status
            app.store.finish_session(sid)
        except Exception as exc:
            report["errors"].append(repr(exc))
        finally:
            if engine and engine.running:
                engine.stop(timeout=3)
            done.set()
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    deadline = time.perf_counter() + 120
    def check():
        if done.is_set() and app.events.empty():
            from PIL import ImageGrab
            panel.window.lift()
            panel.window.attributes("-topmost", True)
            panel.window.update()
            x, y = panel.window.winfo_rootx(), panel.window.winfo_rooty()
            ImageGrab.grab((x, y, x + panel.window.winfo_width(), y + panel.window.winfo_height())).save(output / "real-question-replay.png")
            report["limits"] = ["Two clean synthetic questions; no noisy Zoom audio, overlap, or human alignment.",
                                "Speech end uses the same 10 ms RMS definition as prior fixture reports.",
                                "Includes paced injection, VAD, ASR, speaker model, SQLite, Tk queue, and candidate rendering.",
                                "No cloud submission; model answer latency remains unavailable."]
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            app._destroy()
            return
        if time.perf_counter() > deadline:
            report["errors"].append("Replay exceeded 120-second watchdog")
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            root.destroy()
            return
        root.after(100, check)
    root.after(100, check)
    root.mainloop()
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 1 if report["errors"] or len(report["candidates"]) < 2 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/assistant-question-replay")
    raise SystemExit(main(ROOT / parser.parse_args().output))
