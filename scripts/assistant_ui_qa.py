"""Synthetic native UI checks. Mock answers only; no model, CLI or network runs."""
from __future__ import annotations

import ctypes
import json
from pathlib import Path
import sys
import threading
import time
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting.offline import enable
enable()
from meeting.assistant_core import load_selected_files
from meeting.assistant_ui import AssistantWindow, fixed_policy_sections
from meeting.codex_transport import CodexTransport
from meeting.ui import App


class MockHandle:
    def __init__(self):
        self.cancelled = threading.Event()
        self.active = True

    def cancel(self):
        self.cancelled.set()
        self.active = False


class MockTransport:
    synthetic = True
    destination = "SYNTHETIC QA mock transport — no network or cloud destination"

    def __init__(self):
        self.available = False
        self.start_calls = []
        self.discovery_calls = 0
        self.disconnect_calls = 0
        self.handles = []
        self.closed = False

    def discover_models(self):
        self.discovery_calls += 1
        self.available = True
        return [{"id": "synthetic-qa-model", "label": "Synthetic mock model", "efforts": ["low"], "is_default": True}]

    def start(self, preview, on_event):
        if not self.available:
            raise RuntimeError("Explicit mock connection is required")
        if any(handle.active for handle in self.handles):
            raise RuntimeError("Mock has no request queue")
        self.start_calls.append(preview)
        handle = MockHandle()
        self.handles.append(handle)
        began = time.perf_counter()

        def answer():
            first = None
            source_id = next((source.id for source in preview.sources if "Friday" in source.text), preview.sources[0].id)
            chunks = ["SYNTHETIC QA DRAFT\n\n", "The release is planned for Friday. ",
                      f"The owner has not been confirmed [{source_id}].\n\n", "Verify the plan with the team before assigning an action."]
            text = ""
            for chunk in chunks:
                if handle.cancelled.wait(.14):
                    on_event({"type": "cancelled"})
                    # A deliberately late callback verifies session/generation isolation.
                    on_event({"type": "token", "text": "SHOULD NOT APPEAR AFTER CANCEL"})
                    return
                first = first or time.perf_counter()
                text += chunk
                on_event({"type": "token", "text": chunk})
            handle.active = False
            on_event({"type": "complete", "text": text, "model": preview.model,
                      "metrics": {"started_at": began, "submitted_at": began, "first_token_at": first,
                                  "completed_at": time.perf_counter()}})
        threading.Thread(target=answer, daemon=True, name="synthetic-answer-only").start()
        return handle

    def disconnect(self):
        self.disconnect_calls += 1
        self.available = False
        for handle in self.handles:
            handle.cancel()

    def close(self):
        self.closed = True
        self.disconnect()


class NoCaptureApp(App):
    def _discover_devices(self):
        pass


class FakeEngine:
    def __init__(self):
        self.running, self.stop_called = True, False

    @property
    def status(self):
        return {"running": self.running, "state": "running" if self.running else "stopped", "elapsed_seconds": 1}

    def stop(self):
        self.stop_called = True
        self.running = False


def main():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    output = ROOT / "artifacts/assistant-qa-v03"
    output.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    app = NoCaptureApp(root, output / "workspace")
    root.withdraw()
    errors = []
    root.report_callback_exception = lambda typ, exc, trace: errors.append(repr(exc))
    sid = app.store.create_session("Synthetic private-assistant QA", simulated=True)
    app.selected_id = sid
    context_event = {"type": "segment", "start": 10.0, "end": 14.0, "source": "remote", "speaker": "Speaker 1",
                     "text": "The release is planned for Friday. The owner has not been confirmed.",
                     "caption_kind": "stable_chunk", "is_partial": False}
    app.store.append_event(sid, context_event)
    question = {"type": "segment", "start": 18.0, "end": 21.0, "source": "remote", "speaker": "Speaker 2",
                "text": "Could you explain the release plan?", "caption_kind": "stable_chunk", "is_partial": False,
                "session_id": sid, "assistant_speech_end_perf": time.perf_counter() - .8}
    question["segment_id"] = app.store.append_event(sid, question)
    real_transport = CodexTransport(ROOT)
    real_calls = {"discovery": 0, "start": 0}

    def forbidden_real_call(name):
        def forbidden(*args, **kwargs):
            real_calls[name] += 1
            raise AssertionError("Synthetic UI QA must never invoke real discovery, auth or cloud")
        return forbidden

    real_transport.discover_models = forbidden_real_call("discovery")
    real_transport.start = forbidden_real_call("start")
    real = AssistantWindow(root, ROOT, app.store, lambda: sid, transport=real_transport)
    mock = None
    transport = MockTransport()
    result = {"synthetic": True, "actual_cloud_requests": 0, "live_capture": False, "real_model_inference": False,
              "checks": {}, "screenshots": []}

    def capture(window, name):
        from PIL import ImageGrab
        window.lift()
        window.attributes("-topmost", True)
        window.update()
        box = (window.winfo_rootx(), window.winfo_rooty(), window.winfo_rootx() + window.winfo_width(),
               window.winfo_rooty() + window.winfo_height())
        path = output / name
        ImageGrab.grab(bbox=box).save(path)
        window.attributes("-topmost", False)
        result["screenshots"].append(str(path))

    def select_first(panel):
        ident = panel.question_tree.get_children()[0]
        panel.question_tree.selection_set(ident)
        panel._select_candidate()

    def guard(callback):
        def wrapped():
            try:
                callback()
            except Exception as exc:
                errors.append(repr(exc))
                finish()
        return wrapped

    def real_checks():
        assert "Codex not connected" in real.mode_var.get()
        assert not real.enable_var.get() and not real.transport.available
        real.feed_segment(question)
        assert not real.detector.pending(), "Detection must be off by default"
        result["checks"]["local_detection_off_by_default"] = True
        real.detect_var.set(True)
        real._toggle_detection()
        real.feed_segment(question)
        select_first(real)
        real._open_preview()
        assert real.preview_manual
        assert real.preview.model == "Not selected — manual Codex session"
        real._send_preview()
        assert not real.inflight
        assert real.transport.capability_report()["payloads_submitted"] == 0
        assert real_calls == {"discovery": 0, "start": 0}
        result["checks"]["disconnected_ui_blocks_dispatch"] = True
        assert real.preview_policy_widgets["MEETING_PAYLOAD"].get("1.0", "end-1c") == real.preview.payload_text
        for name, text in fixed_policy_sections():
            widget = real.preview_policy_widgets[name]
            assert widget.get("1.0", "end-1c") == text and str(widget.cget("state")) == "disabled"
        result["checks"]["exact_fixed_policies_separate_from_meeting_payload"] = True
        capture(real.preview_dialog, "real-manual-review.png")
        real.preview_notebook.select(1)
        capture(real.preview_dialog, "fixed-base-policy-review.png")
        real.preview_notebook.select(0)
        real.preview_reviewed_var.set(True)
        real.preview_sensitive_var.set(True)
        real._save_reviewed_prompt(output / "reviewed-manual-prompt.txt")
        assert (output / "reviewed-manual-prompt.txt").is_file()
        result["checks"]["reviewed_local_prompt_export"] = True
        capture(real.window, "real-unavailable-panel.png")
        real.close()
        real.stop_session()  # Idempotent even after native widgets are destroyed.
        app.assistant = real
        fake = FakeEngine()
        app.engine, app.active_id = fake, sid
        app.worker = None
        app._stop()
        result["fake_engine"] = fake
        root.after(200, guard(start_mock))

    def start_mock():
        nonlocal mock
        assert result.pop("fake_engine").stop_called and app.cancel_start.is_set()
        result["checks"]["closed_assistant_does_not_block_capture_stop"] = True
        mock = AssistantWindow(root, ROOT, app.store, lambda: sid, transport=transport)
        mock.detect_var.set(True)
        mock._toggle_detection()
        mock.feed_segment(question)
        select_first(mock)
        assert not transport.start_calls
        mock.enable_var.set(True)
        mock._toggle_enabled()
        assert not transport.start_calls and transport.discovery_calls == 0
        assert not transport.available
        mock._discover_models()
        root.after(200, guard(review_mock))

    def review_mock():
        assert transport.discovery_calls == 1 and not transport.start_calls
        assert transport.available and "SYNTHETIC" in mock.mode_var.get()
        assert not mock.model_var.get()
        result["checks"]["explicit_mock_connect_without_automatic_send"] = True
        mock.model_var.set(mock._model_label(mock.models[0]))
        mock._model_changed()
        mock.nearby_var.set(True)
        mock.context_vars["meeting_goal"].set("Explain release readiness using verified facts.")
        mock.context_vars["constraints"].set("Do not invent owners or dates.")
        context_file = output / "synthetic-facts.txt"
        context_file.write_text("SYNTHETIC QA FACTS\nRelease: Friday. Owner: unconfirmed.\nContact example@example.test is only a synthetic test address.", encoding="utf-8")
        mock.files = load_selected_files([context_file], allowed_directory=output)
        mock._refresh_files()
        edited = "What is the release plan, and which ownership facts remain unconfirmed?"
        mock.question_text.delete("1.0", "end")
        mock.question_text.insert("1.0", edited)
        mock.question_text.edit_modified(False)
        mock._open_preview()
        preview_id = mock.preview.id
        second = dict(question, segment_id=99901, start=25.0, end=28.0, text="Should we review the budget next?")
        mock.feed_segment(second)
        root.update()
        assert mock.question_text.get("1.0", "end-1c") == edited
        assert mock.preview and mock.preview.id == preview_id and mock.preview_dialog.winfo_exists()
        result["checks"]["new_candidate_preserves_edits_and_preview"] = True
        assert not transport.start_calls
        mock._send_preview()  # Missing reviewed checkbox must block dispatch.
        assert not transport.start_calls
        result["checks"]["no_send_before_exact_preview_approval"] = True
        capture(mock.preview_dialog, "synthetic-exact-payload.png")
        mock.preview_reviewed_var.set(True)
        mock.preview_sensitive_var.set(True)
        mock._send_preview()
        assert len(transport.start_calls) == 1
        mock._send_preview()
        assert len(transport.start_calls) == 1
        result["checks"]["one_use_send_no_automatic_backlog"] = True
        root.after(950, guard(answer_checks))

    def answer_checks():
        assert mock.history[0]["status"] == "complete"
        assert "SYNTHETIC QA DRAFT" in mock.answer_text.get("1.0", "end-1c")
        assert mock.history[0]["intervals_ms"]["approval_to_first_token_ms"] is not None
        result["checks"]["mock_streaming_completion_and_timing"] = True
        capture(mock.window, "synthetic-private-answer.png")
        mock._manual_ask("What should we verify before assigning an owner?")
        mock._open_preview()
        mock.preview_reviewed_var.set(True)
        mock.preview_sensitive_var.set(True)
        mock._send_preview()
        assert len(transport.start_calls) == 2
        mock.stop_session()
        assert transport.handles[-1].cancelled.is_set()
        assert not mock.detector.pending() and not mock.files and not mock.preview
        assert not mock.answer_text.get("1.0", "end-1c")
        assert all(not var.get() for var in mock.context_vars.values())
        root.after(250, guard(cancel_checks))

    def cancel_checks():
        assert "SHOULD NOT APPEAR" not in mock.answer_text.get("1.0", "end-1c")
        assert len(transport.start_calls) == 2
        result["checks"]["stop_cancels_clears_context_and_ignores_late_tokens"] = True
        mock.export_history(output / "synthetic-private-history.json")
        result["mock_request_count"] = len(transport.start_calls)
        result["mock_latency_intervals_ms"] = mock.history[0]["intervals_ms"]
        disconnects = transport.disconnect_calls
        mock.enable_var.set(False)
        mock._toggle_enabled()
        assert transport.disconnect_calls == disconnects + 1 and not transport.available
        assert not mock.models and not mock.preview and not mock.files
        mock._discover_models()
        assert transport.discovery_calls == 1 and len(transport.start_calls) == 2
        result["checks"]["disable_disconnects_and_clears_unsent_context"] = True
        finish()

    def finish():
        result["real_transport_calls"] = dict(real_calls)
        result["callback_errors"] = errors
        result["passed"] = not errors and len(result["checks"]) >= 9
        (output / "assistant-ui-qa.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if mock and not mock.closed:
            mock.close()
        if not real.closed:
            real.close()
        app.assistant = None
        app._destroy()

    root.after(150, guard(real_checks))
    root.mainloop()
    print(output / "assistant-ui-qa.json")
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
