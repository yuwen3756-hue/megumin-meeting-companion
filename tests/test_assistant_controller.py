"""Controller boundary checks without a Tk instance or a real transport."""

import queue
import socket
import subprocess
from types import SimpleNamespace

import pytest

from meeting.assistant_ui import AssistantWindow, fixed_policy_sections, timing_intervals_ms


@pytest.fixture(autouse=True)
def forbid_gui_network_and_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("controller tests must not start GUI, network, or processes")

    monkeypatch.setattr("meeting.assistant_ui.tk.Tk", forbidden)
    monkeypatch.setattr("meeting.assistant_ui.tk.Toplevel", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


def test_missing_timestamps_remain_unknown():
    assert timing_intervals_ms({}) == {
        "speech_to_candidate_ms": None,
        "candidate_to_approval_ms": None,
        "approval_to_first_token_ms": None,
        "approval_to_completion_ms": None,
        "speech_to_first_token_ms": None,
    }
    partial = timing_intervals_ms({"speech_end_at": 10.0, "candidate_at": 10.25})
    assert partial["speech_to_candidate_ms"] == 250.0
    assert all(value is None for key, value in partial.items() if key != "speech_to_candidate_ms")


@pytest.mark.parametrize("invalid", [None, "12.5", True, float("nan"), float("inf"), -float("inf")])
def test_invalid_timestamp_does_not_fabricate_latency(invalid):
    result = timing_intervals_ms({
        "speech_end_at": 10.0, "candidate_at": 10.25,
        "sent_at": 12.0, "first_token_at": invalid, "completed_at": 13.0,
    })
    assert result["approval_to_first_token_ms"] is None
    assert result["speech_to_first_token_ms"] is None
    assert result["speech_to_candidate_ms"] == 250.0
    assert result["candidate_to_approval_ms"] == 1750.0
    assert result["approval_to_completion_ms"] == 1000.0


def test_review_time_is_separate_from_post_approval_model_latency():
    timing = {
        "speech_end_at": 100.0, "candidate_at": 100.4,
        "sent_at": 112.4, "first_token_at": 112.9, "completed_at": 114.4,
    }
    original = timing_intervals_ms(timing)
    assert original == pytest.approx({
        "speech_to_candidate_ms": 400.0,
        "candidate_to_approval_ms": 12000.0,
        "approval_to_first_token_ms": 500.0,
        "approval_to_completion_ms": 2000.0,
        "speech_to_first_token_ms": 12900.0,
    })

    # Twenty extra seconds of human review must not slow the measured model.
    delayed = dict(timing)
    for key in ("sent_at", "first_token_at", "completed_at"):
        delayed[key] += 20.0
    result = timing_intervals_ms(delayed)
    assert result["candidate_to_approval_ms"] == pytest.approx(32000.0)
    assert result["speech_to_first_token_ms"] == pytest.approx(32900.0)
    assert result["approval_to_first_token_ms"] == pytest.approx(original["approval_to_first_token_ms"])
    assert result["approval_to_completion_ms"] == pytest.approx(original["approval_to_completion_ms"])


class _ProtectedPayloadWindow(AssistantWindow):
    @property
    def preview(self):
        raise AssertionError("blocked send must not access the payload")

    @property
    def gate(self):
        raise AssertionError("blocked send must not access the approval gate")


class _TransportThatMustNotStart:
    blocked_reason = "Live transport blocked for this test."

    def __init__(self, available):
        self.available = available
        self.starts = 0

    def start(self, *args, **kwargs):
        self.starts += 1
        raise AssertionError("blocked transport must never start")


@pytest.mark.parametrize("available,manual", [(False, False), (False, True), (True, True)])
def test_unavailable_or_manual_send_blocks_before_payload_and_approval(available, manual):
    controller = _ProtectedPayloadWindow.__new__(_ProtectedPayloadWindow)
    controller.transport = _TransportThatMustNotStart(available)
    controller.preview_manual = manual
    statuses = []
    controller._status = statuses.append

    controller._send_preview()

    assert statuses == [controller.transport.blocked_reason]
    assert controller.transport.starts == 0


@pytest.mark.parametrize("transport", [None, SimpleNamespace()])
def test_missing_transport_or_availability_flag_fails_closed(transport):
    controller = _ProtectedPayloadWindow.__new__(_ProtectedPayloadWindow)
    controller.transport = transport
    controller.preview_manual = False
    statuses = []
    controller._status = statuses.append

    controller._send_preview()

    assert len(statuses) == 1
    assert "unavailable" in statuses[0].lower()


def test_late_answer_from_previous_generation_is_not_displayed():
    controller = AssistantWindow.__new__(AssistantWindow)
    controller.closed = False
    controller.generation = 7
    controller.events = queue.Queue()
    controller.events.put(("answer", 6, {"type": "token", "text": "old response"}))
    controller.events.put(("answer", 7, {"type": "token", "text": "current response"}))
    observed, statuses = [], []
    controller._answer_event = observed.append
    controller._status = statuses.append
    controller.window = SimpleNamespace(after=lambda *args: None)

    controller._poll()

    assert observed == [{"type": "token", "text": "current response"}]
    assert statuses == []


def test_stopping_closed_window_does_not_access_destroyed_widgets():
    controller = AssistantWindow.__new__(AssistantWindow)
    controller.closed = True

    controller.stop_session()

    assert controller.closed is True


class _Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Control:
    def __init__(self):
        self.options = {}

    def configure(self, **options):
        self.options.update(options)


def _connection_controller(*, available=False):
    controller = AssistantWindow.__new__(AssistantWindow)
    controller.transport = SimpleNamespace(available=available, blocked_reason="Preflight did not verify isolation.",
                                           disconnect_calls=0, discovery_calls=0, start_calls=0)

    def disconnect():
        controller.transport.disconnect_calls += 1
        controller.transport.available = False

    controller.transport.disconnect = disconnect
    controller.enable_var = _Value(False)
    controller.model_var, controller.effort_var, controller.mode_var = _Value(), _Value(), _Value()
    for name in ("discover_button", "model_combo", "effort_combo", "review_button"):
        setattr(controller, name, _Control())
    controller.models = []
    controller.inflight = False
    controller.discovery_generation = 0
    controller.pending_context = ["Unsent meeting facts"]
    controller.preview = object()
    controller.statuses = []
    controller._status = controller.statuses.append

    def clear_pending():
        controller.pending_context.clear()
        controller.preview = None

    controller.stop_session = clear_pending
    return controller


def test_enabling_optional_answers_does_not_connect_or_submit():
    controller = _connection_controller()
    controller.enable_var.set(True)

    controller._toggle_enabled()

    assert controller.transport.discovery_calls == 0
    assert controller.transport.start_calls == 0
    assert controller.transport.disconnect_calls == 0
    assert controller.mode_var.get() == "Codex not connected — ASR stays offline"
    assert "Connect / discover explicitly" in controller.statuses[-1]
    assert controller.model_var.get() == ""


def test_disabling_disconnects_and_clears_unsent_context_and_catalog():
    controller = _connection_controller(available=True)
    controller.models = [{"id": "verified-model", "label": "Verified", "efforts": ["low"]}]
    controller.model_var.set("Verified [verified-model]")

    controller._toggle_enabled()

    assert controller.transport.disconnect_calls == 1
    assert not controller.transport.available
    assert controller.pending_context == [] and controller.preview is None
    assert controller.models == [] and controller.model_var.get() == ""
    assert controller.discovery_generation == 1
    assert controller.mode_var.get() == "Codex not connected — ASR stays offline"


def test_successful_preflight_publishes_catalog_without_selecting_or_sending():
    controller = _connection_controller(available=True)
    controller.enable_var.set(True)
    catalog = [{"id": "verified-model", "label": "Verified", "efforts": ["low", "medium"]}]

    controller._discovery_result({"models": catalog})

    assert controller.models == catalog
    assert controller.model_var.get() == "" and controller.effort_var.get() == ""
    assert "Connected" in controller.mode_var.get() and "approved text goes to the cloud" in controller.mode_var.get()
    assert controller.transport.start_calls == 0


def test_catalog_does_not_imply_connection_without_verified_availability():
    controller = _connection_controller(available=False)
    controller.enable_var.set(True)

    controller._discovery_result({"models": [{"id": "unverified", "efforts": ["low"]}]})

    assert controller.models == []
    assert "connection failed" in controller.mode_var.get()
    assert controller.transport.blocked_reason in controller.statuses[-1]
    assert controller.transport.start_calls == 0


def test_connection_failure_shows_actual_error_and_no_retry():
    controller = _connection_controller()
    controller.enable_var.set(True)

    controller._discovery_result({"error": "ChatGPT login status unavailable: exit 1"})

    assert "ChatGPT login status unavailable: exit 1" in controller.statuses[-1]
    assert "no retry is scheduled" in controller.statuses[-1]
    assert "connection failed" in controller.mode_var.get()
    assert controller.transport.discovery_calls == 0


def test_discovery_completed_after_disable_cannot_publish_stale_catalog():
    controller = _connection_controller()
    controller.closed = False
    controller.generation = 0
    controller.discovery_generation = 2
    controller.events = queue.Queue()
    controller.events.put(("discovery", 1, {"models": [{"id": "stale", "efforts": ["low"]}]}))
    controller.window = SimpleNamespace(after=lambda *args: None)

    controller._poll()

    assert controller.models == [] and controller.model_var.get() == ""
    assert controller.statuses == []


def test_fixed_policy_preview_reads_exact_bridge_text_without_connecting(monkeypatch):
    from meeting import codex_bridge

    assert fixed_policy_sections() == (("BASE_INSTRUCTIONS", codex_bridge.BASE_INSTRUCTIONS),
                                       ("DEVELOPER_INSTRUCTIONS", codex_bridge.DEVELOPER_INSTRUCTIONS))
    monkeypatch.setattr(codex_bridge, "BASE_INSTRUCTIONS", "Exact base\n政策 with spacing.  ")
    monkeypatch.setattr(codex_bridge, "DEVELOPER_INSTRUCTIONS", "Exact developer\npolicy.")
    assert fixed_policy_sections() == (("BASE_INSTRUCTIONS", "Exact base\n政策 with spacing.  "),
                                       ("DEVELOPER_INSTRUCTIONS", "Exact developer\npolicy."))


@pytest.mark.parametrize("action", ["_approve_local_preview", "_send_preview"])
def test_changed_fixed_policy_requires_new_review_before_approval(action, monkeypatch):
    class PolicyWindow(AssistantWindow):
        @property
        def gate(self):
            raise AssertionError("Changed policy must not consume meeting approval")

    controller = PolicyWindow.__new__(PolicyWindow)
    controller.transport = _TransportThatMustNotStart(True)
    controller.preview_manual = False
    controller.preview = object()
    controller.inflight = False
    controller.enable_var = _Value(True)
    controller.preview_reviewed_var = _Value(True)
    controller.current_session_id = "session-a"
    controller.get_session_id = lambda: "session-a"
    controller.preview_fixed_policies = (("BASE_INSTRUCTIONS", "old reviewed policy"),)
    invalidations, statuses = [], []
    controller._invalidate_preview = lambda: invalidations.append(True)
    controller._status = statuses.append
    monkeypatch.setattr("meeting.assistant_ui.fixed_policy_sections", lambda: (("BASE_INSTRUCTIONS", "new policy"),))

    assert getattr(controller, action)() is None

    assert invalidations == [True]
    assert "fixed policies changed" in statuses[0]
    assert controller.transport.starts == 0
