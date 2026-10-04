"""Protocol mocks only: this suite never starts Codex or submits cloud turns."""
from dataclasses import replace
import copy
from collections import deque
import io
import queue
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from meeting.assistant_core import QuestionDetector, build_preview
from meeting import codex_bridge as bridge
from meeting.codex_transport import CodexTransport, IsolationUnavailable


class FakeRpc:
    account_type = "chatgpt"
    instruction_sources = []
    mcp_status = []
    complete = True
    fault = None
    instances = []

    def __init__(self, executable, cwd, config):
        self.config, self.cwd = copy.deepcopy(config), cwd
        self.events, self.calls = queue.Queue(), []
        self.closed, self._fatal = False, None
        self.count = 0
        self.__class__.instances.append(self)

    def call(self, method, params, timeout=30):
        self.calls.append((method, copy.deepcopy(params)))
        if self.closed:
            raise IsolationUnavailable("Private connection closed")
        if self.fault == method:
            raise IsolationUnavailable("Synthetic required-method failure")
        if method == "account/read":
            return {"account": {"type": self.account_type, "email": "do-not-log@example.invalid"} if self.account_type else None}
        if method == "config/read":
            return {"config": {**copy.deepcopy(self.config), "mcp_servers": {"test.server": {"command": "DO-NOT-LOG", "headers": {"Authorization": "DO-NOT-LOG"}}}}}
        if method == "model/list":
            return {"data": [{"model": "mock-model", "displayName": "Mock model", "isDefault": True,
                              "supportedReasoningEfforts": [{"reasoningEffort": "low"}, {"reasoningEffort": "medium"}]}]}
        if method == "thread/start":
            self.count += 1
            return {"model": params["model"], "modelProvider": "openai", "approvalPolicy": "never",
                    "sandbox": {"type": "readOnly"}, "runtimeWorkspaceRoots": [],
                    "instructionSources": self.instruction_sources,
                    "thread": {"id": f"thread-{self.count}", "ephemeral": True}}
        if method == "mcpServerStatus/list":
            return {"data": self.mcp_status, "nextCursor": None}
        if method == "turn/start":
            self.last_thread = params["threadId"]
            if self.complete:
                self.finish(self.last_thread)
            return {"turn": {"id": "turn-1", "status": "inProgress"}}
        return {}

    def finish(self, thread_id=None):
        common = {"threadId": thread_id or self.last_thread, "turnId": "turn-1"}
        self.events.put({"method": "item/agentMessage/delta", "params": {**common, "delta": "Private "}})
        self.events.put({"method": "item/agentMessage/delta", "params": {**common, "delta": "answer [S1]."}})
        self.events.put({"method": "turn/completed", "params": {**common, "turn": {"id": "turn-1", "status": "completed"}}})

    def notify(self, method, params=None):
        self.calls.append((method, params))

    def close(self):
        self.closed = True


@pytest.fixture
def fake_environment(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Mock suite attempted a real process or network connection")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="codex-cli 0.146.0\n"))
    monkeypatch.setattr(bridge, "_Rpc", FakeRpc)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "clean-test-cli-profile"))
    FakeRpc.instances = []
    for name, value in (("account_type", "chatgpt"), ("instruction_sources", []), ("mcp_status", []), ("complete", True), ("fault", None)):
        monkeypatch.setattr(FakeRpc, name, value)
    transport = CodexTransport(tmp_path, executable="mock-codex.exe")
    yield transport
    transport.close()


def preview(transport, question="What is the status?", model="mock-model", effort="low", destination=None):
    candidate = QuestionDetector().manual(question)
    return build_preview(candidate, model=model, effort=effort, destination=destination or transport.destination)


def wait_done(handle):
    assert handle._done.wait(3), "Mock answer did not finish"


def test_import_construction_and_disconnected_start_have_no_effects(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unconnected transport attempted an external effect")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    transport = CodexTransport(tmp_path)
    assert transport.available is False
    with pytest.raises(IsolationUnavailable, match="Connect"):
        transport.start(preview(transport), forbidden)
    assert transport.capability_report()["payloads_submitted"] == 0
    transport.close()


def test_discovery_uses_only_sanitized_metadata_and_no_turn(fake_environment):
    transport = fake_environment
    models = transport.discover_models()
    rpc = FakeRpc.instances[-1]
    assert transport.available
    assert models == [{"id": "mock-model", "label": "Mock model", "efforts": ["low", "medium"], "is_default": True}]
    assert "turn/start" not in [x[0] for x in rpc.calls]
    assert "DO-NOT-LOG" not in str(transport.capability_report())
    start = next(params for method, params in rpc.calls if method == "thread/start")
    assert start["config"]["mcp_servers"] == {"test.server": {"enabled": False}}
    assert start["environments"] == start["dynamicTools"] == start["selectedCapabilityRoots"] == []
    assert start["ephemeral"] is True


@pytest.mark.parametrize("account_type", [None, "apiKey", "chatgptAuthTokens"])
def test_only_supported_cli_chatgpt_account_is_accepted(fake_environment, monkeypatch, account_type):
    monkeypatch.setattr(FakeRpc, "account_type", account_type)
    with pytest.raises(IsolationUnavailable, match="sign-in|authentication"):
        fake_environment.discover_models()
    assert FakeRpc.instances[-1].closed
    assert not fake_environment.available
    assert all(method not in ("thread/start", "turn/start") for method, _ in FakeRpc.instances[-1].calls)


def test_version_mismatch_stops_before_appserver(fake_environment, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="codex-cli 0.999.0"))
    with pytest.raises(IsolationUnavailable, match="0.146.0"):
        fake_environment.discover_models()
    assert not FakeRpc.instances


@pytest.mark.parametrize("problem", ["instructions", "mcp"])
def test_inherited_instructions_or_integrations_stop_before_payload(fake_environment, monkeypatch, problem):
    if problem == "instructions":
        monkeypatch.setattr(FakeRpc, "instruction_sources", ["PRIVATE-PATH-DO-NOT-LOG"])
    else:
        monkeypatch.setattr(FakeRpc, "mcp_status", [{"name": "unexpected"}])
    with pytest.raises(IsolationUnavailable) as error:
        fake_environment.discover_models()
    assert "PRIVATE-PATH" not in str(error.value)
    assert not fake_environment.available
    assert not any(method == "turn/start" for method, _ in FakeRpc.instances[-1].calls)
    assert FakeRpc.instances[-1].closed


def test_success_stream_exact_payload_and_fresh_threads(fake_environment):
    transport = fake_environment
    transport.discover_models()
    expected = preview(transport)
    events = []
    handle = transport.start(expected, events.append)
    wait_done(handle)
    rpc = FakeRpc.instances[-1]
    assert "".join(x["text"] for x in events if x["type"] == "token") == "Private answer [S1]."
    complete = next(x for x in events if x["type"] == "complete")
    assert complete["text"] == "Private answer [S1]."
    assert complete["metrics"]["submitted_at"] <= complete["metrics"]["first_token_at"] <= complete["metrics"]["completed_at"]
    turn = next(params for method, params in rpc.calls if method == "turn/start")
    assert turn["input"] == [{"type": "text", "text": expected.payload_text, "text_elements": []}]
    assert turn["environments"] == turn["runtimeWorkspaceRoots"] == []
    assert turn["sandboxPolicy"] == {"type": "readOnly", "networkAccess": False}
    assert turn["threadId"] == "thread-2"  # first thread was payload-free preflight
    wait_done(transport.start(preview(transport, "What changed?"), events.append))
    assert transport.capability_report()["payloads_submitted"] == 2
    assert [params["threadId"] for method, params in rpc.calls if method == "turn/start"] == ["thread-2", "thread-3"]


@pytest.mark.parametrize("question", ["What does $skill do?", "What costs $100?", "Read [$skill](x)?", "What about `$test`?"])
def test_explicit_skill_tokens_rejected_without_modifying_preview(fake_environment, question):
    fake_environment.discover_models()
    candidate = preview(fake_environment, question)
    before = len(FakeRpc.instances[-1].calls)
    with pytest.raises(IsolationUnavailable, match="skill invocation"):
        fake_environment.start(candidate, lambda x: None)
    assert len(FakeRpc.instances[-1].calls) == before


@pytest.mark.parametrize("overrides", [{"model": "unadvertised"}, {"effort": "ultra"}, {"destination": "another service"}])
def test_catalog_and_destination_must_match_review(fake_environment, overrides):
    fake_environment.discover_models()
    with pytest.raises(IsolationUnavailable, match="preview's model"):
        fake_environment.start(preview(fake_environment, **overrides), lambda x: None)


def test_mutated_preview_is_rejected(fake_environment):
    fake_environment.discover_models()
    edited = replace(preview(fake_environment), payload_text="unreviewed change")
    with pytest.raises(PermissionError):
        fake_environment.start(edited, lambda x: None)


def test_one_inflight_cancel_interrupts_and_closes_only_private_rpc(fake_environment, monkeypatch):
    transport = fake_environment
    monkeypatch.setattr(FakeRpc, "complete", False)
    transport.discover_models()
    events = []
    handle = transport.start(preview(transport), events.append)
    deadline = time.monotonic() + 3
    while handle.turn_id is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert handle.turn_id
    with pytest.raises(IsolationUnavailable, match="Only one"):
        transport.start(preview(transport), events.append)
    handle.cancel()
    wait_done(handle)
    rpc = FakeRpc.instances[-1]
    assert rpc.closed
    assert any(method == "turn/interrupt" for method, _ in rpc.calls)
    assert any(x["type"] == "cancelled" for x in events)
    assert not transport.available
    assert transport.capability_report()["payloads_submitted"] == 1


def test_disconnect_then_explicit_reconnect_has_no_backlog(fake_environment):
    transport = fake_environment
    transport.discover_models()
    old = FakeRpc.instances[-1]
    transport.disconnect()
    assert old.closed and not transport.available
    transport.discover_models()
    assert transport.available and FakeRpc.instances[-1] is not old
    assert all(method != "turn/start" for rpc in FakeRpc.instances for method, _ in rpc.calls)
    transport.close()
    with pytest.raises(IsolationUnavailable, match="closed"):
        transport.discover_models()


def test_rpc_failure_does_not_retry_or_leak_error(fake_environment, monkeypatch):
    transport = fake_environment
    transport.discover_models()
    monkeypatch.setattr(FakeRpc, "fault", "turn/start")
    events = []
    wait_done(transport.start(preview(transport), events.append))
    assert sum(method == "turn/start" for method, _ in FakeRpc.instances[-1].calls) == 1
    assert FakeRpc.instances[-1].closed and not transport.available
    assert [event["type"] for event in events][-1] == "error"


def test_unexpected_tool_activity_aborts_connection(fake_environment, monkeypatch):
    transport = fake_environment
    transport.discover_models()
    monkeypatch.setattr(FakeRpc, "complete", False)
    events = []
    handle = transport.start(preview(transport), events.append)
    deadline = time.monotonic() + 3
    while handle.turn_id is None and time.monotonic() < deadline:
        time.sleep(0.01)
    FakeRpc.instances[-1].events.put({"method": "item/started", "params": {
        "threadId": handle.thread_id, "turnId": handle.turn_id,
        "item": {"type": "commandExecution", "command": "MUST-NOT-BE-LOGGED"}}})
    wait_done(handle)
    assert FakeRpc.instances[-1].closed
    assert any("unexpected tool/activity" in x.get("message", "") for x in events)
    assert "MUST-NOT-BE-LOGGED" not in str(events)


def test_diagnostic_classifier_never_returns_credentials():
    assert bridge._safe_error('unknown field `skip_host_skill_discovery`, token=sk-secret') == "Installed Codex rejected isolation control 'skip_host_skill_discovery'."
    for raw in ("Bearer private-secret", "AUTHORIZATION: private-secret", "401 private-secret", "timeout private-secret", "unknown field `private-secret`"):
        assert "private-secret" not in bridge._safe_error(raw)


def test_cli_control_arguments_are_json_toml_values_not_shell_commands():
    config = bridge._controls(__import__("pathlib").Path("C:/test/$safe/assistant.txt"))
    flat = dict(bridge._flatten(config))
    assert flat["agents.enabled"] is False
    assert flat["notify"] == []
    assert flat["features.apps"] is False
    assert flat["features.hooks"] is False
    assert bridge._toml_scalar('C:\\a\\b') == '"C:\\\\a\\\\b"'


def test_private_child_cannot_inherit_custom_auth_routing(tmp_path, monkeypatch):
    keys = ("CODEX_REFRESH_TOKEN_URL_OVERRIDE", "CODEX_REVOKE_TOKEN_URL_OVERRIDE",
            "CODEX_APP_SERVER_LOGIN_CLIENT_ID")
    for key in keys:
        monkeypatch.setenv(key, "synthetic-custom-routing")
    observed = {}

    def intercept(*args, **kwargs):
        observed.update(kwargs["env"])
        raise OSError("Synthetic launch interception; no process started")

    monkeypatch.setattr(subprocess, "Popen", intercept)
    with pytest.raises(OSError, match="Synthetic launch interception"):
        bridge._Rpc("mock-codex.exe", tmp_path, {})
    assert all(key not in observed for key in keys)
    assert observed["CODEX_INTERNAL_APP_SERVER_REMOTE_CONTROL_DISABLED"] == "1"
    assert all(bridge.os.environ[key] == "synthetic-custom-routing" for key in keys)


def bare_rpc(stdout="", stderr=""):
    rpc = bridge._Rpc.__new__(bridge._Rpc)
    rpc.events = queue.Queue(maxsize=2)
    rpc._pending, rpc._counter = {}, 0
    rpc._lock, rpc._write_lock, rpc._close_lock = threading.Lock(), threading.Lock(), threading.Lock()
    rpc._closed = threading.Event()
    rpc._diagnostics, rpc._fatal = deque(maxlen=8), None
    rpc._stderr = SimpleNamespace(join=lambda timeout: None)
    rpc.process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(stdout), stderr=io.StringIO(stderr))
    return rpc


def test_server_initiated_request_is_rejected_before_handling():
    rpc = bare_rpc('{"id":12,"method":"item/commandExecution/requestApproval","params":{"secret":"DO-NOT-LOG"}}\n')
    rpc._read()
    reply = __import__("json").loads(rpc.process.stdin.getvalue())
    assert reply["id"] == 12 and reply["error"]["code"] == -32601
    assert "DO-NOT-LOG" not in str(reply)
    assert rpc._fatal == "Unexpected server action request rejected."
    assert rpc.events.empty()


def test_rpc_notification_backpressure_fails_without_unbounded_buffer():
    rpc = bare_rpc(''.join('{"method":"token","params":{}}\n' for _ in range(3)))
    rpc._read()
    assert rpc.events.qsize() == 2
    assert "buffer is full" in rpc._fatal


def test_rpc_malformed_or_oversized_response_does_not_echo_content():
    for payload in ("SECRET-NOT-JSON\n", "x" * 8_000_001):
        rpc = bare_rpc(payload)
        rpc._read()
        assert rpc._fatal.endswith("response rejected.")
        assert "SECRET" not in rpc._fatal


def test_stderr_keeps_only_bounded_sanitized_classifications():
    rpc = bare_rpc(stderr="unknown field `skip_host_skill_discovery`, secret=DO-NOT-LOG\n" * 20)
    rpc._read_errors()
    assert len(rpc._diagnostics) == 8
    assert "DO-NOT-LOG" not in str(rpc._diagnostics)
    assert "skip_host_skill_discovery" in rpc.failure_reason()


def test_close_reaps_only_owned_child_and_erases_queued_payloads():
    rpc = bare_rpc()
    calls = []
    rpc.process.poll = lambda: None
    rpc.process.terminate = lambda: calls.append("terminate-owned")
    rpc.process.wait = lambda timeout: calls.append("wait-owned")
    rpc.process.kill = lambda: calls.append("kill-owned")
    rpc.events.put({"payload": "erase unsent text"})
    rpc.close()
    rpc.close()
    assert calls == ["terminate-owned", "wait-owned"]
    assert rpc.events.empty()


def test_private_child_is_killed_and_reaped_if_termination_times_out():
    rpc = bare_rpc()
    calls = []
    rpc.process.poll = lambda: None
    rpc.process.terminate = lambda: calls.append("terminate-owned")
    rpc.process.kill = lambda: calls.append("kill-owned")
    def wait(timeout):
        calls.append("wait-owned")
        if len(calls) == 2:
            raise subprocess.TimeoutExpired("owned-child", timeout)
    rpc.process.wait = wait
    rpc.close()
    assert calls == ["terminate-owned", "wait-owned", "kill-owned", "wait-owned"]


@pytest.mark.parametrize("path,value", [
    (("features", "apps"), True), (("agents", "enabled"), True),
    (("skills", "include_instructions"), True), (("orchestrator", "skills", "enabled"), True),
    (("orchestrator", "mcp", "enabled"), True), (("project_doc_max_bytes",), 1000),
    (("notify",), ["unexpected-script"]), (("history", "persistence"), "save"),
    (("developer_instructions",), "inherited instructions"),
])
def test_each_critical_effective_control_is_verified(fake_environment, monkeypatch, path, value):
    original = FakeRpc.__init__
    def changed(self, *args):
        original(self, *args)
        target = self.config
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    monkeypatch.setattr(FakeRpc, "__init__", changed)
    with pytest.raises(IsolationUnavailable, match="not confirmed by effective configuration"):
        fake_environment.discover_models()
    assert not any(method == "thread/start" for method, _ in FakeRpc.instances[-1].calls)


@pytest.mark.parametrize("config", [
    {"openai_base_url": "https://example.invalid/codex?secret=DO-NOT-LOG"},
    {"chatgpt_base_url": "https://example.invalid/backend-api?secret=DO-NOT-LOG"},
    {"model_providers": {"openai": {"base_url": "https://example.invalid/v1"}}},
    {"model_providers": {"openai": {"base_url": "https://api.openai.com@evil.invalid/v1"}}},
])
def test_custom_endpoint_is_refused_without_echoing_values(fake_environment, monkeypatch, config):
    original = FakeRpc.__init__
    def changed(self, *args):
        original(self, *args)
        self.config.update(config)
    monkeypatch.setattr(FakeRpc, "__init__", changed)
    with pytest.raises(IsolationUnavailable, match="official OpenAI destinations") as error:
        fake_environment.discover_models()
    assert "DO-NOT-LOG" not in str(error.value)
    assert "example.invalid" not in str(error.value)
    assert not any(method == "thread/start" for method, _ in FakeRpc.instances[-1].calls)


def test_changed_config_after_discovery_fails_before_question(fake_environment):
    transport = fake_environment
    transport.discover_models()
    rpc = FakeRpc.instances[-1]
    rpc.config["features"]["apps"] = True
    events = []
    wait_done(transport.start(preview(transport), events.append))
    assert any(x["type"] == "error" for x in events)
    assert not any(method == "turn/start" for method, _ in rpc.calls)
    assert rpc.closed and not transport.available


def test_rediscovery_closes_previous_owned_server(fake_environment):
    fake_environment.discover_models()
    old = FakeRpc.instances[-1]
    fake_environment.discover_models()
    assert old.closed
    assert not FakeRpc.instances[-1].closed


def test_disconnect_during_discovery_prevents_late_success(fake_environment, monkeypatch):
    original = FakeRpc.call
    entered, release = threading.Event(), threading.Event()
    def delayed(self, method, params, timeout=30):
        if method == "model/list":
            entered.set()
            assert release.wait(3)
        return original(self, method, params, timeout)
    monkeypatch.setattr(FakeRpc, "call", delayed)
    errors = []
    def connect():
        try:
            fake_environment.discover_models()
        except IsolationUnavailable as exc:
            errors.append(str(exc))
    worker = threading.Thread(target=connect)
    worker.start()
    assert entered.wait(3)
    fake_environment.disconnect()
    release.set()
    worker.join(3)
    assert not worker.is_alive()
    assert errors and not fake_environment.available
    assert FakeRpc.instances[-1].closed
    assert not any(method == "turn/start" for method, _ in FakeRpc.instances[-1].calls)


@pytest.mark.parametrize("filename", ["AGENTS.md", "AGENTS.override.md"])
def test_global_guidance_metadata_refuses_before_any_process_or_content_read(fake_environment, monkeypatch, tmp_path, filename):
    from pathlib import Path
    codex_directory = tmp_path / "profile-with-guidance"
    codex_directory.mkdir()
    (codex_directory / filename).write_text("PRIVATE-GUIDANCE-NOT-READ", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex_directory))
    def forbidden(*args, **kwargs):
        pytest.fail("Metadata preflight attempted a process or file-content read")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(IsolationUnavailable, match="contents were not read") as error:
        fake_environment.discover_models()
    assert "PRIVATE-GUIDANCE" not in str(error.value)
    assert not FakeRpc.instances
    assert fake_environment.capability_report()["subprocesses_started_by_transport"] == 0


def test_relative_codex_home_is_refused_before_any_process(fake_environment, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", "relative-profile")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected subprocess"))
    with pytest.raises(IsolationUnavailable, match="absolute CLI profile"):
        fake_environment.discover_models()
    assert not FakeRpc.instances


@pytest.mark.parametrize("field", ["instructionSources", "runtimeWorkspaceRoots"])
def test_missing_required_context_metadata_is_not_assumed_empty(fake_environment, monkeypatch, field):
    original = FakeRpc.call
    def missing(self, method, params, timeout=30):
        result = original(self, method, params, timeout)
        if method == "thread/start":
            result.pop(field)
        return result
    monkeypatch.setattr(FakeRpc, "call", missing)
    with pytest.raises(IsolationUnavailable):
        fake_environment.discover_models()
    assert not any(method == "turn/start" for method, _ in FakeRpc.instances[-1].calls)
