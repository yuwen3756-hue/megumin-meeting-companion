"""Local question/preview/approval contracts; no model or transport is used."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields, replace
import importlib
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import pytest


@pytest.fixture(autouse=True)
def forbid_transport_and_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("assistant-core tests must never use transport or execute commands")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)


@pytest.fixture
def core():
    return importlib.import_module("meeting.assistant_core")


def segment(text, start=100.0, end=101.0, identifier="segment-question", **changes):
    event = {
        "type": "segment", "caption_kind": "stable_chunk", "is_partial": False,
        "id": identifier, "segment_id": identifier, "source": "remote",
        "speaker": "Remote 1", "start": start, "end": end, "text": text,
    }
    event.update(changes)
    return event


@pytest.fixture
def candidate(core):
    return core.Candidate(
        id="candidate-a", segment_id="segment-question", source="remote",
        speaker="Remote 1", start=100.0, end=101.0,
        text="What is the deployment plan?", detected_at=42.0,
    )


@pytest.fixture
def preview_factory(core, candidate):
    def make(selected_candidate=None, **changes):
        options = {
            "model": "gpt-5-mini", "effort": "low",
            "destination": "https://api.openai.com/v1/responses",
        }
        options.update(changes)
        return core.build_preview(selected_candidate or candidate, **options)

    return make


@pytest.mark.parametrize("question", [
    "What is the next milestone?", "Can you explain the rollout plan?",
    "How should we handle retries", "这个方案什么时候上线？", "这是谁负责的", "这两种方案有什么区别",
])
def test_heuristic_questions_remain_addressing_uncertain(core, question):
    detector = core.QuestionDetector(clock=lambda: 42.5)
    found = detector.observe(segment(question), segment_id="stored-segment-17")
    assert found is not None
    assert found.text == question
    assert found.segment_id == "stored-segment-17"
    assert found.source == "remote" and found.speaker == "Remote 1"
    assert found.addressing_uncertain is True
    assert found.detected_at == pytest.approx(42.5)
    assert found.status == "pending"
    assert detector.pending() == (found,)


@pytest.mark.parametrize("changes", [
    {"type": "warning"}, {"type": "status"}, {"is_partial": True},
    {"caption_kind": "partial"}, {"text": "Deployment completed successfully."},
])
def test_non_questions_and_unstable_events_are_not_candidates(core, changes):
    detector = core.QuestionDetector()
    event = segment("What happens next?")
    event.update(changes)
    assert detector.observe(event) is None
    assert detector.pending() == ()


def test_stored_stable_segments_can_omit_caption_kind(core):
    event = segment("What happens next?")
    event.pop("caption_kind")
    assert core.QuestionDetector().observe(event) is not None


def test_mandarin_alternative_asr_fragment_is_reviewable_without_reconstruction(core):
    observed = "还是记录到的来电"
    found = core.QuestionDetector().observe(segment(observed))
    assert found is not None
    assert found.text == observed and found.addressing_uncertain
    assert found.segment_id == "segment-question"


def test_mandarin_still_statement_is_not_added_by_alternative_clause_rule(core):
    assert core.QuestionDetector().observe(segment("我还是支持原来的方案")) is None


def test_unicode_fuzzy_deduplication_and_cooldown_expiry(core):
    now = [10.0]
    detector = core.QuestionDetector(cooldown_seconds=90, clock=lambda: now[0])
    first = detector.observe(segment("Can you explain the deployment plan?"))
    assert first is not None
    detector.set_status(first.id, "dismissed")
    now[0] = 20.0
    assert detector.observe(segment("ＣＡＮ  you explain the deployment plan？")) is None
    assert detector.observe(segment("Can you explain the deployment plans?")) is None
    now[0] = 99.0
    assert detector.observe(segment("Can you explain the deployment plan?")) is None
    now[0] = 200.0
    repeated = detector.observe(segment("Can you explain the deployment plan?"))
    assert repeated is not None and repeated.id != first.id


@pytest.mark.parametrize("first_question,changed_question", [
    ("Should we retain audit logs for 30 days?", "Should we retain audit logs for 90 days?"),
    ("Should we keep two backup replicas?", "Should we keep three backup replicas?"),
    ("Should we enable the legacy integration for this deployment?",
     "Should we not enable the legacy integration for this deployment?"),
    ("我们需要保留三份备份吗？", "我们需要保留五份备份吗？"),
    ("这个部署应该启用自动重试吗？", "这个部署不应该启用自动重试吗？"),
])
def test_numeric_and_negation_changes_are_not_fuzzy_duplicates(core, first_question, changed_question):
    detector = core.QuestionDetector()
    first = detector.observe(segment(first_question))
    changed = detector.observe(segment(changed_question))
    assert first is not None and changed is not None
    assert first.id != changed.id


@pytest.mark.parametrize("changes", [{"source": "local"}, {"speaker": "Remote 2"}])
def test_deduplication_is_scoped_to_source_and_speaker_label(core, changes):
    detector = core.QuestionDetector()
    question = "What is the deployment plan?"
    first = detector.observe(segment(question))
    other_event = segment(question, **changes)
    other = detector.observe(other_event)
    assert first is not None and other is not None and first.id != other.id
    assert first.addressing_uncertain and other.addressing_uncertain
    assert detector.observe(other_event) is None


def test_live_segment_identifier_and_reviewed_speaker_label_are_preserved(core):
    event = segment("What happens next?", identifier="event-id")
    event.update(segment_id="stored-segment-id", speaker="Speaker 2", speaker_name="Reviewed voice label")
    found = core.QuestionDetector().observe(event)
    assert found is not None
    assert found.segment_id == "stored-segment-id"
    assert found.speaker == "Reviewed voice label"
    assert found.addressing_uncertain is True
    explicit = core.QuestionDetector().observe(event, segment_id="explicit-segment-id")
    assert explicit.segment_id == "explicit-segment-id"


def test_manual_candidates_are_bounded_frozen_and_clearable(core):
    detector = core.QuestionDetector(max_pending=3, clock=lambda: 8.0)
    questions = ["Explain oats.", "Compare satellites.", "Summarize escrow.", "Define recursion."]
    created = [detector.manual(text, start=3.0, end=4.0) for text in questions]
    pending = detector.pending()
    assert isinstance(pending, tuple) and len(pending) <= 3
    assert created[-1] in pending
    with pytest.raises(FrozenInstanceError):
        created[-1].text = "mutated"
    detector.set_status(created[-1].id, "reviewed")
    assert all(item.id != created[-1].id for item in detector.pending())
    detector.clear()
    assert detector.pending() == ()


@pytest.mark.parametrize("max_pending", [0, 101])
def test_pending_configuration_cannot_disable_the_bound(core, max_pending):
    with pytest.raises(ValueError):
        core.QuestionDetector(max_pending=max_pending)


def test_auto_question_excerpts_are_marked_but_long_manual_questions_are_rejected(core):
    detector = core.QuestionDetector()
    question = "Can you explain " + "x" * 1500 + "?"
    found = detector.observe(segment(question))
    assert found is not None and len(found.text) <= 1200
    assert found.excerpt_truncated is True
    with pytest.raises(ValueError):
        detector.manual(question)


def test_selected_files_are_read_only_and_not_discovered_by_scanning(core, tmp_path, monkeypatch):
    selected = tmp_path / "selected.md"
    selected.write_text("SELECTED_EVIDENCE", encoding="utf-8")
    (tmp_path / "unselected.txt").write_text("UNSELECTED_PRIVATE_CONTENT", encoding="utf-8")
    before = selected.read_bytes()

    def no_scan(*args, **kwargs):
        raise AssertionError("only explicit file paths may be read")

    for name in ("iterdir", "glob", "rglob"):
        monkeypatch.setattr(Path, name, no_scan)
    sources = core.load_selected_files([selected], allowed_directory=tmp_path)
    assert isinstance(sources, tuple) and len(sources) == 1
    assert sources[0].text == "SELECTED_EVIDENCE"
    assert sources[0].id.startswith("F")
    assert selected.read_bytes() == before
    with pytest.raises(FrozenInstanceError):
        sources[0].text = "mutated"


@pytest.mark.parametrize("suffix", [".txt", ".md", ".json"])
def test_only_supported_explicit_text_files_are_loaded(core, tmp_path, suffix):
    path = tmp_path / ("evidence" + suffix)
    path.write_text('{"note":"中文 evidence"}', encoding="utf-8")
    assert core.load_selected_files([path])[0].text == '{"note":"中文 evidence"}'


def test_file_selection_rejects_directories_unsupported_and_missing_paths(core, tmp_path):
    unsupported = tmp_path / "report.pdf"
    unsupported.write_bytes(b"not a real PDF")
    with pytest.raises(ValueError):
        core.load_selected_files([tmp_path])
    with pytest.raises(ValueError):
        core.load_selected_files([unsupported])
    with pytest.raises(FileNotFoundError):
        core.load_selected_files([tmp_path / "missing.txt"])


def test_allowed_directory_does_not_authorize_sibling_prefixes_or_parent_traversal(core, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    sibling = tmp_path / "allowed-private"
    sibling.mkdir()
    outside = sibling / "secret.txt"
    outside.write_text("OUTSIDE", encoding="utf-8")
    for picked in (outside, allowed / ".." / "allowed-private" / "secret.txt"):
        with pytest.raises(PermissionError):
            core.load_selected_files([picked], allowed_directory=allowed)


@pytest.mark.parametrize("raw_path", [
    r"\\server\share\notes.txt", "//server/share/notes.txt",
    r"\\?\C:\notes.txt", r"\\.\PIPE\assistant-test",
    r"\??\C:\notes.txt", r"\Device\HarddiskVolume1\notes.txt",
    r"\GLOBAL??\C:\notes.txt",
])
@pytest.mark.parametrize("selection", ["path", "allowed_directory"])
def test_network_and_device_paths_are_rejected_before_resolving_or_statting(
    core, monkeypatch, raw_path, selection
):
    def forbidden(*args, **kwargs):
        raise AssertionError("network/device paths must be rejected before filesystem resolution")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    with pytest.raises(ValueError):
        if selection == "path":
            core.load_selected_files([raw_path])
        else:
            core.load_selected_files([], allowed_directory=raw_path)


def test_selection_count_and_utf8_byte_limits(core, tmp_path):
    files = []
    for index in range(6):
        path = tmp_path / f"file-{index}.txt"
        path.write_text("selected", encoding="utf-8")
        files.append(path)
    assert len(core.load_selected_files(files[:5])) == 5
    with pytest.raises(ValueError):
        core.load_selected_files(files)
    boundary = tmp_path / "boundary.txt"
    boundary.write_bytes(b"x" * (128 * 1024))
    assert len(core.load_selected_files([boundary])[0].text) == 128 * 1024
    oversized = tmp_path / "multibyte.txt"
    oversized.write_text("界" * 44_000, encoding="utf-8")
    assert len(oversized.read_text(encoding="utf-8")) < 128 * 1024
    with pytest.raises(ValueError):
        core.load_selected_files([oversized])
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff\xfe\x80")
    with pytest.raises(ValueError):
        core.load_selected_files([invalid])


def test_preview_selects_only_preceding_transcript_inside_the_window(preview_factory):
    transcript = [
        segment("BOUNDARY_INCLUDED", 40, 41, "boundary"),
        segment("JUST_BEFORE_INCLUDED", 99, 100, "preceding"),
        segment("TOO_OLD_EXCLUDED", 39, 40, "old"),
        segment("FUTURE_EXCLUDED", 100, 102, "future"),
        segment("CANDIDATE_EXCLUDED", 99, 100, "segment-question"),
    ]
    preview = preview_factory(segments=transcript)
    text = "\n".join(source.text for source in preview.sources)
    assert "BOUNDARY_INCLUDED" in text and "JUST_BEFORE_INCLUDED" in text
    assert all(marker not in text for marker in ("TOO_OLD_EXCLUDED", "FUTURE_EXCLUDED", "CANDIDATE_EXCLUDED"))
    assert all(source.id.startswith("S") for source in preview.sources)


def test_preview_limits_transcript_to_the_most_recent_six(preview_factory):
    transcript = [segment(f"EVIDENCE_{index}", 50 + index, 51 + index, str(index)) for index in range(9)]
    preview = preview_factory(segments=transcript)
    sources = [source for source in preview.sources if source.id.startswith("S")]
    assert len(sources) == 6
    text = "\n".join(source.text for source in sources)
    assert all(f"EVIDENCE_{index}" in text for index in range(3, 9))
    assert all(f"EVIDENCE_{index}" not in text for index in range(3))


def test_preview_sources_have_bounded_text_and_explicit_attribution(core, preview_factory):
    files = tuple(core.Source(f"F{index + 1}", "file", f"Selected {index}", "z" * 2500) for index in range(5))
    preview = preview_factory(
        segments=[segment("Prior transcript context", 90, 91, "prior")],
        files=files, user_context="User supplied context " * 1000,
    )
    assert isinstance(preview.sources, tuple)
    assert sum(len(source.text) for source in preview.sources) <= 6000
    assert len([source for source in preview.sources if source.id.startswith("F")]) <= 5
    assert len({source.id for source in preview.sources}) == len(preview.sources)
    assert preview.limitations
    assert all(isinstance(source, core.Source) for source in preview.sources)


def test_edited_question_limit_is_enforced(preview_factory):
    question = "What " + "x" * 1194 + "?"
    assert len(question) == 1200
    assert preview_factory(question=question).question == question
    with pytest.raises(ValueError):
        preview_factory(question=question + "x")


SECRETS = [
    "sk-proj-" + "A1b2C3d4" * 8,
    "sk-" + "aB12" * 12,
    "ghp_" + "B" * 36,
    "github_pat_" + "C" * 82,
    "xoxb-123456789012-123456789012-" + "Z" * 24,
    "AKIA" + "D" * 16,
]


@pytest.mark.parametrize("secret", SECRETS)
def test_recognizable_tokens_are_redacted_from_preview_and_labels(core, preview_factory, secret):
    source = core.Source("F1", "file", "Selected notes", f"Service token: {secret}")
    preview = preview_factory(files=(source,), user_context=f"Copy of token: {secret}")
    assert secret not in preview.payload_text
    assert secret not in " ".join(preview.redactions)
    assert preview.redactions
    assert any("heuristic" in text.lower() or "complete" in text.lower() for text in preview.limitations)


def test_bearer_password_and_private_key_values_are_redacted(preview_factory):
    bearer = "abcd1234efgh5678ijkl9012mnop3456"
    password = "violetquartz alphaomega"
    pem_body = "MIIBSyntheticPrivateKeyMaterialOnly123456789"
    context = (
        f"Authorization: Bearer {bearer}\npassword = \"{password}\"\n"
        f"-----BEGIN PRIVATE KEY-----\n{pem_body}\n-----END PRIVATE KEY-----"
    )
    preview = preview_factory(user_context=context)
    for secret in (bearer, "violetquartz", "alphaomega", pem_body):
        assert secret not in preview.payload_text
        assert secret not in " ".join(preview.redactions)
    assert preview.redactions


def test_secrets_in_questions_are_redacted_before_payload_construction(preview_factory):
    secret = SECRETS[0]
    preview = preview_factory(question=f"Why did credential {secret} fail?")
    assert secret not in preview.payload_text
    assert secret not in " ".join(preview.redactions)


def test_context_truncation_does_not_expose_a_partial_secret(preview_factory):
    secret = SECRETS[0]
    preview = preview_factory(user_context="x" * 5990 + " " + secret)
    assert sum(len(source.text) for source in preview.sources) <= 6000
    assert secret[:9] not in preview.payload_text


def test_contact_information_requires_explicit_sensitive_acknowledgment(core, preview_factory):
    preview = preview_factory(user_context="Contact alice@example.test, phone +1 415 555 2671.")
    assert preview.sensitive_flags
    gate = core.ApprovalGate()
    with pytest.raises(PermissionError):
        gate.approve(preview, reviewed_digest=preview.digest)
    receipt = gate.approve(preview, reviewed_digest=preview.digest, acknowledge_sensitive=True)
    assert gate.consume(preview, receipt) is preview


def test_approval_requires_reviewed_digest_and_is_single_use(core, preview_factory):
    preview = preview_factory()
    gate = core.ApprovalGate()
    with pytest.raises(PermissionError):
        gate.approve(preview, reviewed_digest="not the reviewed preview")
    receipt = gate.approve(preview, reviewed_digest=preview.digest)
    assert gate.consume(preview, receipt) is preview
    with pytest.raises(PermissionError):
        gate.consume(preview, receipt)
    with pytest.raises(FrozenInstanceError):
        preview.model = "another-model"
    with pytest.raises(FrozenInstanceError):
        setattr(receipt, fields(receipt)[0].name, "mutated")


@pytest.mark.parametrize("field,new_value", [
    ("id", "preview-b"), ("candidate_id", "candidate-b"), ("question", "What changed?"),
    ("model", "different-model"), ("effort", "high"),
    ("destination", "https://different.example.test"),
    ("payload_text", "unreviewed payload"), ("digest", "0" * 64),
])
def test_approved_preview_cannot_be_reused_after_field_tampering(
    core, preview_factory, field, new_value
):
    preview = preview_factory()
    gate = core.ApprovalGate()
    receipt = gate.approve(preview, reviewed_digest=preview.digest)
    tampered = replace(preview, **{field: new_value})
    with pytest.raises(PermissionError):
        gate.consume(tampered, receipt)


def test_changed_source_content_cannot_reuse_a_reviewed_receipt(core, preview_factory):
    preview = preview_factory(user_context="Reviewed manual evidence.")
    assert preview.sources
    gate = core.ApprovalGate()
    receipt = gate.approve(preview, reviewed_digest=preview.digest)
    altered_source = replace(preview.sources[0], text="Unreviewed replacement evidence.")
    tampered = replace(preview, sources=(altered_source,) + preview.sources[1:])
    with pytest.raises(PermissionError):
        gate.consume(tampered, receipt)


def test_tampered_preview_cannot_be_approved_with_its_old_digest(core, preview_factory):
    preview = preview_factory()
    tampered = replace(preview, model="unreviewed-model")
    with pytest.raises(PermissionError):
        core.ApprovalGate().approve(tampered, reviewed_digest=preview.digest)


def test_sensitive_ack_cannot_be_bypassed_by_replacing_flags(core, preview_factory):
    preview = preview_factory(user_context="Contact alice@example.test.")
    assert preview.sensitive_flags
    tampered = replace(preview, sensitive_flags=())
    with pytest.raises(PermissionError):
        core.ApprovalGate().approve(tampered, reviewed_digest=preview.digest)


def test_new_approval_revokes_old_same_candidate_without_revoking_other_candidates(
    core, preview_factory, candidate
):
    first = preview_factory()
    other = preview_factory(replace(candidate, id="candidate-b"))
    edited = preview_factory(question="What is the revised deployment plan?")
    gate = core.ApprovalGate()
    old_receipt = gate.approve(first, reviewed_digest=first.digest)
    other_receipt = gate.approve(other, reviewed_digest=other.digest)
    new_receipt = gate.approve(edited, reviewed_digest=edited.digest)
    with pytest.raises(PermissionError):
        gate.consume(first, old_receipt)
    assert gate.consume(edited, new_receipt) is edited
    assert gate.consume(other, other_receipt) is other


def test_invalidation_is_scoped_and_can_revoke_all(core, preview_factory, candidate):
    first = preview_factory()
    other = preview_factory(replace(candidate, id="candidate-b"))
    gate = core.ApprovalGate()
    first_receipt = gate.approve(first, reviewed_digest=first.digest)
    other_receipt = gate.approve(other, reviewed_digest=other.digest)
    gate.invalidate(first.candidate_id)
    with pytest.raises(PermissionError):
        gate.consume(first, first_receipt)
    assert gate.consume(other, other_receipt) is other
    first_receipt = gate.approve(first, reviewed_digest=first.digest)
    other_receipt = gate.approve(other, reviewed_digest=other.digest)
    gate.invalidate()
    for preview, receipt in ((first, first_receipt), (other, other_receipt)):
        with pytest.raises(PermissionError):
            gate.consume(preview, receipt)


def test_receipts_are_not_authority_in_a_different_gate(core, preview_factory):
    preview = preview_factory()
    receipt = core.ApprovalGate().approve(preview, reviewed_digest=preview.digest)
    with pytest.raises(PermissionError):
        core.ApprovalGate().consume(preview, receipt)


def test_concurrent_receipt_consumption_has_exactly_one_success(core, preview_factory):
    preview = preview_factory()
    gate = core.ApprovalGate()
    receipt = gate.approve(preview, reviewed_digest=preview.digest)
    barrier = threading.Barrier(3)

    def consume():
        barrier.wait(timeout=3)
        try:
            return gate.consume(preview, receipt) is preview
        except PermissionError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(consume) for _ in range(2)]
        barrier.wait(timeout=3)
        assert sum(future.result(timeout=3) for future in futures) == 1


def test_detection_timestamp_includes_fuzzy_comparison_work(core, monkeypatch):
    now = [100.0]

    class TimedComparison:
        def __init__(self, *args, **kwargs):
            pass

        def ratio(self):
            now[0] += 0.007
            return 0.0

    monkeypatch.setattr(core, "SequenceMatcher", TimedComparison)
    detector = core.QuestionDetector(clock=lambda: now[0])
    assert detector.observe(segment("What is the deployment plan?")) is not None
    found = detector.observe(segment("What is the deployment risk?"))
    assert found is not None
    assert now[0] == pytest.approx(100.007)
    assert found.detected_at == pytest.approx(now[0])
    assert found.received_at == pytest.approx(100.0)
    assert found.detection_ms == pytest.approx(7.0)


def test_thousand_stable_events_keep_pending_bounded_with_scalar_timing(core, record_property):
    detector = core.QuestionDetector(max_pending=100)
    count = 1000
    total_observe_ms = maximum_observe_ms = maximum_candidate_delay_ms = 0.0
    maximum_pending = 0
    for index in range(count):
        # Short, distinct questions avoid conflating this bound check with fuzzy matching.
        event = segment(f"Why {index}?", start=index * 2, end=index * 2 + 1, identifier=str(index))
        received_at = time.perf_counter()
        found = detector.observe(event)
        finished_at = time.perf_counter()
        assert found is not None
        assert received_at <= found.received_at <= found.detected_at <= finished_at
        assert found.detection_ms == pytest.approx((found.detected_at - found.received_at) * 1000)
        observe_ms = (finished_at - received_at) * 1000
        candidate_delay_ms = (found.detected_at - received_at) * 1000
        total_observe_ms += observe_ms
        maximum_observe_ms = max(maximum_observe_ms, observe_ms)
        maximum_candidate_delay_ms = max(maximum_candidate_delay_ms, candidate_delay_ms)
        maximum_pending = max(maximum_pending, len(detector.pending()))
        assert maximum_pending <= 100

    pending = detector.pending()
    assert len(pending) == 100
    assert pending[0].text == "Why 900?" and pending[-1].text == "Why 999?"
    metrics = {
        "synthetic_stable_events": count,
        "maximum_pending_candidates": maximum_pending,
        "mean_observe_ms": round(total_observe_ms / count, 6),
        "maximum_observe_ms": round(maximum_observe_ms, 6),
        "maximum_event_to_candidate_ms": round(maximum_candidate_delay_ms, 6),
    }
    for key, value in metrics.items():
        record_property(key, value)
    print("Synthetic detector timing (no ASR):", metrics)
