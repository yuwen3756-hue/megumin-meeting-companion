"""Measure local candidate overhead, not audio or cloud latency.

Synthetic text only. The existing ASR logs are kept independent: adding these
microbenchmarks to old ASR latencies would not be an end-to-end measurement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meeting.assistant_core import QuestionDetector, build_preview
from meeting.codex_transport import CodexTransport
from scripts.replay_benchmark import distribution


def run(output: Path):
    from meeting.offline import enable
    enable()
    output.mkdir(parents=True, exist_ok=False)
    clock = [0.0]
    detector = QuestionDetector(clock=lambda: clock[0])
    elapsed, candidates, records, preview_times = [], 0, [], []
    # Alternate scripts and move the virtual receipt clock by one second per
    # row. This creates sustained 90-row dedup pressure without waiting hours.
    # It does not model audio timing, human approval, networking, or generation.
    for index in range(2000):
        clock[0] += 1
        question = (f"What is the next milestone for project {index}?" if index % 2 == 0
                    else f"项目{index}的下一个里程碑是什么？")
        event = {"type": "segment", "segment_id": index + 1, "source": "remote",
                 "speaker": "Unknown", "start": float(index * 4),
                 "end": float(index * 4 + 3), "text": question,
                 "caption_kind": "stable_chunk", "is_partial": False}
        started = time.perf_counter()
        candidate = detector.observe(event)
        finished = time.perf_counter()
        elapsed.append((finished - started) * 1000)
        if candidate:
            candidates += 1
            if index < 4:
                records.append({"candidate": asdict(candidate), "call_ms": elapsed[-1]})
            if index % 20 == 0:
                mark = time.perf_counter()
                preview = build_preview(candidate, segments=[],
                                        user_context="Synthetic project fact: no delivery date is agreed.",
                                        model="Not selected — manual Codex session",
                                        effort="Not selected", destination="Manual transfer to Codex (cloud)")
                preview_times.append((time.perf_counter() - mark) * 1000)
                assert "Synthetic project fact" in preview.payload_text
        assert len(detector.pending()) <= 100
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Synthetic text microbenchmark on this hardware; no capture, ASR, or cloud calls",
        "source_sha256": hashlib.sha256((ROOT / "meeting/assistant_core.py").read_bytes()).hexdigest(),
        "input_rows": 2000, "candidate_count": candidates,
        "pending_after_run": len(detector.pending()),
        "event_to_candidate_call_ms": distribution(elapsed),
        "preview_build_ms": distribution(preview_times),
        "question_audio_end_to_candidate_ms": None,
        "submitted_to_first_token_ms": None,
        "submitted_to_complete_ms": None,
        "human_review_ms": None,
        "cloud_submission": CodexTransport(ROOT).capability_report(),
        "sample_candidates": records,
        "limits": ["Virtual time advances only the deduplication window; runtime is measured with perf_counter.",
                   "This excludes Tk scheduling and audio/ASR latency. Do not add it to past ASR timings as an end-to-end claim.",
                   "No cloud turn was submitted; token timing is unavailable, not zero."],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in {
        "input_rows", "candidate_count", "pending_after_run", "event_to_candidate_call_ms", "preview_build_ms"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/assistant-detector-benchmark")
    run(ROOT / parser.parse_args().output)
