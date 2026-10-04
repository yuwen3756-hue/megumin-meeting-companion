"""Derive version-2 replay measurements from immutable report/JSONL evidence.

No models, audio devices, GPU runtimes, GUI, network, or third-party packages are
loaded. The raw report and journal are never modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.replay_benchmark import distribution, quality_score


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def signature(event):
    return tuple(event.get(key) for key in ("source", "start", "end", "text", "received_monotonic"))


def derive(report: dict, records: list[dict]) -> dict:
    cases = report.get("cases", [])
    captions = [event for event in records if event.get("type") == "segment" and str(event.get("text", "")).strip()]
    resources = [event for event in records if event.get("type") == "resources"]
    report_segment_lookup = defaultdict(list)
    segment_case_lookup = defaultdict(list)
    for case in cases:
        for event in case.get("segments", []):
            report_segment_lookup[signature(event)].append(event)
            segment_case_lookup[signature(event)].append(case)
    ui_by_id = defaultdict(list)
    for event in records:
        if event.get("type") == "ui_applied" and finite(event.get("elapsed_seconds")):
            ui_by_id[str(event["segment_id"])].append(event["elapsed_seconds"])
    warnings = []
    ambiguous = []
    event_rows = []

    def actual_bound(case, audio_time):
        if case and all(finite(value) for value in (audio_time, case.get("injection_started_elapsed"), case.get("timeline_start"))):
            return case["injection_started_elapsed"] + audio_time - case["timeline_start"]
        return None

    def delay(value, origin):
        return (value - origin) * 1000 if finite(value) and finite(origin) else None

    for event in captions:
        key = signature(event)
        matches = report_segment_lookup.get(key, [])
        case_matches = segment_case_lookup.get(key, [])
        segment_ids = {str(match["segment_id"]) for match in matches if "segment_id" in match}
        if len(segment_ids) > 1 or len({case["id"] for case in case_matches}) > 1:
            ambiguous.append({"signature": key, "segment_ids": sorted(segment_ids)})
        case = case_matches[0] if len({case["id"] for case in case_matches}) == 1 else None
        ui_times = [value for ident in segment_ids for value in ui_by_id.get(ident, [])] if len(segment_ids) == 1 else []
        ui_time = min(ui_times) if ui_times else None
        callback_time = event.get("received_elapsed_seconds")
        speech_end = actual_bound(case, event.get("speech_last_voiced_end"))
        utterance_start = actual_bound(case, event.get("utterance_start"))
        event_rows.append({"source": event.get("source"), "segment_id": next(iter(segment_ids)) if len(segment_ids) == 1 else None,
                           "case_id": case["id"] if case else None, "text": event["text"],
                           "utterance_id": event.get("utterance_id"), "chunk_index": event.get("chunk_index"),
                           "first_in_utterance": bool(event.get("first_in_utterance")),
                           "final_in_utterance": bool(event.get("final_in_utterance")),
                           "callback_elapsed_seconds": callback_time, "ui_applied_elapsed_seconds": ui_time,
                           "actual_speech_end_elapsed_seconds": speech_end,
                           "actual_utterance_start_elapsed_seconds": utterance_start,
                           "callback_after_speech_end_ms": delay(callback_time, speech_end),
                           "ui_after_speech_end_ms": delay(ui_time, speech_end),
                           "callback_from_utterance_onset_ms": delay(callback_time, utterance_start),
                           "ui_from_utterance_onset_ms": delay(ui_time, utterance_start),
                           "ui_delivery_ms": delay(ui_time, callback_time)})

    clip_groups = defaultdict(list)
    omitted_clip_metadata = 0
    for event in captions:
        if finite(event.get("clip_start")) and finite(event.get("clip_submitted_at")):
            clip_groups[(event.get("source"), event["clip_start"], event["clip_submitted_at"])].append(event)
        else:
            omitted_clip_metadata += 1
    clips = []
    disagreements = []
    for key, events in clip_groups.items():
        starts = [event["clip_start"] for event in events]
        ends = [event["clip_end"] for event in events if finite(event.get("clip_end"))]
        duration = max(ends) - min(starts) if ends else None
        clip = {"source": key[0], "clip_start": key[1], "clip_submitted_at": key[2],
                "clip_duration_seconds": duration, "emitted_caption_count": len(events)}
        for metric in ("asr_ms", "speaker_ms", "worker_ms", "queue_wait_ms"):
            values = [event[metric] for event in events if finite(event.get(metric))]
            clip[metric] = max(values) if values else None
            # Worker time can legitimately increase across emitted ASR segments
            # because earlier callbacks are part of the same worker invocation.
            if metric != "worker_ms" and len(set(values)) > 1:
                disagreements.append({"clip": key, "metric": metric, "values": values})
        clips.append(clip)

    waveform_seconds = sum(case.get("audio_seconds", 0) for case in cases if finite(case.get("audio_seconds")))
    paced_source_seconds = waveform_seconds + .8 * len(cases)
    wall_seconds = report.get("replay_wall_seconds")
    costs = {}
    for metric in ("asr_ms", "speaker_ms", "worker_ms"):
        valid = [clip for clip in clips if finite(clip.get(metric)) and finite(clip.get("clip_duration_seconds"))
                 and clip["clip_duration_seconds"] > 0]
        cost_seconds = sum(clip[metric] for clip in valid) / 1000
        processed_audio_seconds = sum(clip["clip_duration_seconds"] for clip in valid)
        costs[metric] = {"unique_clip_count": len(valid), "total_seconds": cost_seconds,
                         "processed_audio_seconds": processed_audio_seconds,
                         "pooled_processed_audio_rtf": cost_seconds / processed_audio_seconds if processed_audio_seconds else None,
                         "source_waveform_rtf": cost_seconds / waveform_seconds if waveform_seconds else None,
                         "paced_source_rtf": cost_seconds / paced_source_seconds if paced_source_seconds else None,
                         "wall_utilization_fraction": cost_seconds / wall_seconds if finite(wall_seconds) and wall_seconds > 0 else None,
                         "per_clip_ms": distribution([clip[metric] for clip in valid])}

    snapshots = [event.get("engine", {}) for event in resources] + [report.get("final_engine_status", {})]
    maxima = {}
    for key in ("queued_audio", "queued_segments", "oldest_audio_queue_age_ms", "oldest_segment_queue_age_ms",
                "dropped_audio_blocks", "dropped_audio_seconds", "dropped_segments", "failed_segments", "event_callback_errors"):
        values = [snapshot[key] for snapshot in snapshots if finite(snapshot.get(key))]
        maxima[key] = max(values) if values else None
    maxima["queue_wait_ms"] = distribution([clip.get("queue_wait_ms") for clip in clips])
    maxima["counter_interpretation"] = "Drop/failure totals are maxima of cumulative session counters, not sums of repeated snapshots."

    fixture_rows = []
    for case in cases:
        associated = [event for event in event_rows if event["case_id"] == case["id"]]
        callbacks = [event["callback_elapsed_seconds"] for event in associated if finite(event["callback_elapsed_seconds"])]
        ui = [event["ui_applied_elapsed_seconds"] for event in associated if finite(event["ui_applied_elapsed_seconds"])]
        onset, end = actual_bound(case, case.get("speech_onset")), actual_bound(case, case.get("speech_end"))
        observed = case.get("observed", "")
        quality = quality_score(case.get("expected", ""), observed, case["language"])
        fixture_rows.append({"id": case["id"], "source": case["source"], "repeat": case["repeat"],
                             "fixture": case["fixture"], "language": case["language"],
                             "expected": case.get("expected", ""), "observed": observed, "quality": quality,
                             "actual_speech_onset_seconds": onset, "actual_speech_end_seconds": end,
                             "caption_count": len(associated), "ui_applied_caption_count": len(ui),
                             "ui_observation_complete": bool(associated) and len(ui) == len(associated),
                             "first_callback_elapsed_seconds": min(callbacks) if callbacks else None,
                             "last_callback_elapsed_seconds": max(callbacks) if callbacks else None,
                             "first_ui_elapsed_seconds": min(ui) if ui else None,
                             "last_ui_elapsed_seconds": max(ui) if ui else None,
                             "first_callback_from_speech_onset_ms": delay(min(callbacks), onset) if callbacks else None,
                             "last_callback_after_speech_end_ms": delay(max(callbacks), end) if callbacks else None,
                             "first_ui_from_speech_onset_ms": delay(min(ui), onset) if ui else None,
                             "last_ui_after_speech_end_ms": delay(max(ui), end) if ui else None})

    aggregate_quality = []
    grouped_fixtures = defaultdict(list)
    grouped_repeats = defaultdict(list)
    for row in fixture_rows:
        grouped_fixtures[(row["source"], row["fixture"], row["language"])].append(row)
        grouped_repeats[(row["source"], row["repeat"])].append(row)
    for key, rows in grouped_fixtures.items():
        expected, observed = " ".join(row["expected"] for row in rows), " ".join(row["observed"] for row in rows)
        aggregate_quality.append({"source": key[0], "fixture": key[1], "language": key[2],
                                  "repetitions": len(rows), "expected": expected, "observed": observed,
                                  **quality_score(expected, observed, key[2]),
                                  "empty_repetitions": sum(row["quality"]["empty_output"] for row in rows)})
    per_repeat = []
    timing_keys = ("first_callback_from_speech_onset_ms", "last_callback_after_speech_end_ms",
                   "first_ui_from_speech_onset_ms", "last_ui_after_speech_end_ms")
    for key, rows in grouped_repeats.items():
        per_repeat.append({"source": key[0], "repeat": key[1], "fixture_count": len(rows),
                           "empty_fixtures": sum(row["quality"]["empty_output"] for row in rows),
                           "fixture_latency_ms": {metric: distribution([row[metric] for row in rows]) for metric in timing_keys},
                           "fixtures": [{name: row[name] for name in ("fixture", "quality", *timing_keys)} for row in rows]})

    latency = {key: distribution([row[key] for row in event_rows]) for key in
               ("callback_after_speech_end_ms", "ui_after_speech_end_ms", "ui_delivery_ms")}
    for prefix, flag in (("first_stable", "first_in_utterance"), ("final_stable", "final_in_utterance")):
        for sink in ("callback", "ui"):
            source_metric = f"{sink}_from_utterance_onset_ms" if prefix == "first_stable" else f"{sink}_after_speech_end_ms"
            latency[prefix + "_" + source_metric] = distribution([row[source_metric] for row in event_rows if row[flag]])
    if ambiguous:
        warnings.append("Ambiguous report-to-journal caption matches were excluded from UI timing.")
    if omitted_clip_metadata:
        warnings.append(f"{omitted_clip_metadata} caption events lack clip deduplication metadata.")
    if disagreements:
        warnings.append("Some repeated clip costs disagree; maximum cost was retained and differences are recorded.")
    if any(row["quality"]["empty_output"] for row in fixture_rows):
        warnings.append("One or more fixture occurrences produced no recognized text.")
    if report.get("ui_requested") and any(not row["ui_observation_complete"] for row in fixture_rows):
        warnings.append("UI timing is incomplete for one or more fixture occurrences.")
    final_engine = report.get("final_engine_status", {})
    drained = not final_engine.get("running", True) and final_engine.get("state") == "stopped"
    return {"measurement_version": 2, "raw_status": report.get("status"), "profile": report.get("profile"),
            "no_live_capture": report.get("no_live_capture"), "raw_ui_requested": report.get("ui_requested"),
            "generated_utc": datetime.now(timezone.utc).isoformat(), "warnings": warnings,
            "definitions": {
                "activity_clock_correction": "actual_elapsed = injection_started_elapsed + (source_audio_timestamp - timeline_start)",
                "caption": "Nonempty stable Engine segment; no provisional streaming text is counted.",
                "callback": "Python callback receipt, before journal fsync and SQLite persistence.",
                "ui": "Tk widget insertion/application, after persistence; not physical screen-paint timing.",
                "fixture_final": "Last observed caption for the fixture. Missing final words may produce negative delays, which are preserved.",
                "clip_deduplication": "(source, clip_start, clip_submitted_at); ASR/speaker/worker are not added together.",
                "worker_cost": "Maximum worker_ms per clip includes earlier callbacks when a clip emits multiple captions.",
                "cost_coverage": "Only clips that emitted captions have cost evidence. Work on blank/rejected/failed clips is not reconstructed.",
                "pooled_rtf": "sum(cost seconds) / sum(matched processed clip duration); overlap differs from unique source duration.",
                "drain_evidenced": drained,
                "latency_units": "milliseconds; null means missing evidence, never zero-filled",
                "clock_precision": "Engine/event clocks use Python time.monotonic on Windows (GetTickCount64), approximately 15.625 ms resolution in these runs. Zero queue/overhead samples mean below clock resolution, not proven zero work. Clocks were not changed between comparison runs.",
                "analysis_host_monotonic_resolution_seconds": time.get_clock_info("monotonic").resolution,
                "queue_sampling": "Queue depth/age maxima are sampled at approximately 1 second and may miss brief spikes."},
            "audio_seconds": {"source_waveform_sum": waveform_seconds, "paced_source_sum_including_0_8s_gaps": paced_source_seconds,
                              "wall_elapsed": wall_seconds},
            "counts": {"captions": len(captions), "unique_emitted_clips": len(clips),
                       "ui_matched_captions": sum(finite(row["ui_applied_elapsed_seconds"]) for row in event_rows),
                       "fixture_occurrences": len(fixture_rows)},
            "clip_costs": costs, "queue_and_drop_counters": maxima, "caption_latency_ms": latency,
            "raw_engine_caption_latency_ms": {
                key: distribution([event.get(key) for event in captions]) for key in
                ("caption_delay_from_speech_end_ms", "caption_delay_from_utterance_start_ms", "caption_delay_ms")},
            "per_repeat": per_repeat, "quality_by_fixture": aggregate_quality,
            "fixture_occurrences": fixture_rows, "caption_measurements": event_rows, "clips": clips,
            "ambiguous_matches": ambiguous, "clip_cost_disagreements": disagreements,
            "resources_from_raw_report": report.get("resources"), "network_from_raw_report": report.get("network")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="report.json or its benchmark artifact folder")
    parser.add_argument("--output", type=Path, help="Derived summary path; default summary-v2.json beside report.json")
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly permit analysis of an unfinished/crashed snapshot")
    args = parser.parse_args(argv)
    report_path = args.report / "report.json" if args.report.is_dir() else args.report
    journal_path = report_path.parent / "events.jsonl"
    report_bytes, journal_bytes = report_path.read_bytes(), journal_path.read_bytes()
    report = json.loads(report_bytes.decode("utf-8-sig"))
    if not args.allow_partial and report.get("status") not in ("completed", "incomplete", "no_recognized_text", "error"):
        parser.error("Replay is unfinished; wait for completion or pass --allow-partial to label a partial snapshot")
    records, malformed = [], []
    for index, line in enumerate(journal_bytes.decode("utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed.append(index)
    summary = derive(report, records)
    summary["source_evidence"] = {"report": str(report_path.resolve()), "journal": str(journal_path.resolve()),
                                  "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                                  "journal_sha256": hashlib.sha256(journal_bytes).hexdigest(),
                                  "malformed_journal_lines": malformed, "partial_snapshot_allowed": args.allow_partial}
    if malformed:
        summary["warnings"].append(f"Malformed/truncated journal lines skipped: {malformed}")
    output = args.output or report_path.parent / "summary-v2.json"
    if output.resolve() in (report_path.resolve(), journal_path.resolve()):
        parser.error("Derived output must not overwrite raw evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Measurement v2: {output}")
    print("Captions/clips/UI matches:", summary["counts"])
    for name in ("callback_after_speech_end_ms", "ui_after_speech_end_ms", "ui_delivery_ms"):
        print(name, summary["caption_latency_ms"][name])
    for metric, cost in summary["clip_costs"].items():
        print(metric, "source RTF:", cost["source_waveform_rtf"], "wall fraction:", cost["wall_utilization_fraction"])
    print("Queue/drop observations:", summary["queue_and_drop_counters"])
    for score in summary["quality_by_fixture"]:
        print(score["source"], score["fixture"], score["metric"], score["error_rate"], "empty repetitions", score["empty_repetitions"])
    for warning in summary["warnings"]:
        print("WARNING:", warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
