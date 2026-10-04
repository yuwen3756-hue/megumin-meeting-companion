"""Replay only generated fixtures. Never opens an audio device."""
from pathlib import Path
import argparse
import json
import os
import socket
import sys
import threading
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.signal import resample_poly
import psutil


def read_audio(path):
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        data = data.reshape(-1, w.getnchannels()).mean(axis=1)
        rate = w.getframerate()
    import math
    divisor = math.gcd(rate, 16000)
    return resample_poly(data, 16000 // divisor, rate // divisor).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["auto", "arc", "nvidia"], default="auto")
    parser.add_argument("--model", choices=["turbo", "small"], default="turbo")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--beam-size", type=int, choices=[1, 5], default=1, help="Experimental comparison; neither setting is certified")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--dual-source", action="store_true", help="Paced duplicate fixture streams test worst-case two-source load, not diarization accuracy")
    args = parser.parse_args()
    if args.dual_source and not args.realtime:
        parser.error("--dual-source requires --realtime")
    attempts = []
    def blocked(*a, **kw):
        attempts.append(str(a[:1]))
        raise RuntimeError("Benchmark network guard blocked socket operation")
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    from meeting.inference import Recognizer, SpeakerTracker
    from meeting.engine import Engine
    from meeting.profiles import get_profile, detect_gpus
    from dataclasses import replace, asdict
    hardware = detect_gpus()
    profile = replace(get_profile(args.target, args.model, gpus=hardware, cpu=args.cpu), beam_size=args.beam_size)
    process = psutil.Process()
    peak = [process.memory_info().rss]
    gpu_samples = []
    try:
        from gpu_memory import GpuMemory
        gpu = GpuMemory()
        gpu_samples.append(gpu.sample())
        gpu_error = None
    except Exception as exc:
        gpu, gpu_error = None, str(exc)
    done = threading.Event()
    def sample():
        while not done.wait(.1):
            peak[0] = max(peak[0], process.memory_info().rss)
            if gpu:
                gpu_samples.append(gpu.sample())
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    report = dict(profile=asdict(profile), observed_gpus=hardware, synthetic=True, dual_source=args.dual_source,
                  no_live_capture=True, network_guard="Python socket connect and DNS denied", cases=[])
    sources = json.loads((ROOT / "artifacts/fixtures/sources.json").read_text(encoding="utf-8-sig"))
    for source in sources:
        recognizer = Recognizer(str(ROOT / "models" / profile.model_dir), language=source["language"], **profile.recognizer_options())
        tracker = SpeakerTracker(str(ROOT / "models/speaker.onnx"))
        audio = read_audio(ROOT / "artifacts/fixtures" / source["file"])
        # Actual utterance-sized inference, including CPU speaker extraction.
        wall = time.perf_counter()
        chunks = []
        for start in range(0, len(audio), 6 * 16000):
            clip = audio[start:start + 6 * 16000]
            mark = time.perf_counter()
            segments = recognizer.transcribe(clip)
            speaker = tracker.classify(clip)
            chunks.append(dict(audio_seconds=len(clip)/16000, seconds=time.perf_counter()-mark,
                               segments=segments, speaker=speaker))
        seconds = time.perf_counter() - wall
        report["cases"].append(dict(fixture=source["file"], language=source["language"], expected=source["text"],
              load_seconds=recognizer.load_seconds, audio_seconds=len(audio)/16000,
              wall_seconds=seconds, rtf=seconds/(len(audio)/16000), chunks=chunks))
        if args.realtime:
            events = []
            engine = Engine(recognizer, tracker, events.append, ROOT / "artifacts/benchmark_audio")
            wall = time.perf_counter()
            if args.dual_source:
                engine.start()  # Fixture-only; no device enumeration or open.
                local = threading.Thread(target=engine.feed_fixture, args=(audio,"local",True))
                local.start()
                engine.feed_fixture(audio, source="remote", realtime=True)
                local.join()
            else:
                engine.feed_fixture(audio, source="remote", realtime=True)
            engine.stop(timeout=60)
            segments = [e for e in events if e.get("type") == "segment"]
            latency = [e["caption_delay_ms"] for e in segments if e.get("caption_delay_valid")]
            report["cases"][-1]["engine_replay"] = dict(wall_seconds=time.perf_counter()-wall,
                segments=segments, warnings=[e for e in events if e.get("type")=="warning"],
                caption_delay_p50_ms=float(np.percentile(latency, 50)) if latency else None,
                caption_delay_p95_ms=float(np.percentile(latency, 95)) if latency else None,
                final_status=engine.status)
        del recognizer
    done.set()
    thread.join(timeout=2)
    report["peak_process_rss_mb"] = peak[0] / 2**20
    report["network_attempts"] = attempts
    report["vram"] = dict(source="Windows PDH dedicated memory counters", error=gpu_error,
        baseline=gpu_samples[0] if gpu_samples else {}, sample_count=len(gpu_samples),
        adapter_peak_bytes=max((x.get("adapter_dedicated_bytes",0) for x in gpu_samples),default=None),
        process_peak_bytes=max((x.get("process_dedicated_bytes",0) for x in gpu_samples),default=None))
    if gpu:
        gpu.close()
    source_mode = "dual" if args.dual_source else "single"
    path = ROOT / "artifacts" / f"benchmark-{profile.target}-{profile.model}-{profile.device}-beam{profile.beam_size}-{source_mode}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    for case in report["cases"]:
        print(case["fixture"], "RTF", round(case["rtf"], 3), "load", round(case["load_seconds"], 2))
    print("network attempts", len(attempts), "peak RSS MB", round(report["peak_process_rss_mb"]))


if __name__ == "__main__":
    main()
