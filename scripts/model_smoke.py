"""Actual model smoke test of project-generated SAPI fixtures; no capture."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting.offline import enable
enable()
from benchmark import read_audio
from meeting.inference import Recognizer, SpeakerTracker
from meeting.profiles import get_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['turbo', 'small'], default='turbo')
    args = parser.parse_args()
    profile = get_profile('arc', args.model)
    report = {'model': args.model, 'no_live_capture': True, 'synthetic': True, 'cases': []}
    path = ROOT / 'artifacts' / f'smoke-{args.model}.json'
    started = time.perf_counter()
    recognizer = Recognizer(str(ROOT / 'models' / profile.model_dir), **profile.recognizer_options())
    report['load_seconds'] = time.perf_counter() - started
    tracker = SpeakerTracker(str(ROOT / 'models' / 'speaker.onnx'))
    print(f'Model load {report["load_seconds"]:.3f}s', flush=True)
    fixtures = json.loads((ROOT / 'artifacts/fixtures/sources.json').read_text(encoding='utf-8-sig'))
    for repeat in range(2):
        for fixture in fixtures:
            recognizer.language = fixture['language']
            audio = read_audio(ROOT / 'artifacts/fixtures' / fixture['file'])
            case = {'repeat': repeat, 'fixture': fixture['file'], 'language': fixture['language'],
                    'expected': fixture['text'], 'chunks': []}
            for offset in range(0, len(audio), 96000):
                clip = audio[offset:offset + 96000]
                mark = time.perf_counter()
                segments = recognizer.transcribe(clip)
                asr = time.perf_counter() - mark
                mark = time.perf_counter()
                speaker = tracker.classify(clip)
                speaker_seconds = time.perf_counter() - mark
                chunk = {'audio_seconds': len(clip)/16000, 'asr_seconds': asr,
                         'speaker_seconds': speaker_seconds, 'segments': segments, 'speaker': speaker}
                case['chunks'].append(chunk)
                print(json.dumps(dict(repeat=repeat, fixture=fixture['file'], **chunk), ensure_ascii=True), flush=True)
            report['cases'].append(case)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(str(path), flush=True)


if __name__ == '__main__':
    main()
