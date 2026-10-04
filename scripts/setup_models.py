"""Explicit, one-time official downloads; never imported by the meeting app."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting.profiles import get_profile


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["auto", "arc", "nvidia"], default="auto")
    parser.add_argument("--model", choices=["turbo", "small"], default="turbo")
    parser.add_argument("--plan", action="store_true", help="Print sources only; no download")
    parser.add_argument("--yes", action="store_true", help="Explicit consent to official one-time downloads")
    args = parser.parse_args()
    profile = get_profile(args.target, args.model)
    print("One-time setup connects to PyPI/Hugging Face and GitHub for code and official model assets.")
    print("No recordings, transcripts, or notes are uploaded. The meeting app performs no downloads.")
    print("ASR: upstream OpenAI Whisper MIT; OpenVINO small conversion card Apache-2.0, turbo MIT.")
    print("WeSpeaker VoxCeleb weights: CC BY 4.0; attribution retained in the manifest and guide.")
    print(f"Selected {profile.target}/{profile.model}: {profile.repository}; destination models/{profile.model_dir}")
    if profile.target == "nvidia":
        print("CT2 conversion publisher is Dropbox Dash for turbo, SYSTRAN for small; both use upstream OpenAI Whisper weights (MIT).")
    if args.plan:
        return
    if not args.yes and input("Download local models? [y/N] ").strip().lower() != "y":
        return
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
    from huggingface_hub import HfApi, snapshot_download
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    repositories = [(profile.repository, profile.model_dir)]
    manifest_path = models / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"sources": [], "files": {}}
    for repo, name in repositories:
        info = HfApi().model_info(repo)
        revision = info.sha
        print(f"Downloading {repo} at {revision}", flush=True)
        target = models / name
        snapshot_download(repo, revision=revision, local_dir=target,
                          allow_patterns=["*.json", "*.xml", "*.bin", "*.txt", "*.model", "README.md", "LICENSE*"],
                          max_workers=4)
        manifest["sources"] = [s for s in manifest["sources"] if s.get("folder") != name]
        manifest["sources"].append({"repository": repo, "resolved_repository":info.id,"revision": revision, "folder": name})
    speaker_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx"
    speaker = models / "speaker.onnx"
    if not speaker.is_file():
        partial = speaker.with_suffix(".partial")
        print("Downloading official sherpa-onnx WeSpeaker conversion (about 26 MB)", flush=True)
        urllib.request.urlretrieve(speaker_url, partial)
        if partial.stat().st_size < 1000000:
            raise RuntimeError("Speaker download too small")
        partial.replace(speaker)
    manifest["sources"] = [s for s in manifest["sources"] if s.get("url") != speaker_url]
    manifest["sources"].append({"url": speaker_url, "license": "CC BY 4.0", "attribution": "WeSpeaker team; VoxCeleb ResNet34, sherpa-onnx ONNX conversion"})
    vad_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    vad = models / "silero_vad.onnx"
    vad_hash = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"
    if not vad.is_file():
        partial = vad.with_suffix(".partial")
        urllib.request.urlretrieve(vad_url, partial)
        if sha256(partial) != vad_hash:
            raise RuntimeError("Silero release asset hash changed; inspect the official source before proceeding.")
        partial.replace(vad)
    if sha256(vad) != vad_hash:
        raise RuntimeError("Local Silero VAD hash mismatch; rerun setup after inspecting the model file.")
    manifest["sources"] = [s for s in manifest["sources"] if s.get("url") != vad_url]
    manifest["sources"].append({"url": vad_url, "license": "MIT", "sha256": vad_hash})
    for path in models.rglob("*"):
        if path.is_file() and ".cache" not in path.parts and path.name != "manifest.json":
            manifest["files"][path.relative_to(models).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    temporary = models / "manifest.partial.json"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print("Models ready. Launch with Launch.cmd. Setup does not start recording.")


if __name__ == "__main__":
    main()
