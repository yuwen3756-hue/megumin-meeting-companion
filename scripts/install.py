"""Explicit project-only setup. --plan reads metadata and never installs/downloads."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting.profiles import get_profile, detect_gpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["auto", "arc", "nvidia"], default="auto")
    parser.add_argument("--model", choices=["turbo", "small"], default="turbo")
    parser.add_argument("--yes", action="store_true", help="Consent to dependency and model downloads")
    parser.add_argument("--plan", action="store_true", help="Show selection; no network or environment changes")
    args = parser.parse_args()
    gpus = detect_gpus()
    profile = get_profile(args.target, args.model, gpus=gpus)
    requirements = ROOT / f"requirements-{profile.target}.txt"
    print(json.dumps({"detected_gpus":gpus,"target":profile.target,"model":profile.model,
        "repository":profile.repository,"requirements":str(requirements),"destination":str(ROOT),
        "measured_on_this_machine":False},indent=2))
    print("Setup downloads code from PyPI, model data from Hugging Face and GitHub.")
    print("Only this project .venv, .cache and models folders are written. No global install or driver changes.")
    if profile.target == "nvidia":
        print("NVIDIA route includes proprietary CUDA12 cuBLAS/cuDNN9 runtime wheels; no full CUDA toolkit or PyTorch.")
        print("Turbo is the Dropbox Dash CT2 conversion reached by faster-whisper's official Mobius model alias, not an OpenAI-published conversion.")
    print("No audio/transcripts/notes are uploaded. Model choices are unbenchmarked candidates.")
    if args.plan:
        return
    if not args.yes and input("Install this local setup and download these models? [y/N] ").strip().lower() != "y":
        return
    python = ROOT / ".venv/Scripts/python.exe"
    if not python.is_file():
        subprocess.run([sys.executable,"-m","venv",str(ROOT/".venv")],check=True)
    env = dict(os.environ, PIP_CACHE_DIR=str(ROOT/".cache/pip"), PIP_DISABLE_PIP_VERSION_CHECK="1")
    subprocess.run([str(python),"-m","pip","install","--only-binary=:all:","-r",str(requirements)],check=True,env=env)
    subprocess.run([str(python),str(ROOT/"scripts/setup_models.py"),"--target",profile.target,"--model",profile.model,"--yes"],check=True,env=env)
    # Settings stay hardware adaptive when copied to the other machine.
    settings = {"target":args.target,"model":args.model,"cpu":False}
    (ROOT/"settings.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    installed = subprocess.run([str(python),"-m","pip","freeze"],check=True,capture_output=True,text=True,env=env)
    (ROOT/"artifacts").mkdir(exist_ok=True)
    (ROOT/"artifacts"/f"installed-{profile.target}.txt").write_text(installed.stdout,encoding="utf-8")
    print("Setup completed. Use Launch.cmd. Capture remains off until Start is pressed.")


if __name__ == "__main__":
    main()
