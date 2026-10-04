from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"


def main():
    parser = argparse.ArgumentParser(description="MeguminMeet — offline Windows meeting notebook")
    parser.add_argument("--demo", action="store_true", help="Open a clearly labeled synthetic transcript without capture")
    args = parser.parse_args()
    # No HTTP server; block Python IP connections as defence in depth.
    from meeting.offline import enable
    enable()
    from meeting.instance import InstanceLock
    lock = InstanceLock(ROOT / "data")
    from meeting.ui import launch
    try:
        launch(ROOT, demo=args.demo)
    finally:
        lock.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        (ROOT / "artifacts").mkdir(exist_ok=True)
        (ROOT / "artifacts" / "startup-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"MeguminMeet could not start: {exc}\nSee artifacts/startup-error.txt", file=sys.stderr)
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, str(exc) + "\nSee artifacts/startup-error.txt", "MeguminMeet could not start", 0x10)
        except Exception:
            pass
        raise
