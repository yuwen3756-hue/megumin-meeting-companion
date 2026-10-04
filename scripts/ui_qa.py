"""Native UI exercise with only synthetic text; capture controls never invoked."""
from pathlib import Path
import sys
import json
import tkinter as tk
import ctypes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meeting.ui import App
from meeting.offline import enable
enable()

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except (AttributeError, OSError):
    ctypes.windll.user32.SetProcessDPIAware()
root = tk.Tk()
qa_root = ROOT / "artifacts/ui-qa"
qa_root.mkdir(parents=True, exist_ok=True)
app = App(root, qa_root, demo=True)
root.geometry("1360x920+20+20")
errors = []
root.report_callback_exception = lambda typ, exc, trace: errors.append(str(exc))


def capture():
    from PIL import ImageGrab
    root.update_idletasks()
    root.lift()
    root.attributes("-topmost", True)
    root.update()
    root.after(150)
    box = (root.winfo_rootx(), root.winfo_rooty(), root.winfo_rootx() + root.winfo_width(),
           root.winfo_rooty() + root.winfo_height())
    ImageGrab.grab(bbox=box).save(ROOT / "artifacts/ui-demo.png")
    root.attributes("-topmost", False)
    sid = app.selected_id
    rows = app.store.segments(sid)
    assert len(rows) == 8, "Demo did not render"
    assert not app.active_id and not app.engine, "QA must not capture audio"
    # Exercise the same persisted corrections and exports shown in the UI.
    app.store.correct_segment(rows[0]["id"], "Synthetic corrected text.", speaker="Local")
    app.store.rename_speaker(sid, "Local", "Demo host")
    assert app.store.segments(sid, search="corrected")[0]["speaker_name"] == "Demo host"
    exports = app.store.export_session(sid, qa_root / "exports")
    report = dict(native_tk_rendered=True, live_capture=False, synthetic_segments=len(rows),
                  correction_search_rename=True, exports=[str(x) for x in exports],
                  callback_errors=errors, window_width=root.winfo_width(),window_height=root.winfo_height())
    (ROOT / "artifacts/ui-qa.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    app._destroy()


def guarded():
    try:
        capture()
    except Exception as exc:
        (ROOT / "artifacts/ui-qa-error.txt").write_text(repr(exc),encoding="utf-8")
        try:
            app._destroy()
        except Exception:
            root.destroy()

root.after(1800, guarded)
root.mainloop()
