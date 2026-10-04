"""Read-only hardware and legacy device enumeration; never opens capture."""
import ctypes
from ctypes import wintypes
import json
import platform
from pathlib import Path
import winreg
import psutil

ROOT = Path(__file__).resolve().parents[1]
report = dict(host=platform.node(), windows=platform.platform(), python=platform.python_version(),
              ram_bytes=psutil.virtual_memory().total, logical_cpus=psutil.cpu_count())
with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
    report["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
report["gpus"] = []
with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Video") as video:
    for index in range(winreg.QueryInfoKey(video)[0]):
        name = winreg.EnumKey(video, index)
        try:
            with winreg.OpenKey(video, name + r"\0000") as key:
                item = {"name":winreg.QueryValueEx(key,"DriverDesc")[0],
                        "vram_bytes":winreg.QueryValueEx(key,"HardwareInformation.qwMemorySize")[0]}
                try:
                    item["driver_version"] = winreg.QueryValueEx(key,"DriverVersion")[0]
                except OSError:
                    pass
                if item not in report["gpus"]:
                    report["gpus"].append(item)
        except OSError:
            pass

class WaveCaps(ctypes.Structure):
    _fields_ = [("mid",wintypes.WORD),("pid",wintypes.WORD),("version",wintypes.DWORD),
                ("name",wintypes.WCHAR*32),("formats",wintypes.DWORD),("channels",wintypes.WORD),
                ("reserved",wintypes.WORD),("support",wintypes.DWORD)]

report["legacy_audio_devices"] = []
for kind, count_fn, caps_fn in (("mic", "waveInGetNumDevs", "waveInGetDevCapsW"),
                               ("output", "waveOutGetNumDevs", "waveOutGetDevCapsW")):
    for i in range(getattr(ctypes.windll.winmm, count_fn)()):
        caps = WaveCaps()
        error = getattr(ctypes.windll.winmm, caps_fn)(i,ctypes.byref(caps),ctypes.sizeof(caps))
        if not error:
            report["legacy_audio_devices"].append(dict(kind=kind,id=i,name=caps.name,channels=caps.channels))
try:
    import sys
    sys.path.insert(0, str(ROOT))
    from meeting.telemetry import disable_openvino_telemetry
    disable_openvino_telemetry()
    import openvino
    core = openvino.Core()
    report["openvino"] = openvino.__version__
    report["openvino_devices"] = [{"id":d,"name":str(core.get_property(d,"FULL_DEVICE_NAME"))} for d in core.available_devices]
except ImportError:
    report["openvino"] = "not installed"
try:
    import sys
    sys.path.insert(0,str(ROOT))
    from meeting.audio import list_devices
    report["wasapi_devices"] = list_devices()
except Exception as exc:
    report["wasapi_enumeration_error"] = str(exc)
output = ROOT / "artifacts/hardware.json"
output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
