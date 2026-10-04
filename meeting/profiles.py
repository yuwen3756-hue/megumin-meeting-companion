"""Offline, hardware-adaptive candidate profiles. Selection is not a benchmark."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json


@dataclass(frozen=True)
class Profile:
    target: str
    model: str
    backend: str
    device: str
    model_dir: str
    repository: str
    compute_type: str | None = None
    beam_size: int = 1
    measured: bool = False

    @property
    def label(self):
        hardware = "Intel GPU" if self.device == "GPU" else "NVIDIA GPU" if self.device == "cuda" else "CPU"
        return f"Whisper {self.model} · {hardware} · local inference"

    def recognizer_options(self):
        return dict(backend=self.backend, device=self.device, compute_type=self.compute_type, beam_size=self.beam_size)


PROFILES = {
    ("arc", "turbo"): Profile("arc", "turbo", "openvino", "GPU", "whisper-large-v3-turbo-int8-ov", "OpenVINO/whisper-large-v3-turbo-int8-ov"),
    ("arc", "small"): Profile("arc", "small", "openvino", "GPU", "whisper-small-int8-ov", "OpenVINO/whisper-small-int8-ov"),
    ("nvidia", "turbo"): Profile("nvidia", "turbo", "faster-whisper", "cuda", "faster-whisper-large-v3-turbo", "dropbox-dash/faster-whisper-large-v3-turbo", "int8_float16"),
    ("nvidia", "small"): Profile("nvidia", "small", "faster-whisper", "cuda", "faster-whisper-small", "Systran/faster-whisper-small", "int8_float16"),
}


def detect_gpus() -> list[dict]:
    """Read Windows adapter metadata only; do not load GPU runtimes or record."""
    devices = []
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Video") as parent:
            for i in range(winreg.QueryInfoKey(parent)[0]):
                name = winreg.EnumKey(parent, i)
                try:
                    with winreg.OpenKey(parent, name + r"\0000") as key:
                        desc = winreg.QueryValueEx(key, "DriverDesc")[0]
                        try:
                            memory = winreg.QueryValueEx(key, "HardwareInformation.qwMemorySize")[0]
                        except OSError:
                            memory = None
                        device = {"name": str(desc), "vram_bytes": memory}
                        if device not in devices:
                            devices.append(device)
                except OSError:
                    continue
    except (ImportError, OSError):
        pass
    return devices


def detected_target(gpus: list[dict] | None = None) -> str | None:
    devices = detect_gpus() if gpus is None else gpus
    names = " ".join(str(x.get("name", "")) for x in devices).lower()
    # Prefer the discrete NVIDIA route on hybrid NVIDIA/Intel laptops.
    if "nvidia" in names or "geforce" in names:
        return "nvidia"
    if "intel" in names:
        return "arc"
    return None


def get_profile(target="auto", model="turbo", *, gpus=None, cpu=False) -> Profile:
    if target == "auto":
        target = detected_target(gpus)
        if target is None:
            raise RuntimeError("No supported GPU identified. Select Intel or NVIDIA explicitly in setup, or configure a CPU profile.")
    try:
        profile = PROFILES[(target, model)]
    except KeyError as exc:
        raise ValueError("Choose target auto/arc/nvidia and model turbo/small.") from exc
    if cpu:
        profile = replace(profile, device="CPU" if profile.backend == "openvino" else "cpu",
                          compute_type=None if profile.backend == "openvino" else "int8")
    return profile


def load_settings(root: Path) -> dict:
    path = Path(root) / "settings.json"
    settings = {"target": "auto", "model": "turbo", "cpu": False}
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not set(data).issubset(settings):
            raise ValueError("settings.json supports only target, model, and cpu.")
        settings.update(data)
    if settings["target"] not in {"auto", "arc", "nvidia"} or settings["model"] not in {"turbo", "small"} or not isinstance(settings["cpu"], bool):
        raise ValueError("Invalid local runtime settings")
    return settings


def resolve_profile(root: Path, *, gpus=None) -> tuple[Profile, str]:
    """Resolve settings, using an already installed same-backend small fallback."""
    settings = load_settings(root)
    profile = get_profile(**settings, gpus=gpus)
    if not (Path(root) / "models" / profile.model_dir).is_dir() and profile.model == "turbo":
        smaller = get_profile(profile.target, "small", cpu=settings["cpu"])
        if (Path(root) / "models" / smaller.model_dir).is_dir():
            return smaller, "Turbo is not installed; using the existing small model. See docs/BENCHMARK.md for tested hardware and limits."
    return profile, "See docs/BENCHMARK.md for measured hardware, caption delay, and remaining limits."


def runtime_inventory(root: Path) -> dict:
    gpus = detect_gpus()
    result = {"gpus": gpus, "detected_target": detected_target(gpus), "settings": load_settings(root)}
    try:
        profile, notice = resolve_profile(root, gpus=gpus)
        result.update(profile=asdict(profile), notice=notice, model_present=(Path(root) / "models" / profile.model_dir).is_dir())
    except (RuntimeError, ValueError) as exc:
        result["error"] = str(exc)
    return result
