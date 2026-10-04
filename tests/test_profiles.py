import json
import pytest
from meeting.profiles import get_profile, resolve_profile, load_settings, detected_target


def test_arc_and_hybrid_4070_route_to_distinct_offline_backends():
    arc = get_profile(gpus=[{"name":"Intel(R) Arc(TM) A770 Graphics","vram_bytes":16*2**30}])
    laptop = get_profile(gpus=[{"name":"Intel Iris Xe"},{"name":"NVIDIA GeForce RTX 4070 Laptop GPU","vram_bytes":8*2**30}])
    assert arc.backend == "openvino" and arc.device == "GPU"
    assert laptop.backend == "faster-whisper" and laptop.compute_type == "int8_float16"
    assert laptop.beam_size == arc.beam_size == 1
    assert not arc.measured and not laptop.measured
    assert "turbo" in arc.model_dir and "turbo" in laptop.model_dir


def test_unknown_hardware_does_not_silently_claim_gpu_support():
    assert detected_target([{"name":"AMD Radeon"}]) is None
    with pytest.raises(RuntimeError, match="No supported GPU"):
        get_profile(gpus=[])
    assert get_profile("nvidia", "small", cpu=True).device == "cpu"
    assert get_profile("arc", "small", cpu=True).device == "CPU"


def test_only_installed_same_backend_small_model_is_fallback(tmp_path):
    (tmp_path / "models" / "whisper-small-int8-ov").mkdir(parents=True)
    profile, notice = resolve_profile(tmp_path,gpus=[{"name":"Intel Arc A770"}])
    assert profile.model == "small" and "existing" in notice
    profile, notice = resolve_profile(tmp_path,gpus=[{"name":"NVIDIA RTX 4070"}])
    assert profile.model == "turbo" and profile.target == "nvidia"


def test_settings_remain_portable_and_validate(tmp_path):
    assert load_settings(tmp_path)["target"] == "auto"
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"target":"nvidia","model":"small","cpu":True}))
    profile, _ = resolve_profile(tmp_path, gpus=[])
    assert profile.device == "cpu" and profile.compute_type == "int8"
    settings.write_text('{"cpu":"yes"}')
    with pytest.raises(ValueError):
        load_settings(tmp_path)
