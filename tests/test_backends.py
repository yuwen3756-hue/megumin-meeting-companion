"""Backend contracts exercised with local marker files and Python-only stubs.

No test downloads a model, imports a native inference backend, opens a device,
or registers a real DLL directory. GPU behavior is not validated here.
"""

from __future__ import annotations

import builtins
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest


MODEL_FILES = {
    "faster-whisper": ("config.json", "tokenizer.json", "model.bin"),
    "openvino": (
        "config.json", "tokenizer.json", "openvino_encoder_model.xml",
        "openvino_encoder_model.bin", "openvino_decoder_model.xml",
        "openvino_decoder_model.bin",
        "openvino_tokenizer.xml", "openvino_tokenizer.bin",
        "openvino_detokenizer.xml", "openvino_detokenizer.bin",
        "generation_config.json", "preprocessor_config.json",
    ),
}


def speech(seconds=1.0):
    samples = np.arange(round(16_000 * seconds), dtype=np.float64) / 16_000
    return (0.15 * np.sin(2 * np.pi * 173 * samples)).astype(np.float32)


@pytest.fixture
def local_model(tmp_path):
    def make(backend, missing=None):
        directory = tmp_path / "models" / backend
        directory.mkdir(parents=True, exist_ok=True)
        for name in MODEL_FILES[backend]:
            if name != missing:
                (directory / name).write_bytes(b"{}" if name.endswith(".json") else b"stub")
        if missing != "silero_vad.onnx":
            (directory.parent / "silero_vad.onnx").write_bytes(b"stub")
        return directory

    return make


@pytest.fixture
def backend_stubs(monkeypatch, tmp_path):
    state = types.SimpleNamespace(
        fw_initializations=[], fw_transcribes=[], ov_initializations=[],
        ov_generates=[], vad_configs=[], vads=[], vad_speech=True,
        fw_init_error=None, fw_call_error=None, ov_init_error=None,
        fw_generator_factory=None, fw_consumed=False, fw_yielded=0,
        fw_segments=[types.SimpleNamespace(start=0.0, end=0.5, text=" fixture words ")],
        ov_result=types.SimpleNamespace(
            chunks=[types.SimpleNamespace(start_ts=0.0, end_ts=0.5, text=" fixture words ")],
            texts=[],
        ),
        dll_registrations=[], dll_handles=[],
    )

    class WhisperModel:
        def __init__(self, path, **kwargs):
            state.fw_initializations.append((path, kwargs))
            if state.fw_init_error is not None:
                raise state.fw_init_error

        def transcribe(
            self, samples, *, language, task, beam_size,
            condition_on_previous_text, vad_filter, word_timestamps,
            without_timestamps, max_new_tokens, temperature, best_of,
        ):
            kwargs = {
                "language": language, "task": task, "beam_size": beam_size,
                "condition_on_previous_text": condition_on_previous_text,
                "vad_filter": vad_filter, "word_timestamps": word_timestamps,
                "without_timestamps": without_timestamps,
                "max_new_tokens": max_new_tokens, "temperature": temperature,
                "best_of": best_of,
            }
            state.fw_transcribes.append((np.array(samples, copy=True), kwargs))
            if state.fw_call_error is not None:
                raise state.fw_call_error

            def generate():
                for segment in state.fw_segments:
                    state.fw_yielded += 1
                    yield segment
                state.fw_consumed = True

            iterator = state.fw_generator_factory() if state.fw_generator_factory else generate()
            return iterator, types.SimpleNamespace(language="en")

    class WhisperPipeline:
        def __init__(self, path, device):
            state.ov_initializations.append((path, device))
            if state.ov_init_error is not None:
                raise state.ov_init_error

        def generate(self, samples, **kwargs):
            state.ov_generates.append((np.array(samples, copy=True), kwargs))
            return state.ov_result

    class VadModelConfig:
        def __init__(self):
            self.silero_vad = types.SimpleNamespace()
            # An explicit non-CPU default detects accidental provider inheritance.
            self.provider = "cuda"

        def validate(self):
            return True

    class VoiceActivityDetector:
        def __init__(self, config, buffer_size_in_seconds):
            state.vad_configs.append(config)
            state.vads.append(self)
            self.buffer_size = buffer_size_in_seconds
            self.has_nonzero = False
            self.frames = 0
            self.received = []
            self.resets = 0

        def reset(self):
            self.has_nonzero = False
            self.resets += 1

        def accept_waveform(self, frame):
            self.frames += 1
            self.received.append(np.array(frame, copy=True))
            self.has_nonzero = self.has_nonzero or bool(np.any(frame))

        def flush(self):
            pass

        def empty(self):
            return not (state.vad_speech and self.has_nonzero)

    class DllHandle:
        def __init__(self, path):
            self.path = Path(path).resolve()
            self.closed = False

        def close(self):
            self.closed = True

    def add_dll_directory(path):
        handle = DllHandle(path)
        state.dll_registrations.append(handle.path)
        state.dll_handles.append(handle)
        return handle

    fw_module = types.ModuleType("faster_whisper")
    fw_module.WhisperModel = WhisperModel
    ov_module = types.ModuleType("openvino_genai")
    ov_module.WhisperPipeline = WhisperPipeline
    vad_module = types.ModuleType("sherpa_onnx")
    vad_module.VadModelConfig = VadModelConfig
    vad_module.VoiceActivityDetector = VoiceActivityDetector
    for name, module in (
        ("faster_whisper", fw_module), ("openvino_genai", ov_module),
        ("sherpa_onnx", vad_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    inference = importlib.import_module("meeting.inference")
    monkeypatch.setattr(inference, "_PROJECT_ROOT", tmp_path / "project", raising=False)
    monkeypatch.setattr(inference.os, "add_dll_directory", add_dll_directory, raising=False)
    # CUDA initialization may alter this process's PATH; restore it per test.
    monkeypatch.setenv("PATH", inference.os.environ.get("PATH", ""))
    state.inference = inference
    return state


@pytest.mark.parametrize("language", ["auto", "en", "zh"])
def test_faster_whisper_language_and_local_cuda_options(backend_stubs, local_model, language):
    state = backend_stubs
    directory = local_model("faster-whisper")
    recognizer = state.inference.Recognizer(str(directory), backend="faster-whisper", language=language)
    output = recognizer.transcribe(speech())

    assert state.fw_initializations == [(str(directory.resolve()), {
        "local_files_only": True, "device": "cuda", "device_index": 0,
        "compute_type": "int8_float16", "cpu_threads": 4, "num_workers": 1,
    })]
    samples, options = state.fw_transcribes[0]
    assert samples.dtype == np.float32 and samples.ndim == 1
    expected = {
        "language": None if language == "auto" else language,
        "task": "transcribe", "beam_size": 1,
        "condition_on_previous_text": False, "vad_filter": False,
        "word_timestamps": False, "without_timestamps": False,
        "max_new_tokens": 180, "temperature": 0.0, "best_of": 1,
    }
    assert {key: options[key] for key in expected} == expected
    assert state.fw_consumed and state.fw_yielded == 1
    assert output == [{"start": 0.0, "end": 0.5, "text": "fixture words"}]
    assert state.ov_initializations == []
    assert state.vad_configs[0].provider == "cpu"
    assert Path(state.vad_configs[0].silero_vad.model) == directory.parent / "silero_vad.onnx"


@pytest.mark.parametrize("language", ["auto", "en", "zh"])
def test_openvino_language_and_generation_options(backend_stubs, local_model, language):
    state = backend_stubs
    directory = local_model("openvino")
    recognizer = state.inference.Recognizer(str(directory), language=language)
    output = recognizer.transcribe(speech())

    assert state.ov_initializations == [(str(directory.resolve()), "GPU")]
    samples, options = state.ov_generates[0]
    assert samples.dtype == np.float32 and samples.ndim == 1
    expected = {"task": "transcribe", "return_timestamps": True, "max_new_tokens": 180}
    if language != "auto":
        expected["language"] = f"<|{language}|>"
    assert options == expected
    assert output == [{"start": 0.0, "end": 0.5, "text": "fixture words"}]
    assert state.fw_initializations == []
    assert state.vad_configs[0].provider == "cpu"


@pytest.mark.parametrize("device,expected_device,compute_type", [
    ("GPU", "cuda", "int8_float16"), ("CUDA", "cuda", "int8_float16"),
    ("cuda", "cuda", "int8_float16"), ("CPU", "cpu", "int8"),
    ("cpu", "cpu", "int8"),
])
def test_faster_whisper_device_defaults(
    backend_stubs, local_model, device, expected_device, compute_type
):
    state = backend_stubs
    state.inference.Recognizer(str(local_model("faster-whisper")), backend="faster-whisper", device=device)
    _, options = state.fw_initializations[0]
    assert options["device"] == expected_device
    assert options["compute_type"] == compute_type


def test_explicit_faster_whisper_compute_type_is_preserved(backend_stubs, local_model):
    state = backend_stubs
    state.inference.Recognizer(
        str(local_model("faster-whisper")), backend="faster-whisper", compute_type="float16"
    )
    assert state.fw_initializations[0][1]["compute_type"] == "float16"


@pytest.mark.parametrize("backend", ["openvino", "faster-whisper"])
def test_beam_size_override_reaches_selected_backend(backend_stubs, local_model, backend):
    state = backend_stubs
    recognizer = state.inference.Recognizer(str(local_model(backend)), backend=backend, beam_size=4)
    recognizer.transcribe(speech())
    calls = state.ov_generates if backend == "openvino" else state.fw_transcribes
    key = "num_beams" if backend == "openvino" else "beam_size"
    assert calls[0][1][key] == 4


@pytest.mark.parametrize("backend,missing", [
    (backend, filename)
    for backend, filenames in MODEL_FILES.items()
    for filename in (*filenames, "silero_vad.onnx")
])
def test_missing_artifacts_fail_before_backend_imports(
    backend_stubs, local_model, monkeypatch, backend, missing
):
    state = backend_stubs
    directory = local_model(backend, missing=missing)
    original_import = builtins.__import__
    attempted = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"faster_whisper", "openvino_genai", "sherpa_onnx"}:
            attempted.append(name)
            raise AssertionError("incomplete models must fail before backend imports")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(FileNotFoundError):
        state.inference.Recognizer(str(directory), backend=backend)
    assert attempted == []
    assert state.fw_initializations == state.ov_initializations == state.vad_configs == []


@pytest.mark.parametrize("backend", ["openvino", "faster-whisper"])
def test_timestamp_bounds_and_empty_text_are_normalized(backend_stubs, local_model, backend):
    state = backend_stubs
    rows = [(-2.0, 0.3, " leading "), (0.2, 4.0, " trailing "), (0.3, 0.4, "  ")]
    state.fw_segments = [types.SimpleNamespace(start=start, end=end, text=text) for start, end, text in rows]
    state.ov_result = types.SimpleNamespace(chunks=[
        types.SimpleNamespace(start_ts=start, end_ts=end, text=text) for start, end, text in rows
    ])
    recognizer = state.inference.Recognizer(str(local_model(backend)), backend=backend)

    assert recognizer.transcribe(speech()) == [
        {"start": 0.0, "end": 0.3, "text": "leading"},
        {"start": 0.2, "end": 1.0, "text": "trailing"},
    ]


def test_faster_whisper_timing_includes_complete_generator_consumption(
    backend_stubs, local_model, monkeypatch
):
    state = backend_stubs
    clock = [100.0]
    monkeypatch.setattr(state.inference, "time", types.SimpleNamespace(perf_counter=lambda: clock[0]))

    def segments():
        clock[0] += 0.25
        yield types.SimpleNamespace(start=0.0, end=0.2, text=" first ")
        clock[0] += 0.5
        yield types.SimpleNamespace(start=0.2, end=0.5, text=" second ")
        clock[0] += 0.125
        state.fw_consumed = True

    state.fw_generator_factory = segments
    recognizer = state.inference.Recognizer(str(local_model("faster-whisper")), backend="faster-whisper")
    output = recognizer.transcribe(speech())

    assert [segment["text"] for segment in output] == ["first", "second"]
    assert state.fw_consumed
    assert recognizer.last_transcribe_seconds == pytest.approx(0.875)


def test_lazy_generator_failure_has_backend_and_cuda_context(backend_stubs, local_model):
    state = backend_stubs

    def segments():
        yield types.SimpleNamespace(start=0.0, end=0.2, text=" partial ")
        raise RuntimeError("cudnn64_9.dll could not be loaded")

    state.fw_generator_factory = segments
    recognizer = state.inference.Recognizer(str(local_model("faster-whisper")), backend="faster-whisper")
    with pytest.raises(RuntimeError) as failure:
        recognizer.transcribe(speech())
    message = str(failure.value).lower().replace("_", "-")
    assert "faster-whisper" in message
    assert "cuda" in message
    assert any(word in message for word in ("setup", "install", "cpu"))


@pytest.mark.parametrize("backend", ["openvino", "faster-whisper"])
def test_silence_never_reaches_asr(backend_stubs, local_model, backend):
    state = backend_stubs
    recognizer = state.inference.Recognizer(str(local_model(backend)), backend=backend)
    assert recognizer.transcribe(np.zeros(16_000, dtype=np.float32)) == []
    assert state.fw_transcribes == state.ov_generates == []


@pytest.mark.parametrize("backend", ["openvino", "faster-whisper"])
def test_neural_vad_rejection_never_reaches_asr(backend_stubs, local_model, backend):
    state = backend_stubs
    state.vad_speech = False
    recognizer = state.inference.Recognizer(str(local_model(backend)), backend=backend)
    assert recognizer.transcribe(speech()) == []
    assert state.vads[0].frames > 0
    assert state.fw_transcribes == state.ov_generates == []


@pytest.mark.parametrize("backend", ["openvino", "faster-whisper"])
def test_vad_frame_padding_does_not_change_asr_audio(backend_stubs, local_model, backend):
    state = backend_stubs
    audio = speech(1601 / 16_000)
    recognizer = state.inference.Recognizer(str(local_model(backend)), backend=backend)
    recognizer.transcribe(audio)

    vad = state.vads[0]
    assert len(vad.received) == 4
    assert all(frame.shape == (512,) for frame in vad.received)
    np.testing.assert_array_equal(vad.received[-1][:65], audio[-65:])
    assert np.count_nonzero(vad.received[-1][65:]) == 0
    assert vad.resets >= 2
    assert not vad.has_nonzero
    calls = state.ov_generates if backend == "openvino" else state.fw_transcribes
    np.testing.assert_array_equal(calls[0][0], audio)


@pytest.mark.parametrize("backend,module_name", [
    ("openvino", "openvino_genai"), ("faster-whisper", "faster_whisper"),
    ("openvino", "sherpa_onnx"), ("faster-whisper", "sherpa_onnx"),
])
def test_missing_backend_dependencies_have_setup_guidance(
    backend_stubs, local_model, monkeypatch, backend, module_name
):
    state = backend_stubs
    monkeypatch.setitem(sys.modules, module_name, None)
    with pytest.raises(RuntimeError) as failure:
        state.inference.Recognizer(str(local_model(backend)), backend=backend)
    message = str(failure.value).lower()
    assert any(word in message for word in ("setup", "install"))
    expected_context = ("sherpa", "vad") if module_name == "sherpa_onnx" else (
        "openvino",) if backend == "openvino" else ("faster-whisper", "faster_whisper")
    assert any(word in message for word in expected_context)


def test_cuda_dll_initialization_failure_has_actionable_context(backend_stubs, local_model):
    state = backend_stubs
    state.fw_init_error = RuntimeError("Library cublas64_12.dll is not found")
    with pytest.raises(RuntimeError) as failure:
        state.inference.Recognizer(str(local_model("faster-whisper")), backend="faster-whisper")
    message = str(failure.value).lower()
    assert "cuda" in message
    assert any(word in message for word in ("setup", "install", "cpu"))
    assert any(word in message for word in ("cublas", "cudnn", "dll"))


@pytest.mark.parametrize("packages", [("cublas",), ("cublas", "cudnn")])
def test_only_existing_project_cuda_dll_directories_are_registered_and_held(
    backend_stubs, local_model, monkeypatch, tmp_path, packages
):
    state = backend_stubs
    root = tmp_path / "trusted-project"
    monkeypatch.setattr(state.inference, "_PROJECT_ROOT", root)
    library_root = root / ".venv" / "Lib" / "site-packages" / "nvidia"
    expected = []
    for package in packages:
        directory = library_root / package / "bin"
        directory.mkdir(parents=True)
        expected.append(directory.resolve())

    decoy = tmp_path / "untrusted-cuda"
    decoy.mkdir()
    (library_root / "unrelated" / "bin").mkdir(parents=True)
    monkeypatch.setenv("CUDA_PATH", str(decoy))
    monkeypatch.setenv("PATH", str(decoy), prepend=state.inference.os.pathsep)
    recognizer = state.inference.Recognizer(str(local_model("faster-whisper")), backend="faster-whisper")

    assert set(state.dll_registrations) == set(expected)
    assert len(state.dll_registrations) == len(expected)
    assert len(recognizer._dll_handles) == len(expected)
    assert all(handle in recognizer._dll_handles for handle in state.dll_handles)
    assert all(not handle.closed for handle in state.dll_handles)


@pytest.mark.parametrize("backend,device", [("openvino", "GPU"), ("faster-whisper", "cpu")])
def test_non_cuda_backends_do_not_register_nvidia_directories(
    backend_stubs, local_model, backend, device
):
    state = backend_stubs
    for package in ("cublas", "cudnn"):
        (state.inference._PROJECT_ROOT / ".venv" / "Lib" / "site-packages" /
         "nvidia" / package / "bin").mkdir(parents=True)
    recognizer = state.inference.Recognizer(str(local_model(backend)), backend=backend, device=device)
    assert state.dll_registrations == []
    assert recognizer._dll_handles == []


def test_failed_cuda_constructor_closes_registered_directory_handles(backend_stubs, local_model):
    state = backend_stubs
    for package in ("cublas", "cudnn"):
        (state.inference._PROJECT_ROOT / ".venv" / "Lib" / "site-packages" /
         "nvidia" / package / "bin").mkdir(parents=True)
    state.fw_init_error = RuntimeError("synthetic CUDA initialization failure")
    with pytest.raises(RuntimeError):
        state.inference.Recognizer(str(local_model("faster-whisper")), backend="faster-whisper")
    assert len(state.dll_handles) == 2
    assert all(handle.closed for handle in state.dll_handles)


@pytest.mark.parametrize("already_in_path", [False, True])
def test_cuda_path_prefix_is_trusted_preserves_suffix_and_is_idempotent(
    backend_stubs, monkeypatch, tmp_path, already_in_path
):
    state = backend_stubs
    directories = []
    for package in ("cublas", "cudnn"):
        directory = (state.inference._PROJECT_ROOT / ".venv" / "Lib" /
                     "site-packages" / "nvidia" / package / "bin")
        directory.mkdir(parents=True)
        directories.append(str(directory.resolve()))

    separator = state.inference.os.pathsep
    preserved = [str(tmp_path / "original-first"), "", str(tmp_path / "original with spaces"),
                 str(tmp_path / "original-first")]
    original_entries = list(preserved)
    if already_in_path:
        original_entries.insert(1, directories[1])
        original_entries.insert(3, directories[0].upper())
        original_entries.append(directories[1])
    monkeypatch.setenv("PATH", separator.join(original_entries))
    original_cuda_path = str(tmp_path / "external-cuda")
    monkeypatch.setenv("CUDA_PATH", original_cuda_path)
    expected_path = separator.join(directories + preserved)

    first_handles = state.inference._register_nvidia_dlls()
    assert state.inference.os.environ["PATH"] == expected_path
    second_handles = state.inference._register_nvidia_dlls()
    assert state.inference.os.environ["PATH"] == expected_path
    assert state.inference.os.environ["CUDA_PATH"] == original_cuda_path
    assert state.dll_registrations == [Path(path) for path in directories] * 2
    assert len(first_handles) == len(second_handles) == 2
    assert all(not handle.closed for handle in first_handles + second_handles)
