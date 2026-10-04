"""File-only inference. Downloads belong exclusively to scripts/setup_models.py."""
from __future__ import annotations

import os
from pathlib import Path
import time
import numpy as np

# Defence in depth: no model hub fallback or telemetry during a meeting.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

from .telemetry import disable_openvino_telemetry
disable_openvino_telemetry()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _register_nvidia_dlls() -> list:
    """Add only the two optional NVIDIA package bins in this project's venv.

    Keep the returned Windows DLL-directory handles alive while the recognizer
    exists. Legacy LoadLibraryA callers also need these exact bins prepended to
    this process's PATH. No persistent environment edits, arbitrary directory
    search, CUDA_PATH changes, package imports, or installation are performed.
    """
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []
    project = _PROJECT_ROOT.resolve()
    site_packages = (project / ".venv" / "Lib" / "site-packages").resolve()
    if not site_packages.is_relative_to(project):
        raise RuntimeError("The project's NVIDIA DLL directory resolves outside the project.")
    handles = []
    directories = []
    try:
        for package in ("cublas", "cudnn"):
            directory = site_packages / "nvidia" / package / "bin"
            if not directory.is_dir():
                continue
            resolved = directory.resolve()
            if not resolved.is_relative_to(site_packages):
                raise RuntimeError("An NVIDIA package DLL directory resolves outside the project environment.")
            handles.append(os.add_dll_directory(str(resolved)))
            directories.append(str(resolved))
        if directories:
            # CTranslate2 may call legacy LoadLibraryA, which does not reliably
            # use AddDllDirectory's search set. Change only this process; do not
            # write user/system environment settings or add duplicate entries
            # when another local recognizer is constructed.
            registered = {os.path.normcase(os.path.normpath(path)) for path in directories}
            original = os.environ.get("PATH", "")
            entries = original.split(os.pathsep) if original else []
            preserved = [path for path in entries if os.path.normcase(os.path.normpath(path)) not in registered]
            os.environ["PATH"] = os.pathsep.join(directories + preserved)
    except Exception:
        for handle in handles:
            handle.close()
        raise
    return handles


class Recognizer:
    """One sequential local ASR model with a shared CPU speech gate.

    ``compute_type`` selects faster-whisper's arithmetic; OpenVINO precision is
    determined by its exported local model. Beam sizes above one are optional
    candidates, without a claim of measured quality or speed on either GPU.
    """

    def __init__(self, model_path: str, backend: str = "openvino", device: str = "GPU",
                 language: str = "auto", compute_type: str | None = None, beam_size: int = 1):
        self.path = Path(model_path).resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(f"ASR model is missing: {self.path}. Run Setup.cmd first.")
        if language not in {"auto", "en", "zh"}:
            raise ValueError("Choose auto, en, or zh.")
        backend = str(backend).strip().lower().replace("_", "-")
        if backend not in {"openvino", "faster-whisper"}:
            raise ValueError("Choose the openvino or faster-whisper inference backend.")
        if isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size < 1:
            raise ValueError("beam_size must be a positive integer.")
        if compute_type is not None and (not isinstance(compute_type, str) or not compute_type.strip()):
            raise ValueError("compute_type must be a nonempty string or None.")
        self.backend, self.language, self.beam_size = backend, language, beam_size
        if backend == "faster-whisper":
            device_key = str(device).strip().lower()
            if device_key not in {"gpu", "cuda", "cpu"}:
                raise ValueError("faster-whisper device must be cuda/GPU or cpu/CPU.")
            self.device = "cuda" if device_key in {"cuda", "gpu"} else "cpu"
            self.compute_type = compute_type or ("int8_float16" if self.device == "cuda" else "int8")
            required = ("config.json", "tokenizer.json", "model.bin")
        else:
            self.device, self.compute_type = str(device).strip().upper(), compute_type
            required = (
                "openvino_encoder_model.xml", "openvino_encoder_model.bin",
                "openvino_decoder_model.xml", "openvino_decoder_model.bin",
                "openvino_tokenizer.xml", "openvino_tokenizer.bin",
                "openvino_detokenizer.xml", "openvino_detokenizer.bin",
                "config.json", "tokenizer.json", "generation_config.json", "preprocessor_config.json",
            )
        # A local directory alone is insufficient: faster-whisper can otherwise
        # attempt tokenizer fallback. Validate all assets before importing any
        # inference backend or registering native DLLs.
        for name in required:
            if not (self.path / name).is_file():
                label = "OpenVINO" if backend == "openvino" else "faster-whisper"
                raise FileNotFoundError(f"Incomplete {label} model: {name}. Run Setup.cmd for this profile.")
        vad_path = self.path.parent / "silero_vad.onnx"
        if not vad_path.is_file():
            raise FileNotFoundError("Missing local Silero VAD model. Run Setup.cmd.")

        self._dll_handles = []
        self.last_transcribe_seconds = 0.0
        self.detected_language = None
        started = time.perf_counter()
        try:
            import sherpa_onnx
            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = str(vad_path)
            config.silero_vad.threshold = 0.5
            config.silero_vad.min_speech_duration = 0.16
            config.silero_vad.min_silence_duration = 0.25
            config.sample_rate = 16000
            config.num_threads = 1
            config.provider = "cpu"
            self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=12)
        except Exception as exc:
            raise RuntimeError(
                f"Local CPU Silero VAD is unavailable: {exc}. "
                "Check the sherpa-onnx dependency and local VAD model using the approved setup. "
                "Nothing was downloaded or installed."
            ) from exc
        try:
            if backend == "openvino":
                import openvino_genai
                self.model = openvino_genai.WhisperPipeline(str(self.path), self.device)
            else:
                if self.device == "cuda":
                    self._dll_handles = _register_nvidia_dlls()
                from faster_whisper import WhisperModel
                self.model = WhisperModel(
                    str(self.path), local_files_only=True, device=self.device,
                    device_index=0, compute_type=self.compute_type,
                    cpu_threads=4, num_workers=1,
                )
        except Exception as exc:
            for handle in self._dll_handles:
                handle.close()
            self._dll_handles.clear()
            raise self._backend_error("initialization", exc) from exc
        self.load_seconds = time.perf_counter() - started

    def _backend_error(self, operation: str, exc: Exception) -> RuntimeError:
        if self.backend == "faster-whisper" and self.device == "cuda":
            guidance = (
                "Check the faster-whisper/CTranslate2 dependencies and NVIDIA driver. "
                "CUDA execution requires CUDA 12 cuBLAS and cuDNN 9 DLLs; missing-DLL errors "
                "can be addressed through the approved setup, or choose a CPU profile."
            )
        elif self.backend == "faster-whisper":
            guidance = "Check the faster-whisper/CTranslate2 dependencies and the selected local CPU model in the approved setup."
        else:
            guidance = "Check openvino-genai, the selected local model and the Intel graphics driver, or choose an OpenVINO CPU profile."
        return RuntimeError(
            f"{self.backend} {self.device} {operation} failed: {exc}. "
            f"{guidance} Nothing was downloaded or installed."
        )

    def _has_speech(self, samples: np.ndarray) -> bool:
        # The stream energy gate supplies boundaries. This shared CPU neural VAD
        # rejects noise-only clips without shifting the original clip timestamps.
        self._vad.reset()
        try:
            for offset in range(0, len(samples), 512):
                frame = samples[offset:offset + 512]
                if len(frame) < 512:
                    frame = np.pad(frame, (0, 512 - len(frame)))
                self._vad.accept_waveform(frame)
            self._vad.flush()
            return not self._vad.empty()
        finally:
            self._vad.reset()

    @staticmethod
    def _segment(start, end, text, duration) -> dict | None:
        text = str(text).strip()
        if not text:
            return None
        try:
            start = float(start)
        except (TypeError, ValueError):
            start = 0.0
        start = max(0.0, min(duration, start)) if np.isfinite(start) else 0.0
        try:
            end = float(end)
        except (TypeError, ValueError):
            end = duration
        if not np.isfinite(end) or end < start:
            end = duration
        return dict(start=start, end=max(start, min(duration, end)), text=text)

    def transcribe(self, audio: np.ndarray) -> list[dict]:
        started = time.perf_counter()
        try:
            samples = np.asarray(audio, dtype=np.float32).reshape(-1)
            if len(samples) < 1600 or not np.isfinite(samples).all():
                return []
            if float(np.max(np.abs(samples))) <= 1e-6 or not self._has_speech(samples):
                return []
            duration = len(samples) / 16000
            output = []
            try:
                if self.backend == "openvino":
                    options = dict(task="transcribe", return_timestamps=True, max_new_tokens=180)
                    if self.language != "auto":
                        options["language"] = f"<|{self.language}|>"
                    if self.beam_size > 1:
                        options["num_beams"] = self.beam_size
                    result = self.model.generate(samples, **options)
                    chunks = getattr(result, "chunks", None)
                    if chunks:
                        for chunk in chunks:
                            segment = self._segment(chunk.start_ts, chunk.end_ts, chunk.text, duration)
                            if segment is not None:
                                output.append(segment)
                    else:
                        segment = self._segment(0.0, duration, " ".join(getattr(result, "texts", [])), duration)
                        if segment is not None:
                            output.append(segment)
                else:
                    segments, info = self.model.transcribe(
                        samples, language=None if self.language == "auto" else self.language,
                        task="transcribe", beam_size=self.beam_size,
                        temperature=0.0, best_of=1,
                        condition_on_previous_text=False, vad_filter=False,
                        word_timestamps=False, without_timestamps=False, max_new_tokens=180,
                    )
                    self.detected_language = getattr(info, "language", None)
                    # faster-whisper returns a lazy generator. Consume it here,
                    # inside both the error handler and the measured interval.
                    # WhisperModel processes this single clip sequentially; no
                    # BatchedInferencePipeline or extra worker is constructed.
                    for chunk in segments:
                        segment = self._segment(chunk.start, chunk.end, chunk.text, duration)
                        if segment is not None:
                            output.append(segment)
            except Exception as exc:
                raise self._backend_error("transcription", exc) from exc
            return output
        finally:
            self.last_transcribe_seconds = time.perf_counter() - started


class SpeakerTracker:
    """Anonymous, per-session voice clustering; similarity is not probability.

    Three overlapping embedding windows must agree before assignment. This is
    deliberately conservative and does not claim to detect mixed-voice overlap.
    """
    def __init__(self, model_path: str | None = None, threshold: float = 0.70):
        self.extractor = None
        self.threshold = threshold
        self.reset()
        if model_path and Path(model_path).is_file():
            import sherpa_onnx
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(Path(model_path).resolve()),
                                                               num_threads=2, provider="cpu")
            if not cfg.validate():
                raise ValueError("Invalid local speaker model")
            self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def reset(self):
        self.centroids: list[np.ndarray] = []
        self.counts: list[int] = []

    def _embedding(self, audio):
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=np.asarray(audio, dtype=np.float32))
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            raise ValueError("Insufficient speech for speaker embedding")
        vector = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm < 1e-8 or not np.isfinite(vector).all():
            raise ValueError("Invalid speaker embedding")
        return vector / norm

    def classify(self, audio) -> dict:
        unknown = dict(speaker="Unknown", confidence=None, warning="")
        if self.extractor is None:
            return dict(unknown, warning="Speaker model unavailable; remote voices are unassigned.")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(samples) < 24000:
            return dict(unknown, warning="Short turn; assign a speaker manually if known.")
        try:
            # Inspect voice consistency across the utterance, never infer identities.
            window = min(len(samples), 40000)
            offsets = sorted(set([0, (len(samples) - window) // 2, len(samples) - window]))
            vectors = [self._embedding(samples[p:p + window]) for p in offsets]
            coherence = min(float(np.dot(a, b)) for a in vectors for b in vectors)
            if coherence < 0.60:
                return dict(unknown, confidence=round(coherence, 3),
                            warning="Changing or uncertain voice; may contain multiple speakers. Review manually.")
            vector = np.mean(vectors, axis=0)
            vector /= np.linalg.norm(vector)
            if not self.centroids:
                self.centroids.append(vector)
                self.counts.append(1)
                return dict(speaker="Speaker 1", confidence=None, warning="New anonymous voice; provisional.")
            similarities = np.asarray([float(np.dot(vector, c)) for c in self.centroids])
            index = int(np.argmax(similarities))
            score = float(similarities[index])
            gap = score - float(np.partition(similarities, -2)[-2]) if len(similarities) > 1 else 1.
            if score >= self.threshold and gap >= 0.08:
                # Slow updates reduce accidental merging of distinct voices.
                combined = self.centroids[index] * min(self.counts[index], 20) + vector
                self.centroids[index] = combined / np.linalg.norm(combined)
                self.counts[index] += 1
                return dict(speaker=f"Speaker {index + 1}", confidence=round(score, 3), warning="Provisional voice match.")
            if score < 0.48 and len(self.centroids) < 16:
                self.centroids.append(vector)
                self.counts.append(1)
                return dict(speaker=f"Speaker {len(self.centroids)}", confidence=None, warning="New anonymous voice; provisional.")
            return dict(unknown, confidence=round(score, 3), warning="Ambiguous voice match; manual correction available.")
        except Exception as exc:
            return dict(unknown, warning=f"Speaker inference unavailable: {type(exc).__name__}")
