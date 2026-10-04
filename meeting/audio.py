"""Windows WASAPI capture. Importing this module never opens an audio device."""
from __future__ import annotations

import math
import threading
import time
from typing import Callable

import numpy as np
from scipy.signal import resample_poly


def _pyaudio():
    try:
        import pyaudiowpatch
    except ImportError as exc:
        raise RuntimeError("PyAudioWPatch is unavailable; install the offline application dependencies.") from exc
    return pyaudiowpatch


def list_devices() -> list[dict]:
    """Enumerate WASAPI inputs and loopbacks without opening recording streams."""
    module = _pyaudio()
    pa = module.PyAudio()
    try:
        api = pa.get_host_api_info_by_type(module.paWASAPI)
        devices = []
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if int(info.get("hostApi", -1)) != int(api["index"]):
                continue
            channels = int(info.get("maxInputChannels", 0))
            if channels < 1:
                continue
            devices.append({
                "id": index, "name": str(info["name"]),
                "kind": "loopback" if info.get("isLoopbackDevice", False) else "mic",
                "sample_rate": int(round(info["defaultSampleRate"])),
                "channels": channels,
            })
        return devices
    except OSError as exc:
        raise RuntimeError(f"Windows audio devices are unavailable: {exc}") from exc
    finally:
        pa.terminate()


class StreamingResampler:
    """Polyphase conversion with retained filter context and an exact sample clock.

    Unlike resampling independent callbacks, this keeps both left context and a
    short right lookahead. Trimming only at a rational-rate phase boundary avoids
    cumulative timestamp drift, including 44.1 kHz input.
    """

    def __init__(self, input_rate: int, output_rate: int = 16000):
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("Sample rates must be positive")
        self.input_rate, self.output_rate = int(input_rate), int(output_rate)
        divisor = math.gcd(self.input_rate, self.output_rate)
        self.up, self.down = self.output_rate // divisor, self.input_rate // divisor
        self._guard = math.ceil(10 * max(self.up, self.down) / self.up) + 2
        self._buffer = np.empty(0, dtype=np.float32)
        self._base = self._total = self._emitted = 0
        self._flushed = False

    def process(self, samples: np.ndarray) -> np.ndarray:
        if self._flushed:
            raise RuntimeError("Cannot append samples after resampler flush")
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all():
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        if self.up == self.down:
            self._total += len(samples)
            self._emitted += len(samples)
            return samples.copy()
        self._buffer = np.concatenate((self._buffer, samples))
        self._total += len(samples)
        stable = max(0, (self._total - self._guard) * self.up // self.down)
        return self._extract(stable)

    def _extract(self, target: int) -> np.ndarray:
        if target <= self._emitted:
            return np.empty(0, dtype=np.float32)
        converted = resample_poly(self._buffer, self.up, self.down)
        offset = self._base * self.up // self.down
        result = converted[self._emitted - offset:target - offset].astype(np.float32, copy=True)
        self._emitted += len(result)
        keep_from = max(self._base, self._emitted * self.down // self.up - 2 * self._guard)
        keep_from = (keep_from // self.down) * self.down
        remove = max(0, keep_from - self._base)
        if remove:
            self._buffer = self._buffer[remove:]
            self._base += remove
        return result

    def flush(self) -> np.ndarray:
        if self._flushed:
            return np.empty(0, dtype=np.float32)
        result = self._extract(self._total * self.up // self.down)
        self._flushed = True
        self._buffer = np.empty(0, dtype=np.float32)
        return result


class AudioCapture:
    """One explicit-start native stream; callback performs no inference or disk IO."""

    def __init__(self, device_id: int, source: str, started_at: float,
                 on_audio: Callable, on_warning: Callable, on_error: Callable):
        self.device_id, self.source, self.started_at = device_id, source, started_at
        self.on_audio, self.on_warning, self.on_error = on_audio, on_warning, on_error
        self._pa = self._stream = self._resampler = None
        self._running = False
        self._offset = None
        self._samples = 0
        self._last_overflow_warning = 0.0
        self._close_thread = None
        self._close_lock = threading.Lock()
        self.closed = threading.Event()
        self.closed.set()

    def start(self):
        module = _pyaudio()
        self._pa = module.PyAudio()
        self.closed.clear()
        try:
            info = self._pa.get_device_info_by_index(int(self.device_id))
            loopback = bool(info.get("isLoopbackDevice", False))
            if int(info.get("maxInputChannels", 0)) < 1:
                raise ValueError("Selected device has no input channels")
            if loopback != (self.source == "remote"):
                raise ValueError("Select a WASAPI loopback for remote audio and a microphone for local audio")
            rate = int(round(info["defaultSampleRate"]))
            channels = int(info["maxInputChannels"])
            self._resampler = StreamingResampler(rate)
            self._running = True

            def callback(data, frame_count, time_info, status):
                if not self._running:
                    return (None, module.paComplete)
                try:
                    if status and time.monotonic() - self._last_overflow_warning > 2:
                        self._last_overflow_warning = time.monotonic()
                        self.on_warning(f"{self.source}: audio driver reported status {status}; samples may be missing.")
                    native = np.frombuffer(data, dtype=np.float32).reshape(-1, channels).mean(axis=1)
                    if self._offset is None:
                        self._offset = max(0.0, time.monotonic() - self.started_at - len(native) / rate)
                    output = self._resampler.process(native)
                    if len(output):
                        self.on_audio(self.source, output, self._offset + self._samples / 16000)
                        self._samples += len(output)
                    return (None, module.paContinue)
                except Exception as exc:
                    self._running = False
                    self.on_error(f"{self.source}: capture failed ({exc}); select an available device and restart.")
                    return (None, module.paAbort)

            self._stream = self._pa.open(
                format=module.paFloat32, channels=channels, rate=rate, input=True,
                input_device_index=int(self.device_id), frames_per_buffer=max(1, rate // 20),
                stream_callback=callback, start=False,
            )
            self._stream.start_stream()
        except Exception:
            self._running = False
            self._close()
            raise

    @property
    def active(self) -> bool:
        try:
            return bool(self._running and self._stream is not None and self._stream.is_active())
        except Exception:
            return False

    def _close(self):
        try:
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                finally:
                    self._stream.close()
            if self._resampler is not None:
                tail = self._resampler.flush()
                if len(tail) and self._offset is not None:
                    self.on_audio(self.source, tail, self._offset + self._samples / 16000)
        except Exception as exc:
            self.on_warning(f"{self.source}: audio stream close reported {exc}")
        finally:
            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception as exc:
                    self.on_warning(f"{self.source}: audio driver cleanup reported {exc}")
            self._stream = self._pa = None
            self.closed.set()

    def stop(self, timeout: float = 2.0) -> bool:
        self._running = False
        if self.closed.is_set():
            return True
        with self._close_lock:
            if self._close_thread is None:
                self._close_thread = threading.Thread(target=self._close, name=f"audio-close-{self.source}", daemon=True)
                self._close_thread.start()
        return self.closed.wait(max(0.0, timeout))
