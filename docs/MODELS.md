# Model routes and third-party components

Setup installs the selected backend in `.venv` and model files under `models`. It records pinned model revisions and SHA-256 hashes, permits model/configuration data files, and does not enable model-supplied Python code. Models are downloaded separately; they are not a promise of identical performance on every GPU.

The provenance below comes from the project's source checks dated October 3–4, 2026. It is not a claim about the newest upstream releases. Keep complete local snapshots, including tokenizers and configuration files, for offline loading.

| Target | Default candidate | Smaller fallback | Validation |
| --- | --- | --- | --- |
| Intel Arc | OpenVINO Whisper large-v3-turbo INT8, GPU, beam 1 | OpenVINO Whisper-small INT8 | Limited actual-model replay on Arc A770 16 GB. |
| NVIDIA | faster-whisper/CTranslate2 large-v3-turbo, CUDA `int8_float16`, beam 1 | faster-whisper small | Implemented route; RTX 4070 Laptop 8 GB execution remains unverified. |

One ASR model and one worker process the two source queues sequentially. CPU Silero VAD filters speech, and CPU WeSpeaker embeddings support anonymous speaker estimates. Notes use extractive rules, not another language model. Batch 1 and quantization reduce resource demands but impose no hard VRAM cap.

## Setup and configuration

Use `Setup.cmd --plan` to inspect the route before installation. `--target arc` or `--target nvidia` selects it explicitly; `--model small` selects the smaller model. `settings.json` accepts:

```json
{"target": "auto", "model": "turbo", "cpu": false}
```

Supported targets are `auto`, `arc`, and `nvidia`; supported model choices are `turbo` and `small`. An already-installed same-backend small model can substitute if turbo is absent, with a notice. GPU load failure stops startup rather than silently switching to CPU. Setting `cpu` to true is an explicit, unbenchmarked fallback.

The Arc route does not install CUDA. The NVIDIA route uses project-local CTranslate2/faster-whisper and NVIDIA CUDA 12 cuBLAS/cuDNN 9 runtime wheels. It does not install a full CUDA Toolkit or change a system driver. Runtime DLL directories are added for the app process, not permanently to Windows PATH. A compatible existing driver and native runtime remain necessary; the laptop stack needs actual initialization and workload testing.

## Download provenance and model licenses

Keep the notices distributed with each downloaded model and dependency. Model licenses are separate from one another and from the application.

| Component | Source and license information |
| --- | --- |
| Whisper | Upstream code and weights are MIT: [OpenAI Whisper](https://github.com/openai/whisper). |
| Arc turbo conversion | [OpenVINO/whisper-large-v3-turbo-int8-ov](https://huggingface.co/OpenVINO/whisper-large-v3-turbo-int8-ov), conversion card MIT. |
| Arc small conversion | [OpenVINO/whisper-small-int8-ov](https://huggingface.co/OpenVINO/whisper-small-int8-ov), conversion card Apache-2.0 while upstream Whisper is MIT; preserve both notices rather than treating the distinction as resolved. |
| NVIDIA small conversion | [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small); retain the conversion's distributed notices and upstream Whisper attribution. |
| NVIDIA turbo conversion | The recorded request to `mobiuslabsgmbh/faster-whisper-large-v3-turbo` resolved to [dropbox-dash/faster-whisper-large-v3-turbo](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo), a third-party MIT conversion. See the [official faster-whisper alias mapping](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/utils.py). This is not an OpenAI-published conversion. Runtime INT8 does not reduce its FP16 download size. |
| Silero VAD | MIT: [Silero project](https://github.com/snakers4/silero-vad); the selected ONNX asset is distributed through [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx). |
| WeSpeaker VoxCeleb ResNet34 | Pretrained weights require CC BY 4.0 attribution: [WeSpeaker model documentation](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md). Selected [ONNX asset](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx). Mandarin attribution accuracy remains unverified. |

NVIDIA runtime libraries have separate vendor license terms. See [Windows CUDA runtime-wheel guidance](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#pip-wheels) and [CTranslate2 installation requirements](https://opennmt.net/CTranslate2/installation.html). Check the setup disclosure and pinned package requirements rather than assuming the newest upstream versions form a validated combination.

The implemented speaker model is WeSpeaker. Alternative models mentioned in research are not automatically installed or covered by its weight license. Full Whisper large-v3 and beam-5 accuracy experiments are not additional validated live modes.
