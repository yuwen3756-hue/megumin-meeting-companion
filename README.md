# Megumin Companion

A Windows desktop companion for interview practice, presentation rehearsal, and meeting notes. Speech recognition runs locally from downloaded models. Preparation, transcripts, corrections, and notes stay in local files unless you deliberately export or share them.

| Workflow | Implemented features |
| --- | --- |
| Interview practice | Separate preparation profile, fixed local practice prompts, typed answers, a STAR checklist, and saved practice history. |
| Presentation rehearsal | Import selected PPTX, TXT, Markdown, or JSON files; choose specific excerpts; rehearse questions and answers. |
| Live meeting assistance | Separate microphone and system-output capture, English/Mandarin transcription, anonymous speaker estimates, editable notes, and transcript exports. |

The optional private assistant prepares questions and context for review. It never automatically forwards transcript questions. Its experimental Codex bridge has **not been validated with a real cloud answer**; local preparation and transcription do not require it.

## Windows setup

Use Windows with **Python 3.12 x64, including Tk**, and a compatible installed GPU driver. Open PowerShell in the project folder:

```powershell
.\Setup.cmd --plan
.\Setup.cmd
.\Launch.cmd
```

Setup previews its selected route and download disclosure before installation. It creates a project-local `.venv` and downloads models under `models`. The application itself does not download models. Explicit setup choices are available:

```powershell
.\Setup.cmd --target arc
.\Setup.cmd --target nvidia
```

Choose one route for the current computer. `--model small` selects the smaller fallback. Recreate the virtual environment on another computer rather than copying an existing `.venv`. OpenVINO and CTranslate2 ASR models use different formats. See [models and dependencies](docs/MODELS.md).

To inspect the simulated interface after creating the environment:

```powershell
.\.venv\Scripts\python.exe main.py --demo
```

The demo is labeled synthetic and performs no inference or capture.

## Start a session

1. Choose a workflow and open **Prepare / practice**. This starts neither recording nor a cloud connection.
2. For live transcription, select a microphone, a system-output loopback, or both. Confirm participant agreement, then explicitly start capture.
3. Review the recording indicator and queue/error status. Captions arrive as completed speech chunks, up to roughly six seconds plus processing time; these are not word-by-word captions.
4. Stop capture before changing workflows. Correct transcript text, assign speaker names yourself, edit notes, or export TXT, Markdown, JSON, CSV, and SRT.

System loopback captures **all applications on the selected playback output**, not one meeting application. Microphone capture is independent of a meeting application's mute control. Headphones help reduce duplicated audio; acoustic echo cancellation is not implemented. Audio retention is off by default.

## What has been tested

Local Intel Arc A770 16 GB inference was exercised with paced synthetic English/Mandarin audio, including two simultaneous source queues and the native interface. This is limited replay evidence. **Live Zoom/microphone capture, RTX 4070 Laptop performance, and real Codex answers remain unverified.** Speaker estimates and overlapping speech need particular care. See [validation scope](docs/VALIDATION.md).

- [Preparation and practice guide](docs/COMPANION.md)
- [Privacy, storage, and optional cloud review](docs/PRIVACY.md)
- [Model routes, downloads, and third-party licenses](docs/MODELS.md)

For automated checks, install the development requirements into the project environment and run `python -m pytest tests -q` using that environment. These checks do not replace testing the actual audio devices and hardware you intend to use.

## License status

An application license has not yet been specified. Third-party dependencies and downloaded models retain their own licenses and notices; see [model and component information](docs/MODELS.md).
