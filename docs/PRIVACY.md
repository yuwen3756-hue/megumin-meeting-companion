# Local data and optional cloud review

## Capture and local storage

Capture starts only after an explicit Start action and participant-agreement confirmation. Both audio sources and audio retention are off by default. Opening preparation or practice does not start recording.

- Microphone and system-output loopback are separate sources. Loopback captures all applications using the selected output; microphone recording is independent of a meeting application's mute control.
- Transcripts, original recognition, user corrections, speaker labels, and diagnostics are committed to `data/meetings.sqlite3` using SQLite WAL. Interrupted sessions remain readable on restart.
- Preparation, copied material text, typed answers, and practice history are saved in `data/companion.json`.
- With **Keep audio** off, audio is held in bounded processing queues. Enabling it writes per-source mono 16 kHz WAV files under `data/audio/<session>/recordings`.
- Exports remain where you save them. A damaged final WAV after an abrupt exit does not imply that committed transcript events were lost.

These are ordinary local files, not application-encrypted storage. Windows permissions, backups, and any folder-sync software still apply. Removing imported material does not remove earlier exports. There is no built-in permanent session-delete workflow; manage local data deliberately with the app closed.

Speaker clustering produces anonymous estimates, not identity recognition. User-assigned names and source labels are distinct. Short or ambiguous speech may remain Unknown; cosine similarity scores are not calibrated probabilities. Extracted action/decision suggestions are candidate quotes, not confirmed commitments, owners, or deadlines.

## What the network controls do

Normal speech-recognition operation loads complete local model files. Setup is a separate, explicit download step. Runtime guards restrict Python socket connections and disable Hugging Face telemetry. A process-local OpenVINO guard disables the inspected telemetry consent/sender paths before runtime imports; it does not modify user telemetry settings.

These controls are **not a Windows firewall or a packet-level isolation guarantee**. They do not cover arbitrary native networking or unrelated processes, and they depend on the inspected dependency APIs. Changing dependencies requires renewed validation.

## Optional private assistant

Local question detection is off by default. It uses stable transcript text and cannot establish whom a question addresses. You can edit, dismiss, or manually enter a question. Selecting a folder only scopes explicit file choices; it does not scan the folder.

The assistant displays the exact redacted meeting payload, destination, model/effort, and fixed base/developer policies before a one-use Send approval. Credential-pattern redaction and sensitive-content warnings are heuristics; review the entire payload. Changing the request requires a new preview. There is no automatic forwarding, retry, or cloud question backlog.

The Codex bridge is experimental and has not completed real cloud validation. Enabling its controls does not connect. Explicit **Connect / discover** performs preflight through a separate installed Codex CLI process; metadata/authentication operations can themselves use the network. The bridge refuses unsupported configuration and inherited instruction sources it cannot isolate, including relevant `AGENTS.md` guidance. It does not authorize that guidance automatically, copy credentials, or reuse the current desktop conversation.

If connection validation succeeds in a separately tested setup, each text request still needs its own preview approval. The optional native child uses networking independently of the ASR Python guard. Cancellation cannot recall submitted text; ephemeral local threads do not establish service-side zero retention. Model availability, cloud latency, and answer quality are unverified.

Saving or copying a reviewed prompt is local. Pasting and submitting it to a cloud service is a separate deliberate transfer governed by that service's data handling. The app does not post answers to meeting chat or speak them aloud. Its private window can still be visible when sharing the whole desktop.
