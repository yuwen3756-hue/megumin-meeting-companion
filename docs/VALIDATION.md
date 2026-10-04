# Validation scope

This project combines local inference, native UI workflows, and an optional experimental cloud bridge. Evidence for one does not establish the others.

The sanitized public source snapshot passed **327 automated tests** in local validation on October 4, 2026. Compilation also passed. These checks do not establish live capture or real cloud connectivity.

The public snapshot's native Windows demo also rendered eight generic synthetic segments and passed correction, search, speaker-renaming, and five-format export checks with no UI callback errors or audio capture. Generated screenshots, session data, exports and local reports are excluded from this repository.

| Area | Established scope | Still unverified |
| --- | --- | --- |
| Intel Arc ASR | Actual OpenVINO turbo/small inference on Arc A770 16 GB with paced synthetic English/Mandarin replay, VAD, speaker embeddings, persistence, notes, and native UI. Single- and two-source queues were exercised. | Real meeting acoustics, live Zoom/microphone capture, broad recognition accuracy, and performance on other systems. |
| NVIDIA ASR | Backend selection, local-runtime setup path, and configuration are implemented. | RTX 4070 Laptop 8 GB initialization, throughput, thermal behavior, memory use, and two-source headroom. |
| Preparation and review UI | Local profiles, explicit material selection, typed answers, persistence, review controls, and cancellation were exercised with synthetic fixtures. | Whether fixed practice prompts or checklists improve interview or presentation outcomes. |
| Speaker estimates | Anonymous embedding/clustering path exercised in replay. | Reliable simultaneous-speaker separation, precise turns, identity recognition, and Mandarin/noisy-meeting attribution accuracy. |
| Codex bridge | Protocol controls and streaming/cancellation behavior tested with simulated transports and canned replies. | Real connection acceptance, model catalog availability, cloud answer quality, latency, throughput, and token usage. |

Recorded replay timing separates model loading, first-call cost, warm inference, queues, and UI insertion. Synthetic dual-source replay stresses queue load; it does not establish diarization accuracy. Word/character errors on a small scripted corpus are not general accuracy guarantees. Private recordings, presentation content, account details, and machine-specific logs are not included in this public documentation.

Automated tests are useful regression checks, not proof of zero network traffic or production readiness. Before relying on a hardware route, measure startup, queue growth, caption delay, CPU/RAM/VRAM, and representative English/Mandarin speech on that computer. Validate each intended capture device separately with participant agreement. Keep real cloud evaluation distinct from local ASR and mock-response UI timings.
