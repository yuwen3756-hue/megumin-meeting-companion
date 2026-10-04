# Prepare, practice, and take notes

Open **Prepare / practice** from the main window. Interview practice, presentation rehearsal, and live meeting assistance keep separate profiles. Each holds a topic, role, goal, selected materials, question, typed answer, notes, and practice history.

## A local practice workflow

1. Enter the topic, your role, and the goal.
2. Optionally add a material file and select relevant excerpts.
3. Open **Practice & review** and choose **Make practice questions**, or type your own question.
4. Type an answer, use the displayed checklist, and choose **Save practice answer**.
5. Edit **Local notes** and copy, export, or append them to the local meeting workspace.

Practice questions are a small fixed bank adapted with your topic. They are rule-based prompts, not model-generated questions or automatic answer evaluations. The interview checklist uses STAR; presentation practice emphasizes evidence and assumptions; meeting preparation separates confirmed facts from unresolved questions.

## Materials are explicitly selected

**Add material** reads only the chosen file, copies its extracted text into local preparation storage, and selects no excerpts automatically. Ctrl-click to select multiple excerpts. The preview shows their copied text; **Clear selection** excludes them from the next review.

| Input | Limit and behavior |
| --- | --- |
| UTF-8 TXT, Markdown, JSON | Up to 128 KiB; JSON is literal text, not executable configuration. |
| PPTX | Up to 12 MiB, 100 sections, and 60,000 extracted characters. Slide text in presentation order; no speaker notes, OCR, embedded objects, or image interpretation. |
| Per workflow | Up to five imported files. |
| Per review | Up to five explicitly selected excerpts, totaling 5,000 characters. |

Long text is split into bounded sections. Reimport a changed file to use its newer text. Original files are never edited. PDF and DOCX import are not implemented. Network shares and mapped network drives are rejected; a local folder may still be synchronized by another application.

## Open a review draft

**Review question** passes the question, mode/topic/role/goal, and selected excerpts to the private assistant. It excludes unselected material, practice history, notes, and the typed answer. **Review my answer** deliberately includes the typed answer in a feedback draft.

The combined review draft must fit 1,200 characters. Oversized drafts are refused without truncation; local answers can be up to 8,000 characters. Opening a draft does not connect or send. The assistant offers exact payload review and local copy/save; the experimental cloud bridge has separate connection and approval controls. Read [privacy and optional cloud review](PRIVACY.md).

Switching workflows clears unsent assistant context and starts a fresh session view. Switching is refused while capture or a private answer is active; stop or cancel it first.

## Save and recover

**Save preparation locally**, a mode switch, or a normal close saves `data/companion.json` atomically. Edits are not continuously autosaved. Each workflow supports 50 saved practice entries and 12,000 characters of local notes.

Notes export contains user-authored fields, material provenance, and saved practice answers. Appending notes twice creates two snapshots. Imported text remains in preparation storage after its original file is removed; remove the material and save to remove that copied text. Earlier exports can retain it.

A damaged preparation file is reported and not overwritten. Export new notes before discarding unsaved changes, then recover the original from a backup. Meeting transcripts are stored separately in `data/meetings.sqlite3`.
