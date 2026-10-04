"""Optional private, review-first answer panel. No transport starts on import."""
from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk

from .assistant_core import ApprovalGate, QuestionDetector, build_preview, load_selected_files
from .storage import timestamp

BG, PAPER, INK, MUTED, ACCENT = "#F3F5FA", "#FFFFFF", "#182238", "#677187", "#6250D9"


def fixed_policy_sections() -> tuple[tuple[str, str], ...]:
    """Read the bridge's exact fixed text without constructing a connection."""
    from .codex_bridge import BASE_INSTRUCTIONS, DEVELOPER_INSTRUCTIONS
    return (("BASE_INSTRUCTIONS", BASE_INSTRUCTIONS), ("DEVELOPER_INSTRUCTIONS", DEVELOPER_INSTRUCTIONS))


def timing_intervals_ms(timing: dict) -> dict:
    """All timestamps use perf_counter; user review time is reported separately."""
    result = {}
    for label, start, end in (("speech_to_candidate_ms", "speech_end_at", "candidate_at"),
                              ("candidate_to_approval_ms", "candidate_at", "sent_at"),
                              ("approval_to_first_token_ms", "sent_at", "first_token_at"),
                              ("approval_to_completion_ms", "sent_at", "completed_at"),
                              ("speech_to_first_token_ms", "speech_end_at", "first_token_at")):
        values = (timing.get(start), timing.get(end))
        result[label] = ((values[1] - values[0]) * 1000 if all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in values) else None)
    return result


class AssistantWindow:
    """Main-thread Tk facade; injected transport owns any optional cloud work.

    ``feed_segment`` must receive stable events on Tk's thread. ``stop_session``
    cancels outstanding work and invalidates every unsent preview. Models come
    only from the transport's explicit, read-only discovery operation.
    """

    def __init__(self, root, project_dir: Path, store, get_session_id, *, transport=None, models=()):
        self.root, self.project_dir, self.store = root, Path(project_dir), store
        if transport is None:
            from .codex_transport import CodexTransport
            transport = CodexTransport(self.project_dir)
        self.get_session_id, self.transport = get_session_id, transport
        self.detector, self.gate = QuestionDetector(), ApprovalGate()
        self.current_session_id = None
        self.candidates, self.candidate_timing = {}, {}
        self.files, self.allowed_directory = (), None
        self.events = queue.Queue()
        self.models = list(models)
        self.discovery_generation = 0
        self.generation = 0
        self.handle = None
        self.inflight = False
        self.closed = False
        self.selected_candidate = None
        self.preview = None
        self.preview_manual = True
        self.preview_fixed_policies = None
        self.preview_dialog = None
        self.history = []
        self.current_request = None
        self.window = tk.Toplevel(root)
        self.window.title("Megumin Meet · Private assistant · Review before sending")
        self.window.geometry("1150x850")
        self.window.minsize(1040, 780)
        self.window.configure(bg=BG)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self._set_models(self.models)
        self._toggle_enabled()
        self.window.after(80, self._poll)

    def _build(self):
        style = ttk.Style(self.window)
        style.configure("Assistant.TFrame", background=BG)
        style.configure("AssistantCard.TFrame", background=PAPER)
        style.configure("Assistant.TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.configure("AssistantMuted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("AssistantCard.TLabel", background=PAPER, foreground=INK, font=("Segoe UI", 10))
        style.configure("Assistant.TButton", padding=(9, 7), font=("Segoe UI", 10))
        style.configure("AssistantAccent.TButton", padding=(11, 8), background=ACCENT, foreground="white")
        style.map("AssistantAccent.TButton", background=[("active", "#5141BD"), ("disabled", "#BBB5DF")])
        style.configure("Assistant.Treeview", rowheight=36, font=("Segoe UI", 10), background=PAPER, fieldbackground=PAPER)
        frame = ttk.Frame(self.window, style="Assistant.TFrame", padding=20)
        frame.pack(fill="both", expand=True)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(3, weight=1)
        head = ttk.Frame(frame, style="Assistant.TFrame")
        head.grid(row=0, column=0, sticky="ew")
        ttk.Label(head, text="Private assistant", style="Assistant.TLabel", font=("Segoe UI", 23, "bold")).pack(side="left")
        ttk.Button(head, text="Export local history…", style="Assistant.TButton", command=self._export_history).pack(side="right")
        ttk.Button(head, text="Clear pending / context", style="Assistant.TButton", command=self.stop_session).pack(side="right", padx=8)
        self.mode_var = tk.StringVar(value="SYNTHETIC QA · mock answers · no network" if getattr(self.transport, "synthetic", False)
                                     else "Codex not connected — ASR stays offline")
        local_controls = ttk.Frame(frame, style="Assistant.TFrame")
        local_controls.grid(row=1, column=0, sticky="ew", pady=(5, 9))
        ttk.Label(local_controls, textvariable=self.mode_var, style="AssistantMuted.TLabel").pack(side="left")
        self.detect_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(local_controls, text="Detect question candidates locally", variable=self.detect_var,
                        command=self._toggle_detection).pack(side="right")
        controls = ttk.Frame(frame, style="Assistant.TFrame")
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 13))
        self.enable_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Enable optional Codex answers", variable=self.enable_var,
                        command=self._toggle_enabled).pack(side="left")
        self.discover_button = ttk.Button(controls, text="Connect / discover", style="Assistant.TButton", command=self._discover_models)
        self.discover_button.pack(side="left", padx=(10, 12))
        self.model_var, self.effort_var = tk.StringVar(), tk.StringVar()
        self.model_combo = ttk.Combobox(controls, textvariable=self.model_var, state="disabled", width=31)
        self.model_combo.pack(side="left")
        self.model_combo.bind("<<ComboboxSelected>>", self._model_changed)
        ttk.Label(controls, text="Effort", style="Assistant.TLabel").pack(side="left", padx=(12, 5))
        self.effort_combo = ttk.Combobox(controls, textvariable=self.effort_var, state="disabled", width=9)
        self.effort_combo.pack(side="left")
        self.effort_var.trace_add("write", lambda *_: self._invalidate_preview())

        panes = ttk.Panedwindow(frame, orient="horizontal")
        panes.grid(row=3, column=0, sticky="nsew")
        left = ttk.Frame(panes, style="AssistantCard.TFrame", padding=13)
        right = ttk.Frame(panes, style="AssistantCard.TFrame", padding=13)
        panes.add(left, weight=1)
        panes.add(right, weight=3)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)
        ttk.Label(left, text="Question candidates", style="AssistantCard.TLabel", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")
        self.count_var = tk.StringVar(value="0 to review · no automatic requests")
        ttk.Label(left, textvariable=self.count_var, style="AssistantCard.TLabel", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=(6, 10))
        self.question_tree = ttk.Treeview(left, columns=("time", "question"), show="headings", selectmode="browse", style="Assistant.Treeview")
        self.question_tree.heading("time", text="TIME")
        self.question_tree.heading("question", text="QUESTION EXCERPT")
        self.question_tree.column("time", width=68, minwidth=60, stretch=False)
        self.question_tree.column("question", width=220, minwidth=170)
        self.question_tree.grid(row=2, column=0, sticky="nsew")
        self.question_tree.bind("<<TreeviewSelect>>", self._select_candidate)
        buttons = ttk.Frame(left, style="AssistantCard.TFrame")
        buttons.grid(row=3, column=0, sticky="ew", pady=(10, 8))
        ttk.Button(buttons, text="Manual ask", style="Assistant.TButton", command=self._manual_ask).pack(side="left")
        ttk.Button(buttons, text="Dismiss", style="Assistant.TButton", command=self._dismiss).pack(side="right")
        ttk.Label(left, text="Detection uses stable transcript only.\nSpeaker labels and who a question is\naddressed to remain uncertain.",
                  style="AssistantCard.TLabel", font=("Segoe UI", 9)).grid(row=4, column=0, sticky="w")

        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(4, weight=1)
        ttk.Label(right, text="Review the question", style="AssistantCard.TLabel", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")
        self.question_text = tk.Text(right, height=3, wrap="word", font=("Segoe UI", 11), relief="flat", bg="#F5F6FC", padx=9, pady=7)
        self.question_text.grid(row=1, column=0, sticky="ew", pady=(9, 7))
        self.question_text.bind("<<Modified>>", self._question_edited)
        self.source_var = tk.StringVar(value="Select a candidate or enter a manual question. No identity is inferred.")
        ttk.Label(right, textvariable=self.source_var, style="AssistantCard.TLabel", font=("Segoe UI", 9), wraplength=670).grid(row=2, column=0, sticky="w", pady=(0, 10))
        review_controls = ttk.Frame(right, style="AssistantCard.TFrame")
        review_controls.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.review_button = ttk.Button(review_controls, text="Review exact payload…", style="AssistantAccent.TButton", command=self._open_preview)
        self.review_button.pack(side="left")
        self.cancel_button = ttk.Button(review_controls, text="Cancel answer", style="Assistant.TButton", command=self._cancel_answer, state="disabled")
        self.cancel_button.pack(side="right")
        notebook = ttk.Notebook(right)
        notebook.grid(row=4, column=0, sticky="nsew")
        answer_tab = ttk.Frame(notebook, style="AssistantCard.TFrame", padding=10)
        context_tab = ttk.Frame(notebook, style="AssistantCard.TFrame", padding=10)
        notebook.add(answer_tab, text="Private draft answer")
        notebook.add(context_tab, text="Request context & selected files")
        answer_tab.grid_columnconfigure(0, weight=1)
        answer_tab.grid_rowconfigure(2, weight=1)
        self.answer_question_var = tk.StringVar(value="An answer appears only after you approve one request.")
        ttk.Label(answer_tab, textvariable=self.answer_question_var, style="AssistantCard.TLabel", wraplength=630).grid(row=0, column=0, sticky="w")
        self.source_ids_var = tk.StringVar(value="Source IDs appear with each approved draft.")
        ttk.Label(answer_tab, textvariable=self.source_ids_var, style="AssistantCard.TLabel", font=("Segoe UI", 9), wraplength=630).grid(row=1, column=0, sticky="w", pady=(7, 8))
        self.answer_text = tk.Text(answer_tab, wrap="word", font=("Segoe UI", 11), relief="flat", bg="#FAFAFD", padx=10, pady=10, state="disabled")
        self.answer_text.grid(row=2, column=0, sticky="nsew")
        ttk.Label(answer_tab, text="Draft for your review. It is never posted to Zoom, spoken aloud, or sent to another person.",
                  style="AssistantCard.TLabel", font=("Segoe UI", 9), wraplength=630).grid(row=3, column=0, sticky="w", pady=(8, 0))
        context_tab.grid_columnconfigure(1, weight=1)
        self.context_vars = {}
        for row, (key, label) in enumerate((("meeting_goal", "Meeting goal"), ("my_role", "My role"), ("constraints", "Constraints / facts"))):
            ttk.Label(context_tab, text=label, style="AssistantCard.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
            var = tk.StringVar()
            self.context_vars[key] = var
            ttk.Entry(context_tab, textvariable=var).grid(row=row, column=1, sticky="ew", pady=5)
            var.trace_add("write", lambda *_: self._invalidate_preview())
        self.nearby_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(context_tab, text="Include nearby preceding transcript (up to 6 excerpts)", variable=self.nearby_var,
                        command=self._invalidate_preview).grid(row=3, column=0, columnspan=2, sticky="w", pady=(7, 9))
        file_buttons = ttk.Frame(context_tab, style="AssistantCard.TFrame")
        file_buttons.grid(row=4, column=0, columnspan=2, sticky="ew")
        ttk.Button(file_buttons, text="Restrict folder…", style="Assistant.TButton", command=self._choose_folder).pack(side="left")
        ttk.Button(file_buttons, text="Select files…", style="Assistant.TButton", command=self._choose_files).pack(side="left", padx=7)
        ttk.Button(file_buttons, text="Clear files", style="Assistant.TButton", command=self._clear_files).pack(side="left")
        self.folder_var = tk.StringVar(value="No folder restriction. Files are read only when explicitly selected.")
        ttk.Label(context_tab, textvariable=self.folder_var, style="AssistantCard.TLabel", font=("Segoe UI", 9), wraplength=620).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 8))
        self.files_var = tk.StringVar(value="No file context selected.")
        ttk.Label(context_tab, textvariable=self.files_var, style="AssistantCard.TLabel", wraplength=620, justify="left").grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Label(context_tab, text="Only chosen .txt, .md or .json files are read. The folder is never scanned.\nThe exact redacted content appears in the approval preview.",
                  style="AssistantCard.TLabel", font=("Segoe UI", 9), wraplength=620).grid(row=7, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.latency_var = tk.StringVar(value="Timing appears after an approved request. Review time is separated from model time.")
        ttk.Label(right, textvariable=self.latency_var, style="AssistantCard.TLabel", font=("Segoe UI", 9), wraplength=670).grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.status_var = tk.StringVar(value="Nothing has been sent. Enable optional answers, then explicitly discover available models.")
        ttk.Label(frame, textvariable=self.status_var, style="AssistantMuted.TLabel", wraplength=1080).grid(row=4, column=0, sticky="w", pady=(12, 0))

    def _status(self, text):
        self.status_var.set(str(text))

    def _live_available(self):
        return bool(self.transport and getattr(self.transport, "available", False))

    def _update_connection_mode(self, *, connecting=False, error=None):
        if getattr(self.transport, "synthetic", False):
            text = "SYNTHETIC QA · mock answers · no network"
        elif connecting:
            text = "Checking the Codex connection — ASR stays offline"
        elif error:
            text = "Codex connection failed — ASR stays offline"
        elif self._live_available():
            text = "Connected · separate Codex session · approved text goes to the cloud"
        else:
            text = "Codex not connected — ASR stays offline"
        self.mode_var.set(text)

    def _toggle_detection(self):
        if not self.detect_var.get():
            self.detector.clear()
            self._refresh_candidates()
        self._status("Local question detection enabled. Nothing is sent automatically." if self.detect_var.get()
                     else "Local question detection disabled. Manual questions and local review remain available.")

    def show(self):
        if not self.closed:
            self.window.deiconify()
            self.window.lift()
            self.window.focus_set()

    def _toggle_enabled(self):
        enabled = self.enable_var.get()
        self.discover_button.configure(state="normal" if enabled and self.transport else "disabled")
        self.model_combo.configure(state="readonly" if enabled and self.models else "disabled")
        self.effort_combo.configure(state="readonly" if enabled and self.models else "disabled")
        self.review_button.configure(state="normal" if not self.inflight else "disabled")
        if not enabled:
            self.discovery_generation += 1
            self.stop_session()
            self._set_models([])
            disconnect = getattr(self.transport, "disconnect", None)
            disconnect_error = None
            if callable(disconnect):
                try:
                    disconnect()
                except Exception as exc:
                    disconnect_error = str(exc)
            self._update_connection_mode(error=disconnect_error)
            self._status(f"Disconnect reported: {disconnect_error}. Pending context and approvals were cleared; no retry is scheduled."
                         if disconnect_error else "Optional answers disconnected. Pending context and approvals were cleared. Local review remains available.")
        else:
            self._update_connection_mode()
            self._status("Optional answers enabled locally. Use Connect / discover explicitly; it does not submit meeting text. "
                         "Each question still requires a reviewed payload and one-use Send approval.")

    @staticmethod
    def _model_label(model):
        return f"{model.get('label') or model['id']} [{model['id']}]"

    def _set_models(self, models):
        self.models = [model for model in models if isinstance(model, dict) and model.get("id") and isinstance(model.get("efforts"), list)]
        self.model_combo.configure(values=[self._model_label(model) for model in self.models])
        self.model_var.set("")
        self.effort_var.set("")
        self.effort_combo.configure(values=[])
        self.model_combo.configure(state="readonly" if self.enable_var.get() and self.models else "disabled")
        self.effort_combo.configure(state="disabled")

    def _discover_models(self):
        if not self.enable_var.get() or not self.transport:
            return
        if self.inflight:
            self._status("Finish or cancel the approved answer before checking a new connection. No request is queued.")
            return
        self.discovery_generation += 1
        generation = self.discovery_generation
        self._invalidate_preview()
        self.discover_button.configure(state="disabled")
        self._update_connection_mode(connecting=True)
        self._status("Checking the separate Codex session and reading supported models. No meeting text is included.")

        def discover():
            try:
                models = self.transport.discover_models()
                self.events.put(("discovery", generation, {"models": models}))
            except Exception as exc:
                self.events.put(("discovery", generation, {"error": str(exc)}))
        threading.Thread(target=discover, daemon=True, name="assistant-model-discovery").start()

    def _discovery_result(self, event):
        self.discover_button.configure(state="normal")
        error = event.get("error")
        if not error and not self._live_available():
            error = getattr(self.transport, "blocked_reason", "The transport did not confirm a usable connection.")
        if error:
            self._set_models([])
            self._update_connection_mode(error=error)
            self._status(f"Codex connection failed: {error}. No meeting text was sent; no retry is scheduled. Manual prompt review remains available.")
            return
        self._set_models(event["models"])
        self._update_connection_mode()
        self._status(f"Connected to a separate Codex session; {len(self.models)} supported models discovered. "
                     "Select one explicitly. Only a separately approved payload will be sent to the cloud.")

    def _model_changed(self, _event=None):
        model = self._selected_model()
        efforts = model.get("efforts", []) if model else []
        self.effort_combo.configure(values=efforts, state="readonly" if efforts else "disabled")
        self.effort_var.set("low" if "low" in efforts else efforts[0] if efforts else "")
        self._invalidate_preview()

    def _selected_model(self):
        return next((model for model in self.models if self._model_label(model) == self.model_var.get()), None)

    def _ensure_session(self):
        session_id = self.get_session_id()
        if session_id != self.current_session_id:
            self.stop_session()
            self.current_session_id = session_id
        return session_id

    def feed_segment(self, event):
        if self.closed or not self.detect_var.get() or event.get("type") != "segment":
            return
        session_id = self._ensure_session()
        if not session_id or event.get("session_id", session_id) != session_id:
            return
        candidate = self.detector.observe(event, segment_id=event.get("segment_id", event.get("id")))
        if candidate:
            self.candidates[candidate.id] = candidate
            speech_end = event.get("assistant_speech_end_perf")
            self.candidate_timing[candidate.id] = {"speech_end_at": speech_end, "candidate_at": candidate.detected_at,
                                                    "speech_end_clock_source": "explicit_perf_counter" if speech_end is not None else "unavailable",
                                                    "detector_processing_ms": candidate.detection_ms}
            self._refresh_candidates()
            if not self.selected_candidate:
                self.source_var.set("Select a question candidate to review its source, speaker label, and timestamps.")
            if not self.inflight:
                self._status("New local question candidate available. Review it explicitly; nothing has been sent.")

    def _refresh_candidates(self):
        selected = self.question_tree.selection()
        self.question_tree.delete(*self.question_tree.get_children())
        pending = self.detector.pending()
        pending_ids = {candidate.id for candidate in pending}
        self.candidates = {key: value for key, value in self.candidates.items() if key in pending_ids}
        self.candidate_timing = {key: value for key, value in self.candidate_timing.items() if key in pending_ids}
        if self.selected_candidate and self.selected_candidate.id not in pending_ids:
            self.selected_candidate = None
            self._invalidate_preview()
            self.question_text.delete("1.0", "end")
            self.source_var.set("Choose a pending candidate or add a manual question.")
        for candidate in pending:
            self.question_tree.insert("", "end", iid=candidate.id,
                                      values=(timestamp(candidate.start).split(".")[0], candidate.text.replace("\n", " ")))
        self.count_var.set(f"{len(pending)} to review · no automatic requests")
        if selected and self.question_tree.exists(selected[0]):
            self.question_tree.selection_set(selected[0])

    def _select_candidate(self, _event=None):
        selected = self.question_tree.selection()
        if not selected or selected[0] not in self.candidates:
            return
        if self.selected_candidate and selected[0] == self.selected_candidate.id:
            return  # Queue refreshes must not overwrite edits or invalidate an open preview.
        self._invalidate_preview()
        self.selected_candidate = self.candidates[selected[0]]
        candidate = self.selected_candidate
        self.question_text.delete("1.0", "end")
        self.question_text.insert("1.0", candidate.text)
        self.question_text.edit_modified(False)
        self.source_var.set(f"Source: {candidate.source} · Speaker: {candidate.speaker} (unverified) · "
                            f"{timestamp(candidate.start).split('.')[0]}–{timestamp(candidate.end).split('.')[0]}\n"
                            "Addressing is uncertain. This is an excerpt, not evidence that the question is directed at you.")

    def _question_edited(self, _event=None):
        if self.question_text.edit_modified():
            self.question_text.edit_modified(False)
            self._invalidate_preview()

    def _manual_ask(self, text=None):
        if not self._ensure_session():
            self._status("Choose a meeting session before adding a manual question.")
            return
        if text is None:
            text = simpledialog.askstring("Manual question", "What would you like to ask privately?", parent=self.window)
        if not text:
            return
        try:
            recent = self.store.segments(self.current_session_id, limit=1, newest=True)
            candidate = self.detector.manual(text, start=recent[0]["end"] if recent else 0)
        except Exception as exc:
            self._status(str(exc))
            return
        self.candidates[candidate.id] = candidate
        self.candidate_timing[candidate.id] = {"speech_end_at": None, "candidate_at": candidate.detected_at}
        self._refresh_candidates()
        self.question_tree.selection_set(candidate.id)
        self._select_candidate()

    def _dismiss(self):
        if self.selected_candidate:
            try:
                self.detector.set_status(self.selected_candidate.id, "dismissed")
            except KeyError:
                pass
            self.selected_candidate = None
            self.question_text.delete("1.0", "end")
            self._invalidate_preview()
            self._refresh_candidates()

    def _choose_folder(self):
        path = filedialog.askdirectory(parent=self.window, title="Restrict future explicit file selections to this folder")
        if path:
            try:
                load_selected_files((), allowed_directory=path)  # Validate the folder without reading any file text.
            except Exception as exc:
                self._status(f"Folder restriction was not accepted: {exc}")
                return
            self.allowed_directory = Path(path)  # Core validates locality before any file read; never resolve UNC paths here.
            self._clear_files()
            self.folder_var.set(f"Allowed folder: {self.allowed_directory} · no directory scan")

    def _choose_files(self):
        paths = filedialog.askopenfilenames(parent=self.window, title="Select only the files you want in the reviewed request",
                                           initialdir=str(self.allowed_directory or self.project_dir),
                                           filetypes=[("Text, Markdown and JSON", "*.txt *.md *.json")])
        if not paths:
            return
        try:
            self.files = load_selected_files(paths, allowed_directory=self.allowed_directory)
            self._refresh_files()
            self._invalidate_preview()
        except Exception as exc:
            self._status(f"File context was not loaded: {exc}")

    def _refresh_files(self):
        self.files_var.set("\n".join(f"{source.id} · {source.title} · {len(source.text):,} characters" for source in self.files)
                           or "No file context selected.")

    def _clear_files(self):
        self.files = ()
        self._refresh_files()
        self._invalidate_preview()

    def _invalidate_preview(self):
        self.gate.invalidate()
        self.preview = None
        self.preview_fixed_policies = None
        if self.preview_dialog and self.preview_dialog.winfo_exists():
            self.preview_dialog.destroy()
        self.preview_dialog = None

    def _open_preview(self):
        session_id = self._ensure_session()
        if self.inflight:
            self._status("One answer is in progress. Cancel or finish it before approving another request. No request is queued.")
            return
        model = self._selected_model()
        manual = not self._live_available() or not self.enable_var.get()
        if not manual and (not model or self.effort_var.get() not in model.get("efforts", [])):
            self._status("Explicitly discover and select a supported model and effort first.")
            return
        if not session_id or not self.selected_candidate:
            self._status("Select a question candidate or add a manual question first.")
            return
        question = self.question_text.get("1.0", "end-1c").strip()
        try:
            segments = self.store.preceding_segments(session_id, before=self.selected_candidate.start, seconds=60.0, limit=6) if self.nearby_var.get() else ()
            context = json.dumps({key: var.get().strip() for key, var in self.context_vars.items() if var.get().strip()}, ensure_ascii=False)
            self._invalidate_preview()
            self.preview_manual = manual
            self.preview = build_preview(self.selected_candidate, question=question, segments=segments, files=self.files,
                                         user_context=context if context != "{}" else "",
                                         model="Not selected — manual Codex session" if manual else model["id"],
                                         effort="Not selected" if manual else self.effort_var.get(),
                                         destination="Manual transfer to Codex (cloud) — no automatic connection" if manual else self.transport.destination)
            self.preview_fixed_policies = fixed_policy_sections()
        except Exception as exc:
            self._status(f"Preview could not be built: {exc}")
            return
        preview = self.preview
        dialog = tk.Toplevel(self.window)
        self.preview_dialog = dialog
        dialog.title("Review the exact outgoing payload · one-use approval")
        dialog.geometry("1040x790")
        dialog.transient(self.window)
        dialog.grab_set()
        frame = ttk.Frame(dialog, style="Assistant.TFrame", padding=16)
        frame.pack(fill="both", expand=True)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(3, weight=1)
        ttk.Label(frame, text="Approve one exact request", style="Assistant.TLabel", font=("Segoe UI", 19, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text=f"Destination: {preview.destination}\nModel: {preview.model} · Effort: {preview.effort} · Payload digest: {preview.digest[:16]}",
                  style="Assistant.TLabel", wraplength=990).grid(row=1, column=0, sticky="w", pady=(7, 10))
        summary = "Redactions: " + (", ".join(preview.redactions) or "none detected")
        summary += "\nSensitive flags: " + (", ".join(preview.sensitive_flags) or "none detected; automated detection is incomplete")
        if manual:
            summary += "\nLIVE SEND UNAVAILABLE · Save or copy locally, then deliberately paste into a separate Codex session. Manual transfer may send content to the cloud."
        summary += "\nFixed policies are shown separately from meeting content. Copy/save exports only the meeting payload; manual-session instructions are managed in that session."
        ttk.Label(frame, text=summary, style="AssistantMuted.TLabel", wraplength=990).grid(row=2, column=0, sticky="w", pady=(0, 9))
        self.preview_notebook = ttk.Notebook(frame)
        self.preview_notebook.grid(row=3, column=0, sticky="nsew")
        self.preview_policy_widgets = {}
        sections = [("MEETING_PAYLOAD", "Meeting payload · reviewed content", preview.payload_text),
                    *[(name, "Fixed base policy" if name == "BASE_INSTRUCTIONS" else "Fixed developer policy", text)
                      for name, text in self.preview_fixed_policies]]
        for name, title, text in sections:
            page = ttk.Frame(self.preview_notebook, style="Assistant.TFrame")
            page.grid_columnconfigure(0, weight=1)
            page.grid_rowconfigure(1, weight=1)
            note = ("Exact approved meeting question and selected source content." if name == "MEETING_PAYLOAD"
                    else f"{name} · fixed application policy supplied separately by the Codex bridge; this is not meeting evidence.")
            ttk.Label(page, text=note, style="AssistantMuted.TLabel", wraplength=960).grid(row=0, column=0, columnspan=2, sticky="w", pady=7)
            content = tk.Text(page, wrap="word", font=("Consolas", 10), bg=PAPER, fg=INK, padx=10, pady=10)
            content.insert("1.0", text)
            content.configure(state="disabled")
            content.grid(row=1, column=0, sticky="nsew")
            scrollbar = ttk.Scrollbar(page, orient="vertical", command=content.yview)
            scrollbar.grid(row=1, column=1, sticky="ns")
            content.configure(yscrollcommand=scrollbar.set)
            self.preview_notebook.add(page, text=title)
            self.preview_policy_widgets[name] = content
        ttk.Label(frame, text=" · ".join(preview.limitations), style="AssistantMuted.TLabel", wraplength=990).grid(row=4, column=0, sticky="w", pady=(9, 7))
        self.preview_reviewed_var = tk.BooleanVar(value=False)
        self.preview_sensitive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="I reviewed the exact meeting payload, both fixed policies, model and destination.", variable=self.preview_reviewed_var).grid(row=5, column=0, sticky="w")
        if preview.sensitive_flags:
            ttk.Checkbutton(frame, text="I explicitly approve sending the flagged sensitive content in this preview.",
                            variable=self.preview_sensitive_var).grid(row=6, column=0, sticky="w", pady=(4, 0))
        buttons = ttk.Frame(frame, style="Assistant.TFrame")
        buttons.grid(row=7, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(buttons, text="Edit inputs / discard preview", style="Assistant.TButton", command=self._invalidate_preview).pack(side="left")
        ttk.Button(buttons, text="Send once", style="AssistantAccent.TButton", command=self._send_preview,
                   state="disabled" if manual else "normal").pack(side="right")
        ttk.Button(buttons, text="Save reviewed prompt…", style="Assistant.TButton", command=self._save_reviewed_prompt).pack(side="right", padx=7)
        ttk.Button(buttons, text="Copy reviewed prompt", style="Assistant.TButton", command=self._copy_reviewed_prompt).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", self._invalidate_preview)

    def _approve_local_preview(self):
        if not self.preview or not self.preview_reviewed_var.get():
            self._status("Review the exact payload, model and destination before copying or saving it.")
            return None
        if self.get_session_id() != self.current_session_id:
            self.stop_session()
            return None
        if self.preview_fixed_policies != fixed_policy_sections():
            self._invalidate_preview()
            self._status("The fixed policies changed. Open and review a new preview; nothing was sent or copied.")
            return None
        try:
            receipt = self.gate.approve(self.preview, reviewed_digest=self.preview.digest,
                                        acknowledge_sensitive=self.preview_sensitive_var.get())
            return self.gate.consume(self.preview, receipt)
        except Exception as exc:
            self._status(f"Local approval rejected: {exc}")
            return None

    def _record_local_transfer(self, preview, action):
        self.history.append({"session_id": self.current_session_id, "candidate_id": preview.candidate_id,
                             "status": action, "model": preview.model, "effort": preview.effort,
                             "destination": preview.destination, "payload_digest": preview.digest,
                             "payload_text": preview.payload_text, "source_ids": [source.id for source in preview.sources],
                             "fixed_policies": dict(self.preview_fixed_policies or ()),
                             "cloud_request_sent": False, "answer": ""})
        self._invalidate_preview()
        self._status("Reviewed prompt copied/saved locally. No automatic forwarding occurred. Manual paste into Codex may send it to the cloud.")

    def _copy_reviewed_prompt(self):
        preview = self._approve_local_preview()
        if preview:
            self.window.clipboard_clear()
            self.window.clipboard_append(preview.payload_text)
            self._record_local_transfer(preview, "copied_for_manual_transfer")

    def _save_reviewed_prompt(self, path=None):
        if not self.preview or not self.preview_reviewed_var.get():
            self._status("Review the exact payload, model and destination before saving it locally.")
            return
        if path is None:
            path = filedialog.asksaveasfilename(parent=self.preview_dialog, title="Save reviewed prompt for deliberate manual transfer",
                                              initialfile="reviewed-codex-prompt.txt", defaultextension=".txt",
                                              filetypes=[("Text", "*.txt")])
        if not path:
            return
        preview = self._approve_local_preview()
        if preview:
            try:
                Path(path).write_text(preview.payload_text, encoding="utf-8")
                self._record_local_transfer(preview, "saved_for_manual_transfer")
            except Exception as exc:
                self._status(f"Local save failed: {exc}. Nothing was sent.")

    def _send_preview(self):
        if not self._live_available() or self.preview_manual:
            self._status(getattr(self.transport, "blocked_reason", "Live Codex unavailable. Only local review and manual prompt export are supported."))
            return
        preview = self.preview
        if not preview or self.inflight or not self.enable_var.get() or not self.preview_reviewed_var.get():
            self._status("A reviewed, unused preview is required. Nothing was sent.")
            return
        if self.get_session_id() != self.current_session_id:
            self.stop_session()
            return
        if self.preview_fixed_policies != fixed_policy_sections():
            self._invalidate_preview()
            self._status("The fixed policies changed. Open and review a new preview; nothing was sent.")
            return
        if preview.candidate_id not in {candidate.id for candidate in self.detector.pending()}:
            self._invalidate_preview()
            self._status("This candidate is no longer pending. Select or enter a question and review it again.")
            return
        try:
            receipt = self.gate.approve(preview, reviewed_digest=preview.digest,
                                        acknowledge_sensitive=self.preview_sensitive_var.get())
            approved = self.gate.consume(preview, receipt)
        except Exception as exc:
            self._status(f"Approval rejected: {exc}. Nothing was sent.")
            return
        self.preview = None
        if self.preview_dialog:
            self.preview_dialog.destroy()
            self.preview_dialog = None
        self.inflight = True
        self.generation += 1
        generation = self.generation
        self.review_button.configure(state="disabled")
        self.discover_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self._set_answer("")
        self.answer_question_var.set(f"PRIVATE DRAFT · {approved.question}")
        self.source_ids_var.set("Reviewed source IDs: " + ", ".join(source.id for source in approved.sources))
        timing = dict(self.candidate_timing.get(approved.candidate_id, {}))
        timing["sent_at"] = time.perf_counter()
        request = {"session_id": self.current_session_id, "candidate_id": approved.candidate_id,
                   "model": approved.model, "effort": approved.effort, "destination": approved.destination,
                   "payload_digest": approved.digest, "payload_text": approved.payload_text,
                   "fixed_policies": dict(self.preview_fixed_policies or ()),
                   "source_ids": [source.id for source in approved.sources], "timing": timing,
                   "status": "sending", "answer": "", "synthetic": bool(getattr(self.transport, "synthetic", False))}
        self.current_request = request
        self.history.append(request)
        self._status("Sending this approved request only. No other candidate is queued.")
        try:
            self.detector.set_status(approved.candidate_id, "sent")
            self._refresh_candidates()
            self.handle = self.transport.start(approved, lambda event: self.events.put(("answer", generation, dict(event))))
        except Exception as exc:
            self._answer_event({"type": "error", "message": str(exc)})

    def _set_answer(self, text, *, append=False):
        self.answer_text.configure(state="normal")
        if not append:
            self.answer_text.delete("1.0", "end")
        self.answer_text.insert("end", text)
        self.answer_text.see("end")
        self.answer_text.configure(state="disabled")

    def _answer_event(self, event):
        if not self.current_request:
            return
        request, kind = self.current_request, event.get("type")
        if kind == "token":
            request["timing"].setdefault("first_token_at", time.perf_counter())
            request["answer"] += str(event.get("text", ""))
            self._set_answer(str(event.get("text", "")), append=True)
            self._status("Receiving a private draft. Verify it against the reviewed source IDs.")
        elif kind == "status":
            self._status(event.get("message", "Working on the approved request…"))
        elif kind in ("complete", "error", "cancelled"):
            request["status"] = kind
            request["timing"]["completed_at"] = time.perf_counter()
            if kind == "complete":
                if "text" in event:
                    request["answer"] = str(event["text"])
                    self._set_answer(request["answer"])
                request["transport_metrics"] = event.get("metrics", {})
                for key in ("first_token_at", "completed_at", "submitted_at"):
                    if isinstance(event.get("metrics", {}).get(key), (int, float)):
                        request["timing"][key] = event["metrics"][key]
                self._status("Private draft complete. Nothing was posted or spoken. A new request needs a new approval.")
            elif kind == "error":
                request["error"] = str(event.get("message", "Request failed"))
                self._status(f"Answer failed: {request['error']}. No retry or resend is scheduled.")
            else:
                self._status("Answer cancelled. No retry or resend is scheduled.")
            self.inflight = False
            self.handle = None
            self.cancel_button.configure(state="disabled")
            self.review_button.configure(state="normal")
            self.discover_button.configure(state="normal" if self.enable_var.get() else "disabled")
            self._update_connection_mode()
        request["intervals_ms"] = timing_intervals_ms(request["timing"])
        values = request["intervals_ms"]
        def seconds(key):
            return f"{values[key] / 1000:.2f}s" if values.get(key) is not None else "—"
        self.latency_var.set(f"Speech → candidate {seconds('speech_to_candidate_ms')} · Review {seconds('candidate_to_approval_ms')} · "
                             f"Send → first token {seconds('approval_to_first_token_ms')} · Complete {seconds('approval_to_completion_ms')}")

    def _cancel_answer(self):
        self.generation += 1
        if self.handle:
            try:
                self.handle.cancel()
            except Exception as exc:
                self._status(f"Cancellation reported: {exc}")
        if self.inflight and self.current_request:
            self.current_request["status"] = "cancelled"
            self.current_request["timing"]["completed_at"] = time.perf_counter()
            self.current_request["intervals_ms"] = timing_intervals_ms(self.current_request["timing"])
        self.handle, self.inflight = None, False
        self.cancel_button.configure(state="disabled")
        self.review_button.configure(state="normal")
        self.discover_button.configure(state="normal" if self.enable_var.get() and self.transport else "disabled")

    def stop_session(self):
        if self.closed:
            return
        self._cancel_answer()
        self._invalidate_preview()
        self.detector.clear()
        self.candidates.clear()
        self.candidate_timing.clear()
        self.selected_candidate = None
        self.current_session_id = None
        self.current_request = None
        self.files = ()
        self.allowed_directory = None
        self.folder_var.set("No folder restriction. Files are read only when explicitly selected.")
        self._refresh_files()
        for var in self.context_vars.values():
            var.set("")
        self.nearby_var.set(False)
        self.question_text.delete("1.0", "end")
        self.source_var.set("No pending question. A stopped or changed session clears unsent context.")
        self._set_answer("")
        self.answer_question_var.set("An answer appears only after you approve one request.")
        self.source_ids_var.set("Source IDs appear with each approved draft.")
        self.latency_var.set("Timing appears after an approved request. Review time is separated from model time.")
        self._refresh_candidates()
        self._status("Session reset. Pending candidates and approvals were cleared; any active answer was cancelled.")

    def export_history(self, path: Path):
        payload = {"schema_version": 1, "private_local_history": True,
                   "notice": "Review-first requests and private drafts; no automatic meeting upload.", "requests": self.history}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _export_history(self):
        path = filedialog.asksaveasfilename(parent=self.window, title="Export private request history locally", defaultextension=".json",
                                          initialfile="private-assistant-history.json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                self.export_history(Path(path))
                self._status(f"Private history exported locally to {path}")
            except Exception as exc:
                self._status(f"History export failed: {exc}")

    def _poll(self):
        if self.closed:
            return
        try:
            for _ in range(150):
                try:
                    lane, generation, event = self.events.get_nowait()
                except queue.Empty:
                    break
                if lane == "discovery" and generation == self.discovery_generation and self.enable_var.get():
                    self._discovery_result(event)
                elif lane == "answer" and generation == self.generation:
                    self._answer_event(event)
        except Exception as exc:
            self._status(f"Assistant display error: {exc}. No automatic retry or resend.")
        finally:
            if not self.closed:
                self.window.after(80, self._poll)

    def close(self):
        if self.closed:
            return
        self.stop_session()
        self.discovery_generation += 1
        self.closed = True
        if self.transport:
            try:
                self.transport.close()
            except Exception:
                pass
        self.window.destroy()
