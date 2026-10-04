"""Native local meeting workspace. Model and capture work never run on Tk's thread."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from .notes import suggest_notes
from .storage import Store, timestamp
from .companion import CompanionRepository, MODES


BG = "#F3F5FA"
PAPER = "#FFFFFF"
INK = "#182238"
MUTED = "#677187"
PURPLE = "#6250D9"
SIDEBAR = "#171E32"
RED = "#D9344E"
PAGE_SIZE = 200


class App:
    def __init__(self, root: tk.Tk, project_dir: Path, demo: bool = False):
        self.root = root
        self.project_dir = Path(project_dir)
        self.data_dir = self.project_dir / "data"
        self.store = Store(self.data_dir / "meetings.sqlite3")
        self.events: queue.Queue = queue.Queue()
        self.selected_id: str | None = None
        self.active_id: str | None = None
        self.engine = None
        self.cancel_start = threading.Event()
        self.worker: threading.Thread | None = None
        self.stop_worker: threading.Thread | None = None
        self.session_incomplete = False
        self.session_error = False
        self.recording_since: float | None = None
        self.offset = 0
        self.devices = []
        self.history_ids = []
        self.notes_save_job = None
        self.search_job = None
        self.closing = False
        self.closed = False
        self.diagnostics = []
        self.last_latency = None
        self.last_caption_delay = None
        self.assistant = None
        self.companion = None
        self.companion_mode = "meeting"
        preparation_error = None
        try:
            self.companion_mode = CompanionRepository(self.data_dir / "companion.json").load()["mode"]
        except (OSError, ValueError) as exc:
            preparation_error = f"Preparation workspace could not be loaded: {exc}. Existing file was preserved."
        self._build()
        self._refresh_history()
        if preparation_error:
            self._notice(preparation_error)
        if self.store.recovered_count:
            self._notice(f"Recovered {self.store.recovered_count} interrupted session(s). Saved transcript is available in history.")
        self.root.after(80, self._poll)
        self.root.after(300, self._discover_devices)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        if demo:
            self.root.after(100, self._demo)

    def _build(self):
        self.root.title("Megumin Companion · Practice, present, meet")
        self.root.geometry("1220x820")
        self.root.minsize(1080, 820)
        self.root.configure(bg=BG)
        self.root.option_add("*Font", "{Segoe UI} 10")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PAPER)
        style.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=PAPER, foreground=INK)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("CardMuted.TLabel", background=PAPER, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("TButton", padding=(12, 8), background="#E9ECF5", foreground=INK,
                        borderwidth=0, font=("Segoe UI", 10))
        style.map("TButton", background=[("active", "#DCE1F1")], foreground=[("disabled", "#949AAB")])
        style.configure("Accent.TButton", background=PURPLE, foreground="white")
        style.map("Accent.TButton", background=[("active", "#5141BD"), ("disabled", "#BBB5DF")],
                  foreground=[("disabled", "white")])
        style.configure("Danger.TButton", background="#FFE8ED", foreground=RED)
        style.configure("Small.TButton", padding=(5, 8), font=("Segoe UI", 9))
        style.configure("TCheckbutton", background=PAPER, foreground=INK, padding=2)
        style.map("TCheckbutton", background=[("active", PAPER)])
        style.configure("TCombobox", padding=5, fieldbackground="white", foreground=INK)
        style.configure("TEntry", padding=7, fieldbackground="white", foreground=INK)
        style.configure("Treeview", background=PAPER, fieldbackground=PAPER, foreground=INK,
                        rowheight=35, borderwidth=0, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#EDF0F7", foreground=MUTED,
                        padding=(7, 7), relief="flat", font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#E8E4FF")], foreground=[("selected", INK)])
        style.configure("TPanedwindow", background=BG)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        sidebar = tk.Frame(self.root, bg=SIDEBAR, width=248)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(5, weight=1)
        brand = tk.Frame(sidebar, bg=SIDEBAR)
        brand.grid(row=0, column=0, padx=22, pady=(27, 25), sticky="ew")
        tk.Label(brand, text="m", bg=PURPLE, fg="white", font=("Segoe UI", 24, "bold"),
                 width=2).pack(side="left")
        tk.Label(brand, text="Megumin\nCompanion", justify="left", bg=SIDEBAR, fg="white",
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=6)
        tk.Label(sidebar, text="YOUR LOCAL WORKSPACE", bg=SIDEBAR, fg="#949CB4",
                 font=("Segoe UI", 9, "bold")).grid(row=1, column=0, padx=22, sticky="w")
        ttk.Button(sidebar, text="+  New session", style="Accent.TButton", command=self._new_view).grid(
            row=2, column=0, padx=20, pady=(15, 10), sticky="ew")
        ttk.Button(sidebar, text="Demo — no capture", style="Small.TButton", command=self._demo).grid(
            row=3, column=0, padx=20, pady=(0, 26), sticky="ew")
        tk.Label(sidebar, text="RECENT SESSIONS", bg=SIDEBAR, fg="#949CB4",
                 font=("Segoe UI", 9, "bold")).grid(row=4, column=0, padx=22, pady=(0, 10), sticky="w")
        self.history = tk.Listbox(sidebar, bg=SIDEBAR, fg="#E2E6F2", selectbackground="#333754",
                                  selectforeground="white", borderwidth=0, highlightthickness=0,
                                  font=("Segoe UI", 10), activestyle="none", exportselection=False)
        self.history.grid(row=5, column=0, padx=13, sticky="nsew")
        self.history.bind("<<ListboxSelect>>", self._select_history)
        sidebar_tools = tk.Frame(sidebar, bg=SIDEBAR)
        sidebar_tools.grid(row=6, column=0, padx=20, pady=(12, 0), sticky="ew")
        tk.Label(sidebar_tools, text="SESSION MODE", bg=SIDEBAR, fg="#949CB4",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 6))
        self.companion_mode_var = tk.StringVar(value=MODES[self.companion_mode])
        self.companion_mode_combo = ttk.Combobox(sidebar_tools, textvariable=self.companion_mode_var,
                                                values=list(MODES.values()), state="readonly", width=23)
        self.companion_mode_combo.pack(fill="x")
        self.companion_mode_combo.bind("<<ComboboxSelected>>", self._choose_companion_mode)
        ttk.Button(sidebar_tools, text="Prepare / practice", style="Accent.TButton",
                   command=self._open_companion).pack(fill="x", pady=8)
        ttk.Button(sidebar_tools, text="Private Q&A review", command=self._open_assistant).pack(fill="x")
        tk.Label(sidebar, text="LOCAL BY DEFAULT\nAudio and notes stay on this device.\nOptional Codex: approve each question.",
                 bg=SIDEBAR, fg="#9DA6C0", justify="left", wraplength=187,
                 font=("Segoe UI", 9), pady=23).grid(row=7, column=0, padx=22, sticky="sw")

        main = ttk.Frame(self.root, padding=(25, 14, 25, 12))
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(3, weight=1)
        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.grid_columnconfigure(0, weight=1)
        self.workspace_var = tk.StringVar(value=MODES[self.companion_mode].upper())
        ttk.Label(header, textvariable=self.workspace_var, foreground=PURPLE,
                  font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.title_var = tk.StringVar(value="Ready when you are.")
        ttk.Label(header, textvariable=self.title_var, font=("Segoe UI", 24, "bold")).grid(
            row=1, column=0, sticky="w", pady=(4, 2))
        self.subtitle_var = tk.StringVar(value="Prepare privately, or select audio sources to capture a session.")
        ttk.Label(header, textvariable=self.subtitle_var, style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        self.indicator = tk.Label(header, text="●  READY", bg="#E6EAF4", fg=MUTED,
                                  padx=11, pady=6, font=("Segoe UI", 9, "bold"))
        self.indicator.grid(row=0, column=1, rowspan=2, sticky="e")
        self.timer_var = tk.StringVar(value="00:00:00")
        ttk.Label(header, textvariable=self.timer_var, foreground=MUTED,
                  font=("Consolas", 12)).grid(row=2, column=1, sticky="e")

        capture = ttk.Frame(main, style="Card.TFrame", padding=10)
        capture.grid(row=1, column=0, sticky="ew")
        for col in (0, 1):
            capture.grid_columnconfigure(col, weight=1)
        ttk.Label(capture, text="MICROPHONE", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(capture, text="SYSTEM AUDIO", style="CardMuted.TLabel").grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(capture, text="LANGUAGE", style="CardMuted.TLabel").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.mic_var = tk.StringVar(value="Off")
        self.loopback_var = tk.StringVar(value="Off")
        self.language_var = tk.StringVar(value="Auto")
        self.mic_combo = ttk.Combobox(capture, textvariable=self.mic_var, values=["Off"], state="readonly", width=24)
        self.mic_combo.grid(row=1, column=0, sticky="ew", pady=(5, 8))
        self.loopback_combo = ttk.Combobox(capture, textvariable=self.loopback_var, values=["Off"], state="readonly", width=24)
        self.loopback_combo.grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=(5, 8))
        self.language_combo = ttk.Combobox(capture, textvariable=self.language_var, values=["Auto", "English", "中文"],
                                           state="readonly", width=9)
        self.language_combo.grid(row=1, column=2, padx=(12, 0), pady=(5, 8))
        ttk.Button(capture, text="↻", width=3, command=self._discover_devices).grid(row=1, column=3, padx=(8, 0), pady=(5, 8))
        ttk.Label(capture, text="System audio captures all apps on the selected output. Headphones recommended.",
                  style="CardMuted.TLabel").grid(row=2, column=0, columnspan=4, sticky="w")
        choices = ttk.Frame(capture, style="Card.TFrame")
        choices.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(7, 0))
        self.consent_var = tk.BooleanVar(value=False)
        self.retain_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(choices, text="Participants have agreed to recording", variable=self.consent_var).pack(side="left")
        ttk.Checkbutton(choices, text="Keep source audio locally", variable=self.retain_var).pack(side="right")

        controls = ttk.Frame(main)
        controls.grid(row=2, column=0, sticky="ew", pady=(10, 11))
        self.start_button = ttk.Button(controls, text="●  Start capture", style="Accent.TButton", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="■  Stop", style="Danger.TButton", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.profile_var = tk.StringVar(value="Hardware-adaptive offline speech · awaiting model setup")
        ttk.Label(controls, textvariable=self.profile_var, style="Muted.TLabel", wraplength=340).pack(side="left", padx=4)
        ttk.Button(controls, text="Export…", command=self._export).pack(side="right")
        ttk.Button(controls, text="Rename", command=self._rename_session).pack(side="right", padx=7)

        panes = ttk.Panedwindow(main, orient="horizontal")
        panes.grid(row=3, column=0, sticky="nsew")
        transcript = ttk.Frame(panes, style="Card.TFrame", padding=12)
        notes = ttk.Frame(panes, style="Card.TFrame", padding=14)
        panes.add(transcript, weight=3)
        panes.add(notes, weight=2)
        transcript.grid_columnconfigure(0, weight=1)
        transcript.grid_rowconfigure(2, weight=1, minsize=140)
        titleline = ttk.Frame(transcript, style="Card.TFrame")
        titleline.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(titleline, text="Transcript", style="Card.TLabel", font=("Segoe UI", 14, "bold")).pack(side="left")
        self.count_var = tk.StringVar(value="0 segments")
        ttk.Label(titleline, textvariable=self.count_var, style="CardMuted.TLabel").pack(side="right")
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(transcript, textvariable=self.search_var)
        self.search_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 6))
        self.search_var.trace_add("write", self._search_changed)
        self.search_entry.insert(0, "")
        columns = ("time", "source", "speaker", "text")
        self.tree = ttk.Treeview(transcript, columns=columns, show="headings", selectmode="browse", height=10)
        for key, text, width in [("time", "TIME", 74), ("source", "SOURCE", 67),
                                 ("speaker", "SPEAKER", 94), ("text", "WORDS  ·  SEARCH ABOVE", 245)]:
            self.tree.heading(key, text=text)
            self.tree.column(key, width=width, minwidth=width if key != "text" else 130,
                             stretch=key == "text", anchor="w")
        self.tree.grid(row=2, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(transcript, orient="vertical", command=self.tree.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.tag_configure("alternate", background="#F7F8FC")
        self.tree.bind("<<TreeviewSelect>>", self._segment_selected)
        self.tree.bind("<Double-1>", self._edit_segment)
        self.tree.bind("<Return>", self._edit_segment)
        self.detail_var = tk.StringVar(value="Select a segment to see its full text and confidence.")
        self.detail = tk.Text(transcript, height=1, wrap="word", bg="#F7F8FC", fg=MUTED,
                              relief="flat", padx=9, pady=6, font=("Segoe UI", 9), state="disabled")
        self.detail.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 5))
        self._detail("Your transcript will appear here. Audio source and speaker identity are shown separately.")
        navigation = ttk.Frame(transcript, style="Card.TFrame")
        navigation.grid(row=4, column=0, columnspan=2, sticky="ew")
        self.older_button = ttk.Button(navigation, text="← Older", command=lambda: self._page(1), padding=5)
        self.older_button.pack(side="left")
        self.newer_button = ttk.Button(navigation, text="Newer →", command=lambda: self._page(-1), padding=5)
        self.newer_button.pack(side="left", padx=4)
        ttk.Button(navigation, text="Edit / speaker", command=self._edit_segment, padding=5).pack(side="right")
        self.page_var = tk.StringVar(value="0 segments")
        ttk.Label(navigation, textvariable=self.page_var, style="CardMuted.TLabel").pack(side="left", padx=8)

        notes.grid_columnconfigure(0, weight=1)
        notes.grid_rowconfigure(3, weight=1)
        ttk.Label(notes, text="Session notes", style="Card.TLabel", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(notes, text="Your words, saved automatically.", style="CardMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(5, 12))
        ttk.Button(notes, text="+ Find candidate quotes", command=self._suggest_notes).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.notes = tk.Text(notes, wrap="word", undo=True, bg="#FAFAFD", fg=INK,
                             insertbackground=PURPLE, relief="flat", padx=13, pady=13,
                             font=("Segoe UI", 11), spacing1=3, spacing3=5, width=29)
        self.notes.grid(row=3, column=0, sticky="nsew")
        self.notes.bind("<<Modified>>", self._notes_changed)
        self.notes_status = tk.StringVar(value="Choose or start a session to write notes.")
        ttk.Label(notes, textvariable=self.notes_status, style="CardMuted.TLabel").grid(row=4, column=0, sticky="w", pady=(10, 3))
        notes_hint = ttk.Label(notes, text="Candidate quotes are extractive. Review them before treating them as decisions or actions.",
                               style="CardMuted.TLabel", wraplength=280, justify="left")
        notes_hint.grid(row=5, column=0, sticky="ew", pady=(2, 0))
        notes.bind("<Configure>", lambda event: notes_hint.configure(wraplength=max(160, event.width - 28)))
        footer = ttk.Frame(main)
        footer.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        footer.grid_columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Ready · All files stored locally")
        self.status_label = ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel", wraplength=730)
        self.status_label.grid(row=0, column=0, sticky="w")
        ttk.Button(footer, text="Diagnostics", command=self._show_diagnostics, padding=5).grid(row=0, column=1, sticky="e")
        self.performance_var = tk.StringVar(value="Unknown speaker = insufficient evidence · Labels are estimates")
        ttk.Label(footer, textvariable=self.performance_var, style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.root.bind("<Control-f>", lambda _: self.search_entry.focus_set())
        self.root.bind("<Control-s>", lambda _: self._save_notes())
        self._notes_enabled(False)

    def _notice(self, message: str):
        self.status_var.set(message)
        self.diagnostics.append(f"{datetime.now():%H:%M:%S}  {message}")
        self.diagnostics = self.diagnostics[-500:]

    def _detail(self, text: str, *, expanded: bool = False):
        self.detail.configure(state="normal", height=3 if expanded else 1)
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", text)
        self.detail.configure(state="disabled")

    def _notes_enabled(self, enabled: bool):
        self.notes.configure(state="normal" if enabled else "disabled")

    def _refresh_history(self):
        sessions = self.store.list_sessions()
        self.history_ids = [s["id"] for s in sessions]
        self.history.delete(0, "end")
        for session in sessions:
            marker = "◈" if session["simulated"] else "●" if session["id"] == self.active_id else "○"
            self.history.insert("end", f" {marker}  {session['title'][:24]}")
            self.history.insert("end", f"      {session['created_at'][5:10]}  ·  {session['segment_count']} segments")
        if self.selected_id in self.history_ids:
            self.history.selection_set(self.history_ids.index(self.selected_id) * 2)

    def _select_history(self, _event=None):
        selected = self.history.curselection()
        if selected and selected[0] // 2 < len(self.history_ids):
            self._load_session(self.history_ids[selected[0] // 2])

    def _load_session(self, session_id: str):
        if session_id != self.selected_id:
            self._reset_assistant()
        self._save_notes()
        self.selected_id = session_id
        self.offset = 0
        self.search_var.set("")
        session = self.store.get_session(session_id)
        self.title_var.set(session["title"])
        self.profile_var.set(session["profile"] or "Hardware-adaptive offline speech · awaiting model setup")
        self.subtitle_var.set(("SIMULATED DEMO · No audio was captured" if session["simulated"] else
                               f"{session['created_at'][:16].replace('T', ' ')} UTC · {session['status'].capitalize()}")
                              + " · Saved on this device")
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", session["notes"])
        self.notes.edit_modified(False)
        self.notes_status.set("Saved locally")
        self._detail("Select a segment for full text and confidence. Double-click to edit words or speaker.")
        if not self.active_id:
            self.indicator.configure(text="◈  SIMULATED" if session["simulated"] else "●  SAVED",
                                     bg="#EDE9FC" if session["simulated"] else "#E6EAF4", fg=PURPLE if session["simulated"] else MUTED)
            self.timer_var.set(timestamp(session["duration"]).split(".")[0])
        self._refresh_transcript()

    def _new_view(self):
        if self.active_id:
            self._load_session(self.active_id)
            self._notice("A session is active. Stop it before starting another meeting.")
            return
        self._reset_assistant()
        self._save_notes()
        self.selected_id = None
        self.title_var.set("Ready when you are.")
        self.subtitle_var.set("Prepare privately, or select audio sources to capture a session.")
        self.profile_var.set("Hardware-adaptive offline speech · awaiting model setup")
        self.indicator.configure(text="●  READY", bg="#E6EAF4", fg=MUTED)
        self.timer_var.set("00:00:00")
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        self.notes.edit_modified(False)
        self._notes_enabled(False)
        self.notes_status.set("Choose or start a session to write notes.")
        self.tree.delete(*self.tree.get_children())
        self.count_var.set("0 segments")
        self.history.selection_clear(0, "end")
        self._detail("Select your audio sources, confirm participant consent, then start a meeting.")

    def _refresh_transcript(self):
        if not self.selected_id:
            return
        selected = self.tree.selection()
        rows = self.store.segments(self.selected_id, search=self.search_var.get(), limit=PAGE_SIZE,
                                   offset=self.offset, newest=True)
        count = self.store.segment_count(self.selected_id, self.search_var.get())
        if self.offset and not rows:
            self.offset = max(0, self.offset - PAGE_SIZE)
            return self._refresh_transcript()
        self.tree.delete(*self.tree.get_children())
        for index, segment in enumerate(reversed(rows)):
            source = {"local": "Mic", "mic": "Mic", "remote": "System", "loopback": "System"}.get(segment["source"], segment["source"])
            self.tree.insert("", "end", iid=str(segment["id"]), values=(timestamp(segment["start"]).split(".")[0], source,
                              segment["speaker_name"], segment["text"].replace("\n", " ")),
                             tags=("alternate",) if index % 2 else ())
        self.count_var.set(f"{count:,} segments")
        self.page_var.set(f"{self.offset + 1 if count else 0}–{min(self.offset + PAGE_SIZE, count)} / {count:,}")
        self.older_button.configure(state="normal" if count > self.offset + PAGE_SIZE else "disabled")
        self.newer_button.configure(state="normal" if self.offset else "disabled")
        if selected and self.tree.exists(selected[0]):
            self.tree.selection_set(selected[0])
        elif rows and self.offset == 0 and self.selected_id == self.active_id:
            self.tree.see(str(rows[0]["id"]))
        else:
            self.tree.yview_moveto(0)

    def _search_changed(self, *_):
        if self.search_job:
            self.root.after_cancel(self.search_job)
        self.search_job = self.root.after(200, self._apply_search)

    def _apply_search(self):
        self.search_job = None
        self.offset = 0
        self._refresh_transcript()

    def _page(self, direction: int):
        self.offset = max(0, self.offset + direction * PAGE_SIZE)
        self._refresh_transcript()

    def _segment_selected(self, _event=None):
        selected = self.tree.selection()
        if not selected:
            return
        segment = self.store.get_segment(int(selected[0]))
        confidence = segment["confidence"]
        confidence_text = "unavailable" if confidence is None else f"{confidence:.2f} (model estimate)"
        changed = " · User corrected" if segment["edited_at"] else ""
        text = f"{segment['text']}\nSource: {segment['source']} · Speaker: {segment['speaker_name']} · Confidence: {confidence_text}{changed}"
        if segment.get("speech_end_delay_ms") is not None:
            text += f"\nStable caption after detected speech end: {segment['speech_end_delay_ms'] / 1000:.1f}s"
        elif segment.get("caption_delay_ms") is not None:
            text += f"\nLegacy delay from ASR timestamp: {segment['caption_delay_ms'] / 1000:.1f}s"
        if segment.get("latency_ms") is not None:
            text += f" · Queue and processing: {segment['latency_ms'] / 1000:.1f}s"
        if segment["warning"]:
            text += "\n" + segment["warning"]
        self._detail(text, expanded=True)

    def _edit_segment(self, _event=None):
        selected = self.tree.selection()
        if not selected:
            self._notice("Select a transcript segment to edit its words or speaker.")
            return
        segment = self.store.get_segment(int(selected[0]))
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit transcript · original retained")
        dialog.geometry("620x490")
        dialog.configure(bg=BG)
        dialog.transient(self.root)
        dialog.grab_set()
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Correct the transcript", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(frame, text=f"{timestamp(segment['start'])}  ·  Source: {segment['source']}  ·  Original text is preserved",
                  style="Muted.TLabel").pack(anchor="w", pady=(5, 12))
        text = tk.Text(frame, wrap="word", height=6, font=("Segoe UI", 11), padx=9, pady=9)
        text.insert("1.0", segment["text"])
        text.pack(fill="both", expand=True)
        ttk.Label(frame, text="Speaker label for this segment").pack(anchor="w", pady=(12, 4))
        labels = sorted({s["speaker"] for s in self.store.segments(segment["session_id"])})
        speaker_var = tk.StringVar(value=segment["speaker"])
        combo = ttk.Combobox(frame, textvariable=speaker_var, values=labels)
        combo.pack(fill="x")
        ttk.Label(frame, text="Display name (user supplied; applies to this speaker label throughout the session)",
                  style="Muted.TLabel").pack(anchor="w", pady=(12, 4))
        name_var = tk.StringVar(value=segment["speaker_name"] if segment["speaker_name"] != segment["speaker"] else "")
        ttk.Entry(frame, textvariable=name_var).pack(fill="x")
        ttk.Label(frame, text="Automatic labels do not identify real people. Unknown may contain several voices.",
                  style="Muted.TLabel").pack(anchor="w", pady=10)

        def save():
            speaker = speaker_var.get().strip() or "Unknown"
            self.store.correct_segment(segment["id"], text.get("1.0", "end-1c"), speaker=speaker)
            self.store.rename_speaker(segment["session_id"], speaker, name_var.get())
            self._refresh_transcript()
            self._segment_selected()
            self._notice("Correction saved. Original text and original speaker label remain in JSON and CSV exports.")
            dialog.destroy()

        ttk.Button(frame, text="Save correction", style="Accent.TButton", command=save).pack(side="right")
        ttk.Button(frame, text="Cancel", command=dialog.destroy).pack(side="right", padx=8)

    def _notes_changed(self, _event=None):
        if not self.notes.edit_modified():
            return
        self.notes.edit_modified(False)
        if self.selected_id:
            self.notes_status.set("Saving…")
            if self.notes_save_job:
                self.root.after_cancel(self.notes_save_job)
            self.notes_save_job = self.root.after(500, self._save_notes)

    def _save_notes(self):
        if self.notes_save_job:
            self.root.after_cancel(self.notes_save_job)
            self.notes_save_job = None
        if self.selected_id and not self.closed:
            self.store.update_session(self.selected_id, notes=self.notes.get("1.0", "end-1c"))
            self.notes_status.set("Saved locally")

    def _suggest_notes(self):
        if not self.selected_id:
            self._notice("Start a meeting or select a saved session first.")
            return
        segments = self.store.segments(self.selected_id)
        candidate_text = suggest_notes(segments)
        existing = self.notes.get("1.0", "end-1c").rstrip()
        self.notes.insert("end", ("\n\n" if existing else "") + candidate_text)
        self._save_notes()
        self._notice("Candidate quotes appended to your notes. Verify context, decisions, and assignments yourself.")

    def _rename_session(self):
        if not self.selected_id:
            self._notice("Start a meeting or choose a saved session to rename it.")
            return
        name = simpledialog.askstring("Rename meeting", "Meeting title:", parent=self.root, initialvalue=self.title_var.get())
        if name and name.strip():
            self.store.update_session(self.selected_id, title=name.strip())
            self.title_var.set(name.strip())
            self._refresh_history()

    def _export(self):
        if not self.selected_id:
            self._notice("Select a session to export its transcript and notes.")
            return
        destination = filedialog.askdirectory(parent=self.root, title="Export TXT, Markdown, JSON, CSV and SRT to this folder")
        if not destination:
            return
        self._save_notes()
        try:
            paths = self.store.export_session(self.selected_id, destination)
            self._notice(f"Exported {len(paths)} files to {destination}")
        except Exception as exc:
            self._notice(f"Export failed: {exc}")
            messagebox.showerror("Export failed", str(exc), parent=self.root)

    def _discover_devices(self):
        if self.active_id:
            self._notice("Audio devices can be refreshed after stopping the active meeting.")
            return

        def discover():
            try:
                from .audio import list_devices
                self.events.put({"type": "devices", "devices": list_devices()})
            except Exception as exc:
                self.events.put({"type": "warning", "message": f"Audio devices unavailable: {exc}"})

        threading.Thread(target=discover, daemon=True, name="device-discovery").start()

    def _device_id(self, choice: str, kind: str):
        for device in self.devices:
            if device["kind"] == kind and self._device_label(device) == choice:
                return device["id"]
        return None

    @staticmethod
    def _device_label(device: dict) -> str:
        return f"{device['name']} [{device['id']}]"

    def _set_capture_controls(self, active: bool):
        self.start_button.configure(state="disabled" if active else "normal")
        self.stop_button.configure(state="normal" if active else "disabled")
        for combo in (self.mic_combo, self.loopback_combo, self.language_combo, self.companion_mode_combo):
            combo.configure(state="disabled" if active else "readonly")

    def _start(self):
        if self.active_id:
            return
        if not self.consent_var.get():
            self._notice("Confirm participant agreement before recording.")
            return
        mic_id = self._device_id(self.mic_var.get(), "mic")
        loopback_id = self._device_id(self.loopback_var.get(), "loopback")
        if mic_id is None and loopback_id is None:
            self._notice("Select a microphone, a system audio output, or both.")
            return
        try:
            from .profiles import resolve_profile
            profile, profile_notice = resolve_profile(self.project_dir)
        except Exception as exc:
            self._notice(f"Cannot select a local speech profile: {exc}. Check settings.json and run Setup for your Intel or NVIDIA target.")
            return
        self.profile_var.set(profile.label)
        self._notice(profile_notice)
        model_path = self.project_dir / "models" / profile.model_dir
        if not model_path.is_dir():
            self._notice(f"Local model missing: {model_path.name}. Run the documented Setup for target '{profile.target}', "
                         "then start again. The synthetic demo works now.")
            return
        language = {"Auto": "auto", "English": "en", "中文": "zh"}[self.language_var.get()]
        retain_audio = self.retain_var.get()
        session_id = self.store.create_session(f"{MODES[self.companion_mode]} · {datetime.now():%b %d, %H:%M}", language=language,
                                               retain_audio=retain_audio, profile=profile.label)
        self.active_id = session_id
        self.session_incomplete = False
        self.session_error = False
        self.last_latency = None
        self.last_caption_delay = None
        self.cancel_start.clear()
        self._load_session(session_id)
        self._refresh_history()
        self._set_capture_controls(True)
        self.indicator.configure(text="●  LOADING", bg="#FFF1D6", fg="#9C6C12")
        self._notice("Loading the local model… audio capture has not started.")

        def on_event(event):
            event = dict(event)
            event["session_id"] = session_id
            try:
                segment_id = self.store.append_event(session_id, event)
                if segment_id is not None:
                    event["segment_id"] = segment_id
            except Exception as exc:
                event = {"type": "persistence_error", "message": f"Saving failed: {exc}", "session_id": session_id}
            self.events.put(event)

        def load_and_start():
            try:
                from .inference import Recognizer, SpeakerTracker
                from .engine import Engine
                try:
                    recognizer = Recognizer(str(model_path), language=language, **profile.recognizer_options())
                except Exception as model_error:
                    raise RuntimeError(
                        f"Selected candidate could not load ({profile.label}): {model_error}. "
                        f"Check Setup for target '{profile.target}', or explicitly set cpu to true in settings.json "
                        "if you intend to use CPU. CPU speed must be measured before live use."
                    ) from model_error
                if self.cancel_start.is_set():
                    self.events.put({"type": "finished", "session_id": session_id, "status": "cancelled"})
                    return
                speaker_path = self.project_dir / "models" / "speaker.onnx"
                tracker = SpeakerTracker(str(speaker_path) if speaker_path.exists() else None)
                engine = Engine(recognizer, tracker, on_event, self.data_dir / "audio" / session_id)
                self.engine = engine
                if self.cancel_start.is_set():
                    self.events.put({"type": "finished", "session_id": session_id, "status": "cancelled"})
                    return
                result = engine.start(mic_id=mic_id, loopback_id=loopback_id, retain_audio=retain_audio)
                if not result.get("running") or result.get("state") == "error":
                    self.events.put({"type": "finished", "session_id": session_id, "status": "error"})
                    return
                if self.cancel_start.is_set():
                    engine.stop()
                    # UI polls the engine until all in-flight work has exited.
                    return
                self.store.update_session(session_id, status="recording", profile=profile.label)
                on_event({"type": "started", "profile_label": profile.label})
            except Exception as exc:
                on_event({"type": "warning", "message": f"Meeting could not start: {exc}"})
                if self.engine and self.engine.running:
                    self.engine.stop()
                self.events.put({"type": "finished", "session_id": session_id, "status": "error"})

        self.worker = threading.Thread(target=load_and_start, daemon=True, name="model-loader")
        self.worker.start()

    def _stop(self):
        # Capture cancellation must not depend on optional assistant widgets.
        if self.active_id:
            self.cancel_start.set()
        self._reset_assistant()
        if not self.active_id:
            return
        self.cancel_start.set()
        self.stop_button.configure(state="disabled")
        self.indicator.configure(text="●  STOPPING", bg="#FFF1D6", fg="#9C6C12")
        self._notice("Stopping capture and finishing queued speech…")
        session_id = self.active_id
        try:
            self.store.update_session(session_id, status="stopping")
        except Exception as exc:
            self.session_error = True
            self._notice(f"Cannot save session status ({exc}); capture is stopping.")
        # The loader owns cancellation until it has finished starting the engine.
        if self.worker and self.worker.is_alive():
            return

        def stop_engine():
            try:
                if self.engine:
                    self.engine.stop()
            except Exception as exc:
                self.session_error = True
                self.events.put({"type": "warning", "message": f"Capture stop reported: {exc}"})
            if not self.engine:
                self.events.put({"type": "finished", "session_id": session_id, "status": "cancelled"})

        self.stop_worker = threading.Thread(target=stop_engine, daemon=True, name="capture-stop")
        self.stop_worker.start()

    def _finish_current(self, session_id: str, status: str, engine_status: dict | None = None):
        if session_id != self.active_id:
            return
        engine_status = engine_status or {}
        if self.engine and self.engine.running:
            if status == "error":
                self.session_error = True
            return
        self._reset_assistant()
        duration = engine_status.get("elapsed_seconds")
        if duration is None and self.recording_since:
            duration = time.monotonic() - self.recording_since
        if self.session_error or engine_status.get("state") == "error" or engine_status.get("device_error"):
            status = "error"
        elif self.session_incomplete or any(engine_status.get(key, 0) for key in
                                           ("dropped_audio_blocks", "dropped_segments", "failed_segments", "event_callback_errors")):
            status = "incomplete"
        save_error = None
        try:
            self.store.append_event(session_id, {"type": "final_status", "status": status, "engine": engine_status})
            self.store.finish_session(session_id, status=status, duration=duration)
        except Exception as exc:
            save_error = exc
            status = "error"
            self.session_error = True
            # Keep the error visible if closing triggered this finalization.
            self.closing = False
        # Capture is already stopped. A database error must not retain a stale
        # engine or leave the controls claiming that recording is still active.
        self.active_id = None
        self.engine = None
        self.recording_since = None
        self._set_capture_controls(False)
        self.indicator.configure(text="●  UNSAVED ERROR" if save_error else "●  SAVED" if status == "completed" else "●  " + status.upper(),
                                 bg="#FFE4EB" if status in ("error", "incomplete") else "#E6EAF4",
                                 fg=RED if status in ("error", "incomplete") else MUTED)
        if save_error:
            self._notice(f"Capture stopped, but the final session state could not be saved: {save_error}. "
                         "Previously committed transcript remains in the database. Open Diagnostics for details.")
        else:
            self._notice("Meeting stopped. Transcript and notes are saved locally." if status == "completed"
                         else f"Session {status}. Saved content is available in history; open Diagnostics for details.")
        if duration is not None:
            self.timer_var.set(timestamp(duration).split(".")[0])
        if self.selected_id == session_id:
            self.subtitle_var.set("Capture stopped · Final session state was not saved" if save_error
                                  else f"Session {status} · Saved on this device")
        for refresh in (self._refresh_history, self._refresh_transcript):
            try:
                refresh()
            except Exception as exc:
                message = f"Stopped session display could not refresh: {exc}. Open Diagnostics for save errors."
                if save_error:
                    message = f"Final session state was not saved ({save_error}). " + message
                self._notice(message)

    def _demo(self):
        if self.active_id:
            self._notice("Stop the current meeting before opening the synthetic demo.")
            return
        session_id = self.store.create_session("Product sync · demo", language="auto", simulated=True,
                                               profile="Synthetic transcript · no model or audio capture")
        examples = [
            (0.8, 5.3, "local", "Speaker 1", "Let's focus on the onboarding experience today.", .94),
            (6.1, 11.4, "remote", "Speaker 2", "The first screen should explain where recordings are stored.", .89),
            (12.0, 17.8, "local", "Speaker 1", "I'll draft the welcome copy and share it for review.", .95),
            (18.6, 23.2, "remote", "Speaker 2", "我们需要同时支持中文和英文的会议记录。", .91),
            (24.0, 29.7, "remote", "Unknown", "Could we make the recording indicator easier to see?", .62),
            (30.5, 35.4, "local", "Speaker 1", "Let's test that with the team before the next meeting.", .92),
            (36.1, 41.3, "remote", "Speaker 2", "I'll review the transcript export tomorrow.", .93),
            (42.0, 47.2, "local", "Speaker 1", "Thanks everyone. We'll confirm the next steps in our notes.", .96),
        ]
        for start, end, source, speaker, text, confidence in examples:
            self.store.append_event(session_id, {"type": "segment", "start": start, "end": end, "source": source,
                                                  "speaker": speaker, "text": text, "confidence": confidence,
                                                  "warning": "Simulated data; accuracy and performance are not measured.", "latency_ms": None})
        self.store.finish_session(session_id, duration=47.2)
        self.store.update_session(session_id, notes="SYNTHETIC DEMO\nNo meeting was recorded.\n\nDiscussion\n• Make local storage clear on the welcome screen.\n• Support English and Chinese transcripts.\n• Review the visibility of recording controls.\n\nTry it\nSelect a segment to read the full text. Double-click to correct it or assign a speaker name.\n\nUse “Find candidate quotes” to collect verbatim passages for review.")
        self._load_session(session_id)
        self._refresh_history()
        self.profile_var.set("Synthetic transcript · no audio capture")
        self._notice("Synthetic demo · No capture or inference has run. All transcript content and confidence values are simulated.")

    def _poll(self):
        if self.closed:
            return
        try:
            self._poll_once()
        except Exception as exc:
            self._notice(f"UI update failed: {exc}. The interface remains active; check Diagnostics for save errors.")
        finally:
            if not self.closed:
                self.root.after(100, self._poll)

    def _poll_once(self):
        refresh = False
        for _ in range(250):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event.get("type")
            if kind == "devices":
                self.devices = event["devices"]
                for combo, var, device_kind in [(self.mic_combo, self.mic_var, "mic"),
                                                 (self.loopback_combo, self.loopback_var, "loopback")]:
                    values = ["Off"] + [self._device_label(d) for d in self.devices if d["kind"] == device_kind]
                    combo.configure(values=values)
                    if var.get() not in values:
                        var.set("Off")
                if not self.devices:
                    self._notice("No capture devices found. Connect an audio device and refresh, or open the synthetic demo.")
            elif kind == "segment":
                refresh |= event.get("session_id") == self.selected_id
                if (self.assistant and not self.assistant.closed
                        and event.get("session_id") == self.selected_id and not self.cancel_start.is_set()):
                    try:
                        self.assistant.feed_segment(event)
                    except Exception as exc:
                        self._notice(f"Question assistant update failed: {exc}. Transcription continues.")
                self.last_latency = event.get("latency_ms")
                self.last_caption_delay = event.get("caption_delay_from_speech_end_ms") if event.get("caption_delay_valid") else None
            elif kind == "started":
                self.recording_since = time.monotonic()
                self.indicator.configure(text="●  RECORDING", bg="#FFE4EB", fg=RED)
                self.profile_var.set(event["profile_label"])
                self._notice("Recording · Transcript is saved as speech segments arrive.")
            elif kind == "finished":
                self._finish_current(event["session_id"], event["status"], self.engine.status if self.engine else {})
                refresh = True
            elif kind == "persistence_error":
                self._notice(event["message"])
                self.session_error = True
                self._stop()
            elif kind in ("warning", "error"):
                self._notice(str(event.get("message", event)))
            elif kind == "status":
                if event.get("state") == "degraded":
                    self.session_incomplete = True
                elif event.get("state") == "error":
                    self.session_error = True
                self._notice(str(event.get("message") or event.get("state") or "Working…"))
            else:
                if "message" in event:
                    self._notice(str(event["message"]))
        if refresh:
            try:
                self._refresh_transcript()
            except Exception as exc:
                self._notice(f"Transcript display could not refresh: {exc}. Check Diagnostics for save errors.")
        if self.recording_since:
            self.timer_var.set(timestamp(time.monotonic() - self.recording_since).split(".")[0])
        backlog = self.events.qsize()
        if self.active_id:
            if self.engine:
                stats = self.engine.status
                drops = stats.get("dropped_audio_blocks", 0) + stats.get("dropped_segments", 0)
                failures = stats.get("failed_segments", 0)
                if drops or failures or stats.get("event_callback_errors", 0):
                    self.session_incomplete = True
                delay = f" · Caption delay {self.last_caption_delay / 1000:.1f}s" if self.last_caption_delay is not None else ""
                self.performance_var.set(f"Queued audio: {stats.get('queued_audio', 0)} · Speech: {stats.get('queued_segments', 0)}"
                                         f" · Dropped: {drops} · Failed: {failures}{delay} · UI: {backlog}")
                if stats.get("state") == "stopping":
                    self.indicator.configure(text="●  STOPPING", bg="#FFF1D6", fg="#9C6C12")
                    self.stop_button.configure(state="disabled")
                loader_done = not self.worker or not self.worker.is_alive()
                stopper_done = not self.stop_worker or not self.stop_worker.is_alive()
                if self.cancel_start.is_set() and stats.get("running") and stats.get("state") != "stopping" and loader_done and stopper_done:
                    self._stop()
                if not stats.get("running") and stats.get("state") in ("stopped", "error") and loader_done and stopper_done:
                    self._finish_current(self.active_id, "completed", stats)
            else:
                self.performance_var.set(f"Loading local models · UI queue: {backlog} events")
        if self.closing and not self.active_id:
            self._destroy()
            return

    def _open_companion(self):
        from .companion_ui import CompanionWindow
        if self.companion and not self.companion.closed:
            self.companion.show()
            return
        try:
            self.companion = CompanionWindow(
                self.root, self.project_dir, on_review=self._companion_review,
                on_notes=self._companion_notes, on_mode=self._set_companion_mode,
                initial_mode=self.companion_mode)
            self._set_companion_mode(self.companion.mode)
        except (OSError, ValueError) as exc:
            self._notice(f"Preparation workspace could not be opened: {exc}. Existing data was preserved.")

    def _choose_companion_mode(self, _event=None):
        chosen = next((mode for mode, label in MODES.items() if label == self.companion_mode_var.get()), None)
        self.companion_mode_var.set(MODES[self.companion_mode])
        if chosen is None:
            return
        self._open_companion()
        if self.companion and not self.companion.closed:
            self.companion.set_mode(chosen)

    def _set_companion_mode(self, mode):
        if mode not in MODES:
            raise ValueError("Unknown companion mode.")
        if mode == self.companion_mode:
            return True
        if self.active_id:
            self._notice("Stop the current capture before switching session modes.")
            return False
        if self.assistant and not self.assistant.closed and self.assistant.inflight:
            self._notice("Finish or cancel the current private answer before switching modes.")
            return False
        self._new_view()
        self.companion_mode = mode
        self.companion_mode_var.set(MODES[mode])
        self.workspace_var.set(MODES[mode].upper())
        self._notice(f"{MODES[mode]}. Preparation stays local; capture requires Start capture.")
        return True

    def _ensure_prepared_session(self, mode):
        if not self.selected_id:
            session_id = self.store.create_session(
                f"{MODES[mode]} · {datetime.now():%b %d, %H:%M}", language="auto",
                profile="Local preparation · no capture")
            self.store.update_session(session_id, status="prepared")
            self._load_session(session_id)
            self._refresh_history()
        return self.selected_id

    def _companion_review(self, question, profile, sources):
        if self.assistant and not self.assistant.closed and self.assistant.inflight:
            self._notice("Finish or cancel the current private answer before reviewing another question.")
            return False
        mode = profile.get("mode", self.companion_mode)
        if not self._set_companion_mode(mode):
            return False
        self._ensure_prepared_session(mode)
        self._open_assistant()
        assistant = self.assistant
        # A session reset must happen before adding this deliberately selected context.
        assistant._ensure_session()
        assistant._invalidate_preview()
        assistant.nearby_var.set(False)
        assistant.files = tuple(sources)
        assistant.allowed_directory = None
        assistant.folder_var.set("Preparation: only the explicitly selected excerpts below.")
        assistant._refresh_files()
        assistant.context_vars["meeting_goal"].set(profile.get("goal", ""))
        assistant.context_vars["my_role"].set(profile.get("role", ""))
        assistant.context_vars["constraints"].set(
            f"Mode: {MODES[mode]}. Topic: {profile.get('topic', '')}. "
            "Ground the draft in the reviewed sources. Mark missing information; do not invent facts or outcomes.")
        assistant._manual_ask(question)
        assistant._status("Prepared locally. Review the exact payload; each cloud request needs separate approval.")
        assistant.show()
        return True

    def _companion_notes(self, mode, markdown):
        if not self._set_companion_mode(mode):
            return False
        self._ensure_prepared_session(mode)
        existing = self.notes.get("1.0", "end-1c").rstrip()
        self.notes.insert("end", ("\n\n---\n\n" if existing else "") + markdown)
        self._save_notes()
        self._notice("Preparation notes appended to this local session. Export includes the notes.")
        return True

    def _open_assistant(self):
        from .assistant_ui import AssistantWindow
        if self.assistant and not self.assistant.closed:
            self.assistant.show()
            return
        self.assistant = AssistantWindow(self.root, self.project_dir, self.store,
                                         lambda: self.selected_id)

    def _reset_assistant(self):
        if self.assistant and not self.assistant.closed:
            try:
                self.assistant.stop_session()
            except Exception as exc:
                self._notice(f"Question assistant cleanup failed: {exc}. Capture controls remain available.")

    def _show_diagnostics(self):
        try:
            import json
            from .profiles import runtime_inventory
            inventory = json.dumps(runtime_inventory(self.project_dir), ensure_ascii=False, indent=2)
        except Exception as exc:
            inventory = f"Hardware/profile inventory unavailable: {exc}"
        dialog = tk.Toplevel(self.root)
        dialog.title("Local diagnostics")
        dialog.geometry("850x450")
        text = tk.Text(dialog, wrap="word", padx=15, pady=15, font=("Consolas", 10))
        text.pack(fill="both", expand=True)
        text.insert("1.0", f"Database: {self.store.path}\nUI event backlog: {self.events.qsize()}\n"
                    "Speaker labels and confidence are model estimates, not verified identity.\n"
                    "Caption delay is measured after speech ends; processing latency includes the inference queue.\n"
                    + (f"Engine status: {self.engine.status}\n" if self.engine else "")
                    + "\nHardware and local candidate profile (selection is not a benchmark):\n" + inventory
                    + "\n\n" + "\n".join(self.diagnostics))
        text.configure(state="disabled")

    def _close(self):
        self._save_notes()
        if self.active_id:
            self.closing = True
            self._stop()
            self._notice("Finishing the active session before closing…")
        else:
            self._destroy()

    def _destroy(self):
        if self.companion and not self.companion.closed:
            self.companion.close()
            if not self.companion.closed:
                self._notice("Preparation could not be saved. Resolve its error before closing the app.")
                return
        self.closed = True
        if self.assistant:
            try:
                self.assistant.close()
            except Exception as exc:
                self._notice(f"Question assistant cleanup failed while closing: {exc}")
        self.store.close()
        self.root.destroy()


def launch(project_dir: Path, demo: bool = False):
    # Match Tk coordinates to physical pixels on scaled Windows displays.
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    App(root, project_dir, demo=demo)
    root.mainloop()
