"""Local preparation and practice for three independent companion workflows."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .companion import (MODES, CompanionRepository, default_state, load_material,
                        notes_markdown, question_bank, review_sources)

BG, PAPER, INK, MUTED = "#F3F5FA", "#FFFFFF", "#182238", "#667187"
GUIDES = {
    "interview": ("Prepare your interview", "Set the role and goal, choose a practice question, then type your answer. Save it locally or open a feedback draft for review.",
                  "STAR check: describe the situation, your task, the action you took, and the result. Separate your contribution from the team's; use evidence you can support."),
    "presentation": ("Rehearse your presentation", "Add your slides or notes and select the exact excerpts you want to discuss. Practice a question and review your answer before sharing any context.",
                     "Presentation check: answer the question directly, cite the relevant slide or excerpt, explain assumptions, and say when the supplied material does not establish a claim."),
    "meeting": ("Prepare your meeting", "Write your goal and role, select any useful excerpts, and keep local notes. A question opens the existing private assistant for a separate review.",
                "Meeting check: distinguish facts, open questions and possible next steps. Leave owners and dates unassigned unless they were explicitly confirmed."),
}


class CompanionWindow:
    def __init__(self, root, project_dir, *, on_review, on_notes, on_mode, initial_mode="meeting"):
        self.root, self.project_dir = root, Path(project_dir)
        self.on_review, self.on_notes, self.on_mode = on_review, on_notes, on_mode
        self.closed = False
        self.loading = False
        self.load_error = None
        path = self.project_dir / "data/companion.json"
        existed = path.exists()
        self.repository = CompanionRepository(path)
        try:
            self.state = self.repository.load()
        except Exception as exc:
            self.state = default_state()
            self.load_error = str(exc)
        if not existed and initial_mode in MODES:
            self.state["mode"] = initial_mode
        self.material_sections = {}
        self.window = tk.Toplevel(root)
        self.window.title("Megumin Companion — Practice & prepare")
        self.window.geometry("1150x850")
        self.window.minsize(1040, 780)
        self.window.configure(bg=BG)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self._load_profile()
        if self.load_error:
            self._status(f"Saved preparation could not be read: {self.load_error}. Existing data will not be overwritten; local export remains available.")

    @property
    def mode(self):
        return self.state["mode"]

    def _build(self):
        style = ttk.Style(self.window)
        style.configure("Companion.TFrame", background=BG)
        style.configure("CompanionCard.TFrame", background=PAPER)
        style.configure("Companion.TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.configure("CompanionMuted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Companion.TButton", padding=(10, 7), font=("Segoe UI", 10))
        style.configure("Companion.Treeview", rowheight=29, font=("Segoe UI", 10))
        frame = ttk.Frame(self.window, style="Companion.TFrame", padding=20)
        frame.pack(fill="both", expand=True)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(3, weight=1)
        title = ttk.Frame(frame, style="Companion.TFrame")
        title.grid(row=0, column=0, sticky="ew")
        ttk.Label(title, text="Practice & prepare", style="Companion.TLabel", font=("Segoe UI", 23, "bold")).pack(side="left")
        ttk.Label(title, text="Saved on this computer · no recording or automatic sending", style="CompanionMuted.TLabel").pack(side="right")
        modes = ttk.Frame(frame, style="Companion.TFrame")
        modes.grid(row=1, column=0, sticky="ew", pady=(16, 10))
        self.mode_var = tk.StringVar(value=self.mode)
        for key, label in MODES.items():
            ttk.Radiobutton(modes, text=label, value=key, variable=self.mode_var,
                            command=lambda: self.set_mode(self.mode_var.get())).pack(side="left", padx=(0, 30), ipady=5)
        self.guide_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.guide_var, style="Companion.TLabel", wraplength=1060).grid(row=2, column=0, sticky="w", pady=(0, 12))
        self.notebook = ttk.Notebook(frame)
        self.notebook.grid(row=3, column=0, sticky="nsew")
        self.prepare_tab = ttk.Frame(self.notebook, style="Companion.TFrame", padding=15)
        self.practice_tab = ttk.Frame(self.notebook, style="Companion.TFrame", padding=15)
        self.notes_tab = ttk.Frame(self.notebook, style="Companion.TFrame", padding=15)
        for tab, label in ((self.prepare_tab, "1  Prepare & choose context"), (self.practice_tab, "2  Practice & review"), (self.notes_tab, "3  Local notes")):
            self.notebook.add(tab, text=label)
        self._build_prepare()
        self._build_practice()
        self._build_notes()
        footer = ttk.Frame(frame, style="Companion.TFrame")
        footer.grid(row=4, column=0, sticky="ew", pady=(13, 0))
        ttk.Button(footer, text="Save preparation locally", style="Companion.TButton", command=self._save_preparation).pack(side="right")
        self.status_var = tk.StringVar(value="Choose a workflow. Each keeps its own preparation, selected context and practice answers.")
        ttk.Label(footer, textvariable=self.status_var, style="CompanionMuted.TLabel", wraplength=780).pack(side="left", fill="x", expand=True)

    def _text(self, parent, *, height=5, readonly=False):
        frame = ttk.Frame(parent, style="Companion.TFrame")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        widget = tk.Text(frame, height=height, wrap="word", font=("Segoe UI", 10), bg=PAPER, fg=INK,
                         relief="flat", padx=9, pady=8, undo=not readonly)
        widget.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        widget.configure(yscrollcommand=scroll.set, state="disabled" if readonly else "normal")
        return frame, widget

    def _build_prepare(self):
        tab = self.prepare_tab
        tab.grid_columnconfigure(0, weight=2)
        tab.grid_columnconfigure(1, weight=3)
        tab.grid_rowconfigure(0, weight=1)
        left = ttk.Frame(tab, style="Companion.TFrame", padding=(0, 0, 18, 0))
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        ttk.Label(left, text="Your preparation", style="Companion.TLabel", font=("Segoe UI", 15, "bold")).grid(row=0, column=0, sticky="w")
        self.topic_var, self.role_var = tk.StringVar(), tk.StringVar()
        for row, label, variable in ((1, "Topic / position / meeting", self.topic_var), (3, "Your role", self.role_var)):
            ttk.Label(left, text=label, style="Companion.TLabel").grid(row=row, column=0, sticky="w", pady=(13, 5))
            ttk.Entry(left, textvariable=variable).grid(row=row + 1, column=0, sticky="ew")
        ttk.Label(left, text="Goal and facts you want to establish", style="Companion.TLabel").grid(row=5, column=0, sticky="w", pady=(15, 5))
        box, self.goal_text = self._text(left, height=8)
        box.grid(row=6, column=0, sticky="nsew")
        left.grid_rowconfigure(6, weight=1)
        ttk.Label(left, text="These fields become visible context only when you open a review draft.\nImported files are not included until you select their excerpts.",
                  style="CompanionMuted.TLabel", wraplength=390).grid(row=7, column=0, sticky="w", pady=(12, 0))
        right = ttk.Frame(tab, style="Companion.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=2)
        right.grid_rowconfigure(6, weight=1)
        ttk.Label(right, text="Choose your material excerpts", style="Companion.TLabel", font=("Segoe UI", 15, "bold")).grid(row=0, column=0, sticky="w")
        tools = ttk.Frame(right, style="Companion.TFrame")
        tools.grid(row=1, column=0, sticky="ew", pady=(10, 7))
        ttk.Button(tools, text="Add material…", style="Companion.TButton", command=self._import_material).pack(side="left")
        ttk.Button(tools, text="Remove file", style="Companion.TButton", command=self._remove_material).pack(side="left", padx=6)
        ttk.Button(tools, text="Clear selection", style="Companion.TButton", command=self._clear_selection).pack(side="right")
        ttk.Label(right, text="Choose up to 5 excerpts / 5,000 characters. Ctrl-click for multiple. Adding a file selects nothing.",
                  style="CompanionMuted.TLabel", wraplength=440).grid(row=2, column=0, sticky="w", pady=(0, 7))
        table = ttk.Frame(right, style="Companion.TFrame")
        table.grid(row=3, column=0, sticky="nsew")
        table.grid_columnconfigure(0, weight=1)
        table.grid_rowconfigure(0, weight=1)
        self.sections_tree = ttk.Treeview(table, columns=("file", "section"), show="headings", selectmode="extended", height=7, style="Companion.Treeview")
        self.sections_tree.heading("file", text="FILE")
        self.sections_tree.heading("section", text="SLIDE / EXCERPT")
        self.sections_tree.column("file", width=145, minwidth=110)
        self.sections_tree.column("section", width=310, minwidth=210)
        self.sections_tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.sections_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.sections_tree.configure(yscrollcommand=scroll.set)
        self.sections_tree.bind("<<TreeviewSelect>>", self._section_selected)
        self.selection_var = tk.StringVar()
        ttk.Label(right, textvariable=self.selection_var, style="CompanionMuted.TLabel").grid(row=4, column=0, sticky="w", pady=7)
        ttk.Label(right, text="Excerpt preview", style="Companion.TLabel").grid(row=5, column=0, sticky="w", pady=(0, 5))
        box, self.material_preview = self._text(right, height=5, readonly=True)
        box.grid(row=6, column=0, sticky="nsew")

    def _build_practice(self):
        tab = self.practice_tab
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=3)
        tab.grid_rowconfigure(0, weight=1)
        left = ttk.Frame(tab, style="Companion.TFrame", padding=(0, 0, 16, 0))
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)
        ttk.Button(left, text="Make practice questions", style="Companion.TButton", command=self._generate_questions).grid(row=0, column=0, sticky="ew")
        ttk.Label(left, text="Fixed local prompts, adapted with your topic.\nNo model or cloud request.", style="CompanionMuted.TLabel", wraplength=255).grid(row=1, column=0, sticky="w", pady=10)
        self.questions_list = tk.Listbox(left, font=("Segoe UI", 10), bg=PAPER, fg=INK, width=27, height=9, exportselection=False)
        self.questions_list.grid(row=2, column=0, sticky="nsew")
        question_scroll = ttk.Scrollbar(left, orient="horizontal", command=self.questions_list.xview)
        question_scroll.grid(row=3, column=0, sticky="ew")
        self.questions_list.configure(xscrollcommand=question_scroll.set)
        self.questions_list.bind("<<ListboxSelect>>", self._choose_question)
        self.practice_count_var = tk.StringVar()
        ttk.Label(left, textvariable=self.practice_count_var, style="CompanionMuted.TLabel", wraplength=255).grid(row=4, column=0, sticky="w", pady=(10, 0))
        right = ttk.Frame(tab, style="Companion.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)
        ttk.Label(right, text="Question to practice or review", style="Companion.TLabel", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")
        box, self.question_text = self._text(right, height=3)
        box.grid(row=1, column=0, sticky="ew", pady=(8, 12))
        ttk.Label(right, text="Your typed answer — stays local until included in a reviewed draft", style="Companion.TLabel").grid(row=2, column=0, sticky="w", pady=(0, 6))
        box, self.answer_text = self._text(right, height=8)
        box.grid(row=3, column=0, sticky="nsew")
        self.checklist_var = tk.StringVar()
        ttk.Label(right, textvariable=self.checklist_var, style="CompanionMuted.TLabel", wraplength=715).grid(row=4, column=0, sticky="w", pady=(12, 10))
        controls = ttk.Frame(right, style="Companion.TFrame")
        controls.grid(row=5, column=0, sticky="ew")
        ttk.Button(controls, text="Save practice answer", style="Companion.TButton", command=self._save_practice).pack(side="left")
        ttk.Button(controls, text="Review my answer…", style="Companion.TButton", command=self._review_answer).pack(side="right")
        ttk.Button(controls, text="Review question…", style="Companion.TButton", command=self._review_question).pack(side="right", padx=7)
        ttk.Label(right, text="Review opens a draft in the private assistant. You choose context, connection and approval there; this window never sends it.",
                  style="CompanionMuted.TLabel", wraplength=715).grid(row=6, column=0, sticky="w", pady=(10, 0))

    def _build_notes(self):
        tab = self.notes_tab
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        ttk.Label(tab, text="Your notes, questions and next steps", style="Companion.TLabel", font=("Segoe UI", 15, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))
        box, self.notes_text = self._text(tab, height=16)
        box.grid(row=1, column=0, sticky="nsew")
        controls = ttk.Frame(tab, style="Companion.TFrame")
        controls.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(controls, text="Copy local notes", style="Companion.TButton", command=self._copy_notes).pack(side="left")
        ttk.Button(controls, text="Export notes…", style="Companion.TButton", command=self._export_notes).pack(side="left", padx=7)
        ttk.Button(controls, text="Use in meeting notes", style="Companion.TButton", command=self._send_notes).pack(side="right")
        ttk.Label(tab, text="Notes and saved practice answers remain local. This button hands them to the meeting notes workspace; it does not send a message.",
                  style="CompanionMuted.TLabel", wraplength=1010).grid(row=3, column=0, sticky="w", pady=(10, 0))

    def _status(self, text):
        self.status_var.set(str(text))

    def _replace_text(self, widget, value):
        readonly = str(widget.cget("state")) == "disabled"
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        if readonly:
            widget.configure(state="disabled")

    def _flush(self):
        profile = self.state["profiles"][self.mode]
        values = {"topic": self.topic_var.get().strip(), "role": self.role_var.get().strip()}
        values.update({key: widget.get("1.0", "end-1c").strip() for key, widget in
                       (("goal", self.goal_text), ("question", self.question_text), ("answer", self.answer_text), ("notes", self.notes_text))})
        for key, limit in (("topic", 300), ("role", 300), ("goal", 2000), ("question", 1200), ("answer", 8000), ("notes", 12000)):
            if len(values[key]) > limit:
                raise ValueError(f"{key.capitalize()} exceeds {limit:,} characters. Shorten it before saving or switching modes.")
        profile.update(values)
        return profile

    def _load_profile(self):
        self.loading = True
        profile = self.state["profiles"][self.mode]
        self.mode_var.set(self.mode)
        self.topic_var.set(profile["topic"])
        self.role_var.set(profile["role"])
        for key, widget in (("goal", self.goal_text), ("question", self.question_text), ("answer", self.answer_text), ("notes", self.notes_text)):
            self._replace_text(widget, profile[key])
        self.guide_var.set(GUIDES[self.mode][1])
        self.checklist_var.set(GUIDES[self.mode][2])
        self.questions_list.delete(0, "end")
        self.practice_count_var.set(f"{len(profile['practice_log'])} saved practice answers in this workflow")
        self._refresh_materials()
        self.loading = False

    def set_mode(self, mode):
        if self.closed or mode not in MODES:
            return False
        if mode == self.mode:
            self.mode_var.set(mode)
            return True
        try:
            self._flush()
            if self.on_mode(mode) is False:
                self.mode_var.set(self.mode)
                self._status("Finish or stop the active capture before changing workflows. Your current preparation is retained.")
                return False
        except Exception as exc:
            self.mode_var.set(self.mode)
            self._status(str(exc))
            return False
        self.state["mode"] = mode
        self._load_profile()
        self._save_preparation()
        return True

    def _refresh_materials(self):
        previous_loading = self.loading
        self.loading = True
        self.sections_tree.delete(*self.sections_tree.get_children())
        self.material_sections.clear()
        profile = self.state["profiles"][self.mode]
        for material in profile["materials"]:
            for section in material["sections"]:
                self.material_sections[section["id"]] = (material, section)
                self.sections_tree.insert("", "end", iid=section["id"], values=(material["name"], section["title"]))
        self.sections_tree.selection_set([key for key in profile["selected_ids"] if key in self.material_sections])
        self._selection_summary()
        self._replace_text(self.material_preview, "Select an excerpt to preview it. Only highlighted excerpts are included in a review draft.")
        self.loading = previous_loading

    def _selection_summary(self):
        selected = self.state["profiles"][self.mode]["selected_ids"]
        count = sum(len(self.material_sections[key][1]["text"]) for key in selected if key in self.material_sections)
        self.selection_var.set(f"{len(selected)} selected · {count:,} / 5,000 characters")

    def _section_selected(self, _event=None):
        if self.loading or self.closed:
            return
        profile = self.state["profiles"][self.mode]
        previous = list(profile["selected_ids"])
        profile["selected_ids"] = list(self.sections_tree.selection())
        try:
            review_sources(profile)
        except Exception as exc:
            profile["selected_ids"] = previous
            self.sections_tree.selection_set(previous)
            self._status(f"Selection was not changed: {exc}")
        selected = self.sections_tree.selection()
        focus = self.sections_tree.focus()
        key = focus if focus in self.material_sections else (selected[-1] if selected else None)
        if key:
            material, section = self.material_sections[key]
            self._replace_text(self.material_preview, f"{material['name']} — {section['title']}\n\n{section['text']}")
        self._selection_summary()

    def _clear_selection(self):
        self.state["profiles"][self.mode]["selected_ids"] = []
        self.sections_tree.selection_set([])
        self._selection_summary()

    def _import_material(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(parent=self.window, title="Choose one local material file",
                                             filetypes=[("Supported materials", "*.txt *.md *.json *.pptx")])
        if not path:
            return False
        try:
            profile = self._flush()
            if len(profile["materials"]) >= 5:
                raise ValueError("Use at most five material files per workflow; remove one before adding another.")
            material = load_material(path)
            if any(item["sha256"] == material["sha256"] for item in profile["materials"]):
                raise ValueError("This material is already in this workflow.")
            profile["materials"].append(material)
            self._refresh_materials()
            self._status(f"Added {material['name']} locally; no excerpts were selected automatically. {material.get('notice', '')}")
            return True
        except Exception as exc:
            self._status(f"Material was not added: {exc}")
            return False

    def _remove_material(self):
        selected = self.sections_tree.selection()
        focus = self.sections_tree.focus()
        key = focus if focus in self.material_sections else (selected[0] if selected else None)
        if not key:
            self._status("Choose an excerpt from the file you want to remove.")
            return
        material, _section = self.material_sections[key]
        profile = self.state["profiles"][self.mode]
        removed_ids = {section["id"] for section in material["sections"]}
        profile["materials"] = [item for item in profile["materials"] if item is not material]
        profile["selected_ids"] = [value for value in profile["selected_ids"] if value not in removed_ids]
        self._refresh_materials()
        self._status(f"Removed {material['name']} from this preparation. The original file was not changed.")

    def _generate_questions(self):
        try:
            questions = question_bank(self.mode, self.topic_var.get().strip())
            self.questions_list.delete(0, "end")
            for question in questions:
                self.questions_list.insert("end", question.removeprefix("[Rule-based practice] "))
            self._status("Local practice prompts are ready. Select one to edit; they are rule-based prompts, not generated answers.")
        except Exception as exc:
            self._status(f"Practice prompts could not be prepared: {exc}")

    def _choose_question(self, _event=None):
        selected = self.questions_list.curselection()
        if selected:
            self._replace_text(self.question_text, self.questions_list.get(selected[0]))

    def _save_preparation(self):
        try:
            self._flush()
            if self.load_error:
                raise ValueError(f"The existing preparation file could not be loaded and will not be overwritten: {self.load_error}")
            self.repository.save(self.state)
            self._status(f"{MODES[self.mode]} preparation saved locally.")
            return True
        except Exception as exc:
            self._status(f"Preparation was not saved: {exc}")
            return False

    def _save_practice(self):
        try:
            profile = self._flush()
            if not profile["question"] or not profile["answer"]:
                raise ValueError("Enter a question and your answer before saving a practice entry.")
            if len(profile["practice_log"]) >= 50:
                raise ValueError("This workflow has 50 saved practice entries. Export your notes to retain a local copy.")
            entry = {key: profile[key] for key in ("question", "answer", "notes")}
            entry["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            profile["practice_log"].append(entry)
            if not self._save_preparation():
                profile["practice_log"].pop()
                return False
            self.practice_count_var.set(f"{len(profile['practice_log'])} saved practice answers in this workflow")
            self._status("Practice answer saved locally with your question and notes.")
            return True
        except Exception as exc:
            self._status(str(exc))
            return False

    def _review(self, feedback=False):
        try:
            profile = self._flush()
            if not profile["question"]:
                raise ValueError("Enter or choose a question first.")
            draft = profile["question"]
            if feedback:
                if not profile["answer"]:
                    raise ValueError("Type an answer first so you can review a feedback request.")
                draft = (f"Review my {MODES[self.mode].lower()} answer. Give concise feedback on clarity, evidence and uncertainty. "
                         "Do not invent facts, personal experience or achievements.\n\n"
                         f"Question: {profile['question']}\n\nMy typed answer:\n{profile['answer']}")
            if len(draft) > 1200:
                raise ValueError("The review draft exceeds 1,200 characters. Shorten the question or typed answer explicitly; nothing was truncated or sent.")
            sources = review_sources(profile)
            context = {key: deepcopy(profile[key]) for key in ("topic", "role", "goal", "selected_ids")}
            context["mode"] = self.mode
            if self.on_review(draft, context, sources) is False:
                self._status("The review draft was not opened. Finish the active workflow or resolve the meeting workspace message, then try again.")
                return False
            self._status("Draft opened for a separate review. No answer request was sent from this window.")
            return True
        except Exception as exc:
            self._status(f"Review draft was not opened: {exc}")
            return False

    def _review_question(self):
        return self._review(False)

    def _review_answer(self):
        return self._review(True)

    def _notes(self):
        return notes_markdown(self.mode, self._flush())

    def _copy_notes(self):
        try:
            text = self._notes()
            self.window.clipboard_clear()
            self.window.clipboard_append(text)
            self._status("Local notes copied to the clipboard.")
        except Exception as exc:
            self._status(f"Notes could not be copied: {exc}")

    def _export_notes(self, path=None):
        try:
            text = self._notes()
            if path is None:
                path = filedialog.asksaveasfilename(parent=self.window, title="Export local preparation notes", defaultextension=".md",
                                                   initialfile=f"{self.mode}-notes.md", filetypes=[("Markdown notes", "*.md"), ("Text", "*.txt")])
            if not path:
                return False
            Path(path).write_text(text, encoding="utf-8")
            self._status(f"Notes exported locally to {Path(path).name}.")
            return True
        except Exception as exc:
            self._status(f"Notes were not exported: {exc}")
            return False

    def _send_notes(self):
        try:
            if self.on_notes(self.mode, self._notes()) is False:
                self._status("The meeting workspace did not accept these notes. Your preparation remains here locally.")
                return False
            self._status("Notes handed to the local meeting workspace. No message was sent.")
            return True
        except Exception as exc:
            self._status(f"Local notes could not be opened: {exc}")
            return False

    def show(self):
        if not self.closed:
            self.window.deiconify()
            self.window.lift()
            self.window.focus_set()

    def close(self):
        if self.closed:
            return True
        if not self._save_preparation():
            self.window.lift()
            if not messagebox.askyesno("Unsaved preparation",
                                       "Preparation could not be saved. Use Export notes to keep a local copy before closing.\n\n"
                                       "Close and discard these unsaved changes? The existing preparation file will not be overwritten.",
                                       parent=self.window, icon="warning", default="no"):
                return False
        self.closed = True
        self.window.destroy()
        return True
