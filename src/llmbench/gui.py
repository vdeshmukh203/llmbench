"""Tkinter GUI for llmbench.

Launch via ``llmbench gui`` or programmatically::

    from llmbench.gui import launch
    launch()

The GUI requires only Python's stdlib ``tkinter`` module.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import List, Optional

from .spec import BenchmarkSpec, SAMPLE_TASKS
from .runner import BenchmarkResult, BenchmarkRunner


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
_BG = "#1e1e2e"
_FG = "#cdd6f4"
_ACCENT = "#89b4fa"
_GREEN = "#a6e3a1"
_RED = "#f38ba8"
_YELLOW = "#f9e2af"
_SURFACE = "#313244"
_SURFACE2 = "#45475a"
_PANEL = "#181825"
_FONT = ("Helvetica", 10)
_MONO = ("Courier", 9)


def _score_colour(value: float) -> str:
    if value >= 0.75:
        return _GREEN
    if value >= 0.40:
        return _YELLOW
    return _RED


class LLMBenchApp:
    """Main application window."""

    def __init__(self, root) -> None:
        import tkinter as tk
        import tkinter.ttk as ttk
        import tkinter.filedialog as fd
        import tkinter.messagebox as mb
        import tkinter.scrolledtext as st

        self._tk = tk
        self._ttk = ttk
        self._fd = fd
        self._mb = mb
        self._st = st

        self.root = root
        self.root.title("llmbench — LLM Benchmark Runner")
        self.root.configure(bg=_BG)
        try:
            self.root.geometry("1100x720")
        except Exception:
            pass

        self._spec = BenchmarkSpec()
        self._results: List[BenchmarkResult] = []
        self._running = False

        self._build_styles(ttk)
        self._build_menu()
        self._build_layout()
        self._populate_task_tree()

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def _build_styles(self, ttk) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=_BG, foreground=_FG, font=_FONT)
        style.configure("TFrame", background=_BG)
        style.configure("TLabel", background=_BG, foreground=_FG, font=_FONT)
        style.configure("TButton", background=_SURFACE, foreground=_FG, font=_FONT, relief="flat", padding=4)
        style.map("TButton", background=[("active", _ACCENT), ("disabled", _SURFACE2)])
        style.configure(
            "Treeview",
            background=_SURFACE,
            foreground=_FG,
            fieldbackground=_SURFACE,
            rowheight=22,
            font=_FONT,
        )
        style.configure("Treeview.Heading", background=_SURFACE2, foreground=_ACCENT, font=_FONT)
        style.map("Treeview", background=[("selected", _ACCENT)], foreground=[("selected", _PANEL)])
        style.configure("TNotebook", background=_BG, tabmargins=[2, 5, 2, 0])
        style.configure("TNotebook.Tab", background=_SURFACE, foreground=_FG, padding=[10, 4])
        style.map("TNotebook.Tab", background=[("selected", _ACCENT)], foreground=[("selected", _PANEL)])
        style.configure("TProgressbar", troughcolor=_SURFACE, background=_ACCENT, thickness=8)
        style.configure("TEntry", fieldbackground=_SURFACE, foreground=_FG, insertcolor=_FG)
        style.configure("TCombobox", fieldbackground=_SURFACE, foreground=_FG)
        style.configure("TLabelframe", background=_BG, foreground=_ACCENT)
        style.configure("TLabelframe.Label", background=_BG, foreground=_ACCENT, font=_FONT)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        tk = self._tk
        mb_widget = tk.Menu(self.root, bg=_SURFACE, fg=_FG, tearoff=False)
        self.root.config(menu=mb_widget)

        file_menu = tk.Menu(mb_widget, bg=_SURFACE, fg=_FG, tearoff=False)
        mb_widget.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Task File…", command=self._open_task_file)
        file_menu.add_separator()
        file_menu.add_command(label="Export CSV…", command=self._export_csv)
        file_menu.add_command(label="Export JSONL…", command=self._export_jsonl)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit)

        run_menu = tk.Menu(mb_widget, bg=_SURFACE, fg=_FG, tearoff=False)
        mb_widget.add_cascade(label="Run", menu=run_menu)
        run_menu.add_command(label="Run Offline (echo mock)", command=self._run_offline_echo)
        run_menu.add_command(label="Run Offline (word-count mock)", command=self._run_offline_wc)
        run_menu.add_separator()
        run_menu.add_command(label="Run OpenAI API…", command=self._show_api_dialog)

        help_menu = tk.Menu(mb_widget, bg=_SURFACE, fg=_FG, tearoff=False)
        mb_widget.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        tk = self._tk
        ttk = self._ttk

        # Top toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=6, pady=(6, 0))

        ttk.Button(toolbar, text="Open Tasks", command=self._open_task_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="▶  Run Offline", command=self._run_offline_echo).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="▶  Run OpenAI", command=self._show_api_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Export CSV", command=self._export_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Export JSONL", command=self._export_jsonl).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Clear Results", command=self._clear_results).pack(side=tk.LEFT, padx=2)

        # Task counter label on the right
        self._task_label = ttk.Label(toolbar, text="")
        self._task_label.pack(side=tk.RIGHT, padx=6)
        self._update_task_label()

        # Progress bar
        self._progress = ttk.Progressbar(self.root, mode="determinate", style="TProgressbar")
        self._progress.pack(fill=tk.X, padx=6, pady=(4, 0))
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self._status_var).pack(anchor=tk.W, padx=8)

        # Main paned window (left=tasks, right=detail+results)
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # --- Left: task browser ---
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        ttk.Label(left, text="Tasks", foreground=_ACCENT).pack(anchor=tk.W)
        self._task_tree = ttk.Treeview(
            left,
            columns=("id", "cat", "prompt"),
            show="headings",
            selectmode="extended",
        )
        self._task_tree.heading("id", text="ID")
        self._task_tree.heading("cat", text="Category")
        self._task_tree.heading("prompt", text="Prompt (preview)")
        self._task_tree.column("id", width=80, stretch=False)
        self._task_tree.column("cat", width=90, stretch=False)
        self._task_tree.column("prompt", width=180)
        self._task_tree.pack(fill=tk.BOTH, expand=True)
        _add_scrollbar(tk, ttk, left, self._task_tree)

        # --- Right: notebook ---
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        nb = ttk.Notebook(right)
        nb.pack(fill=tk.BOTH, expand=True)

        # Results tab
        results_frame = ttk.Frame(nb)
        nb.add(results_frame, text="Results")
        self._result_tree = ttk.Treeview(
            results_frame,
            columns=("id", "cat", "em", "rouge", "bleu", "f1", "comp", "lat", "err"),
            show="headings",
            selectmode="browse",
        )
        cols = [
            ("id", "Task ID", 85),
            ("cat", "Category", 90),
            ("em", "EM", 52),
            ("rouge", "ROUGE-L", 68),
            ("bleu", "BLEU-1", 60),
            ("f1", "F1", 52),
            ("comp", "Composite", 78),
            ("lat", "Latency (s)", 82),
            ("err", "Error", 55),
        ]
        for key, heading, width in cols:
            self._result_tree.heading(key, text=heading)
            self._result_tree.column(key, width=width, anchor=tk.CENTER, stretch=(key == "id"))
        self._result_tree.pack(fill=tk.BOTH, expand=True)
        _add_scrollbar(tk, ttk, results_frame, self._result_tree)
        self._result_tree.bind("<<TreeviewSelect>>", self._on_result_select)

        # Detail tab
        detail_frame = ttk.Frame(nb)
        nb.add(detail_frame, text="Detail")
        self._detail_text = self._st.ScrolledText(
            detail_frame, bg=_SURFACE, fg=_FG, font=_MONO, state="disabled", wrap=tk.WORD
        )
        self._detail_text.pack(fill=tk.BOTH, expand=True)

        # Summary tab
        summary_frame = ttk.Frame(nb)
        nb.add(summary_frame, text="Summary")
        self._summary_text = self._st.ScrolledText(
            summary_frame, bg=_SURFACE, fg=_FG, font=_MONO, state="disabled", wrap=tk.WORD
        )
        self._summary_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Task tree
    # ------------------------------------------------------------------

    def _populate_task_tree(self) -> None:
        self._task_tree.delete(*self._task_tree.get_children())
        for t in self._spec.tasks:
            preview = t.prompt[:70].replace("\n", " ")
            self._task_tree.insert("", "end", values=(t.task_id, t.category, preview))
        self._update_task_label()

    def _update_task_label(self) -> None:
        n = len(self._spec)
        self._task_label.config(text=f"{n} task{'s' if n != 1 else ''} loaded")

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _open_task_file(self) -> None:
        path = self._fd.askopenfilename(
            title="Open Task File",
            filetypes=[
                ("All supported", "*.json *.jsonl *.yaml *.yml"),
                ("JSON", "*.json"),
                ("JSONL", "*.jsonl"),
                ("YAML", "*.yaml *.yml"),
                ("All files", "*"),
            ],
        )
        if not path:
            return
        try:
            self._spec = BenchmarkSpec.from_file(path)
            self._populate_task_tree()
            self._set_status(f"Loaded {len(self._spec)} tasks from {Path(path).name}")
        except Exception as exc:
            self._mb.showerror("Load Error", str(exc))

    def _export_csv(self) -> None:
        if not self._results:
            self._mb.showinfo("Nothing to export", "Run a benchmark first.")
            return
        path = self._fd.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if not path:
            return
        runner = BenchmarkRunner()
        runner.results = self._results
        runner.export_csv(path)
        self._set_status(f"Exported CSV → {path}")

    def _export_jsonl(self) -> None:
        if not self._results:
            self._mb.showinfo("Nothing to export", "Run a benchmark first.")
            return
        path = self._fd.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL", "*.jsonl"), ("All files", "*")],
        )
        if not path:
            return
        runner = BenchmarkRunner()
        runner.results = self._results
        runner.export_jsonl(path)
        self._set_status(f"Exported JSONL → {path}")

    # ------------------------------------------------------------------
    # Run offline
    # ------------------------------------------------------------------

    def _run_offline_echo(self) -> None:
        self._run_async(lambda p: p.split("?")[0].strip() if "?" in p else p[:40], "Echo mock")

    def _run_offline_wc(self) -> None:
        def _wc(p: str) -> str:
            words = p.split()
            return f"{len(words)} words"

        self._run_async(_wc, "Word-count mock")

    def _run_async(self, model_fn, label: str) -> None:
        if self._running:
            self._mb.showwarning("Busy", "A benchmark is already running.")
            return
        self._running = True
        self._set_status(f"Running {label}…")
        self._progress["maximum"] = len(self._spec)
        self._progress["value"] = 0

        def worker() -> None:
            runner = BenchmarkRunner(self._spec)
            results: List[BenchmarkResult] = []
            for i, task in enumerate(self._spec.tasks, 1):
                batch = runner.run_offline(model_fn, [task])
                results.extend(batch)
                self.root.after(0, lambda v=i: self._progress.configure(value=v))
            self.root.after(0, lambda: self._on_run_complete(results, runner))

        threading.Thread(target=worker, daemon=True).start()

    def _on_run_complete(self, results: List[BenchmarkResult], runner: BenchmarkRunner) -> None:
        self._results.extend(results)
        self._running = False
        self._populate_result_tree()
        self._update_summary(runner)
        self._set_status(f"Done — {len(results)} tasks scored.")

    # ------------------------------------------------------------------
    # OpenAI dialog
    # ------------------------------------------------------------------

    def _show_api_dialog(self) -> None:
        tk = self._tk
        ttk = self._ttk

        dlg = tk.Toplevel(self.root)
        dlg.title("Run OpenAI API")
        dlg.configure(bg=_BG)
        dlg.resizable(False, False)
        dlg.grab_set()

        def _row(label, default=""):
            f = ttk.Frame(dlg)
            f.pack(fill=tk.X, padx=12, pady=4)
            ttk.Label(f, text=label, width=14).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            ttk.Entry(f, textvariable=var, width=36).pack(side=tk.LEFT)
            return var

        api_var = _row("API Key", os.environ.get("OPENAI_API_KEY", ""))
        model_var = _row("Model", "gpt-3.5-turbo")
        temp_var = _row("Temperature", "0.0")
        tokens_var = _row("Max Tokens", "256")

        def _start() -> None:
            key = api_var.get().strip()
            if not key:
                self._mb.showerror("Missing API Key", "Please enter an API key.", parent=dlg)
                return
            model = model_var.get().strip() or "gpt-3.5-turbo"
            try:
                temp = float(temp_var.get())
                max_tok = int(tokens_var.get())
            except ValueError:
                self._mb.showerror("Invalid input", "Temperature must be a float, max_tokens an int.", parent=dlg)
                return
            dlg.destroy()
            self._run_openai_async(key, model, temp, max_tok)

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(pady=8)
        ttk.Button(btn_frame, text="Run", command=_start).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=4)

    def _run_openai_async(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        if self._running:
            return
        self._running = True
        self._progress["maximum"] = len(self._spec)
        self._progress["value"] = 0
        self._set_status(f"Querying {model}…")

        def worker() -> None:
            runner = BenchmarkRunner(self._spec)
            results: List[BenchmarkResult] = []
            for i, task in enumerate(self._spec.tasks, 1):
                batch = runner.run_openai(
                    api_key=api_key, model=model,
                    tasks=[task], temperature=temperature, max_tokens=max_tokens,
                )
                results.extend(batch)
                self.root.after(0, lambda v=i: self._progress.configure(value=v))
            self.root.after(0, lambda: self._on_run_complete(results, runner))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Results display
    # ------------------------------------------------------------------

    def _clear_results(self) -> None:
        self._results = []
        self._result_tree.delete(*self._result_tree.get_children())
        self._set_detail("")
        self._set_summary_text("")
        self._progress["value"] = 0
        self._set_status("Results cleared.")

    def _populate_result_tree(self) -> None:
        self._result_tree.delete(*self._result_tree.get_children())
        for r in self._results:
            has_err = "yes" if r.error else ""
            self._result_tree.insert(
                "",
                "end",
                iid=r.task_id,
                values=(
                    r.task_id,
                    r.category,
                    f"{r.exact_match:.2f}",
                    f"{r.rouge_l:.3f}",
                    f"{r.bleu_1:.3f}",
                    f"{r.f1:.3f}",
                    f"{r.composite_score:.3f}",
                    f"{r.latency_s:.3f}",
                    has_err,
                ),
            )
            fg = _score_colour(r.composite_score)
            self._result_tree.tag_configure(r.task_id, foreground=fg)
            self._result_tree.item(r.task_id, tags=(r.task_id,))

    def _on_result_select(self, _event) -> None:
        sel = self._result_tree.selection()
        if not sel:
            return
        task_id = sel[0]
        result = next((r for r in self._results if r.task_id == task_id), None)
        if result is None:
            return
        lines = [
            f"Task ID   : {result.task_id}",
            f"Category  : {result.category}",
            f"",
            f"--- Prompt ---",
            result.prompt,
            f"",
            f"--- Reference ---",
            result.reference,
            f"",
            f"--- Prediction ---",
            result.prediction or "(empty)",
            f"",
            f"--- Scores ---",
            f"  Exact Match       : {result.exact_match:.4f}",
            f"  Exact Match Norm  : {result.exact_match_norm:.4f}",
            f"  ROUGE-L           : {result.rouge_l:.4f}",
            f"  BLEU-1            : {result.bleu_1:.4f}",
            f"  F1                : {result.f1:.4f}",
            f"  Composite         : {result.composite_score:.4f}",
            f"  Latency (s)       : {result.latency_s:.4f}",
            f"  Approx tokens     : {result.approx_tokens}",
        ]
        if result.error:
            lines += ["", f"--- Error ---", result.error]
        self._set_detail("\n".join(lines))

    def _set_detail(self, text: str) -> None:
        widget = self._detail_text
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.config(state="disabled")

    def _update_summary(self, runner: BenchmarkRunner) -> None:
        summary = runner.summarize(self._results)
        self._set_summary_text(json.dumps(summary, indent=2))

    def _set_summary_text(self, text: str) -> None:
        widget = self._summary_text
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.config(state="disabled")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)

    def _show_about(self) -> None:
        from . import __version__
        self._mb.showinfo(
            "About llmbench",
            f"llmbench {__version__}\n\n"
            "Lightweight reproducible LLM benchmarking framework.\n\n"
            "Metrics: ROUGE-L, BLEU-1, F1, Exact Match\n"
            "Stdlib-only — no external dependencies required.\n\n"
            "https://github.com/vdeshmukh203/llmbench",
        )


# ---------------------------------------------------------------------------
# Scrollbar helper
# ---------------------------------------------------------------------------

def _add_scrollbar(tk, ttk, parent, tree) -> None:
    sb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=sb.set)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def launch() -> None:
    """Create and run the Tkinter main loop.

    Raises
    ------
    ImportError
        If tkinter is not available in the current Python installation.
    """
    try:
        import tkinter as tk
    except ImportError as exc:
        raise ImportError(
            "tkinter is required for the GUI. On Debian/Ubuntu: sudo apt install python3-tk"
        ) from exc

    root = tk.Tk()
    LLMBenchApp(root)
    root.mainloop()
