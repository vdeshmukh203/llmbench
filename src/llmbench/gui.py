"""Tkinter graphical interface for llmbench.

Launch via ``llmbench gui`` or ``llmbench-gui``, or call :func:`main` directly.
Requires Python's stdlib ``tkinter`` (included with most Python distributions).
"""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import List, Optional

from .runner import BenchmarkRunner
from .spec import BenchmarkResult, BenchmarkSpec


def main() -> None:
    """Entry point: create and run the GUI event loop."""
    app = LLMBenchApp()
    app.mainloop()


class LLMBenchApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("llmbench — LLM Benchmark Runner")
        self.geometry("920x660")
        self.minsize(700, 520)
        self._runner: Optional[BenchmarkRunner] = None
        self._results: List[BenchmarkResult] = []
        self._sort_col: Optional[str] = None
        self._sort_reverse: bool = False
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_menu()
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._nb = nb

        self._tab_run = ttk.Frame(nb)
        self._tab_results = ttk.Frame(nb)
        self._tab_summary = ttk.Frame(nb)
        nb.add(self._tab_run, text=" Run ")
        nb.add(self._tab_results, text=" Results ")
        nb.add(self._tab_summary, text=" Summary ")

        self._build_run_tab()
        self._build_results_tab()
        self._build_summary_tab()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Export JSONL…", command=self._export_jsonl)
        file_menu.add_command(label="Export CSV…", command=self._export_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

    # ------ Run tab -------------------------------------------------------

    def _build_run_tab(self) -> None:
        f = self._tab_run
        f.columnconfigure(1, weight=1)

        pad = {"padx": 10, "pady": 5}

        # Mode
        ttk.Label(f, text="Mode:").grid(row=0, column=0, sticky=tk.W, **pad)
        self._mode = tk.StringVar(value="demo")
        mode_frame = ttk.Frame(f)
        mode_frame.grid(row=0, column=1, sticky=tk.W, **pad)
        ttk.Radiobutton(
            mode_frame, text="Demo  (no API key)", variable=self._mode,
            value="demo", command=self._on_mode_change,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            mode_frame, text="OpenAI API", variable=self._mode,
            value="openai", command=self._on_mode_change,
        ).pack(side=tk.LEFT, padx=12)

        # API key
        ttk.Label(f, text="API Key:").grid(row=1, column=0, sticky=tk.W, **pad)
        self._api_key_var = tk.StringVar()
        self._api_key_entry = ttk.Entry(f, textvariable=self._api_key_var, show="*", width=52)
        self._api_key_entry.grid(row=1, column=1, sticky=tk.EW, **pad)
        self._api_key_entry.config(state=tk.DISABLED)

        # Model
        ttk.Label(f, text="Model:").grid(row=2, column=0, sticky=tk.W, **pad)
        self._model_var = tk.StringVar(value="gpt-3.5-turbo")
        self._model_entry = ttk.Entry(f, textvariable=self._model_var, width=32)
        self._model_entry.grid(row=2, column=1, sticky=tk.W, **pad)
        self._model_entry.config(state=tk.DISABLED)

        # Task source
        ttk.Label(f, text="Tasks:").grid(row=3, column=0, sticky=tk.W, **pad)
        task_frame = ttk.Frame(f)
        task_frame.grid(row=3, column=1, sticky=tk.EW, **pad)
        task_frame.columnconfigure(0, weight=1)
        self._task_path_var = tk.StringVar(value="(built-in sample tasks)")
        ttk.Entry(task_frame, textvariable=self._task_path_var, state="readonly").grid(
            row=0, column=0, sticky=tk.EW
        )
        ttk.Button(task_frame, text="Browse…", command=self._browse_tasks).grid(
            row=0, column=1, padx=(6, 0)
        )
        ttk.Button(task_frame, text="Reset", command=self._reset_tasks).grid(
            row=0, column=2, padx=(4, 0)
        )

        # Run button
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)
        self._run_btn = ttk.Button(btn_frame, text="▶  Run Benchmark", command=self._run)
        self._run_btn.pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Clear Log", command=self._clear_log).pack(
            side=tk.LEFT, padx=6
        )

        # Progress bar
        ttk.Label(f, text="Progress:").grid(row=5, column=0, sticky=tk.W, padx=10)
        self._progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(f, variable=self._progress_var, maximum=100).grid(
            row=5, column=1, sticky=tk.EW, padx=10
        )
        self._progress_label = ttk.Label(f, text="")
        self._progress_label.grid(row=6, column=0, columnspan=2, pady=2)

        # Log
        ttk.Label(f, text="Log:").grid(row=7, column=0, sticky=tk.NW, padx=10, pady=(4, 0))
        self._log = scrolledtext.ScrolledText(
            f, height=13, state=tk.DISABLED, wrap=tk.WORD, font=("Courier", 10)
        )
        self._log.grid(row=7, column=1, sticky=tk.NSEW, padx=10, pady=4)
        f.rowconfigure(7, weight=1)

    # ------ Results tab ---------------------------------------------------

    def _build_results_tab(self) -> None:
        f = self._tab_results
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)

        cols = (
            "task_id", "category", "em", "em_norm",
            "rouge_l", "bleu_1", "f1", "composite", "latency_s", "tokens", "error",
        )
        headers = (
            "Task ID", "Category", "EM", "EM Norm",
            "ROUGE-L", "BLEU-1", "F1", "Composite", "Latency (s)", "Tokens", "Error",
        )
        self._results_tree = ttk.Treeview(f, columns=cols, show="headings", height=22)
        for col, hdr in zip(cols, headers):
            self._results_tree.heading(
                col, text=hdr, command=lambda c=col: self._sort_results(c)
            )
            width = 110 if col in ("task_id", "category") else 74
            if col == "error":
                width = 180
            self._results_tree.column(col, width=width, anchor=tk.CENTER, minwidth=50)
        self._results_tree.column("error", anchor=tk.W)

        vsb = ttk.Scrollbar(f, orient="vertical", command=self._results_tree.yview)
        hsb = ttk.Scrollbar(f, orient="horizontal", command=self._results_tree.xview)
        self._results_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._results_tree.grid(row=0, column=0, sticky=tk.NSEW)
        vsb.grid(row=0, column=1, sticky=tk.NS)
        hsb.grid(row=1, column=0, sticky=tk.EW)

        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=8)
        ttk.Button(btn_frame, text="Export JSONL…", command=self._export_jsonl).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(btn_frame, text="Export CSV…", command=self._export_csv).pack(
            side=tk.LEFT, padx=6
        )

    # ------ Summary tab ---------------------------------------------------

    def _build_summary_tab(self) -> None:
        f = self._tab_summary
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=1)

        ttk.Label(f, text="Benchmark Summary", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=10, pady=(10, 4)
        )
        self._summary_text = scrolledtext.ScrolledText(
            f, state=tk.DISABLED, wrap=tk.WORD, font=("Courier", 11)
        )
        self._summary_text.grid(row=1, column=0, sticky=tk.NSEW, padx=10, pady=(0, 10))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_mode_change(self) -> None:
        state = tk.NORMAL if self._mode.get() == "openai" else tk.DISABLED
        self._api_key_entry.config(state=state)
        self._model_entry.config(state=state)

    def _browse_tasks(self) -> None:
        path = filedialog.askopenfilename(
            title="Select JSONL task file",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
        if path:
            self._task_path_var.set(path)

    def _reset_tasks(self) -> None:
        self._task_path_var.set("(built-in sample tasks)")

    def _clear_log(self) -> None:
        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)

    def _log_append(self, text: str) -> None:
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, text + "\n")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Benchmark execution (runs in background thread)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        self._run_btn.config(state=tk.DISABLED)
        self._progress_var.set(0.0)
        self._progress_label.config(text="")
        self._clear_log()
        self._results = []

        task_path = self._task_path_var.get()
        if task_path == "(built-in sample tasks)":
            spec = BenchmarkSpec()
        else:
            try:
                spec = BenchmarkSpec.from_jsonl(task_path)
            except Exception as exc:
                messagebox.showerror("Task Load Error", f"Failed to load tasks:\n{exc}")
                self._run_btn.config(state=tk.NORMAL)
                return

        self._runner = BenchmarkRunner(spec)
        mode = self._mode.get()

        def _progress_cb(done: int, total: int, result: BenchmarkResult) -> None:
            pct = done / total * 100
            status = (
                f"[{done}/{total}] {result.task_id}: composite={result.composite_score:.4f}"
            )
            if result.error:
                status += f"  ERROR: {result.error}"
            self.after(0, lambda: self._progress_var.set(pct))
            self.after(0, lambda s=status: self._log_append(s))
            self.after(
                0,
                lambda d=done, t=total: self._progress_label.config(
                    text=f"{d} / {t} tasks complete"
                ),
            )

        def _worker() -> None:
            try:
                if mode == "openai":
                    api_key = self._api_key_var.get().strip()
                    model = self._model_var.get().strip()
                    if not api_key:
                        self.after(
                            0,
                            lambda: messagebox.showerror(
                                "Missing API Key", "An API key is required for OpenAI mode."
                            ),
                        )
                        return
                    res = self._runner.run_openai(
                        api_key=api_key,
                        model=model,
                        progress_callback=_progress_cb,
                    )
                else:
                    res = self._runner.run_offline(
                        lambda p: p.split("?")[0].strip() if "?" in p else p[:40],
                        progress_callback=_progress_cb,
                    )
                self._results = res
                self.after(0, self._on_run_complete)
            except Exception as exc:
                self.after(
                    0,
                    lambda: messagebox.showerror("Benchmark Error", f"Run failed:\n{exc}"),
                )
            finally:
                self.after(0, lambda: self._run_btn.config(state=tk.NORMAL))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_run_complete(self) -> None:
        self._log_append("\nBenchmark complete.")
        self._populate_results(self._results)
        if self._runner:
            self._populate_summary(self._runner.summarize())
        self._nb.select(self._tab_results)

    # ------------------------------------------------------------------
    # Results & Summary population
    # ------------------------------------------------------------------

    def _populate_results(self, results: List[BenchmarkResult]) -> None:
        self._results_tree.delete(*self._results_tree.get_children())
        for r in results:
            self._results_tree.insert(
                "", tk.END,
                values=(
                    r.task_id,
                    r.category,
                    f"{r.exact_match:.3f}",
                    f"{r.exact_match_norm:.3f}",
                    f"{r.rouge_l:.3f}",
                    f"{r.bleu_1:.3f}",
                    f"{r.f1:.3f}",
                    f"{r.composite_score:.3f}",
                    f"{r.latency_s:.3f}",
                    r.approx_tokens,
                    r.error or "",
                ),
            )

    def _sort_results(self, col: str) -> None:
        items = [
            (self._results_tree.set(iid, col), iid)
            for iid in self._results_tree.get_children("")
        ]
        reverse = (self._sort_col == col) and not self._sort_reverse
        try:
            items.sort(key=lambda x: float(x[0]), reverse=reverse)
        except ValueError:
            items.sort(key=lambda x: x[0], reverse=reverse)
        for idx, (_, iid) in enumerate(items):
            self._results_tree.move(iid, "", idx)
        self._sort_col = col
        self._sort_reverse = reverse

    def _populate_summary(self, summary: dict) -> None:
        self._summary_text.config(state=tk.NORMAL)
        self._summary_text.delete("1.0", tk.END)

        def _section(title: str, stats: dict) -> List[str]:
            return [
                "=" * 60,
                title,
                "=" * 60,
                f"  Tasks:              {stats.get('n', 0)}",
                f"  Errors:             {stats.get('errors', 0)}",
                f"  Exact Match:        {stats.get('exact_match', 0):.4f}",
                f"  Exact Match (norm): {stats.get('exact_match_norm', 0):.4f}",
                f"  ROUGE-L:            {stats.get('rouge_l', 0):.4f}",
                f"  BLEU-1:             {stats.get('bleu_1', 0):.4f}",
                f"  F1:                 {stats.get('f1', 0):.4f}",
                f"  Composite:          {stats.get('composite', 0):.4f}",
                f"  Avg Latency (s):    {stats.get('avg_latency_s', 0):.4f}",
                "",
            ]

        lines: List[str] = _section("OVERALL", summary.get("overall", {}))
        for cat, stats in summary.get("by_category", {}).items():
            lines.extend(_section(f"CATEGORY: {cat.upper()}", stats))

        self._summary_text.insert(tk.END, "\n".join(lines))
        self._summary_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_jsonl(self) -> None:
        if not self._results:
            messagebox.showinfo("No Results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl")],
            title="Export results as JSONL",
        )
        if path and self._runner:
            self._runner.export_jsonl(Path(path), self._results)
            messagebox.showinfo("Exported", f"Results saved to:\n{path}")

    def _export_csv(self) -> None:
        if not self._results:
            messagebox.showinfo("No Results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export results as CSV",
        )
        if path and self._runner:
            self._runner.export_csv(Path(path), self._results)
            messagebox.showinfo("Exported", f"Results saved to:\n{path}")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _show_about(self) -> None:
        from . import __version__
        messagebox.showinfo(
            "About llmbench",
            f"llmbench v{__version__}\n\n"
            "Lightweight reproducible LLM benchmarking.\n\n"
            "Metrics: ROUGE-L · BLEU-1 · F1 · Exact Match\n"
            "Composite = 0.40·ROUGE-L + 0.30·F1 + 0.20·EM-norm + 0.10·BLEU-1",
        )
