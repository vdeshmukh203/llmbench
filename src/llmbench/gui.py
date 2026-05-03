"""Tkinter GUI for llmbench — interactive benchmark runner and result viewer.

Launch via:
    llmbench gui
    python -m llmbench.gui
"""

from __future__ import annotations

import json
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from .runner import BenchmarkResult, BenchmarkRunner
from .spec import SAMPLE_TASKS, BenchmarkSpec, Task


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------


class _LabeledText(ttk.LabelFrame):
    """A LabelFrame containing a scrollable, read-only Text widget."""

    def __init__(self, parent: tk.Widget, label: str, height: int = 6, **kw: object) -> None:
        super().__init__(parent, text=label, **kw)
        self.text = tk.Text(self, wrap=tk.WORD, height=height, font=("Courier", 9), state=tk.DISABLED)
        vsb = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=vsb.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def set_text(self, value: str) -> None:
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, value)
        self.text.config(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------


class LLMBenchGUI(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("llmbench — LLM Benchmark Runner")
        self.geometry("960x680")
        self.minsize(720, 520)

        self.results: List[BenchmarkResult] = []
        self._stop_flag = False
        self._key_visible = False

        self._build_menu()
        self._build_notebook()
        self._build_status_bar()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

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

    def _build_notebook(self) -> None:
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 2))

        self._run_tab = ttk.Frame(self._nb)
        self._results_tab = ttk.Frame(self._nb)
        self._detail_tab = ttk.Frame(self._nb)

        self._nb.add(self._run_tab, text="  Configure & Run  ")
        self._nb.add(self._results_tab, text="  Results  ")
        self._nb.add(self._detail_tab, text="  Task Detail  ")

        self._build_run_tab()
        self._build_results_tab()
        self._build_detail_tab()

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(
            self,
            textvariable=self._status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(4, 1),
        ).pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=(0, 4))

    # ------------------------------------------------------------------
    # Configure & Run tab
    # ------------------------------------------------------------------

    def _build_run_tab(self) -> None:
        frame = self._run_tab
        frame.columnconfigure(1, weight=1)

        row = 0
        # Mode selection
        ttk.Label(frame, text="Mode", font=("", 10, "bold")).grid(
            row=row, column=0, sticky=tk.W, padx=12, pady=(14, 2)
        )
        self._mode = tk.StringVar(value="demo")
        ttk.Radiobutton(
            frame,
            text="Demo — offline, no API key required",
            variable=self._mode,
            value="demo",
            command=self._on_mode_change,
        ).grid(row=row, column=1, columnspan=2, sticky=tk.W)

        row += 1
        ttk.Radiobutton(
            frame,
            text="OpenAI API (or compatible endpoint)",
            variable=self._mode,
            value="openai",
            command=self._on_mode_change,
        ).grid(row=row, column=1, columnspan=2, sticky=tk.W)

        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(
            row=row, columnspan=3, sticky=tk.EW, padx=8, pady=8
        )

        # API Key
        row += 1
        ttk.Label(frame, text="API key:").grid(row=row, column=0, sticky=tk.W, padx=12, pady=3)
        self._api_key = tk.StringVar(value=os.environ.get("OPENAI_API_KEY", ""))
        self._api_key_entry = ttk.Entry(frame, textvariable=self._api_key, show="*", width=50)
        self._api_key_entry.grid(row=row, column=1, sticky=tk.EW, padx=4)
        self._show_key_btn = ttk.Button(
            frame, text="Show", command=self._toggle_key_visibility, width=6
        )
        self._show_key_btn.grid(row=row, column=2, padx=(0, 12))

        # Model
        row += 1
        ttk.Label(frame, text="Model:").grid(row=row, column=0, sticky=tk.W, padx=12, pady=3)
        self._model = tk.StringVar(value="gpt-3.5-turbo")
        self._model_entry = ttk.Entry(frame, textvariable=self._model, width=34)
        self._model_entry.grid(row=row, column=1, sticky=tk.W, padx=4)

        # Max tokens
        row += 1
        ttk.Label(frame, text="Max tokens:").grid(row=row, column=0, sticky=tk.W, padx=12, pady=3)
        self._max_tokens = tk.IntVar(value=256)
        ttk.Spinbox(
            frame, textvariable=self._max_tokens, from_=16, to=4096, increment=64, width=10
        ).grid(row=row, column=1, sticky=tk.W, padx=4)

        # Temperature
        row += 1
        ttk.Label(frame, text="Temperature:").grid(row=row, column=0, sticky=tk.W, padx=12, pady=3)
        self._temperature = tk.DoubleVar(value=0.0)
        temp_frame = ttk.Frame(frame)
        temp_frame.grid(row=row, column=1, sticky=tk.W)
        self._temp_val_lbl = ttk.Label(temp_frame, text="0.00", width=5)
        self._temp_val_lbl.pack(side=tk.LEFT)
        ttk.Scale(
            temp_frame,
            from_=0.0,
            to=2.0,
            variable=self._temperature,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda v: self._temp_val_lbl.config(text=f"{float(v):.2f}"),
        ).pack(side=tk.LEFT, padx=4)

        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(
            row=row, columnspan=3, sticky=tk.EW, padx=8, pady=8
        )

        # Task source
        row += 1
        ttk.Label(frame, text="Task set", font=("", 10, "bold")).grid(
            row=row, column=0, sticky=tk.W, padx=12, pady=3
        )
        self._task_source = tk.StringVar(value="sample")
        ttk.Radiobutton(
            frame,
            text="Built-in sample tasks (10 tasks)",
            variable=self._task_source,
            value="sample",
        ).grid(row=row, column=1, columnspan=2, sticky=tk.W)

        row += 1
        ttk.Radiobutton(
            frame, text="Custom JSONL file:", variable=self._task_source, value="file"
        ).grid(row=row, column=1, sticky=tk.W)

        row += 1
        self._task_file = tk.StringVar()
        self._task_file_entry = ttk.Entry(frame, textvariable=self._task_file, width=44)
        self._task_file_entry.grid(row=row, column=1, sticky=tk.EW, padx=4)
        ttk.Button(frame, text="Browse…", command=self._browse_tasks).grid(
            row=row, column=2, padx=(0, 12)
        )

        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(
            row=row, columnspan=3, sticky=tk.EW, padx=8, pady=8
        )

        # Run / Stop buttons
        row += 1
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, padx=12, pady=4, sticky=tk.W)
        self._run_btn = ttk.Button(
            btn_frame, text="▶  Run Benchmark", command=self._run_benchmark, width=22
        )
        self._run_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._stop_btn = ttk.Button(
            btn_frame, text="■  Stop", command=self._stop_benchmark, width=10, state=tk.DISABLED
        )
        self._stop_btn.pack(side=tk.LEFT)

        # Progress bar
        row += 1
        self._progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, mode="determinate")
        self._progress.grid(row=row, columnspan=3, sticky=tk.EW, padx=12, pady=4)

        # Summary JSON output
        row += 1
        self._summary_box = _LabeledText(frame, "Summary (JSON)", height=7)
        self._summary_box.grid(row=row, columnspan=3, sticky=tk.NSEW, padx=12, pady=(4, 8))
        frame.rowconfigure(row, weight=1)

        self._on_mode_change()

    # ------------------------------------------------------------------
    # Results tab
    # ------------------------------------------------------------------

    def _build_results_tab(self) -> None:
        frame = self._results_tab
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # Overall summary row
        stats_frame = ttk.LabelFrame(frame, text="Overall Summary")
        stats_frame.grid(row=0, column=0, sticky=tk.EW, padx=8, pady=6)
        self._stat_vars: Dict[str, tk.StringVar] = {}
        metrics = [
            ("n", "Tasks"),
            ("exact_match", "Exact Match"),
            ("exact_match_norm", "EM (norm)"),
            ("rouge_l", "ROUGE-L"),
            ("bleu_1", "BLEU-1"),
            ("f1", "F1"),
            ("composite", "Composite"),
            ("avg_latency_s", "Avg Latency"),
        ]
        for i, (key, label) in enumerate(metrics):
            col = (i % 4) * 2
            row_idx = i // 4
            ttk.Label(stats_frame, text=label + ":").grid(
                row=row_idx, column=col, sticky=tk.W, padx=(8, 2), pady=3
            )
            var = tk.StringVar(value="—")
            ttk.Label(stats_frame, textvariable=var, font=("", 9, "bold"), width=10).grid(
                row=row_idx, column=col + 1, sticky=tk.W, padx=(0, 12), pady=3
            )
            self._stat_vars[key] = var

        # Results treeview
        tree_frame = ttk.Frame(frame)
        tree_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=8, pady=4)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        cols = (
            "task_id", "category", "composite",
            "rouge_l", "f1", "exact_match_norm",
            "bleu_1", "latency_s", "error",
        )
        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")

        col_cfg = {
            "task_id": ("Task ID", 110, tk.W),
            "category": ("Category", 90, tk.W),
            "composite": ("Composite", 80, tk.CENTER),
            "rouge_l": ("ROUGE-L", 72, tk.CENTER),
            "f1": ("F1", 60, tk.CENTER),
            "exact_match_norm": ("EM (norm)", 72, tk.CENTER),
            "bleu_1": ("BLEU-1", 62, tk.CENTER),
            "latency_s": ("Latency (s)", 80, tk.CENTER),
            "error": ("Error", 140, tk.W),
        }
        for col in cols:
            text, width, anchor = col_cfg[col]
            self._tree.heading(
                col, text=text, command=lambda c=col: self._sort_tree(c)
            )
            self._tree.column(col, width=width, anchor=anchor, minwidth=40)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky=tk.NSEW)
        vsb.grid(row=0, column=1, sticky=tk.NS)
        hsb.grid(row=1, column=0, sticky=tk.EW)

        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Color-coding tags
        self._tree.tag_configure("good", background="#d4edda")
        self._tree.tag_configure("ok", background="#fff3cd")
        self._tree.tag_configure("bad", background="#f8d7da")
        self._tree.tag_configure("err", background="#f5c6cb")

        self._sort_col: Optional[str] = None
        self._sort_reverse = False

        # Export buttons
        export_frame = ttk.Frame(frame)
        export_frame.grid(row=2, column=0, sticky=tk.EW, padx=8, pady=4)
        ttk.Button(export_frame, text="Export JSONL…", command=self._export_jsonl).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(export_frame, text="Export CSV…", command=self._export_csv).pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Task Detail tab
    # ------------------------------------------------------------------

    def _build_detail_tab(self) -> None:
        frame = self._detail_tab
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(
            frame,
            text="Select a row in the Results tab to view full prompt / reference / prediction.",
            font=("", 9, "italic"),
        ).grid(row=0, column=0, sticky=tk.W, padx=12, pady=6)

        pw = ttk.PanedWindow(frame, orient=tk.VERTICAL)
        pw.grid(row=1, column=0, sticky=tk.NSEW, padx=8, pady=4)

        self._detail_prompt = _LabeledText(pw, "Prompt", height=5)
        self._detail_reference = _LabeledText(pw, "Reference", height=4)
        self._detail_prediction = _LabeledText(pw, "Prediction", height=5)
        pw.add(self._detail_prompt, weight=1)
        pw.add(self._detail_reference, weight=1)
        pw.add(self._detail_prediction, weight=1)

        self._detail_scores = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self._detail_scores, font=("Courier", 9)).grid(
            row=2, column=0, sticky=tk.W, padx=12, pady=(2, 6)
        )

    # ------------------------------------------------------------------
    # Event handlers — configuration tab
    # ------------------------------------------------------------------

    def _on_mode_change(self) -> None:
        openai = self._mode.get() == "openai"
        state = tk.NORMAL if openai else tk.DISABLED
        self._api_key_entry.config(state=state)
        self._show_key_btn.config(state=state)
        self._model_entry.config(state=state)

    def _toggle_key_visibility(self) -> None:
        self._key_visible = not self._key_visible
        self._api_key_entry.config(show="" if self._key_visible else "*")
        self._show_key_btn.config(text="Hide" if self._key_visible else "Show")

    def _browse_tasks(self) -> None:
        path = filedialog.askopenfilename(
            title="Select task JSONL file",
            filetypes=[("JSONL files", "*.jsonl"), ("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self._task_file.set(path)
            self._task_source.set("file")

    def _stop_benchmark(self) -> None:
        self._stop_flag = True
        self._status_var.set("Stopping after current task…")

    def _run_benchmark(self) -> None:
        self._stop_flag = False

        if self._mode.get() == "openai" and not self._api_key.get().strip():
            messagebox.showerror("Missing API key", "Please enter your OpenAI API key.")
            return

        # Load tasks
        try:
            if self._task_source.get() == "file":
                path_str = self._task_file.get().strip()
                if not path_str:
                    messagebox.showerror("No file", "Please select a JSONL task file.")
                    return
                spec = BenchmarkSpec.from_jsonl(Path(path_str))
            else:
                spec = BenchmarkSpec.from_sample()
            tasks = spec.tasks
        except Exception as exc:
            messagebox.showerror("Task load error", str(exc))
            return

        self._run_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._progress.configure(maximum=len(tasks), value=0)
        self._status_var.set(f"Running {len(tasks)} task(s)…")
        self._summary_box.set_text("")
        self.results = []

        def _worker() -> None:
            runner = BenchmarkRunner(tasks)
            for i, task in enumerate(tasks):
                if self._stop_flag:
                    break
                self.after(
                    0,
                    lambda i=i, t=task: self._status_var.set(
                        f"Task {i + 1}/{len(tasks)}: {t.task_id}"
                    ),
                )
                if self._mode.get() == "demo":
                    batch = runner.run_offline(
                        lambda p: p.split("?")[0].strip() if "?" in p else p[:40],
                        [task],
                    )
                else:
                    batch = runner.run_openai(
                        api_key=self._api_key.get().strip(),
                        model=self._model.get().strip() or "gpt-3.5-turbo",
                        tasks=[task],
                        max_tokens=self._max_tokens.get(),
                        temperature=self._temperature.get(),
                    )
                self.results.extend(batch)
                self.after(0, lambda v=i + 1: self._progress.configure(value=v))

            self.after(0, self._on_run_complete)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_run_complete(self) -> None:
        self._run_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        if not self.results:
            self._status_var.set("No results produced.")
            return
        runner = BenchmarkRunner([])
        runner.results = self.results
        summary = runner.summarize()
        self._refresh_results(summary)
        n = len(self.results)
        self._status_var.set(
            f"Completed {n} task(s). "
            f"Composite: {summary.get('overall', {}).get('composite', 0):.4f}"
        )

    # ------------------------------------------------------------------
    # Results population
    # ------------------------------------------------------------------

    def _refresh_results(self, summary: dict) -> None:
        # Summary box
        self._summary_box.set_text(json.dumps(summary, indent=2))

        # Stat labels
        ov = summary.get("overall", {})
        for key, var in self._stat_vars.items():
            val = ov.get(key, "—")
            var.set(f"{val:.4f}" if isinstance(val, float) else str(val))

        # Tree
        for iid in self._tree.get_children():
            self._tree.delete(iid)

        for r in self.results:
            if r.error:
                tag = "err"
            elif r.composite_score >= 0.7:
                tag = "good"
            elif r.composite_score >= 0.35:
                tag = "ok"
            else:
                tag = "bad"

            self._tree.insert(
                "",
                tk.END,
                iid=r.task_id,
                values=(
                    r.task_id,
                    r.category,
                    f"{r.composite_score:.3f}",
                    f"{r.rouge_l:.3f}",
                    f"{r.f1:.3f}",
                    f"{r.exact_match_norm:.3f}",
                    f"{r.bleu_1:.3f}",
                    f"{r.latency_s:.3f}",
                    r.error or "",
                ),
                tags=(tag,),
            )

    # ------------------------------------------------------------------
    # Event handlers — results tab
    # ------------------------------------------------------------------

    def _on_tree_select(self, _event: object = None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        task_id = sel[0]
        result = next((r for r in self.results if r.task_id == task_id), None)
        if result is None:
            return

        self._detail_prompt.set_text(result.prompt)
        self._detail_reference.set_text(result.reference)
        self._detail_prediction.set_text(result.prediction or "(no prediction)")

        score_line = (
            f"Composite: {result.composite_score:.4f}"
            f"  |  ROUGE-L: {result.rouge_l:.4f}"
            f"  |  F1: {result.f1:.4f}"
            f"  |  EM-norm: {result.exact_match_norm:.4f}"
            f"  |  BLEU-1: {result.bleu_1:.4f}"
            f"  |  Latency: {result.latency_s:.3f} s"
            f"  |  Tokens: {result.approx_tokens}"
        )
        if result.error:
            score_line += f"  |  Error: {result.error}"
        self._detail_scores.set(score_line)

        self._nb.select(self._detail_tab)

    def _sort_tree(self, col: str) -> None:
        rows = [(self._tree.set(iid, col), iid) for iid in self._tree.get_children("")]
        try:
            rows.sort(key=lambda x: float(x[0]), reverse=self._sort_reverse)
        except ValueError:
            rows.sort(key=lambda x: x[0].lower(), reverse=self._sort_reverse)
        for idx, (_, iid) in enumerate(rows):
            self._tree.move(iid, "", idx)
        self._sort_reverse = not self._sort_reverse

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_jsonl(self) -> None:
        if not self.results:
            messagebox.showinfo("No results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL", "*.jsonl"), ("All files", "*.*")],
            initialfile="results.jsonl",
        )
        if path:
            runner = BenchmarkRunner([])
            runner.results = self.results
            runner.export_jsonl(Path(path))
            self._status_var.set(f"Exported JSONL → {path}")

    def _export_csv(self) -> None:
        if not self.results:
            messagebox.showinfo("No results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile="results.csv",
        )
        if path:
            runner = BenchmarkRunner([])
            runner.results = self.results
            runner.export_csv(Path(path))
            self._status_var.set(f"Exported CSV → {path}")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About llmbench",
            "llmbench v0.1.0\n\n"
            "Lightweight framework for reproducible LLM benchmarking.\n\n"
            "Metrics: ROUGE-L · BLEU-1 · Token-F1 · Exact Match\n"
            "Export: JSONL · CSV\n\n"
            "Author: Vaibhav Deshmukh\n"
            "License: MIT",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = LLMBenchGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
