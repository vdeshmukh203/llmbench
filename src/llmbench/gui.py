"""Tkinter graphical interface for llmbench."""

from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import List, Optional

from .runner import BenchmarkResult, BenchmarkRunner
from .spec import SAMPLE_TASKS, Task


class _SummaryBar(ttk.Frame):
    """One-line summary label used across tabs."""

    def __init__(self, parent: tk.Widget, **kwargs: object) -> None:
        super().__init__(parent, **kwargs)
        self._var = tk.StringVar(value="No results yet.")
        ttk.Label(
            self,
            textvariable=self._var,
            font=("Courier", 10),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=4, pady=2)

    def update(self, summary: dict) -> None:
        ov = summary.get("overall", {})
        self._var.set(
            f"n={ov.get('n', 0)}  "
            f"EM={ov.get('exact_match', 0):.4f}  "
            f"ROUGE-L={ov.get('rouge_l', 0):.4f}  "
            f"F1={ov.get('f1', 0):.4f}  "
            f"Composite={ov.get('composite', 0):.4f}  "
            f"Latency={ov.get('avg_latency_s', 0):.3f}s"
        )


class LLMBenchGUI:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("llmbench — LLM Benchmark Runner")
        self.root.minsize(900, 640)
        self.results: List[BenchmarkResult] = []
        self._build_ui()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._build_run_tab(nb)
        self._build_score_tab(nb)
        self._build_results_tab(nb)
        self._build_tasks_tab(nb)

    # ── Run tab ───────────────────────────────────────────────────────────────

    def _build_run_tab(self, nb: ttk.Notebook) -> None:
        frame = ttk.Frame(nb)
        nb.add(frame, text="Run API Benchmark")

        cfg = ttk.LabelFrame(frame, text="Configuration", padding=8)
        cfg.pack(fill=tk.X, padx=8, pady=(8, 4))

        # API key
        ttk.Label(cfg, text="API Key:").grid(row=0, column=0, sticky=tk.W)
        self._api_key_var = tk.StringVar(value=os.environ.get("OPENAI_API_KEY", ""))
        ttk.Entry(cfg, textvariable=self._api_key_var, width=52, show="*").grid(
            row=0, column=1, columnspan=2, sticky=tk.EW, padx=(6, 0)
        )

        # Model
        ttk.Label(cfg, text="Model:").grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        self._model_var = tk.StringVar(value="gpt-3.5-turbo")
        ttk.Entry(cfg, textvariable=self._model_var, width=30).grid(
            row=1, column=1, sticky=tk.W, padx=(6, 0), pady=(4, 0)
        )

        # Max tokens
        ttk.Label(cfg, text="Max tokens:").grid(row=2, column=0, sticky=tk.W, pady=(4, 0))
        self._max_tokens_var = tk.IntVar(value=256)
        ttk.Spinbox(
            cfg, from_=1, to=4096, textvariable=self._max_tokens_var, width=8
        ).grid(row=2, column=1, sticky=tk.W, padx=(6, 0), pady=(4, 0))

        # Temperature
        ttk.Label(cfg, text="Temperature:").grid(row=3, column=0, sticky=tk.W, pady=(4, 0))
        self._temp_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(
            cfg,
            from_=0.0,
            to=2.0,
            increment=0.1,
            textvariable=self._temp_var,
            width=8,
            format="%.1f",
        ).grid(row=3, column=1, sticky=tk.W, padx=(6, 0), pady=(4, 0))

        # Tasks file
        ttk.Label(cfg, text="Tasks JSONL:").grid(row=4, column=0, sticky=tk.W, pady=(4, 0))
        tasks_row = ttk.Frame(cfg)
        tasks_row.grid(row=4, column=1, columnspan=2, sticky=tk.EW, pady=(4, 0))
        self._tasks_file_var = tk.StringVar(value="(built-in sample tasks)")
        ttk.Entry(tasks_row, textvariable=self._tasks_file_var, width=40).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(tasks_row, text="Browse…", command=self._browse_tasks_file).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Button(
            tasks_row,
            text="Reset",
            command=lambda: self._tasks_file_var.set("(built-in sample tasks)"),
        ).pack(side=tk.LEFT, padx=(4, 0))

        cfg.columnconfigure(1, weight=1)

        # Action buttons
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, padx=8, pady=(4, 0))
        self._run_btn = ttk.Button(
            btn_row, text="▶  Run API Benchmark", command=self._run_api
        )
        self._run_btn.pack(side=tk.LEFT)
        ttk.Button(
            btn_row, text="Run Offline Demo", command=self._run_demo
        ).pack(side=tk.LEFT, padx=(8, 0))

        # Progress bar
        self._progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(
            frame, variable=self._progress_var, maximum=100
        ).pack(fill=tk.X, padx=8, pady=(6, 4))

        # Log area
        ttk.Label(frame, text="Log:").pack(anchor=tk.W, padx=8)
        self._run_log = scrolledtext.ScrolledText(frame, height=14, state=tk.DISABLED)
        self._run_log.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _browse_tasks_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select tasks JSONL file",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
        if path:
            self._tasks_file_var.set(path)

    def _log(self, msg: str) -> None:
        self._run_log.configure(state=tk.NORMAL)
        self._run_log.insert(tk.END, msg + "\n")
        self._run_log.see(tk.END)
        self._run_log.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self._run_log.configure(state=tk.NORMAL)
        self._run_log.delete("1.0", tk.END)
        self._run_log.configure(state=tk.DISABLED)

    def _run_demo(self) -> None:
        self._clear_log()
        self._log("Running offline demo benchmark…")
        self._progress_var.set(0.0)
        runner = BenchmarkRunner()
        results = runner.run_offline(
            lambda p: p.split("?")[0].strip() if "?" in p else p[:40]
        )
        summary = runner.summarize()
        self._progress_var.set(100.0)
        self._log(f"Done — {len(results)} tasks completed.")
        self._log(json.dumps(summary, indent=2))
        self._push_results(results)

    def _run_api(self) -> None:
        api_key = self._api_key_var.get().strip()
        if not api_key:
            messagebox.showerror("Missing API key", "Please enter an API key.")
            return
        model = self._model_var.get().strip() or "gpt-3.5-turbo"

        tasks: Optional[List[Task]] = None
        tasks_path = self._tasks_file_var.get()
        if tasks_path and tasks_path != "(built-in sample tasks)":
            try:
                from .cli import _load_tasks
                tasks = _load_tasks(tasks_path)
            except Exception as exc:
                messagebox.showerror("Task load error", str(exc))
                return

        self._run_btn.configure(state=tk.DISABLED)
        self._progress_var.set(0.0)
        self._clear_log()
        self._log(f"Starting API benchmark — model={model}")

        def _worker() -> None:
            runner = BenchmarkRunner(tasks)

            def _progress(done: int, total: int, result: BenchmarkResult) -> None:
                pct = 100.0 * done / total
                suffix = f"  ERROR: {result.error}" if result.error else ""
                self.root.after(0, self._progress_var.set, pct)
                self.root.after(
                    0,
                    self._log,
                    f"  [{done}/{total}] {result.task_id}"
                    f"  composite={result.composite_score:.4f}{suffix}",
                )

            try:
                results = runner.run_openai(
                    api_key=api_key,
                    model=model,
                    max_tokens=self._max_tokens_var.get(),
                    temperature=self._temp_var.get(),
                    on_progress=_progress,
                )
                summary = runner.summarize()
                self.root.after(0, self._log, f"\nDone — {len(results)} tasks.")
                self.root.after(0, self._log, json.dumps(summary, indent=2))
                self.root.after(0, self._push_results, results)
                self.root.after(0, self._progress_var.set, 100.0)
            except Exception as exc:
                self.root.after(0, self._log, f"Error: {exc}")
            finally:
                self.root.after(0, self._run_btn.configure, {"state": tk.NORMAL})

        threading.Thread(target=_worker, daemon=True).start()

    # ── Score tab ─────────────────────────────────────────────────────────────

    def _build_score_tab(self, nb: ttk.Notebook) -> None:
        frame = ttk.Frame(nb)
        nb.add(frame, text="Score Predictions")

        top = ttk.Frame(frame, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Predictions JSONL:").grid(row=0, column=0, sticky=tk.W)
        self._score_file_var = tk.StringVar()
        ttk.Entry(top, textvariable=self._score_file_var, width=52).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 6)
        )
        ttk.Button(top, text="Browse…", command=self._browse_predictions).grid(row=0, column=2)
        top.columnconfigure(1, weight=1)

        ttk.Button(frame, text="Score", command=self._score_predictions, padding=6).pack(
            anchor=tk.W, padx=8, pady=4
        )
        ttk.Label(frame, text="Summary:").pack(anchor=tk.W, padx=8)
        self._score_output = scrolledtext.ScrolledText(frame, height=20, state=tk.DISABLED)
        self._score_output.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _browse_predictions(self) -> None:
        path = filedialog.askopenfilename(
            title="Select predictions JSONL",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
        if path:
            self._score_file_var.set(path)

    def _score_predictions(self) -> None:
        path_str = self._score_file_var.get().strip()
        if not path_str:
            messagebox.showerror("No file", "Please select a predictions JSONL file.")
            return
        path = Path(path_str)
        if not path.is_file():
            messagebox.showerror("Not found", f"{path} does not exist.")
            return

        results: List[BenchmarkResult] = []
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    pred = obj.get("prediction", "")
                    task = Task(
                        task_id=obj.get("task_id", "?"),
                        category=obj.get("category", "unknown"),
                        prompt=obj.get("prompt", ""),
                        reference=obj.get("reference", ""),
                    )
                    runner = BenchmarkRunner([task])
                    results.extend(runner.run_offline(lambda _, p=pred: p, [task]))
        except Exception as exc:
            messagebox.showerror("Error reading file", str(exc))
            return

        scorer = BenchmarkRunner()
        scorer.results = results
        summary = scorer.summarize()
        self._push_results(results)

        self._score_output.configure(state=tk.NORMAL)
        self._score_output.delete("1.0", tk.END)
        self._score_output.insert(tk.END, json.dumps(summary, indent=2))
        self._score_output.configure(state=tk.DISABLED)

    # ── Results tab ───────────────────────────────────────────────────────────

    def _build_results_tab(self, nb: ttk.Notebook) -> None:
        frame = ttk.Frame(nb)
        nb.add(frame, text="Results")

        self._results_summary = _SummaryBar(frame)
        self._results_summary.pack(fill=tk.X, padx=8, pady=(8, 0))

        cols = (
            "task_id", "category", "composite",
            "rouge_l", "f1", "bleu_1", "em_norm", "latency_s", "error",
        )
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 0))

        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=20)
        col_widths = {
            "task_id": 90, "category": 110, "composite": 90,
            "rouge_l": 80, "f1": 70, "bleu_1": 70, "em_norm": 80,
            "latency_s": 80, "error": 200,
        }
        for col in cols:
            self._tree.heading(col, text=col)
            anchor = tk.W if col in ("task_id", "category", "error") else tk.CENTER
            self._tree.column(col, width=col_widths.get(col, 80), anchor=anchor)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky=tk.NSEW)
        vsb.grid(row=0, column=1, sticky=tk.NS)
        hsb.grid(row=1, column=0, sticky=tk.EW)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=8, pady=(4, 8))
        ttk.Button(btn_frame, text="Export JSONL…", command=self._export_jsonl).pack(
            side=tk.LEFT
        )
        ttk.Button(btn_frame, text="Export CSV…", command=self._export_csv).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(btn_frame, text="Clear", command=self._clear_results).pack(
            side=tk.RIGHT
        )

    def _push_results(self, results: List[BenchmarkResult]) -> None:
        """Populate the results tab with new data."""
        self.results = results
        for item in self._tree.get_children():
            self._tree.delete(item)
        for r in results:
            tag = "error" if r.error else ("good" if r.composite_score >= 0.5 else "")
            self._tree.insert(
                "",
                tk.END,
                values=(
                    r.task_id,
                    r.category,
                    f"{r.composite_score:.4f}",
                    f"{r.rouge_l:.4f}",
                    f"{r.f1:.4f}",
                    f"{r.bleu_1:.4f}",
                    f"{r.exact_match_norm:.4f}",
                    f"{r.latency_s:.3f}",
                    r.error or "",
                ),
                tags=(tag,),
            )
        self._tree.tag_configure("error", foreground="red")
        self._tree.tag_configure("good", foreground="darkgreen")

        runner = BenchmarkRunner()
        runner.results = results
        self._results_summary.update(runner.summarize())

    def _clear_results(self) -> None:
        self.results = []
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._results_summary._var.set("No results yet.")

    def _export_jsonl(self) -> None:
        if not self.results:
            messagebox.showinfo("No results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl")],
        )
        if path:
            runner = BenchmarkRunner()
            runner.results = self.results
            runner.export_jsonl(Path(path))
            messagebox.showinfo("Exported", f"Saved to {path}")

    def _export_csv(self) -> None:
        if not self.results:
            messagebox.showinfo("No results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if path:
            runner = BenchmarkRunner()
            runner.results = self.results
            runner.export_csv(Path(path))
            messagebox.showinfo("Exported", f"Saved to {path}")

    # ── Sample Tasks tab ──────────────────────────────────────────────────────

    def _build_tasks_tab(self, nb: ttk.Notebook) -> None:
        frame = ttk.Frame(nb)
        nb.add(frame, text="Sample Tasks")

        ttk.Label(frame, text="Built-in sample tasks (read-only):").pack(
            anchor=tk.W, padx=8, pady=(8, 4)
        )

        cols = ("task_id", "category", "prompt", "reference")
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=20)
        tree.heading("task_id", text="task_id")
        tree.heading("category", text="category")
        tree.heading("prompt", text="prompt")
        tree.heading("reference", text="reference")
        tree.column("task_id", width=80)
        tree.column("category", width=100)
        tree.column("prompt", width=380)
        tree.column("reference", width=260)

        for task in SAMPLE_TASKS:
            tree.insert(
                "",
                tk.END,
                values=(task.task_id, task.category, task.prompt, task.reference),
            )

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky=tk.NSEW)
        vsb.grid(row=0, column=1, sticky=tk.NS)
        hsb.grid(row=1, column=0, sticky=tk.EW)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)


def launch_gui() -> None:
    """Create and run the main application window."""
    root = tk.Tk()
    LLMBenchGUI(root)
    root.mainloop()


def main() -> None:
    launch_gui()
