"""Tkinter GUI for llmbench.

Launch via:
    python -m llmbench gui
    llmbench gui
    llmbench-gui
"""

from __future__ import annotations

import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path
from typing import List, Optional

from .runner import BenchmarkRunner
from .spec import BenchmarkSpec
from .tasks import BenchmarkResult, SAMPLE_TASKS, Task


class _App:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("llmbench — LLM Benchmark Runner")
        self.root.geometry("960x680")
        self.root.minsize(720, 520)

        self._results: List[BenchmarkResult] = []
        self._tasks: List[Task] = list(SAMPLE_TASKS)

        self._build_menu()
        self._build_notebook()
        self._build_statusbar()

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        mb = tk.Menu(self.root)

        file_menu = tk.Menu(mb, tearoff=0)
        file_menu.add_command(label="Load Spec…", command=self._load_spec)
        file_menu.add_separator()
        file_menu.add_command(label="Export CSV…", command=self._export_csv)
        file_menu.add_command(label="Export JSONL…", command=self._export_jsonl)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        mb.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(mb, tearoff=0)
        help_menu.add_command(label="About llmbench", command=self._show_about)
        mb.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=mb)

    def _build_notebook(self) -> None:
        self._nb = ttk.Notebook(self.root)
        self._nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._build_run_tab()
        self._build_results_tab()
        self._build_summary_tab()

    def _build_run_tab(self) -> None:
        frame = ttk.Frame(self._nb)
        self._nb.add(frame, text="  Run  ")

        # --- Configuration ---
        cfg = ttk.LabelFrame(frame, text="Configuration", padding=10)
        cfg.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(cfg, text="API Key:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self._api_key = tk.StringVar()
        ttk.Entry(cfg, textvariable=self._api_key, width=52, show="*").grid(
            row=0, column=1, sticky=tk.EW, padx=(8, 0), pady=3
        )

        ttk.Label(cfg, text="Model:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self._model = tk.StringVar(value="gpt-3.5-turbo")
        ttk.Combobox(
            cfg,
            textvariable=self._model,
            width=30,
            values=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        ).grid(row=1, column=1, sticky=tk.W, padx=(8, 0), pady=3)

        ttk.Label(cfg, text="Max tokens:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self._max_tokens = tk.IntVar(value=256)
        ttk.Spinbox(cfg, textvariable=self._max_tokens, from_=16, to=4096, width=8).grid(
            row=2, column=1, sticky=tk.W, padx=(8, 0), pady=3
        )

        ttk.Label(cfg, text="Temperature:").grid(row=3, column=0, sticky=tk.W, pady=3)
        self._temperature = tk.DoubleVar(value=0.0)
        ttk.Spinbox(
            cfg, textvariable=self._temperature, from_=0.0, to=2.0, increment=0.1, width=8
        ).grid(row=3, column=1, sticky=tk.W, padx=(8, 0), pady=3)

        cfg.columnconfigure(1, weight=1)

        # --- Mode ---
        mode_frame = ttk.LabelFrame(frame, text="Mode", padding=10)
        mode_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        self._mode = tk.StringVar(value="demo")
        ttk.Radiobutton(
            mode_frame,
            text="Demo  (offline — no API key required)",
            variable=self._mode,
            value="demo",
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            mode_frame,
            text="OpenAI-compatible API",
            variable=self._mode,
            value="api",
        ).pack(anchor=tk.W)

        # --- Progress ---
        prog = ttk.Frame(frame)
        prog.pack(fill=tk.X, padx=8, pady=(0, 4))

        self._progress = tk.DoubleVar(value=0.0)
        ttk.Progressbar(prog, variable=self._progress, maximum=100).pack(fill=tk.X)

        self._progress_label = ttk.Label(prog, text="Ready.")
        self._progress_label.pack(anchor=tk.W, pady=(2, 0))

        # --- Buttons ---
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, padx=8, pady=(4, 0))

        self._run_btn = ttk.Button(btn_row, text="▶  Run Benchmark", command=self._run_benchmark)
        self._run_btn.pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Clear Results", command=self._clear_results).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        # --- Log ---
        log_frame = ttk.LabelFrame(frame, text="Log", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._log = scrolledtext.ScrolledText(
            log_frame, height=8, state=tk.DISABLED, font=("Courier", 9)
        )
        self._log.pack(fill=tk.BOTH, expand=True)

    def _build_results_tab(self) -> None:
        frame = ttk.Frame(self._nb)
        self._nb.add(frame, text="  Results  ")

        cols = ("task_id", "category", "em", "rouge_l", "bleu_1", "f1", "composite", "latency_s", "error")
        self._tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="browse")

        headers = {
            "task_id": "Task ID",
            "category": "Category",
            "em": "EM",
            "rouge_l": "ROUGE-L",
            "bleu_1": "BLEU-1",
            "f1": "F1",
            "composite": "Composite",
            "latency_s": "Latency (s)",
            "error": "Error",
        }
        widths = {
            "task_id": 100, "category": 100, "em": 55, "rouge_l": 75,
            "bleu_1": 60, "f1": 55, "composite": 80, "latency_s": 90, "error": 140,
        }
        for col in cols:
            self._tree.heading(col, text=headers[col])
            self._tree.column(col, width=widths[col], minwidth=40, anchor=tk.CENTER)
        self._tree.column("task_id", anchor=tk.W)
        self._tree.column("error", anchor=tk.W)

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._tree.yview)
        hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._tree.grid(row=0, column=0, sticky=tk.NSEW)
        vsb.grid(row=0, column=1, sticky=tk.NS)
        hsb.grid(row=1, column=0, sticky=tk.EW)

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        # Detail panel
        detail = ttk.LabelFrame(frame, text="Prediction / Reference", padding=4)
        detail.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=4, pady=4)

        self._detail = scrolledtext.ScrolledText(
            detail, height=5, state=tk.DISABLED, font=("Courier", 9), wrap=tk.WORD
        )
        self._detail.pack(fill=tk.BOTH, expand=True)

        self._tree.bind("<<TreeviewSelect>>", self._on_row_select)

    def _build_summary_tab(self) -> None:
        frame = ttk.Frame(self._nb)
        self._nb.add(frame, text="  Summary  ")

        self._summary_text = scrolledtext.ScrolledText(
            frame, state=tk.DISABLED, font=("Courier", 10)
        )
        self._summary_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Button(frame, text="Refresh", command=self._refresh_summary).pack(pady=(0, 8))

    def _build_statusbar(self) -> None:
        self._status = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self._status, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X, padx=2, pady=2
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_line(self, msg: str) -> None:
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, msg + "\n")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _set_status(self, msg: str) -> None:
        self._status.set(msg)
        self.root.update_idletasks()

    # ------------------------------------------------------------------
    # Benchmark execution
    # ------------------------------------------------------------------

    def _run_benchmark(self) -> None:
        self._run_btn.config(state=tk.DISABLED)
        self._progress.set(0)
        self._progress_label.config(text="Starting…")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        try:
            self._do_run()
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))
        finally:
            self.root.after(0, lambda: self._run_btn.config(state=tk.NORMAL))

    def _do_run(self) -> None:
        tasks = self._tasks
        n = len(tasks)
        runner = BenchmarkRunner(tasks)
        results: List[BenchmarkResult] = []

        if self._mode.get() == "demo":
            def _demo(prompt: str) -> str:
                return prompt.split("?")[0].strip() if "?" in prompt else prompt[:40]

            for i, task in enumerate(tasks):
                self.root.after(0, lambda i=i: self._progress.set(i / n * 100))
                self.root.after(0, lambda t=task: self._log_line(f"  {t.task_id}: running…"))
                partial = runner.run_offline(_demo, [task])
                results.extend(partial)
                self.root.after(0, lambda r=partial[0]: self._add_row(r))
        else:
            api_key = self._api_key.get().strip()
            if not api_key:
                self.root.after(
                    0,
                    lambda: messagebox.showerror("API Key Required", "Enter an API key for API mode."),
                )
                return
            model = self._model.get()
            max_tokens = self._max_tokens.get()
            temperature = self._temperature.get()

            for i, task in enumerate(tasks):
                self.root.after(0, lambda i=i: self._progress.set(i / n * 100))
                self.root.after(0, lambda t=task: self._log_line(f"  {t.task_id}: calling API…"))
                partial = runner.run_openai(
                    api_key, model=model, tasks=[task],
                    max_tokens=max_tokens, temperature=temperature,
                )
                results.extend(partial)
                self.root.after(0, lambda r=partial[0]: self._add_row(r))

        self._results = results
        self.root.after(0, lambda: self._progress.set(100))
        self.root.after(0, lambda: self._progress_label.config(text=f"Done — {len(results)} tasks."))
        self.root.after(0, lambda: self._log_line(f"Benchmark complete: {len(results)} result(s)."))
        self.root.after(0, self._refresh_summary)
        self.root.after(0, lambda: self._nb.select(1))
        self.root.after(0, lambda: self._set_status(f"Done — {len(results)} tasks completed."))

    def _add_row(self, r: BenchmarkResult) -> None:
        if r.error:
            tag = "err"
        elif r.composite_score >= 0.5:
            tag = "good"
        else:
            tag = "poor"
        self._tree.insert(
            "", tk.END,
            values=(
                r.task_id, r.category,
                f"{r.exact_match:.2f}", f"{r.rouge_l:.4f}", f"{r.bleu_1:.4f}",
                f"{r.f1:.4f}", f"{r.composite_score:.4f}",
                f"{r.latency_s:.3f}", r.error or "",
            ),
            tags=(tag,),
        )
        self._tree.tag_configure("err", foreground="#c0392b")
        self._tree.tag_configure("good", foreground="#27ae60")
        self._tree.tag_configure("poor", foreground="#d35400")

    def _on_row_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        if idx >= len(self._results):
            return
        r = self._results[idx]
        text = f"PROMPT:\n{r.prompt}\n\nPREDICTION:\n{r.prediction}\n\nREFERENCE:\n{r.reference}"
        if r.error:
            text += f"\n\nERROR:\n{r.error}"
        self._detail.config(state=tk.NORMAL)
        self._detail.delete("1.0", tk.END)
        self._detail.insert("1.0", text)
        self._detail.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _refresh_summary(self) -> None:
        if not self._results:
            return
        reporter = BenchmarkRunner()
        reporter.results = self._results
        summary = reporter.summarize()

        ov = summary.get("overall", {})
        lines = [
            "=" * 54,
            "  BENCHMARK SUMMARY",
            "=" * 54,
            "",
            f"  Total tasks  : {ov.get('n', 0)}",
            f"  Errors       : {ov.get('errors', 0)}",
            f"  Avg latency  : {ov.get('avg_latency_s', 0):.3f} s",
            "",
            f"  {'Metric':<22} {'Overall':>10}",
            "  " + "-" * 34,
        ]
        for metric in ("exact_match", "exact_match_norm", "rouge_l", "bleu_1", "f1", "composite"):
            label = metric.replace("_", " ").title()
            lines.append(f"  {label:<22} {ov.get(metric, 0):>10.4f}")

        lines += ["", "  BY CATEGORY", "  " + "-" * 34]
        for cat, stats in summary.get("by_category", {}).items():
            lines.append(
                f"\n  [{cat}]  n={stats['n']}  errors={stats['errors']}"
            )
            lines.append(
                f"    ROUGE-L={stats['rouge_l']:.4f}  "
                f"F1={stats['f1']:.4f}  "
                f"Composite={stats['composite']:.4f}"
            )

        self._summary_text.config(state=tk.NORMAL)
        self._summary_text.delete("1.0", tk.END)
        self._summary_text.insert("1.0", "\n".join(lines))
        self._summary_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _clear_results(self) -> None:
        self._results = []
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._summary_text.config(state=tk.NORMAL)
        self._summary_text.delete("1.0", tk.END)
        self._summary_text.config(state=tk.DISABLED)
        self._progress.set(0)
        self._progress_label.config(text="Ready.")
        self._set_status("Results cleared.")

    def _load_spec(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Benchmark Spec",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            spec = BenchmarkSpec.from_file(Path(path))
            self._tasks = spec.tasks if spec.tasks else list(SAMPLE_TASKS)
            self._model.set(spec.model)
            self._max_tokens.set(spec.max_tokens)
            self._temperature.set(spec.temperature)
            self._log_line(f"Loaded spec '{spec.name}': {len(self._tasks)} task(s), model={spec.model}")
            messagebox.showinfo(
                "Spec Loaded",
                f"'{spec.name}'\n{len(self._tasks)} task(s) | model: {spec.model}",
            )
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _export_csv(self) -> None:
        if not self._results:
            messagebox.showwarning("No Results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            r = BenchmarkRunner()
            r.results = self._results
            r.export_csv(Path(path))
            self._set_status(f"Exported CSV → {path}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    def _export_jsonl(self) -> None:
        if not self._results:
            messagebox.showwarning("No Results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            r = BenchmarkRunner()
            r.results = self._results
            r.export_jsonl(Path(path))
            self._set_status(f"Exported JSONL → {path}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About llmbench",
            "llmbench  v0.1.0\n\n"
            "Lightweight reproducible LLM benchmarking framework.\n\n"
            "Metrics: Exact Match · ROUGE-L · BLEU-1 · F1 · Composite\n\n"
            "https://github.com/vdeshmukh203/llmbench",
        )


def main() -> None:
    root = tk.Tk()
    _App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
