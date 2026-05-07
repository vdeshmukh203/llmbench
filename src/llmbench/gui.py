"""Tkinter GUI for llmbench. Launch with ``llmbench gui`` or ``launch_gui()``."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import List, Optional

from .models import SAMPLE_TASKS, Task
from .runner import BenchmarkRunner


def launch_gui() -> None:
    """Entry point for the graphical interface."""
    app = LLMBenchGUI()
    app.mainloop()


class LLMBenchGUI(tk.Tk):
    """Main application window for llmbench."""

    def __init__(self) -> None:
        super().__init__()
        self.title("llmbench — LLM Benchmark Runner")
        self.geometry("1050x720")
        self.minsize(800, 550)
        self._tasks: List[Task] = list(SAMPLE_TASKS)
        self._results: List = []
        self._runner: Optional[BenchmarkRunner] = None
        self._stop_flag = False
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True, padx=6, pady=6)

        self._tab_bench = ttk.Frame(self._nb)
        self._tab_results = ttk.Frame(self._nb)
        self._tab_summary = ttk.Frame(self._nb)

        self._nb.add(self._tab_bench, text="  Benchmark  ")
        self._nb.add(self._tab_results, text="  Results  ")
        self._nb.add(self._tab_summary, text="  Summary  ")

        self._build_bench_tab()
        self._build_results_tab()
        self._build_summary_tab()

    def _build_bench_tab(self) -> None:
        f = self._tab_bench
        pad = {"padx": 10, "pady": 5}

        # ── Tasks ──────────────────────────────────────────────────────────
        tasks_lf = ttk.LabelFrame(f, text="Tasks")
        tasks_lf.pack(fill="x", **pad)

        ttk.Label(tasks_lf, text="Tasks file (JSON/JSONL):").grid(
            row=0, column=0, sticky="w", **pad
        )
        self._tasks_var = tk.StringVar(value="<built-in sample tasks>")
        ttk.Entry(tasks_lf, textvariable=self._tasks_var, width=52).grid(
            row=0, column=1, sticky="ew", **pad
        )
        ttk.Button(tasks_lf, text="Browse…", command=self._browse_tasks).grid(
            row=0, column=2, **pad
        )
        ttk.Button(tasks_lf, text="Use Built-in", command=self._use_builtin_tasks).grid(
            row=0, column=3, **pad
        )
        tasks_lf.columnconfigure(1, weight=1)

        # ── Model configuration ────────────────────────────────────────────
        cfg_lf = ttk.LabelFrame(f, text="Model Configuration")
        cfg_lf.pack(fill="x", **pad)

        self._mode_var = tk.StringVar(value="offline")
        ttk.Radiobutton(
            cfg_lf, text="Offline / Demo (no API key needed)",
            variable=self._mode_var, value="offline", command=self._on_mode_change,
        ).grid(row=0, column=0, columnspan=2, sticky="w", **pad)
        ttk.Radiobutton(
            cfg_lf, text="OpenAI API",
            variable=self._mode_var, value="openai", command=self._on_mode_change,
        ).grid(row=0, column=2, sticky="w", **pad)

        ttk.Label(cfg_lf, text="API Key:").grid(row=1, column=0, sticky="w", **pad)
        self._apikey_var = tk.StringVar()
        self._apikey_entry = ttk.Entry(cfg_lf, textvariable=self._apikey_var, show="*", width=48)
        self._apikey_entry.grid(row=1, column=1, columnspan=3, sticky="ew", **pad)

        ttk.Label(cfg_lf, text="Model:").grid(row=2, column=0, sticky="w", **pad)
        self._model_var = tk.StringVar(value="gpt-3.5-turbo")
        _models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
        self._model_combo = ttk.Combobox(
            cfg_lf, textvariable=self._model_var, values=_models, width=28
        )
        self._model_combo.grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(cfg_lf, text="Max tokens:").grid(row=3, column=0, sticky="w", **pad)
        self._max_tokens_var = tk.IntVar(value=256)
        ttk.Spinbox(cfg_lf, textvariable=self._max_tokens_var, from_=1, to=4096, width=8).grid(
            row=3, column=1, sticky="w", **pad
        )

        ttk.Label(cfg_lf, text="Temperature:").grid(row=4, column=0, sticky="w", **pad)
        self._temp_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(
            cfg_lf, textvariable=self._temp_var,
            from_=0.0, to=2.0, increment=0.1, format="%.1f", width=8,
        ).grid(row=4, column=1, sticky="w", **pad)
        cfg_lf.columnconfigure(1, weight=1)

        # ── Controls ───────────────────────────────────────────────────────
        ctrl = ttk.Frame(f)
        ctrl.pack(fill="x", padx=10, pady=4)
        self._run_btn = ttk.Button(ctrl, text="▶  Run Benchmark", command=self._run_benchmark)
        self._run_btn.pack(side="left", padx=4)
        self._stop_btn = ttk.Button(ctrl, text="■  Stop", command=self._request_stop, state="disabled")
        self._stop_btn.pack(side="left", padx=4)

        self._progress = ttk.Progressbar(f, mode="determinate")
        self._progress.pack(fill="x", padx=10, pady=2)

        # ── Log ────────────────────────────────────────────────────────────
        log_lf = ttk.LabelFrame(f, text="Log")
        log_lf.pack(fill="both", expand=True, **pad)
        self._log = scrolledtext.ScrolledText(
            log_lf, height=9, state="disabled", font=("Courier", 10), wrap="word"
        )
        self._log.pack(fill="both", expand=True, padx=4, pady=4)

        self._on_mode_change()

    def _build_results_tab(self) -> None:
        f = self._tab_results
        cols = (
            "task_id", "category",
            "exact_match", "rouge_l", "bleu_1", "f1",
            "composite", "latency_s", "tokens", "error",
        )
        col_cfg = {
            "task_id":     ("Task ID",      110),
            "category":    ("Category",      90),
            "exact_match": ("Exact Match",   80),
            "rouge_l":     ("ROUGE-L",       80),
            "bleu_1":      ("BLEU-1",        80),
            "f1":          ("F1",            80),
            "composite":   ("Composite",     90),
            "latency_s":   ("Latency (s)",   80),
            "tokens":      ("~Tokens",       70),
            "error":       ("Error",        150),
        }
        self._tree = ttk.Treeview(f, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            label, width = col_cfg[c]
            self._tree.heading(c, text=label, command=lambda _c=c: self._sort_by(_c))
            self._tree.column(c, width=width, anchor="center", minwidth=50)

        self._tree.tag_configure("good", foreground="#1a7f2f")
        self._tree.tag_configure("poor", foreground="#cc5500")
        self._tree.tag_configure("error_row", foreground="#cc0000")

        vsb = ttk.Scrollbar(f, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(f, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)

        btn_f = ttk.Frame(f)
        btn_f.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=6)
        ttk.Button(btn_f, text="Export JSONL…", command=self._export_jsonl).pack(side="left", padx=4)
        ttk.Button(btn_f, text="Export CSV…", command=self._export_csv).pack(side="left", padx=4)
        ttk.Button(btn_f, text="Clear Results", command=self._clear_results).pack(side="right", padx=4)

        self._sort_col: Optional[str] = None
        self._sort_rev = False

    def _build_summary_tab(self) -> None:
        f = self._tab_summary
        ttk.Label(f, text="Aggregate Statistics", font=("TkDefaultFont", 12, "bold")).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        self._summary_text = scrolledtext.ScrolledText(
            f, font=("Courier", 11), state="disabled", wrap="none"
        )
        self._summary_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_mode_change(self) -> None:
        openai = self._mode_var.get() == "openai"
        state = "normal" if openai else "disabled"
        self._apikey_entry.configure(state=state)
        self._model_combo.configure(state="readonly" if openai else "disabled")

    def _browse_tasks(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Tasks File",
            filetypes=[
                ("JSON / JSONL", "*.json *.jsonl"),
                ("JSON", "*.json"),
                ("JSONL", "*.jsonl"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._tasks_var.set(path)
            self._load_tasks_file(Path(path))

    def _use_builtin_tasks(self) -> None:
        self._tasks = list(SAMPLE_TASKS)
        self._tasks_var.set("<built-in sample tasks>")
        self._log_msg(f"Loaded {len(self._tasks)} built-in sample tasks.")

    def _load_tasks_file(self, path: Path) -> None:
        from .spec import BenchmarkSpec
        try:
            spec = BenchmarkSpec.from_json(path) if path.suffix == ".json" else BenchmarkSpec.from_jsonl(path)
            self._tasks = spec.tasks
            self._log_msg(f"Loaded {len(self._tasks)} tasks from {path.name}.")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _run_benchmark(self) -> None:
        self._results = []
        self._clear_tree()
        self._progress["value"] = 0
        self._progress["maximum"] = max(len(self._tasks), 1)
        self._stop_flag = False
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        threading.Thread(target=self._worker_thread, daemon=True).start()

    def _request_stop(self) -> None:
        self._stop_flag = True
        self._log_msg("Stop requested — finishing current task…")

    def _worker_thread(self) -> None:
        mode = self._mode_var.get()
        if mode == "openai":
            api_key = self._apikey_var.get().strip()
            if not api_key:
                self.after(0, lambda: messagebox.showerror("Error", "OpenAI API key is required."))
                self.after(0, self._finish_run)
                return
            model = self._model_var.get()
            max_tokens = self._max_tokens_var.get()
            temperature = self._temp_var.get()

            runner = BenchmarkRunner([])
            for i, task in enumerate(self._tasks):
                if self._stop_flag:
                    break
                batch = runner.run_openai(api_key, model, [task], max_tokens, temperature)
                r = batch[0]
                self._results.append(r)
                self.after(0, lambda _r=r, _i=i: self._on_result(_r, _i))
        else:
            def _demo(prompt: str) -> str:
                return prompt.split("?")[0].strip() if "?" in prompt else prompt[:40]

            runner = BenchmarkRunner([])
            for i, task in enumerate(self._tasks):
                if self._stop_flag:
                    break
                batch = runner.run_offline(_demo, [task])
                r = batch[0]
                self._results.append(r)
                self.after(0, lambda _r=r, _i=i: self._on_result(_r, _i))

        self._runner = runner
        self.after(0, self._finish_run)

    def _on_result(self, result, idx: int) -> None:
        self._add_tree_row(result)
        self._progress["value"] = idx + 1
        status = f"ERROR: {result.error}" if result.error else f"composite={result.composite_score:.4f}"
        self._log_msg(f"[{result.task_id}] {status}")

    def _finish_run(self) -> None:
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        if self._results:
            tmp = BenchmarkRunner([])
            tmp.results = self._results
            summary = tmp.summarize()
            self._show_summary(summary)
            self._nb.select(self._tab_results)
            n = len(self._results)
            comp = summary.get("overall", {}).get("composite", 0.0)
            self._log_msg(f"Done — {n} task(s) scored, overall composite = {comp:.4f}.")
        else:
            self._log_msg("No results (benchmark stopped before any tasks completed).")

    # ── tree helpers ──────────────────────────────────────────────────────────

    def _clear_tree(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)

    def _clear_results(self) -> None:
        self._results = []
        self._clear_tree()
        self._show_summary({})
        self._log_msg("Results cleared.")

    def _add_tree_row(self, r) -> None:
        comp = r.composite_score
        if r.error:
            tag = "error_row"
        elif comp >= 0.70:
            tag = "good"
        elif comp < 0.35:
            tag = "poor"
        else:
            tag = ""
        self._tree.insert(
            "", "end",
            values=(
                r.task_id, r.category,
                f"{r.exact_match:.3f}", f"{r.rouge_l:.3f}",
                f"{r.bleu_1:.3f}", f"{r.f1:.3f}",
                f"{comp:.3f}", f"{r.latency_s:.3f}",
                r.approx_tokens, r.error or "",
            ),
            tags=(tag,),
        )

    def _sort_by(self, col: str) -> None:
        """Sort tree rows by the clicked column header."""
        rows = [(self._tree.set(k, col), k) for k in self._tree.get_children("")]
        try:
            rows.sort(key=lambda x: float(x[0]), reverse=self._sort_rev)
        except ValueError:
            rows.sort(key=lambda x: x[0], reverse=self._sort_rev)
        for i, (_, k) in enumerate(rows):
            self._tree.move(k, "", i)
        self._sort_rev = not self._sort_rev

    # ── summary / log ─────────────────────────────────────────────────────────

    def _show_summary(self, summary: dict) -> None:
        self._summary_text.configure(state="normal")
        self._summary_text.delete("1.0", "end")
        if summary:
            self._summary_text.insert("end", json.dumps(summary, indent=2))
        self._summary_text.configure(state="disabled")

    def _log_msg(self, msg: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    # ── export ────────────────────────────────────────────────────────────────

    def _export_jsonl(self) -> None:
        if not self._results:
            messagebox.showinfo("No Results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL", "*.jsonl"), ("All files", "*.*")],
        )
        if path:
            tmp = BenchmarkRunner([])
            tmp.results = self._results
            tmp.export_jsonl(Path(path))
            messagebox.showinfo("Exported", f"Results saved to:\n{path}")

    def _export_csv(self) -> None:
        if not self._results:
            messagebox.showinfo("No Results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if path:
            tmp = BenchmarkRunner([])
            tmp.results = self._results
            tmp.export_csv(Path(path))
            messagebox.showinfo("Exported", f"Results saved to:\n{path}")
