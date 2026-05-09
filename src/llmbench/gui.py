"""Tkinter GUI for llmbench.

Launch via the CLI::

    llmbench gui

or programmatically::

    from llmbench.gui import launch
    launch()

The GUI provides five tabs:

* **Configuration** — provider, API key, model, generation parameters.
* **Tasks** — load a JSONL task file or use the built-in sample tasks.
* **Run** — execute the benchmark with a live progress bar and log.
* **Results** — sortable per-task results table with prediction detail pane.
* **Summary** — aggregate statistics across all categories.

All export formats (JSONL, CSV, Markdown) are available from the File menu.
"""
from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

from ._sample_tasks import SAMPLE_TASKS
from .runner import BenchmarkResult, BenchmarkRunner
from .spec import BenchmarkSpec, Task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MONOSPACE = ("Courier", 9)
_HEADER_FONT = ("TkDefaultFont", 12, "bold")


def _fmt(val, decimals: int = 4) -> str:
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class LLMBenchApp:
    """Root application widget."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("llmbench — LLM Benchmark Runner")
        self.root.geometry("960x700")
        self.root.minsize(800, 580)

        self._tasks: List[Task] = list(SAMPLE_TASKS)
        self._results: List[BenchmarkResult] = []
        self._runner: Optional[BenchmarkRunner] = None
        self._running = False

        self._build_menu()
        self._build_notebook()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        bar = tk.Menu(self.root)

        file_menu = tk.Menu(bar, tearoff=0)
        file_menu.add_command(label="Load Tasks (JSONL)…", command=self._load_tasks)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results (JSONL)…", command=self._export_jsonl)
        file_menu.add_command(label="Export Results (CSV)…", command=self._export_csv)
        file_menu.add_command(label="Export Report (Markdown)…", command=self._export_md)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit)
        bar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(bar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        bar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=bar)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About llmbench",
            "llmbench — Lightweight Reproducible LLM Benchmarking\n\n"
            "Metrics: ROUGE-L, BLEU-1, Token F1, Exact Match\n"
            "Providers: OpenAI, Anthropic\n\n"
            "Results include SHA-256 provenance hashes.",
        )

    # ------------------------------------------------------------------
    # Notebook
    # ------------------------------------------------------------------

    def _build_notebook(self) -> None:
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        frames = {}
        for name in ("Configuration", "Tasks", "Run", "Results", "Summary"):
            f = ttk.Frame(nb)
            nb.add(f, text=f"  {name}  ")
            frames[name] = f

        self._build_config_tab(frames["Configuration"])
        self._build_tasks_tab(frames["Tasks"])
        self._build_run_tab(frames["Run"])
        self._build_results_tab(frames["Results"])
        self._build_summary_tab(frames["Summary"])

    # ------------------------------------------------------------------
    # Configuration tab
    # ------------------------------------------------------------------

    def _build_config_tab(self, f: ttk.Frame) -> None:
        pad = {"padx": 12, "pady": 6}

        ttk.Label(f, text="Provider & Model Settings", font=_HEADER_FONT).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 4)
        )

        # Provider
        ttk.Label(f, text="Provider:").grid(row=1, column=0, sticky="e", **pad)
        self._provider_var = tk.StringVar(value="openai")
        pf = ttk.Frame(f)
        pf.grid(row=1, column=1, sticky="w", **pad)
        for val, lbl in (("openai", "OpenAI"), ("anthropic", "Anthropic")):
            ttk.Radiobutton(pf, text=lbl, variable=self._provider_var, value=val).pack(
                side=tk.LEFT, padx=6
            )

        # API key
        ttk.Label(f, text="API Key:").grid(row=2, column=0, sticky="e", **pad)
        self._api_key_var = tk.StringVar(
            value=os.environ.get("OPENAI_API_KEY", "")
        )
        ttk.Entry(f, textvariable=self._api_key_var, width=52, show="•").grid(
            row=2, column=1, sticky="w", **pad
        )

        # Model
        ttk.Label(f, text="Model:").grid(row=3, column=0, sticky="e", **pad)
        self._model_var = tk.StringVar(value="gpt-3.5-turbo")
        cb = ttk.Combobox(f, textvariable=self._model_var, width=34)
        cb["values"] = (
            "gpt-3.5-turbo",
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
        )
        cb.grid(row=3, column=1, sticky="w", **pad)

        # Max tokens
        ttk.Label(f, text="Max Tokens:").grid(row=4, column=0, sticky="e", **pad)
        self._max_tokens_var = tk.IntVar(value=256)
        ttk.Spinbox(
            f, from_=16, to=4096, textvariable=self._max_tokens_var, width=8
        ).grid(row=4, column=1, sticky="w", **pad)

        # Temperature
        ttk.Label(f, text="Temperature:").grid(row=5, column=0, sticky="e", **pad)
        self._temperature_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(
            f,
            from_=0.0,
            to=2.0,
            increment=0.1,
            textvariable=self._temperature_var,
            format="%.1f",
            width=8,
        ).grid(row=5, column=1, sticky="w", **pad)

        # System prompt (OpenAI only)
        ttk.Label(f, text="System Prompt:").grid(row=6, column=0, sticky="ne", **pad)
        self._system_prompt_var = tk.StringVar(value="Answer concisely.")
        ttk.Entry(f, textvariable=self._system_prompt_var, width=52).grid(
            row=6, column=1, sticky="w", **pad
        )
        ttk.Label(f, text="(OpenAI only)", foreground="gray").grid(
            row=7, column=1, sticky="w", padx=12
        )

        f.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    # Tasks tab
    # ------------------------------------------------------------------

    def _build_tasks_tab(self, f: ttk.Frame) -> None:
        toolbar = ttk.Frame(f)
        toolbar.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(toolbar, text="Load JSONL…", command=self._load_tasks).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="Use Sample Tasks", command=self._use_sample_tasks).pack(
            side=tk.LEFT, padx=2
        )
        self._task_count_lbl = ttk.Label(toolbar, text="")
        self._task_count_lbl.pack(side=tk.RIGHT, padx=8)

        cols = ("task_id", "category", "prompt", "reference")
        tree_frame = ttk.Frame(f)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self._task_tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings", selectmode="browse"
        )
        for col, w in zip(cols, (110, 100, 380, 220)):
            self._task_tree.heading(col, text=col)
            self._task_tree.column(col, width=w, minwidth=60)
        self._task_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._task_tree.yview)
        self._task_tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._refresh_task_tree()

    def _refresh_task_tree(self) -> None:
        self._task_tree.delete(*self._task_tree.get_children())
        for t in self._tasks:
            prompt_p = t.prompt[:90] + "…" if len(t.prompt) > 90 else t.prompt
            ref_p = t.reference[:60] + "…" if len(t.reference) > 60 else t.reference
            self._task_tree.insert("", tk.END, values=(t.task_id, t.category, prompt_p, ref_p))
        n = len(self._tasks)
        self._task_count_lbl.config(text=f"{n} task{'s' if n != 1 else ''} loaded")

    def _load_tasks(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Tasks",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            spec = BenchmarkSpec.from_jsonl(path)
            self._tasks = spec.tasks
            self._refresh_task_tree()
            messagebox.showinfo(
                "Tasks Loaded", f"Loaded {len(self._tasks)} tasks from:\n{path}"
            )
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _use_sample_tasks(self) -> None:
        self._tasks = list(SAMPLE_TASKS)
        self._refresh_task_tree()

    # ------------------------------------------------------------------
    # Run tab
    # ------------------------------------------------------------------

    def _build_run_tab(self, f: ttk.Frame) -> None:
        pad = {"padx": 10, "pady": 6}

        ttk.Label(f, text="Run Benchmark", font=_HEADER_FONT).pack(
            anchor="w", padx=12, pady=(12, 4)
        )

        btn_frame = ttk.Frame(f)
        btn_frame.pack(fill=tk.X, **pad)
        self._run_btn = ttk.Button(
            btn_frame, text="▶  Run (API)", command=self._start_api_run
        )
        self._run_btn.pack(side=tk.LEFT, padx=2)
        self._demo_btn = ttk.Button(
            btn_frame, text="▶  Demo (offline)", command=self._start_demo_run
        )
        self._demo_btn.pack(side=tk.LEFT, padx=6)
        ttk.Label(
            btn_frame,
            text="Demo mode uses a trivial stub — no API key needed.",
            foreground="gray",
        ).pack(side=tk.LEFT, padx=8)

        # Progress
        prog_lf = ttk.LabelFrame(f, text="Progress")
        prog_lf.pack(fill=tk.X, **pad)
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            prog_lf, variable=self._progress_var, maximum=100
        )
        self._progress_bar.pack(fill=tk.X, padx=8, pady=4)
        self._progress_lbl = ttk.Label(prog_lf, text="Idle")
        self._progress_lbl.pack(anchor="w", padx=8, pady=2)

        # Log
        log_lf = ttk.LabelFrame(f, text="Log")
        log_lf.pack(fill=tk.BOTH, expand=True, **pad)
        log_inner = ttk.Frame(log_lf)
        log_inner.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._log_text = tk.Text(
            log_inner, state=tk.DISABLED, wrap=tk.WORD, font=_MONOSPACE
        )
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_sb = ttk.Scrollbar(log_inner, orient=tk.VERTICAL, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_sb.set)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y)

    def _log(self, msg: str) -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _set_running(self, running: bool) -> None:
        self._running = running
        state = tk.DISABLED if running else tk.NORMAL
        self._run_btn.config(state=state)
        self._demo_btn.config(state=state)

    def _start_api_run(self) -> None:
        if self._running:
            return
        if not self._tasks:
            messagebox.showwarning("No Tasks", "Load tasks first (Tasks tab).")
            return
        api_key = self._api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning(
                "No API Key", "Enter an API key in the Configuration tab."
            )
            return
        self._set_running(True)
        self._log("Starting API benchmark…")
        threading.Thread(target=self._api_run_thread, daemon=True).start()

    def _start_demo_run(self) -> None:
        if self._running:
            return
        self._set_running(True)
        self._log("Starting offline demo…")
        threading.Thread(target=self._demo_run_thread, daemon=True).start()

    def _progress_cb(self, done: int, total: int, r: BenchmarkResult) -> None:
        pct = done / total * 100
        msg = (
            f"  [{done}/{total}] {r.task_id}:"
            f" composite={r.composite:.4f} latency={r.latency_s:.2f}s"
            + (f"  ERROR: {r.error}" if r.error else "")
        )
        # schedule UI update on main thread
        self.root.after(0, lambda: self._progress_var.set(pct))
        self.root.after(
            0, lambda: self._progress_lbl.config(text=f"{done}/{total} — {r.task_id}")
        )
        self.root.after(0, lambda: self._log(msg))

    def _api_run_thread(self) -> None:
        try:
            runner = BenchmarkRunner(self._tasks)
            provider = self._provider_var.get()
            api_key = self._api_key_var.get().strip()
            model = self._model_var.get().strip()
            max_tokens = self._max_tokens_var.get()
            temperature = self._temperature_var.get()

            if provider == "anthropic":
                runner.run_anthropic(
                    api_key=api_key,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    progress_callback=self._progress_cb,
                )
            else:
                runner.run_openai(
                    api_key=api_key,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=self._system_prompt_var.get(),
                    progress_callback=self._progress_cb,
                )
            self._runner = runner
            self._results = runner.results
            self.root.after(0, self._on_run_complete)
        except Exception as exc:
            self.root.after(0, lambda: self._log(f"ERROR: {exc}"))
            self.root.after(0, lambda: self._set_running(False))

    def _demo_run_thread(self) -> None:
        try:
            runner = BenchmarkRunner(self._tasks)
            runner.run_offline(
                lambda p: p.split("?")[0].strip() if "?" in p else p[:40],
                progress_callback=self._progress_cb,
            )
            self._runner = runner
            self._results = runner.results
            self.root.after(0, self._on_run_complete)
        except Exception as exc:
            self.root.after(0, lambda: self._log(f"ERROR: {exc}"))
            self.root.after(0, lambda: self._set_running(False))

    def _on_run_complete(self) -> None:
        self._log(f"Done — {len(self._results)} result(s) collected.")
        self._progress_var.set(100)
        self._refresh_results_table()
        self._refresh_summary()
        self._set_running(False)

    # ------------------------------------------------------------------
    # Results tab
    # ------------------------------------------------------------------

    def _build_results_tab(self, f: ttk.Frame) -> None:
        cols = (
            "task_id", "category", "em", "rouge_l",
            "bleu_1", "f1", "composite", "latency", "tokens", "error",
        )
        headers = {
            "task_id":   ("Task ID",     115),
            "category":  ("Category",     90),
            "em":        ("EM",           48),
            "rouge_l":   ("ROUGE-L",      70),
            "bleu_1":    ("BLEU-1",       65),
            "f1":        ("F1",           52),
            "composite": ("Composite",    80),
            "latency":   ("Latency (s)",  85),
            "tokens":    ("Tokens",       58),
            "error":     ("Error",       140),
        }

        tree_frame = ttk.Frame(f)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._res_tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings", selectmode="browse"
        )
        for col in cols:
            lbl, w = headers[col]
            self._res_tree.heading(
                col, text=lbl, command=lambda c=col: self._sort_results(c)
            )
            self._res_tree.column(col, width=w, minwidth=40, anchor="center")
        self._res_tree.column("task_id", anchor="w")
        self._res_tree.column("category", anchor="w")
        self._res_tree.column("error", anchor="w")

        self._res_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        res_sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._res_tree.yview)
        self._res_tree.configure(yscrollcommand=res_sb.set)
        res_sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Detail pane
        detail_lf = ttk.LabelFrame(f, text="Prediction Detail")
        detail_lf.pack(fill=tk.X, padx=6, pady=(0, 6))
        self._detail_text = tk.Text(
            detail_lf, height=5, state=tk.DISABLED, wrap=tk.WORD, font=_MONOSPACE
        )
        self._detail_text.pack(fill=tk.X, padx=4, pady=4)
        self._res_tree.bind("<<TreeviewSelect>>", self._on_result_select)

        # Tag colours
        self._res_tree.tag_configure("odd", background="#f7f7f7")
        self._res_tree.tag_configure("even", background="#ffffff")
        self._res_tree.tag_configure("error_row", background="#ffe4e4")

    def _refresh_results_table(self) -> None:
        self._res_tree.delete(*self._res_tree.get_children())
        for i, r in enumerate(self._results):
            tag = "error_row" if r.error else ("odd" if i % 2 else "even")
            self._res_tree.insert(
                "", tk.END, iid=str(i), tags=(tag,),
                values=(
                    r.task_id, r.category,
                    _fmt(r.exact_match, 2),
                    _fmt(r.rouge_l),
                    _fmt(r.bleu_1),
                    _fmt(r.f1),
                    _fmt(r.composite),
                    _fmt(r.latency_s, 3),
                    r.approx_tokens,
                    r.error or "",
                ),
            )

    def _on_result_select(self, _event=None) -> None:
        sel = self._res_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx >= len(self._results):
            return
        r = self._results[idx]
        text = (
            f"Prompt:     {r.prompt}\n"
            f"Reference:  {r.reference}\n"
            f"Prediction: {r.prediction}\n"
            f"SHA-256:    {r.sha256 or 'N/A'}"
        )
        self._detail_text.config(state=tk.NORMAL)
        self._detail_text.delete("1.0", tk.END)
        self._detail_text.insert(tk.END, text)
        self._detail_text.config(state=tk.DISABLED)

    def _sort_results(self, col: str) -> None:
        items = [
            (self._res_tree.set(iid, col), iid)
            for iid in self._res_tree.get_children()
        ]
        try:
            items.sort(key=lambda x: float(x[0]) if x[0] else -1.0, reverse=True)
        except ValueError:
            items.sort(key=lambda x: x[0])
        for pos, (_, iid) in enumerate(items):
            self._res_tree.move(iid, "", pos)

    # ------------------------------------------------------------------
    # Summary tab
    # ------------------------------------------------------------------

    def _build_summary_tab(self, f: ttk.Frame) -> None:
        ttk.Label(f, text="Benchmark Summary", font=_HEADER_FONT).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        inner = ttk.Frame(f)
        inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self._summary_text = tk.Text(
            inner, state=tk.DISABLED, wrap=tk.WORD, font=("Courier", 10)
        )
        self._summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(inner, orient=tk.VERTICAL, command=self._summary_text.yview)
        self._summary_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def _refresh_summary(self) -> None:
        if not self._runner or not self._results:
            return
        s = self._runner.summarize(self._results)
        ov = s.get("overall", {})
        lines = [
            "=" * 52,
            "OVERALL",
            "=" * 52,
            f"  Tasks:             {ov.get('n', 0)}",
            f"  Errors:            {ov.get('errors', 0)}",
            f"  Exact Match:       {ov.get('exact_match', 0):.4f}",
            f"  Exact Match (n):   {ov.get('exact_match_norm', 0):.4f}",
            f"  ROUGE-L:           {ov.get('rouge_l', 0):.4f}",
            f"  BLEU-1:            {ov.get('bleu_1', 0):.4f}",
            f"  F1:                {ov.get('f1', 0):.4f}",
            f"  Composite:         {ov.get('composite', 0):.4f}",
            f"  Avg Latency (s):   {ov.get('avg_latency_s', 0):.4f}",
            "",
            "BY CATEGORY",
            "-" * 52,
        ]
        for cat, st in s.get("by_category", {}).items():
            lines += [
                f"  [{cat}]  N={st['n']}  errors={st['errors']}",
                f"    ROUGE-L={st['rouge_l']:.4f}  F1={st['f1']:.4f}"
                f"  Composite={st['composite']:.4f}",
                f"    EM={st['exact_match']:.4f}  BLEU-1={st['bleu_1']:.4f}"
                f"  Latency={st['avg_latency_s']:.4f}s",
                "",
            ]
        self._summary_text.config(state=tk.NORMAL)
        self._summary_text.delete("1.0", tk.END)
        self._summary_text.insert(tk.END, "\n".join(lines) + "\n")
        self._summary_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _require_results(self) -> bool:
        if not self._results:
            messagebox.showwarning("No Results", "Run a benchmark first (Run tab).")
            return False
        return True

    def _export_jsonl(self) -> None:
        if not self._require_results():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL", "*.jsonl"), ("All", "*.*")],
        )
        if path:
            BenchmarkRunner().export_jsonl(path, self._results)
            messagebox.showinfo("Exported", f"JSONL saved to:\n{path}")

    def _export_csv(self) -> None:
        if not self._require_results():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if path:
            BenchmarkRunner().export_csv(path, self._results)
            messagebox.showinfo("Exported", f"CSV saved to:\n{path}")

    def _export_md(self) -> None:
        if not self._require_results():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("All", "*.*")],
        )
        if path:
            r = BenchmarkRunner()
            r.results = self._results
            r.export_markdown(path, self._results)
            messagebox.showinfo("Exported", f"Markdown report saved to:\n{path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def launch() -> None:
    """Create the Tk root and run the application event loop."""
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except tk.TclError:
        pass
    LLMBenchApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
