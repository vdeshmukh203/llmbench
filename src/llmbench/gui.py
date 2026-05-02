"""Tkinter GUI for llmbench. Stdlib-only, no external dependencies."""
from __future__ import annotations

import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path
from typing import List, Optional

from .models import BenchmarkResult
from .runner import BenchmarkRunner
from .spec import BenchmarkSpec


class _LLMBenchApp:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("llmbench — LLM Benchmark Runner")
        self.root.geometry("1050x680")
        self.root.minsize(800, 560)

        self._spec: BenchmarkSpec = BenchmarkSpec.builtin()
        self._runner: Optional[BenchmarkRunner] = None
        self._res_sort_col: Optional[str] = None
        self._res_sort_asc: bool = True

        self._build_menu()
        self._build_notebook()
        self._refresh_tasks()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_m = tk.Menu(menubar, tearoff=0)
        file_m.add_command(label="Load Task File…", command=self._load_task_file)
        file_m.add_command(label="Use Built-in Sample Tasks", command=self._use_builtin)
        file_m.add_separator()
        file_m.add_command(label="Export Results (JSONL)…", command=self._export_jsonl)
        file_m.add_command(label="Export Results (CSV)…", command=self._export_csv)
        file_m.add_separator()
        file_m.add_command(label="Quit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_m)
        self.root.config(menu=menubar)

    # ------------------------------------------------------------------
    # Notebook / tabs
    # ------------------------------------------------------------------

    def _build_notebook(self) -> None:
        self._nb = ttk.Notebook(self.root)
        self._nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._build_tasks_tab()
        self._build_run_tab()
        self._build_results_tab()
        self._build_summary_tab()

    # ── Tasks tab ─────────────────────────────────────────────────────

    def _build_tasks_tab(self) -> None:
        frame = ttk.Frame(self._nb)
        self._nb.add(frame, text="Tasks")

        toolbar = ttk.Frame(frame)
        toolbar.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(toolbar, text="Load File…", command=self._load_task_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Use Built-in Sample", command=self._use_builtin).pack(side=tk.LEFT, padx=2)
        self._task_label = tk.StringVar(value="0 tasks")
        ttk.Label(toolbar, textvariable=self._task_label, foreground="#555").pack(side=tk.RIGHT, padx=8)

        cols = ("task_id", "category", "prompt", "reference")
        self._task_tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="browse")
        for col, width in zip(cols, (100, 90, 420, 200)):
            self._task_tree.heading(col, text=col.replace("_", " ").title())
            self._task_tree.column(col, width=width, anchor=tk.W)
        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._task_tree.yview)
        self._task_tree.configure(yscrollcommand=vsb.set)
        self._task_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=4)
        vsb.pack(side=tk.RIGHT, fill=tk.Y, pady=4, padx=(0, 4))

    def _refresh_tasks(self) -> None:
        self._task_tree.delete(*self._task_tree.get_children())
        for t in self._spec.tasks:
            self._task_tree.insert(
                "",
                tk.END,
                values=(
                    t.task_id,
                    t.category,
                    (t.prompt[:80] + "…") if len(t.prompt) > 80 else t.prompt,
                    (t.reference[:60] + "…") if len(t.reference) > 60 else t.reference,
                ),
            )
        self._task_label.set(f"{len(self._spec.tasks)} tasks  [{self._spec.name}]")

    # ── Run tab ────────────────────────────────────────────────────────

    def _build_run_tab(self) -> None:
        frame = ttk.Frame(self._nb)
        self._nb.add(frame, text="Run")

        mode_frame = ttk.LabelFrame(frame, text="Run Mode")
        mode_frame.pack(fill=tk.X, padx=8, pady=6)
        self._run_mode = tk.StringVar(value="offline")
        ttk.Radiobutton(
            mode_frame, text="Offline (echo model — no API needed)",
            variable=self._run_mode, value="offline", command=self._toggle_api_fields,
        ).pack(side=tk.LEFT, padx=10, pady=4)
        ttk.Radiobutton(
            mode_frame, text="OpenAI-compatible API",
            variable=self._run_mode, value="openai", command=self._toggle_api_fields,
        ).pack(side=tk.LEFT, padx=10, pady=4)

        self._api_frame = ttk.LabelFrame(frame, text="API Settings")
        self._api_frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(self._api_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, padx=6, pady=3)
        self._api_key_var = tk.StringVar()
        self._api_key_entry = ttk.Entry(self._api_frame, textvariable=self._api_key_var, width=52, show="*")
        self._api_key_entry.grid(row=0, column=1, sticky=tk.EW, padx=6, pady=3)

        ttk.Label(self._api_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=6, pady=3)
        self._model_var = tk.StringVar(value="gpt-4o-mini")
        self._model_entry = ttk.Entry(self._api_frame, textvariable=self._model_var, width=30)
        self._model_entry.grid(row=1, column=1, sticky=tk.W, padx=6, pady=3)

        ttk.Label(self._api_frame, text="Base URL:").grid(row=2, column=0, sticky=tk.W, padx=6, pady=3)
        self._base_url_var = tk.StringVar(value="https://api.openai.com/v1")
        self._base_url_entry = ttk.Entry(self._api_frame, textvariable=self._base_url_var, width=52)
        self._base_url_entry.grid(row=2, column=1, sticky=tk.EW, padx=6, pady=3)
        self._api_frame.columnconfigure(1, weight=1)

        self._toggle_api_fields()

        ctrl = ttk.Frame(frame)
        ctrl.pack(fill=tk.X, padx=8, pady=6)
        self._run_btn = ttk.Button(ctrl, text="▶  Run Benchmark", command=self._run_benchmark)
        self._run_btn.pack(side=tk.LEFT, padx=4)
        self._progress = ttk.Progressbar(ctrl, mode="indeterminate", length=180)
        self._progress.pack(side=tk.LEFT, padx=8)
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(ctrl, textvariable=self._status_var).pack(side=tk.LEFT, padx=4)

        log_frame = ttk.LabelFrame(frame, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self._log = scrolledtext.ScrolledText(
            log_frame, height=10, font=("Courier", 9), state=tk.DISABLED
        )
        self._log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _toggle_api_fields(self) -> None:
        state = tk.NORMAL if self._run_mode.get() == "openai" else tk.DISABLED
        for widget in (self._api_key_entry, self._model_entry, self._base_url_entry):
            widget.configure(state=state)

    def _log_append(self, msg: str) -> None:
        self._log.configure(state=tk.NORMAL)
        self._log.insert(tk.END, msg + "\n")
        self._log.see(tk.END)
        self._log.configure(state=tk.DISABLED)

    def _run_benchmark(self) -> None:
        if not self._spec.tasks:
            messagebox.showwarning("No Tasks", "Load tasks before running.")
            return
        self._run_btn.configure(state=tk.DISABLED)
        self._progress.start(10)
        mode = self._run_mode.get()
        self._status_var.set("Running…")
        self._log_append(f"Starting: {len(self._spec.tasks)} tasks, mode={mode}")

        def _worker() -> None:
            try:
                runner = BenchmarkRunner(spec=self._spec)
                if mode == "openai":
                    api_key = self._api_key_var.get().strip()
                    if not api_key:
                        self.root.after(
                            0,
                            lambda: messagebox.showerror("Missing API Key", "Enter an API key first."),
                        )
                        self.root.after(0, self._reset_run_ui)
                        return
                    runner.run_openai(
                        api_key=api_key,
                        model=self._model_var.get().strip(),
                        base_url=self._base_url_var.get().strip(),
                    )
                else:
                    runner.run_offline(
                        lambda p: p.split("?")[0].strip() if "?" in p else p[:40]
                    )
                self._runner = runner
                self.root.after(0, self._on_run_complete)
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._on_run_error(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _reset_run_ui(self) -> None:
        self._progress.stop()
        self._run_btn.configure(state=tk.NORMAL)
        self._status_var.set("Ready.")

    def _on_run_complete(self) -> None:
        self._progress.stop()
        self._run_btn.configure(state=tk.NORMAL)
        n = len(self._runner.results) if self._runner else 0
        self._status_var.set(f"Done — {n} results.")
        self._log_append(f"Complete: {n} results collected.")
        self._refresh_results()
        self._refresh_summary()
        self._nb.select(2)  # jump to Results tab

    def _on_run_error(self, exc: Exception) -> None:
        self._progress.stop()
        self._run_btn.configure(state=tk.NORMAL)
        self._status_var.set("Error.")
        self._log_append(f"ERROR: {exc}")
        messagebox.showerror("Run Failed", str(exc))

    # ── Results tab ────────────────────────────────────────────────────

    def _build_results_tab(self) -> None:
        frame = ttk.Frame(self._nb)
        self._nb.add(frame, text="Results")

        cols = ("task_id", "category", "composite", "rouge_l", "f1", "exact_match", "bleu_1", "latency_s", "error")
        self._res_tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="browse")
        widths = (90, 100, 80, 72, 72, 80, 72, 85, 150)
        for col, w in zip(cols, widths):
            self._res_tree.heading(
                col,
                text=col.replace("_", " ").title(),
                command=lambda c=col: self._sort_results(c),
            )
            self._res_tree.column(col, width=w, anchor=tk.CENTER)

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._res_tree.yview)
        hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self._res_tree.xview)
        self._res_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._res_tree.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=(4, 0))
        vsb.grid(row=0, column=1, sticky="ns", pady=(4, 0))
        hsb.grid(row=1, column=0, sticky="ew", padx=(4, 0))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    def _refresh_results(self) -> None:
        self._res_tree.delete(*self._res_tree.get_children())
        if not self._runner:
            return
        for r in self._runner.results:
            tag = "error" if r.error else ""
            self._res_tree.insert(
                "",
                tk.END,
                tags=(tag,),
                values=(
                    r.task_id,
                    r.category,
                    f"{r.composite_score:.4f}",
                    f"{r.rouge_l:.4f}",
                    f"{r.f1:.4f}",
                    f"{r.exact_match:.4f}",
                    f"{r.bleu_1:.4f}",
                    f"{r.latency_s:.3f}s",
                    r.error or "",
                ),
            )
        self._res_tree.tag_configure("error", foreground="#c0392b")

    def _sort_results(self, col: str) -> None:
        if self._res_sort_col == col:
            self._res_sort_asc = not self._res_sort_asc
        else:
            self._res_sort_col = col
            self._res_sort_asc = True
        items = [(self._res_tree.set(k, col), k) for k in self._res_tree.get_children("")]
        try:
            items.sort(key=lambda x: float(x[0].rstrip("s")), reverse=not self._res_sort_asc)
        except ValueError:
            items.sort(key=lambda x: x[0], reverse=not self._res_sort_asc)
        for idx, (_, k) in enumerate(items):
            self._res_tree.move(k, "", idx)

    # ── Summary tab ────────────────────────────────────────────────────

    def _build_summary_tab(self) -> None:
        frame = ttk.Frame(self._nb)
        self._nb.add(frame, text="Summary")

        self._summary_text = scrolledtext.ScrolledText(
            frame, font=("Courier", 10), state=tk.DISABLED, wrap=tk.NONE
        )
        self._summary_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _refresh_summary(self) -> None:
        if not self._runner:
            return
        text = json.dumps(self._runner.summarize(), indent=2)
        self._summary_text.configure(state=tk.NORMAL)
        self._summary_text.delete("1.0", tk.END)
        self._summary_text.insert(tk.END, text)
        self._summary_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _load_task_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Task File",
            filetypes=[("JSON / YAML files", "*.json *.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self._spec = BenchmarkSpec.from_file(Path(path))
            self._refresh_tasks()
            self._log_append(f"Loaded {len(self._spec.tasks)} tasks from {path}")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _use_builtin(self) -> None:
        self._spec = BenchmarkSpec.builtin()
        self._refresh_tasks()
        self._log_append("Loaded built-in sample tasks.")

    def _export_jsonl(self) -> None:
        if not self._runner or not self._runner.results:
            messagebox.showinfo("No Results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
        if not path:
            return
        self._runner.export_jsonl(Path(path))
        messagebox.showinfo("Exported", f"Saved {len(self._runner.results)} results to:\n{path}")

    def _export_csv(self) -> None:
        if not self._runner or not self._runner.results:
            messagebox.showinfo("No Results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self._runner.export_csv(Path(path))
        messagebox.showinfo("Exported", f"Saved {len(self._runner.results)} results to:\n{path}")


def launch_gui() -> None:
    """Launch the llmbench GUI (blocking until window is closed)."""
    root = tk.Tk()
    _LLMBenchApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
