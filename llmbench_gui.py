#!/usr/bin/env python3
"""
llmbench_gui.py — Tkinter GUI for the llmbench benchmark runner.

Launch with:  python llmbench_gui.py
          or: llmbench-gui   (if installed via pip)
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext
import tkinter as tk
from tkinter import ttk

# Allow running directly from the repo root
sys.path.insert(0, str(Path(__file__).parent))
import llmbench as lb


class _App:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("llmbench — LLM Benchmark Runner")
        self.root.minsize(980, 660)

        self._runner = lb.BenchmarkRunner()
        self._results: list[lb.BenchmarkResult] = []
        self._tree_sort_rev: dict[str, bool] = {}

        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        self._build_toolbar()

        self._nb = ttk.Notebook(self.root)
        self._nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))

        cfg_tab = ttk.Frame(self._nb, padding=4)
        res_tab = ttk.Frame(self._nb)
        sum_tab = ttk.Frame(self._nb)
        self._nb.add(cfg_tab, text="  Configuration  ")
        self._nb.add(res_tab, text="  Results  ")
        self._nb.add(sum_tab, text="  Summary  ")

        self._build_config(cfg_tab)
        self._build_results(res_tab)
        self._build_summary(sum_tab)

        self._status = tk.StringVar(value="Ready.")
        ttk.Label(
            self.root, textvariable=self._status,
            relief=tk.SUNKEN, anchor=tk.W, padding=(6, 2),
        ).pack(fill=tk.X, side=tk.BOTTOM, padx=6, pady=(0, 4))

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=6, pady=6)

        self._run_btn = ttk.Button(bar, text="▶  Run Benchmark", command=self._on_run)
        self._run_btn.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="Export CSV",  command=self._export_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Export JSONL", command=self._export_jsonl).pack(side=tk.LEFT, padx=2)

        self._pb = ttk.Progressbar(bar, mode="indeterminate", length=200)
        self._pb.pack(side=tk.RIGHT)

    def _build_config(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        # ── Inference mode ────────────────────────────────────────────
        mf = ttk.LabelFrame(parent, text="Inference Mode", padding=8)
        mf.grid(row=0, column=0, sticky=tk.EW, pady=(4, 4))

        self._mode = tk.StringVar(value="offline")
        ttk.Radiobutton(
            mf, text="Offline demo  (no API key required)",
            variable=self._mode, value="offline", command=self._on_mode,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            mf, text="OpenAI-compatible API",
            variable=self._mode, value="openai", command=self._on_mode,
        ).pack(anchor=tk.W)

        # ── API settings ──────────────────────────────────────────────
        af = ttk.LabelFrame(parent, text="API Settings", padding=8)
        af.grid(row=1, column=0, sticky=tk.EW, pady=4)
        af.columnconfigure(1, weight=1)

        self._api_key  = tk.StringVar()
        self._base_url = tk.StringVar(value="https://api.openai.com/v1/chat/completions")
        self._model    = tk.StringVar(value="gpt-3.5-turbo")
        self._max_tok  = tk.IntVar(value=256)
        self._temp     = tk.DoubleVar(value=0.0)
        self._seed     = tk.StringVar(value="")

        fields = [
            ("API Key:",          ttk.Entry(af, textvariable=self._api_key, show="*")),
            ("Base URL:",         ttk.Entry(af, textvariable=self._base_url)),
            ("Model:",            ttk.Entry(af, textvariable=self._model, width=32)),
            ("Max Tokens:",       ttk.Spinbox(af, from_=1, to=4096, textvariable=self._max_tok, width=8)),
            ("Temperature:",      ttk.Spinbox(af, from_=0.0, to=2.0, increment=0.1,
                                              textvariable=self._temp, format="%.1f", width=8)),
            ("Seed (optional):",  ttk.Entry(af, textvariable=self._seed, width=12)),
        ]
        self._api_widgets: list[tk.Widget] = [w for _, w in fields]
        for i, (label, widget) in enumerate(fields):
            ttk.Label(af, text=label).grid(row=i, column=0, sticky=tk.W, padx=(0, 8), pady=2)
            widget.grid(row=i, column=1, sticky=tk.EW, pady=2)

        # ── Tasks source ──────────────────────────────────────────────
        tf = ttk.LabelFrame(parent, text="Tasks", padding=8)
        tf.grid(row=2, column=0, sticky=tk.EW, pady=4)
        tf.columnconfigure(1, weight=1)

        self._tasks_src = tk.StringVar(value="sample")
        ttk.Radiobutton(
            tf, text="Use built-in sample tasks  (10 tasks)",
            variable=self._tasks_src, value="sample",
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W)

        ttk.Radiobutton(
            tf, text="Load from file:",
            variable=self._tasks_src, value="file",
        ).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))

        self._tasks_path = tk.StringVar()
        ttk.Entry(tf, textvariable=self._tasks_path).grid(
            row=1, column=1, sticky=tk.EW, padx=4, pady=(4, 0))
        ttk.Button(tf, text="Browse…", command=self._browse).grid(
            row=1, column=2, pady=(4, 0))

        ttk.Label(
            tf, text="Formats: JSON array, JSONL, YAML (requires pyyaml)",
            foreground="gray",
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))

        self._on_mode()  # set initial widget states

    def _build_results(self, parent: ttk.Frame) -> None:
        cols = (
            "task_id", "category", "em", "em_norm", "rouge_l",
            "bleu_1", "f1", "composite", "latency_s", "tokens", "error",
        )
        heads = (
            "Task ID", "Category", "EM", "EM-Norm", "ROUGE-L",
            "BLEU-1", "F1", "Composite", "Latency (s)", "Tokens", "Error",
        )
        widths = [80, 110, 62, 68, 68, 62, 62, 80, 80, 60, 200]

        self._tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="browse")
        for col, head, w in zip(cols, heads, widths):
            self._tree.heading(col, text=head, command=lambda c=col: self._sort(c))
            self._tree.column(col, width=w, anchor=tk.CENTER, stretch=(col == "error"))
        self._tree.column("error", anchor=tk.W)

        sy = ttk.Scrollbar(parent, orient=tk.VERTICAL,   command=self._tree.yview)
        sx = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)

        sy.pack(side=tk.RIGHT,  fill=tk.Y)
        sx.pack(side=tk.BOTTOM, fill=tk.X)
        self._tree.pack(fill=tk.BOTH, expand=True)

    def _build_summary(self, parent: ttk.Frame) -> None:
        self._sum_text = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, font=("Courier", 10), state=tk.DISABLED,
        )
        self._sum_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_mode(self) -> None:
        state = tk.NORMAL if self._mode.get() == "openai" else tk.DISABLED
        for w in self._api_widgets:
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select task file",
            filetypes=[
                ("Task files", "*.json *.jsonl *.yaml *.yml"),
                ("All files",  "*.*"),
            ],
        )
        if path:
            self._tasks_path.set(path)
            self._tasks_src.set("file")

    def _on_run(self) -> None:
        self._run_btn.configure(state=tk.DISABLED)
        self._pb.start(10)
        self._status.set("Running benchmark…")
        threading.Thread(target=self._run_thread, daemon=True).start()

    # Runs in a background thread — no direct Tk calls here
    def _run_thread(self) -> None:
        try:
            if self._tasks_src.get() == "file" and self._tasks_path.get():
                tasks = lb.load_tasks_from_file(Path(self._tasks_path.get()))
            else:
                tasks = list(lb.SAMPLE_TASKS)

            self._runner = lb.BenchmarkRunner(tasks)

            if self._mode.get() == "openai":
                api_key = self._api_key.get() or os.environ.get("OPENAI_API_KEY", "")
                if not api_key:
                    self.root.after(0, lambda: messagebox.showerror(
                        "API Key Required",
                        "Enter an API key or set the OPENAI_API_KEY environment variable.",
                    ))
                    return
                seed_str = self._seed.get().strip()
                seed = int(seed_str) if seed_str.isdigit() else None
                self._results = self._runner.run_openai(
                    api_key=api_key,
                    model=self._model.get(),
                    base_url=self._base_url.get(),
                    max_tokens=self._max_tok.get(),
                    temperature=self._temp.get(),
                    seed=seed,
                )
            else:
                self._results = self._runner.run_offline(
                    lambda p: p.split("?")[0].strip() if "?" in p else p[:40]
                )

            self.root.after(0, self._update_ui)
        except Exception as exc:
            msg = str(exc)
            self.root.after(0, lambda: messagebox.showerror("Benchmark Error", msg))
        finally:
            self.root.after(0, self._stop_progress)

    def _stop_progress(self) -> None:
        self._pb.stop()
        self._run_btn.configure(state=tk.NORMAL)
        n = len(self._results)
        self._status.set(f"Completed — {n} task{'s' if n != 1 else ''} evaluated.")

    def _update_ui(self) -> None:
        # Populate results treeview
        for item in self._tree.get_children():
            self._tree.delete(item)
        for r in self._results:
            tag = "err" if r.error else ("hi" if r.composite_score >= 0.5 else "lo")
            self._tree.insert("", tk.END, tags=(tag,), values=(
                r.task_id, r.category,
                f"{r.exact_match:.3f}", f"{r.exact_match_norm:.3f}",
                f"{r.rouge_l:.3f}",    f"{r.bleu_1:.3f}",
                f"{r.f1:.3f}",         f"{r.composite_score:.3f}",
                f"{r.latency_s:.3f}",  r.approx_tokens,
                r.error or "",
            ))
        self._tree.tag_configure("err", background="#ffe0e0")
        self._tree.tag_configure("hi",  background="#e8ffe8")

        # Populate summary tab
        summary = self._runner.summarize()
        self._sum_text.configure(state=tk.NORMAL)
        self._sum_text.delete("1.0", tk.END)
        self._sum_text.insert(tk.END, json.dumps(summary, indent=2))
        self._sum_text.configure(state=tk.DISABLED)

        # Switch to results tab
        self._nb.select(1)

    def _sort(self, col: str) -> None:
        """Sort the results treeview by *col* (toggle ascending/descending)."""
        rev = self._tree_sort_rev.get(col, False)
        rows = [(self._tree.set(k, col), k) for k in self._tree.get_children("")]
        try:
            rows.sort(key=lambda t: float(t[0]), reverse=rev)
        except ValueError:
            rows.sort(key=lambda t: t[0].lower(), reverse=rev)
        for idx, (_, k) in enumerate(rows):
            self._tree.move(k, "", idx)
        self._tree_sort_rev[col] = not rev

    def _export_csv(self) -> None:
        if not self._results:
            messagebox.showwarning("No Results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._runner.export_csv(Path(path))
            self._status.set(f"Exported CSV → {path}")

    def _export_jsonl(self) -> None:
        if not self._results:
            messagebox.showwarning("No Results", "Run a benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL", "*.jsonl"), ("All files", "*.*")],
        )
        if path:
            self._runner.export_jsonl(Path(path))
            sha_path = Path(path).with_suffix(".sha256")
            self._status.set(f"Exported JSONL → {path}  (SHA-256: {sha_path.name})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_gui() -> None:
    """Launch the llmbench graphical interface."""
    root = tk.Tk()
    _App(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
