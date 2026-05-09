"""Command-line interface for llmbench.

Usage::

    llmbench sample                        # print built-in tasks as JSONL
    llmbench run --api-key KEY --model gpt-4o-mini
    llmbench run --provider anthropic --api-key KEY --model claude-3-haiku-20240307
    llmbench score results.jsonl           # score an existing predictions file
    llmbench gui                           # launch the graphical interface
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ._sample_tasks import SAMPLE_TASKS
from .runner import BenchmarkRunner, BenchmarkResult
from .spec import BenchmarkSpec, Task


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llmbench",
        description="Benchmark LLM outputs against reference answers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # ---- run ----
    run_p = sub.add_parser("run", help="Benchmark an OpenAI or Anthropic API.")
    run_p.add_argument(
        "--api-key", default=None,
        help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY).",
    )
    run_p.add_argument("--model", default="gpt-3.5-turbo", help="Model identifier.")
    run_p.add_argument("--tasks", default=None, help="JSONL task file path.")
    run_p.add_argument("--output", "-o", default="results.jsonl", help="Output JSONL path.")
    run_p.add_argument("--csv", default=None, help="Also export CSV.")
    run_p.add_argument("--markdown", default=None, help="Also export Markdown report.")
    run_p.add_argument(
        "--provider", choices=["openai", "anthropic"], default="openai",
        help="API provider (default: openai).",
    )
    run_p.add_argument("--max-tokens", type=int, default=256)
    run_p.add_argument("--temperature", type=float, default=0.0)

    # ---- score ----
    score_p = sub.add_parser(
        "score", help="Score an existing predictions JSONL file."
    )
    score_p.add_argument("predictions", help="JSONL file with prediction/reference pairs.")
    score_p.add_argument(
        "--format", choices=["json", "text"], default="text",
        help="Output format (default: text).",
    )

    # ---- sample ----
    sub.add_parser("sample", help="Print built-in sample tasks as JSONL.")

    # ---- gui ----
    sub.add_parser("gui", help="Launch the graphical interface (requires tkinter).")

    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    # ---- gui ----
    if args.command == "gui":
        try:
            from .gui import launch
        except ImportError as exc:
            print(f"GUI unavailable: {exc}", file=sys.stderr)
            return 1
        launch()
        return 0

    # ---- sample ----
    if args.command == "sample":
        for t in SAMPLE_TASKS:
            print(json.dumps(t.to_dict(), ensure_ascii=False))
        return 0

    # ---- score ----
    if args.command == "score":
        path = Path(args.predictions)
        if not path.is_file():
            print(f"Error: {path} not found", file=sys.stderr)
            return 1
        results: list[BenchmarkResult] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pred = obj.get("prediction", "")
                t = Task(
                    task_id=obj.get("task_id", "?"),
                    category=obj.get("category", "unknown"),
                    prompt=obj.get("prompt", ""),
                    reference=obj.get("reference", ""),
                )
                # capture pred per iteration to avoid closure-over-loop bug
                runner = BenchmarkRunner([t])
                _pred = pred
                results.extend(runner.run_offline(lambda _: _pred, [t]))
        reporter = BenchmarkRunner()
        reporter.results = results
        summary = reporter.summarize()
        ov = summary.get("overall", {})
        if args.format == "json":
            print(json.dumps(summary, indent=2))
        else:
            print(
                f"Tasks: {ov.get('n', 0)}"
                f"  EM: {ov.get('exact_match', 0):.4f}"
                f"  ROUGE-L: {ov.get('rouge_l', 0):.4f}"
                f"  F1: {ov.get('f1', 0):.4f}"
                f"  Composite: {ov.get('composite', 0):.4f}"
            )
        return 0

    # ---- run ----
    if args.command == "run":
        env_key = (
            "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
        )
        api_key = args.api_key or os.environ.get(env_key, "")
        if not api_key:
            print(
                f"Error: provide --api-key or set {env_key}", file=sys.stderr
            )
            return 1
        spec = BenchmarkSpec.from_jsonl(Path(args.tasks)) if args.tasks else BenchmarkSpec(SAMPLE_TASKS)
        runner = BenchmarkRunner(spec=spec)

        def _progress(done: int, total: int, r: BenchmarkResult) -> None:
            print(
                f"  [{done}/{total}] {r.task_id}:"
                f" composite={r.composite:.4f} latency={r.latency_s:.2f}s"
                + (f" ERROR: {r.error}" if r.error else "")
            )

        if args.provider == "anthropic":
            runner.run_anthropic(
                api_key=api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                progress_callback=_progress,
            )
        else:
            runner.run_openai(
                api_key=api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                progress_callback=_progress,
            )
        runner.export_jsonl(Path(args.output))
        print(f"\nResults written to {args.output}")
        if args.csv:
            runner.export_csv(Path(args.csv))
            print(f"CSV written to {args.csv}")
        if args.markdown:
            runner.export_markdown(Path(args.markdown))
            print(f"Markdown report written to {args.markdown}")
        print(json.dumps(runner.summarize(), indent=2))
        return 0

    # no subcommand: offline demo
    runner = BenchmarkRunner(list(SAMPLE_TASKS))
    runner.run_offline(lambda p: p.split("?")[0].strip() if "?" in p else p[:40])
    print(json.dumps(runner.summarize(), indent=2))
    return 0
