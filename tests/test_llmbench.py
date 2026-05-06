"""Tests for the llmbench package.

Covers metrics, Task/BenchmarkSpec, BenchmarkRunner, BenchmarkResult,
CSV/JSONL export, CLI, and the flat llmbench.py compatibility shim.
"""

import json
import sys
import math
import tempfile
from pathlib import Path

import pytest

# ── make sure we import the src-layout package first ───────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import llmbench as lb
from llmbench.metrics import (
    _tokenise,
    exact_match,
    exact_match_normalised,
    rouge_l,
    bleu_1,
    f1_score,
    _approx_tokens,
    contains_code,
)
from llmbench.spec import Task, BenchmarkSpec, SAMPLE_TASKS
from llmbench.runner import BenchmarkResult, BenchmarkRunner


# ===========================================================================
# Metrics
# ===========================================================================

class TestTokenise:
    def test_lowercases(self):
        assert _tokenise("Hello WORLD") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert _tokenise("hello, world!") == ["hello", "world"]

    def test_empty(self):
        assert _tokenise("") == []

    def test_numbers(self):
        assert "2024" in _tokenise("Year 2024")


class TestExactMatch:
    def test_identical(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_different(self):
        assert exact_match("London", "Paris") == 0.0

    def test_strips_whitespace(self):
        assert exact_match("  Paris  ", "Paris") == 1.0

    def test_case_sensitive(self):
        assert exact_match("paris", "Paris") == 0.0


class TestExactMatchNormalised:
    def test_identical(self):
        assert exact_match_normalised("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        assert exact_match_normalised("paris", "Paris") == 1.0

    def test_punctuation_ignored(self):
        assert exact_match_normalised("H2O!", "H2O") == 1.0

    def test_different(self):
        assert exact_match_normalised("London", "Paris") == 0.0


class TestRougeL:
    def test_perfect(self):
        assert rouge_l("the cat sat", "the cat sat") == 1.0

    def test_zero_empty_pred(self):
        assert rouge_l("", "reference") == 0.0

    def test_zero_empty_ref(self):
        assert rouge_l("prediction", "") == 0.0

    def test_partial_overlap(self):
        score = rouge_l("the cat", "the cat sat")
        assert 0.0 < score < 1.0

    def test_symmetry_approx(self):
        a = rouge_l("a b c d", "a b e f")
        b = rouge_l("a b e f", "a b c d")
        assert abs(a - b) < 1e-9


class TestBleu1:
    def test_perfect(self):
        assert bleu_1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_empty_pred(self):
        assert bleu_1("", "reference") == 0.0

    def test_no_overlap(self):
        assert bleu_1("xyz", "abc") == 0.0

    def test_brevity_penalty(self):
        # Short prediction gets penalised
        score_short = bleu_1("cat", "the cat sat on the mat")
        score_full = bleu_1("the cat sat on the mat", "the cat sat on the mat")
        assert score_short < score_full

    def test_returns_float(self):
        assert isinstance(bleu_1("hello", "hello"), float)


class TestF1Score:
    def test_perfect(self):
        assert f1_score("the cat sat", "the cat sat") == 1.0

    def test_zero_empty_pred(self):
        assert f1_score("", "reference") == 0.0

    def test_zero_no_overlap(self):
        assert f1_score("xyz", "abc") == 0.0

    def test_partial(self):
        score = f1_score("cat sat", "the cat sat on the mat")
        assert 0.0 < score < 1.0


class TestApproxTokens:
    def test_basic(self):
        assert _approx_tokens("hello world") == 2

    def test_empty(self):
        assert _approx_tokens("") == 0

    def test_returns_int(self):
        assert isinstance(_approx_tokens("x"), int)


class TestContainsCode:
    def test_detects_backticks(self):
        assert contains_code("```python\nprint('hi')\n```")

    def test_detects_def(self):
        assert contains_code("def foo(): pass")

    def test_plain_text(self):
        assert not contains_code("The weather is nice today.")


# ===========================================================================
# Task & BenchmarkSpec
# ===========================================================================

class TestTask:
    def test_to_dict_keys(self):
        t = Task("t1", "qa", "What?", "Answer")
        d = t.to_dict()
        assert set(d.keys()) == {"task_id", "category", "prompt", "reference", "metadata"}

    def test_default_metadata(self):
        t = Task("t1", "qa", "Q", "A")
        assert t.metadata == {}


class TestBenchmarkSpec:
    def test_default_uses_sample_tasks(self):
        spec = BenchmarkSpec()
        assert len(spec) == len(SAMPLE_TASKS)

    def test_iter(self):
        spec = BenchmarkSpec()
        tasks = list(spec)
        assert all(isinstance(t, Task) for t in tasks)

    def test_from_dicts(self):
        records = [
            {"task_id": "x1", "category": "qa", "prompt": "Q1", "reference": "A1"},
            {"task_id": "x2", "category": "qa", "prompt": "Q2", "reference": "A2"},
        ]
        spec = BenchmarkSpec.from_dicts(records)
        assert len(spec) == 2
        assert spec.tasks[0].task_id == "x1"

    def test_from_file_json(self, tmp_path):
        data = [{"task_id": "j1", "category": "qa", "prompt": "P", "reference": "R"}]
        p = tmp_path / "tasks.json"
        p.write_text(json.dumps(data))
        spec = BenchmarkSpec.from_file(p)
        assert len(spec) == 1
        assert spec.tasks[0].task_id == "j1"

    def test_from_file_jsonl(self, tmp_path):
        lines = [
            json.dumps({"task_id": "j1", "category": "qa", "prompt": "P", "reference": "R"}),
            json.dumps({"task_id": "j2", "category": "qa", "prompt": "P2", "reference": "R2"}),
        ]
        p = tmp_path / "tasks.jsonl"
        p.write_text("\n".join(lines))
        spec = BenchmarkSpec.from_file(p)
        assert len(spec) == 2

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            BenchmarkSpec.from_file("/nonexistent/path/tasks.json")

    def test_from_file_json_tasks_key(self, tmp_path):
        data = {"tasks": [{"task_id": "t1", "category": "qa", "prompt": "P", "reference": "R"}]}
        p = tmp_path / "tasks.json"
        p.write_text(json.dumps(data))
        spec = BenchmarkSpec.from_file(p)
        assert len(spec) == 1


# ===========================================================================
# BenchmarkResult
# ===========================================================================

class TestBenchmarkResult:
    def _make_result(self, **kwargs):
        defaults = dict(
            task_id="t1", category="qa", prompt="Q", reference="A",
            prediction="A", latency_s=0.1,
            exact_match=1.0, exact_match_norm=1.0,
            rouge_l=1.0, bleu_1=1.0, f1=1.0,
            approx_tokens=1, error=None,
        )
        defaults.update(kwargs)
        return BenchmarkResult(**defaults)

    def test_composite_perfect(self):
        r = self._make_result()
        assert r.composite_score == pytest.approx(1.0)

    def test_composite_zero(self):
        r = self._make_result(
            exact_match=0.0, exact_match_norm=0.0,
            rouge_l=0.0, bleu_1=0.0, f1=0.0,
        )
        assert r.composite_score == pytest.approx(0.0)

    def test_composite_weights(self):
        r = self._make_result(rouge_l=1.0, f1=0.0, exact_match_norm=0.0, bleu_1=0.0)
        assert r.composite_score == pytest.approx(0.40)

    def test_to_dict_includes_composite(self):
        r = self._make_result()
        d = r.to_dict()
        assert "composite_score" in d
        assert d["composite_score"] == pytest.approx(1.0)


# ===========================================================================
# BenchmarkRunner
# ===========================================================================

class TestBenchmarkRunner:
    def test_run_offline_returns_results(self):
        runner = BenchmarkRunner()
        results = runner.run_offline(lambda p: "answer")
        assert len(results) == len(SAMPLE_TASKS)
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_run_offline_captures_exception(self):
        def bad_model(p):
            raise ValueError("model error")

        runner = BenchmarkRunner()
        results = runner.run_offline(bad_model)
        assert all(r.error is not None for r in results)
        assert all(r.prediction == "" for r in results)

    def test_run_offline_accumulates_results(self):
        runner = BenchmarkRunner()
        runner.run_offline(lambda p: "x")
        runner.run_offline(lambda p: "y")
        assert len(runner.results) == 2 * len(SAMPLE_TASKS)

    def test_run_offline_perfect_score(self):
        task = Task("t1", "qa", "Q", "hello world")
        runner = BenchmarkRunner([task])
        results = runner.run_offline(lambda p: "hello world")
        assert results[0].exact_match == 1.0
        assert results[0].rouge_l == 1.0

    def test_summarize_empty(self):
        runner = BenchmarkRunner()
        assert runner.summarize() == {}

    def test_summarize_keys(self):
        runner = BenchmarkRunner()
        runner.run_offline(lambda p: "x")
        summary = runner.summarize()
        assert "overall" in summary
        assert "by_category" in summary

    def test_summarize_overall_n(self):
        runner = BenchmarkRunner()
        runner.run_offline(lambda p: "x")
        assert runner.summarize()["overall"]["n"] == len(SAMPLE_TASKS)

    def test_summarize_categories(self):
        runner = BenchmarkRunner()
        runner.run_offline(lambda p: "x")
        cats = runner.summarize()["by_category"].keys()
        assert {"qa", "coding", "summarization"}.issubset(set(cats))

    def test_export_csv(self, tmp_path):
        import csv as _csv
        runner = BenchmarkRunner()
        runner.run_offline(lambda p: "x")
        p = tmp_path / "out.csv"
        runner.export_csv(p)
        with p.open(newline="", encoding="utf-8") as fh:
            rows = list(_csv.reader(fh))
        assert len(rows) == len(SAMPLE_TASKS) + 1  # header + rows

    def test_export_csv_empty(self, tmp_path):
        runner = BenchmarkRunner()
        p = tmp_path / "out.csv"
        runner.export_csv(p)  # should not raise
        assert not p.exists()

    def test_export_jsonl(self, tmp_path):
        runner = BenchmarkRunner()
        runner.run_offline(lambda p: "x")
        p = tmp_path / "out.jsonl"
        runner.export_jsonl(p)
        lines = [json.loads(l) for l in p.read_text().splitlines()]
        assert len(lines) == len(SAMPLE_TASKS)
        assert "composite_score" in lines[0]

    def test_export_jsonl_empty(self, tmp_path):
        runner = BenchmarkRunner()
        p = tmp_path / "out.jsonl"
        runner.export_jsonl(p)  # should not raise
        assert not p.exists()

    def test_custom_task_list(self):
        tasks = [Task("t1", "qa", "Q", "A"), Task("t2", "qa", "Q2", "A2")]
        runner = BenchmarkRunner(tasks)
        results = runner.run_offline(lambda p: "A")
        assert len(results) == 2

    def test_spec_accepted(self):
        spec = BenchmarkSpec.from_dicts([{"task_id": "s1", "category": "qa", "prompt": "P", "reference": "R"}])
        runner = BenchmarkRunner(spec)
        assert len(runner.tasks) == 1


# ===========================================================================
# CLI
# ===========================================================================

class TestCLI:
    def test_sample_command(self, capsys):
        from llmbench.cli import main
        rc = main(["sample"])
        assert rc == 0
        out = capsys.readouterr().out
        lines = [l for l in out.splitlines() if l.strip()]
        assert len(lines) == len(SAMPLE_TASKS)
        obj = json.loads(lines[0])
        assert "task_id" in obj

    def test_score_command_text(self, tmp_path, capsys):
        from llmbench.cli import main
        data = [{"task_id": "t1", "category": "qa", "prompt": "Q", "reference": "Paris", "prediction": "Paris"}]
        p = tmp_path / "preds.jsonl"
        p.write_text(json.dumps(data[0]))
        rc = main(["score", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Composite" in out

    def test_score_command_json(self, tmp_path, capsys):
        from llmbench.cli import main
        data = {"task_id": "t1", "category": "qa", "prompt": "Q", "reference": "Paris", "prediction": "Paris"}
        p = tmp_path / "preds.jsonl"
        p.write_text(json.dumps(data))
        rc = main(["score", str(p), "--format", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "overall" in parsed

    def test_score_missing_file(self, capsys):
        from llmbench.cli import main
        rc = main(["score", "/nonexistent/path.jsonl"])
        assert rc == 1

    def test_no_command_runs_demo(self, capsys):
        from llmbench.cli import main
        rc = main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "overall" in out


# ===========================================================================
# Flat module compatibility (llmbench.py)
# ===========================================================================

class TestFlatModule:
    """Verify that the flat llmbench.py exposes the same public API."""

    def setup_method(self):
        # The flat module is importable because tests/ parent is /repo root
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import importlib
        import llmbench as flat
        # May be the src package or the flat module depending on sys.path order;
        # both must expose BenchmarkRunner and metric functions.
        self.flat = flat

    def test_has_benchmark_runner(self):
        assert hasattr(self.flat, "BenchmarkRunner")

    def test_has_rouge_l(self):
        assert callable(self.flat.rouge_l)

    def test_has_exact_match(self):
        assert callable(self.flat.exact_match)

    def test_has_approx_tokens(self):
        assert callable(self.flat._approx_tokens)

    def test_rouge_l_perfect(self):
        assert self.flat.rouge_l("the cat sat", "the cat sat") == 1.0

    def test_exact_match_values(self):
        assert self.flat.exact_match("hello", "hello") == 1.0
        assert self.flat.exact_match("hello", "world") == 0.0
