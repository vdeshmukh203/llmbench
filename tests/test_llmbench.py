"""Tests for llmbench.

The test suite is split into three sections:
  1. Metrics   – pure scoring functions
  2. Spec      – Task, BenchmarkSpec, BenchmarkResult dataclasses
  3. Runner    – BenchmarkRunner offline execution, summarize, export
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

import llmbench as lb
from llmbench import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSpec,
    Task,
    approx_tokens,
    bleu_1,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)


# ===========================================================================
# 1. Metrics
# ===========================================================================


class TestExactMatch:
    def test_identical(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_different(self):
        assert exact_match("London", "Paris") == 0.0

    def test_whitespace_stripped(self):
        assert exact_match("  Paris  ", "Paris") == 1.0

    def test_case_sensitive(self):
        assert exact_match("paris", "Paris") == 0.0

    def test_empty_both(self):
        assert exact_match("", "") == 1.0

    def test_empty_prediction(self):
        assert exact_match("", "Paris") == 0.0


class TestExactMatchNormalised:
    def test_identical(self):
        assert exact_match_normalised("Hello World", "Hello World") == 1.0

    def test_case_insensitive(self):
        assert exact_match_normalised("hello world", "Hello World") == 1.0

    def test_punctuation_ignored(self):
        assert exact_match_normalised("hello, world!", "hello world") == 1.0

    def test_different(self):
        assert exact_match_normalised("foo", "bar") == 0.0


class TestRougeL:
    def test_perfect(self):
        assert rouge_l("the cat sat", "the cat sat") == 1.0

    def test_zero_empty_pred(self):
        assert rouge_l("", "the cat sat") == 0.0

    def test_zero_empty_ref(self):
        assert rouge_l("the cat sat", "") == 0.0

    def test_partial_overlap(self):
        score = rouge_l("the cat", "the cat sat on the mat")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert rouge_l("dog", "cat") == 0.0

    def test_symmetry_not_required(self):
        # ROUGE-L is not symmetric by definition; just check valid range
        s = rouge_l("a b c", "a b c d e")
        assert 0.0 <= s <= 1.0


class TestBleu1:
    def test_perfect(self):
        assert bleu_1("hello world", "hello world") == 1.0

    def test_empty_prediction(self):
        assert bleu_1("", "hello") == 0.0

    def test_brevity_penalty_applied(self):
        # Short prediction vs long reference → bp < 1
        score = bleu_1("hello", "hello world foo bar baz")
        assert score < 1.0

    def test_partial_overlap(self):
        score = bleu_1("the cat", "the cat sat on the mat")
        assert 0.0 < score <= 1.0


class TestF1Score:
    def test_perfect(self):
        assert f1_score("the cat sat", "the cat sat") == 1.0

    def test_zero_empty(self):
        assert f1_score("", "hello") == 0.0
        assert f1_score("hello", "") == 0.0

    def test_no_overlap(self):
        assert f1_score("dog", "cat") == 0.0

    def test_partial_overlap(self):
        score = f1_score("the cat", "the dog")
        assert 0.0 < score < 1.0

    def test_symmetric(self):
        assert math.isclose(
            f1_score("a b c", "a b d"), f1_score("a b d", "a b c"), rel_tol=1e-9
        )


class TestApproxTokens:
    def test_basic(self):
        assert approx_tokens("hello world") == 2

    def test_empty(self):
        assert approx_tokens("") == 0

    def test_punctuation_ignored(self):
        assert approx_tokens("hello, world!") == 2

    def test_numbers(self):
        assert approx_tokens("H2O is water") == 3

    def test_private_alias(self):
        # Backward-compatible alias
        assert lb._approx_tokens("hello world") == 2


# ===========================================================================
# 2. Spec
# ===========================================================================


class TestTask:
    def test_to_dict(self):
        t = Task("qa_01", "qa", "What?", "Answer")
        d = t.to_dict()
        assert d["task_id"] == "qa_01"
        assert d["category"] == "qa"
        assert d["prompt"] == "What?"
        assert d["reference"] == "Answer"
        assert d["metadata"] == {}


class TestBenchmarkResult:
    def _make(self, **overrides):
        defaults = dict(
            task_id="t1", category="qa", prompt="p", reference="r",
            prediction="r", latency_s=0.1,
            exact_match=1.0, exact_match_norm=1.0,
            rouge_l=1.0, bleu_1=1.0, f1=1.0, approx_tokens=1,
        )
        defaults.update(overrides)
        return BenchmarkResult(**defaults)

    def test_composite_perfect(self):
        r = self._make()
        assert math.isclose(r.composite_score, 1.0, rel_tol=1e-9)

    def test_composite_zero(self):
        r = self._make(
            exact_match=0.0, exact_match_norm=0.0, rouge_l=0.0, bleu_1=0.0, f1=0.0
        )
        assert r.composite_score == 0.0

    def test_composite_weights(self):
        # Only rouge_l = 1, rest 0  → composite = 0.40
        r = self._make(exact_match=0.0, exact_match_norm=0.0, rouge_l=1.0, bleu_1=0.0, f1=0.0)
        assert math.isclose(r.composite_score, 0.40, rel_tol=1e-9)

    def test_to_dict_has_error_key(self):
        r = self._make()
        assert "error" in r.to_dict()


class TestBenchmarkSpec:
    def test_defaults_to_sample_tasks(self):
        spec = BenchmarkSpec()
        assert len(spec.tasks) == len(BenchmarkSpec.SAMPLE_TASKS)

    def test_custom_tasks(self):
        tasks = [Task("x", "qa", "Q?", "A")]
        spec = BenchmarkSpec(tasks)
        assert spec.tasks == tasks

    def test_from_jsonl(self, tmp_path):
        tasks = [
            {"task_id": "a", "category": "qa", "prompt": "Q?", "reference": "A"},
            {"task_id": "b", "category": "qa", "prompt": "Q2?", "reference": "B"},
        ]
        p = tmp_path / "tasks.jsonl"
        p.write_text("\n".join(json.dumps(t) for t in tasks))
        spec = BenchmarkSpec.from_jsonl(p)
        assert len(spec.tasks) == 2
        assert spec.tasks[0].task_id == "a"

    def test_from_jsonl_skips_blank_lines(self, tmp_path):
        p = tmp_path / "tasks.jsonl"
        p.write_text('{"task_id":"a","category":"qa","prompt":"Q","reference":"A"}\n\n')
        spec = BenchmarkSpec.from_jsonl(p)
        assert len(spec.tasks) == 1


# ===========================================================================
# 3. Runner
# ===========================================================================


class TestBenchmarkRunner:
    def _single_task_spec(self, pred="Paris", ref="Paris"):
        return BenchmarkSpec([Task("q1", "qa", "Capital of France?", ref)])

    def test_run_offline_returns_results(self):
        runner = BenchmarkRunner(self._single_task_spec())
        results = runner.run_offline(lambda _: "Paris")
        assert len(results) == 1
        assert results[0].prediction == "Paris"

    def test_run_offline_perfect_scores(self):
        runner = BenchmarkRunner(self._single_task_spec("Paris", "Paris"))
        results = runner.run_offline(lambda _: "Paris")
        r = results[0]
        assert r.exact_match == 1.0
        assert r.rouge_l == 1.0
        assert r.f1 == 1.0
        assert math.isclose(r.composite_score, 1.0, rel_tol=1e-9)

    def test_run_offline_error_caught(self):
        runner = BenchmarkRunner(self._single_task_spec())
        results = runner.run_offline(lambda _: (_ for _ in ()).throw(ValueError("boom")))
        assert results[0].error is not None
        assert "boom" in results[0].error

    def test_run_offline_accumulates_results(self):
        runner = BenchmarkRunner(self._single_task_spec())
        runner.run_offline(lambda _: "Paris")
        runner.run_offline(lambda _: "Paris")
        assert len(runner.results) == 2

    def test_progress_callback_called(self):
        calls = []
        runner = BenchmarkRunner(self._single_task_spec())
        runner.run_offline(lambda _: "Paris", progress_callback=lambda d, t, r: calls.append((d, t)))
        assert calls == [(1, 1)]

    def test_progress_callback_multiple_tasks(self):
        calls = []
        spec = BenchmarkSpec([
            Task("q1", "qa", "A?", "A"),
            Task("q2", "qa", "B?", "B"),
        ])
        runner = BenchmarkRunner(spec)
        runner.run_offline(lambda _: "x", progress_callback=lambda d, t, r: calls.append(d))
        assert calls == [1, 2]

    def test_summarize_overall_keys(self):
        runner = BenchmarkRunner(self._single_task_spec())
        runner.run_offline(lambda _: "Paris")
        summary = runner.summarize()
        for key in ("n", "exact_match", "rouge_l", "bleu_1", "f1", "composite", "avg_latency_s", "errors"):
            assert key in summary["overall"], f"Missing key: {key}"

    def test_summarize_by_category(self):
        spec = BenchmarkSpec([
            Task("q1", "qa", "A?", "A"),
            Task("c1", "coding", "Write a fn.", "def f(): pass"),
        ])
        runner = BenchmarkRunner(spec)
        runner.run_offline(lambda _: "x")
        summary = runner.summarize()
        assert "qa" in summary["by_category"]
        assert "coding" in summary["by_category"]

    def test_summarize_empty(self):
        runner = BenchmarkRunner()
        assert runner.summarize() == {}

    def test_export_jsonl(self, tmp_path):
        runner = BenchmarkRunner(self._single_task_spec())
        runner.run_offline(lambda _: "Paris")
        out = tmp_path / "results.jsonl"
        runner.export_jsonl(out)
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert "composite_score" in lines[0]
        assert lines[0]["task_id"] == "q1"

    def test_export_csv(self, tmp_path):
        import csv
        runner = BenchmarkRunner(self._single_task_spec())
        runner.run_offline(lambda _: "Paris")
        out = tmp_path / "results.csv"
        runner.export_csv(out)
        with out.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert "composite_score" in rows[0]

    def test_export_jsonl_empty(self, tmp_path):
        runner = BenchmarkRunner()
        out = tmp_path / "empty.jsonl"
        runner.export_jsonl(out)
        assert out.read_text() == ""

    def test_lambda_closure_score_command(self):
        """Verify each task gets its own prediction (no late-binding closure bug)."""
        preds = ["A", "B", "C"]
        tasks = [Task(f"t{i}", "qa", "Q?", p) for i, p in enumerate(preds)]
        spec = BenchmarkSpec(tasks)
        runner = BenchmarkRunner(spec)
        results = []
        for task, pred in zip(tasks, preds):
            r = BenchmarkRunner(BenchmarkSpec([task]))
            results.extend(r.run_offline(lambda _, p=pred: p, [task]))
        for result, expected in zip(results, preds):
            assert result.prediction == expected


# ===========================================================================
# 4. Public API surface
# ===========================================================================


def test_public_api_attributes():
    for name in ["BenchmarkRunner", "BenchmarkSpec", "BenchmarkResult", "Task",
                  "rouge_l", "bleu_1", "f1_score", "exact_match",
                  "exact_match_normalised", "approx_tokens"]:
        assert hasattr(lb, name), f"Missing public attribute: {name}"


def test_version():
    assert hasattr(lb, "__version__")
    assert lb.__version__ == "0.1.0"
