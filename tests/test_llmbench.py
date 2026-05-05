"""
Test suite for llmbench.
Covers metrics, dataclasses, BenchmarkRunner, BenchmarkSpec, and CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the src package is on the path regardless of install state.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import llmbench as lb
from llmbench import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSpec,
    SAMPLE_TASKS,
    Task,
    _approx_tokens,
    _tokenise,
    bleu_1,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)


# ===========================================================================
# Tokeniser
# ===========================================================================

class TestTokenise:
    def test_basic(self):
        assert _tokenise("Hello, World!") == ["hello", "world"]

    def test_empty(self):
        assert _tokenise("") == []

    def test_numbers(self):
        assert "42" in _tokenise("answer is 42")

    def test_lowercase(self):
        assert _tokenise("ABC") == ["abc"]

    def test_punctuation_stripped(self):
        assert _tokenise("foo.bar,baz") == ["foo", "bar", "baz"]


# ===========================================================================
# Exact match
# ===========================================================================

class TestExactMatch:
    def test_identical(self):
        assert exact_match("hello", "hello") == 1.0

    def test_different(self):
        assert exact_match("hello", "world") == 0.0

    def test_leading_trailing_whitespace(self):
        assert exact_match("  hello  ", "hello") == 1.0

    def test_case_sensitive(self):
        assert exact_match("Hello", "hello") == 0.0

    def test_empty_both(self):
        assert exact_match("", "") == 1.0

    def test_empty_prediction(self):
        assert exact_match("", "ref") == 0.0


class TestExactMatchNormalised:
    def test_case_insensitive(self):
        assert exact_match_normalised("Hello", "hello") == 1.0

    def test_punctuation_ignored(self):
        assert exact_match_normalised("H2O.", "H2O") == 1.0

    def test_different(self):
        assert exact_match_normalised("foo", "bar") == 0.0

    def test_empty_both(self):
        assert exact_match_normalised("", "") == 1.0


# ===========================================================================
# ROUGE-L
# ===========================================================================

class TestRougeL:
    def test_identical(self):
        assert rouge_l("the cat sat", "the cat sat") == 1.0

    def test_partial_overlap(self):
        score = rouge_l("the cat", "the cat sat on a mat")
        assert 0.0 < score < 1.0

    def test_empty_prediction(self):
        assert rouge_l("", "reference") == 0.0

    def test_empty_reference(self):
        assert rouge_l("prediction", "") == 0.0

    def test_no_overlap(self):
        assert rouge_l("foo bar", "baz qux") == 0.0

    def test_symmetry_approx(self):
        # ROUGE-L is not perfectly symmetric but both directions should be > 0
        assert rouge_l("cat sat", "the cat sat") > 0
        assert rouge_l("the cat sat", "cat sat") > 0


# ===========================================================================
# BLEU-1
# ===========================================================================

class TestBleu1:
    def test_identical(self):
        assert bleu_1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_empty_prediction(self):
        assert bleu_1("", "reference") == 0.0

    def test_partial(self):
        score = bleu_1("the cat", "the cat sat")
        assert 0.0 < score <= 1.0

    def test_brevity_penalty(self):
        # Short prediction vs long reference should incur BP < 1
        score = bleu_1("cat", "the cat sat on a mat")
        assert score < 1.0

    def test_no_overlap(self):
        assert bleu_1("foo", "bar") == 0.0


# ===========================================================================
# F1
# ===========================================================================

class TestF1Score:
    def test_identical(self):
        assert f1_score("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert f1_score("foo bar", "baz qux") == 0.0

    def test_empty_prediction(self):
        assert f1_score("", "reference") == 0.0

    def test_empty_reference(self):
        assert f1_score("prediction", "") == 0.0

    def test_partial(self):
        score = f1_score("cat sat", "the cat sat on a mat")
        assert 0.0 < score < 1.0


# ===========================================================================
# approx_tokens
# ===========================================================================

class TestApproxTokens:
    def test_basic(self):
        assert _approx_tokens("hello world") == 2

    def test_empty(self):
        assert _approx_tokens("") == 0

    def test_punctuation_not_counted(self):
        assert _approx_tokens("hello, world!") == 2

    def test_return_type(self):
        assert isinstance(_approx_tokens("foo bar"), int)


# ===========================================================================
# BenchmarkResult
# ===========================================================================

class TestBenchmarkResult:
    def _make(self, **kwargs):
        defaults = dict(
            task_id="t1", category="qa", prompt="q?", reference="a",
            prediction="a", latency_s=0.1,
            exact_match=1.0, exact_match_norm=1.0, rouge_l=1.0,
            bleu_1=1.0, f1=1.0, approx_tokens=1,
        )
        defaults.update(kwargs)
        return BenchmarkResult(**defaults)

    def test_composite_perfect(self):
        r = self._make()
        assert r.composite_score == pytest.approx(1.0)

    def test_composite_zero(self):
        r = self._make(
            exact_match=0.0, exact_match_norm=0.0, rouge_l=0.0, bleu_1=0.0, f1=0.0
        )
        assert r.composite_score == pytest.approx(0.0)

    def test_composite_weights(self):
        r = self._make(rouge_l=1.0, f1=0.0, exact_match_norm=0.0, bleu_1=0.0)
        assert r.composite_score == pytest.approx(0.40)

    def test_to_dict_keys(self):
        r = self._make()
        d = r.to_dict()
        for key in ("task_id", "category", "prompt", "reference", "prediction",
                    "latency_s", "exact_match", "exact_match_norm", "rouge_l",
                    "bleu_1", "f1", "approx_tokens", "error"):
            assert key in d

    def test_error_field_default_none(self):
        r = self._make()
        assert r.error is None


# ===========================================================================
# Task
# ===========================================================================

class TestTask:
    def test_to_dict(self):
        t = Task("qa_01", "qa", "prompt", "ref")
        d = t.to_dict()
        assert d["task_id"] == "qa_01"
        assert d["metadata"] == {}

    def test_metadata(self):
        t = Task("t1", "qa", "p", "r", metadata={"source": "test"})
        assert t.to_dict()["metadata"]["source"] == "test"


# ===========================================================================
# BenchmarkRunner
# ===========================================================================

class TestBenchmarkRunner:
    def test_default_tasks(self):
        runner = BenchmarkRunner()
        assert len(runner.tasks) == len(SAMPLE_TASKS)

    def test_custom_tasks(self):
        tasks = [Task("x1", "qa", "Q?", "A")]
        runner = BenchmarkRunner(tasks)
        assert len(runner.tasks) == 1

    def test_empty_list_not_replaced(self):
        runner = BenchmarkRunner([])
        assert runner.tasks == []

    def test_run_offline_returns_results(self):
        runner = BenchmarkRunner()
        results = runner.run_offline(lambda p: p)
        assert len(results) == len(SAMPLE_TASKS)
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_run_offline_accumulates(self):
        runner = BenchmarkRunner()
        runner.run_offline(lambda p: "")
        runner.run_offline(lambda p: "")
        assert len(runner.results) == 2 * len(SAMPLE_TASKS)

    def test_run_offline_error_handling(self):
        runner = BenchmarkRunner()
        results = runner.run_offline(lambda p: (_ for _ in ()).throw(ValueError("boom")))
        assert all(r.error is not None for r in results)
        assert all(r.prediction == "" for r in results)

    def test_run_offline_empty_task_list(self):
        runner = BenchmarkRunner()
        results = runner.run_offline(lambda p: "x", tasks=[])
        assert results == []
        # runner.results should not be extended
        assert runner.results == []

    def test_summarize_overall(self):
        runner = BenchmarkRunner()
        runner.run_offline(lambda p: "Paris")
        summary = runner.summarize()
        assert "overall" in summary
        assert summary["overall"]["n"] == len(SAMPLE_TASKS)

    def test_summarize_by_category(self):
        runner = BenchmarkRunner()
        runner.run_offline(lambda p: "")
        summary = runner.summarize()
        cats = set(summary["by_category"].keys())
        assert "qa" in cats
        assert "coding" in cats
        assert "summarization" in cats

    def test_summarize_empty(self):
        runner = BenchmarkRunner()
        assert runner.summarize() == {}

    def test_summarize_explicit_results(self):
        runner = BenchmarkRunner()
        r = runner.run_offline(lambda p: "Paris")
        s = runner.summarize(r)
        assert s["overall"]["n"] == len(SAMPLE_TASKS)

    def test_export_csv(self, tmp_path):
        runner = BenchmarkRunner()
        runner.run_offline(lambda p: "test")
        out = tmp_path / "results.csv"
        runner.export_csv(out)
        assert out.exists()
        content = out.read_text()
        assert "task_id" in content
        assert "composite_score" in content

    def test_export_csv_empty_no_file(self, tmp_path):
        runner = BenchmarkRunner()
        out = tmp_path / "empty.csv"
        runner.export_csv(out)
        assert not out.exists()

    def test_export_jsonl(self, tmp_path):
        runner = BenchmarkRunner()
        runner.run_offline(lambda p: "test")
        out = tmp_path / "results.jsonl"
        runner.export_jsonl(out)
        assert out.exists()
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == len(SAMPLE_TASKS)
        row = json.loads(lines[0])
        assert "sha256" in row
        assert "composite_score" in row

    def test_export_jsonl_checksum_stable(self, tmp_path):
        runner = BenchmarkRunner([Task("t1", "qa", "Q?", "A")])
        runner.run_offline(lambda p: "A")
        out = tmp_path / "r.jsonl"
        runner.export_jsonl(out)
        first = json.loads(out.read_text().strip())["sha256"]
        # Re-export and verify checksum is deterministic
        runner.results = runner.results  # no change
        out2 = tmp_path / "r2.jsonl"
        runner.export_jsonl(out2)
        second = json.loads(out2.read_text().strip())["sha256"]
        assert first == second

    def test_export_jsonl_empty_no_file(self, tmp_path):
        runner = BenchmarkRunner()
        out = tmp_path / "empty.jsonl"
        runner.export_jsonl(out)
        assert not out.exists()


# ===========================================================================
# BenchmarkSpec
# ===========================================================================

class TestBenchmarkSpec:
    def _minimal_dict(self):
        return {
            "name": "test-spec",
            "tasks": [
                {"task_id": "t1", "category": "qa", "prompt": "Q?", "reference": "A"}
            ],
        }

    def test_from_dict_minimal(self):
        spec = BenchmarkSpec.from_dict(self._minimal_dict())
        assert spec.name == "test-spec"
        assert len(spec.tasks) == 1
        assert spec.tasks[0].task_id == "t1"

    def test_from_dict_defaults(self):
        spec = BenchmarkSpec.from_dict(self._minimal_dict())
        assert spec.model == "gpt-3.5-turbo"
        assert spec.max_tokens == 256
        assert spec.temperature == 0.0

    def test_from_dict_custom(self):
        data = self._minimal_dict()
        data.update({"model": "gpt-4o", "max_tokens": 512, "temperature": 0.7})
        spec = BenchmarkSpec.from_dict(data)
        assert spec.model == "gpt-4o"
        assert spec.max_tokens == 512
        assert spec.temperature == pytest.approx(0.7)

    def test_from_dict_missing_name_raises(self):
        with pytest.raises((ValueError, KeyError)):
            BenchmarkSpec.from_dict({"tasks": []})

    def test_from_file(self, tmp_path):
        spec_file = tmp_path / "spec.json"
        spec_file.write_text(json.dumps(self._minimal_dict()))
        spec = BenchmarkSpec.from_file(spec_file)
        assert spec.name == "test-spec"

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            BenchmarkSpec.from_file(Path("/nonexistent/spec.json"))

    def test_save_and_reload(self, tmp_path):
        spec = BenchmarkSpec(
            name="roundtrip",
            tasks=[Task("t1", "qa", "Q?", "A")],
        )
        out = tmp_path / "spec.json"
        spec.save(out)
        reloaded = BenchmarkSpec.from_file(out)
        assert reloaded.name == "roundtrip"
        assert reloaded.tasks[0].task_id == "t1"

    def test_to_dict_round_trip(self):
        spec = BenchmarkSpec(
            name="rt", description="desc", model="gpt-4o",
            tasks=[Task("t1", "qa", "Q?", "A")],
        )
        d = spec.to_dict()
        spec2 = BenchmarkSpec.from_dict(d)
        assert spec2.name == "rt"
        assert spec2.description == "desc"
        assert spec2.model == "gpt-4o"
        assert spec2.tasks[0].task_id == "t1"


# ===========================================================================
# CLI (smoke tests)
# ===========================================================================

class TestCLI:
    def test_sample_command(self, capsys):
        from llmbench.cli import main
        rc = main(["sample"])
        assert rc == 0
        out = capsys.readouterr().out
        lines = [l for l in out.strip().splitlines() if l]
        assert len(lines) == len(SAMPLE_TASKS)
        obj = json.loads(lines[0])
        assert "task_id" in obj

    def test_demo_mode(self, capsys):
        from llmbench.cli import main
        rc = main([])
        assert rc == 0
        out = capsys.readouterr().out
        summary = json.loads(out)
        assert "overall" in summary
        assert summary["overall"]["n"] == len(SAMPLE_TASKS)

    def test_score_command_text(self, tmp_path, capsys):
        from llmbench.cli import main
        pred_file = tmp_path / "preds.jsonl"
        pred_file.write_text(
            json.dumps({"task_id": "t1", "category": "qa", "prompt": "Q?",
                        "prediction": "Paris", "reference": "Paris"}) + "\n"
        )
        rc = main(["score", str(pred_file)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Tasks:" in out
        assert "EM:" in out

    def test_score_command_json(self, tmp_path, capsys):
        from llmbench.cli import main
        pred_file = tmp_path / "preds.jsonl"
        pred_file.write_text(
            json.dumps({"task_id": "t1", "category": "qa", "prompt": "Q?",
                        "prediction": "A", "reference": "A"}) + "\n"
        )
        rc = main(["score", str(pred_file), "--format", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        summary = json.loads(out)
        assert summary["overall"]["n"] == 1

    def test_score_missing_file(self, capsys):
        from llmbench.cli import main
        rc = main(["score", "/nonexistent/preds.jsonl"])
        assert rc == 1

    def test_run_no_api_key(self, capsys):
        from llmbench.cli import main
        import os
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            rc = main(["run"])
            assert rc == 1
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old


# ===========================================================================
# Public API surface (import-level checks)
# ===========================================================================

class TestPublicAPI:
    def test_version(self):
        assert hasattr(lb, "__version__")
        assert isinstance(lb.__version__, str)

    def test_all_exports_accessible(self):
        for name in lb.__all__:
            assert hasattr(lb, name), f"Missing export: {name}"

    def test_benchmark_runner_accessible(self):
        assert hasattr(lb, "BenchmarkRunner")

    def test_benchmark_spec_accessible(self):
        assert hasattr(lb, "BenchmarkSpec")

    def test_sample_tasks_non_empty(self):
        assert len(lb.SAMPLE_TASKS) > 0
        assert all(isinstance(t, Task) for t in lb.SAMPLE_TASKS)
