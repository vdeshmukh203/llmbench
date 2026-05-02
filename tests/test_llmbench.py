"""Tests for the llmbench package."""
import json
import sys
import tempfile
from pathlib import Path

# Prefer the installed package; fall back to src/ directory for editable installs.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import llmbench as lb
from llmbench.models import BenchmarkResult, Task
from llmbench.metrics import (
    _tokenise,
    approx_tokens,
    bleu_1,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)
from llmbench.runner import BenchmarkRunner
from llmbench.spec import BenchmarkSpec


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------

def test_package_exports():
    assert hasattr(lb, "BenchmarkRunner")
    assert hasattr(lb, "BenchmarkSpec")
    assert hasattr(lb, "Task")
    assert hasattr(lb, "BenchmarkResult")
    assert hasattr(lb, "rouge_l")
    assert hasattr(lb, "bleu_1")
    assert hasattr(lb, "f1_score")
    assert hasattr(lb, "exact_match")
    assert hasattr(lb, "exact_match_normalised")
    assert hasattr(lb, "_approx_tokens")  # v0.1 compat alias


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_exact_match_equal():
    assert exact_match("hello", "hello") == 1.0


def test_exact_match_unequal():
    assert exact_match("hello", "world") == 0.0


def test_exact_match_whitespace():
    assert exact_match("  hello  ", "hello") == 1.0


def test_exact_match_normalised_case():
    assert exact_match_normalised("Hello World", "hello world") == 1.0


def test_exact_match_normalised_punct():
    assert exact_match_normalised("hello, world!", "hello world") == 1.0


def test_rouge_l_perfect():
    assert rouge_l("the cat sat", "the cat sat") == 1.0


def test_rouge_l_empty_pred():
    assert rouge_l("", "reference") == 0.0


def test_rouge_l_empty_ref():
    assert rouge_l("prediction", "") == 0.0


def test_rouge_l_partial():
    score = rouge_l("the cat", "the cat sat on the mat")
    assert 0.0 < score < 1.0


def test_bleu_1_perfect():
    assert bleu_1("the cat sat", "the cat sat") == 1.0


def test_bleu_1_empty():
    assert bleu_1("", "reference") == 0.0


def test_bleu_1_no_overlap():
    assert bleu_1("xyz", "abc") == 0.0


def test_bleu_1_partial():
    score = bleu_1("the cat", "the cat sat on the mat")
    assert 0.0 < score <= 1.0


def test_f1_perfect():
    assert f1_score("cat sat mat", "cat sat mat") == 1.0


def test_f1_no_overlap():
    assert f1_score("abc", "xyz") == 0.0


def test_f1_empty_pred():
    assert f1_score("", "reference") == 0.0


def test_f1_partial():
    score = f1_score("cat sat", "cat sat mat")
    assert 0.0 < score < 1.0


def test_approx_tokens_count():
    assert approx_tokens("hello world") == 2


def test_approx_tokens_punctuation():
    assert approx_tokens("hello, world!") == 2


def test_tokenise():
    assert _tokenise("Hello, World!") == ["hello", "world"]


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------

def test_task_checksum_deterministic():
    t = Task("qa_01", "qa", "What is 2+2?", "4")
    assert t.checksum() == t.checksum()


def test_task_checksum_differs():
    t1 = Task("qa_01", "qa", "What is 2+2?", "4")
    t2 = Task("qa_02", "qa", "What is 3+3?", "6")
    assert t1.checksum() != t2.checksum()


def test_task_to_dict():
    t = Task("qa_01", "qa", "What is 2+2?", "4")
    d = t.to_dict()
    assert d["task_id"] == "qa_01"
    assert d["reference"] == "4"


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

def test_composite_score_range():
    r = BenchmarkResult(
        task_id="t", category="qa", prompt="Q", reference="A", prediction="A",
        latency_s=0.1, exact_match=1.0, exact_match_norm=1.0,
        rouge_l=1.0, bleu_1=1.0, f1=1.0, approx_tokens=1,
        task_checksum="abc", run_timestamp="2026-01-01T00:00:00+00:00",
    )
    assert round(r.composite_score, 10) == 1.0


def test_result_checksum_present():
    r = BenchmarkResult(
        task_id="t", category="qa", prompt="Q", reference="A", prediction="A",
        latency_s=0.1, exact_match=1.0, exact_match_norm=1.0,
        rouge_l=1.0, bleu_1=1.0, f1=1.0, approx_tokens=1,
        task_checksum="abc", run_timestamp="2026-01-01T00:00:00+00:00",
    )
    cs = r.result_checksum()
    assert isinstance(cs, str) and len(cs) == 64


def test_result_to_dict_includes_composite():
    r = BenchmarkResult(
        task_id="t", category="qa", prompt="Q", reference="A", prediction="A",
        latency_s=0.1, exact_match=1.0, exact_match_norm=1.0,
        rouge_l=0.8, bleu_1=0.5, f1=0.9, approx_tokens=1,
        task_checksum="abc", run_timestamp="2026-01-01T00:00:00+00:00",
    )
    d = r.to_dict()
    assert "composite_score" in d
    assert d["composite_score"] == round(0.4 * 0.8 + 0.3 * 0.9 + 0.2 * 1.0 + 0.1 * 0.5, 4)


# ---------------------------------------------------------------------------
# BenchmarkSpec
# ---------------------------------------------------------------------------

def test_spec_builtin():
    spec = BenchmarkSpec.builtin()
    assert spec.name == "sample"
    assert len(spec.tasks) == 10


def test_spec_from_dict():
    data = {
        "name": "test_suite",
        "version": "1.0",
        "tasks": [
            {"task_id": "q1", "category": "qa", "prompt": "What is 2+2?", "reference": "4"},
        ],
    }
    spec = BenchmarkSpec._from_dict(data)
    assert spec.name == "test_suite"
    assert len(spec.tasks) == 1
    assert spec.tasks[0].task_id == "q1"


def test_spec_from_json_file():
    data = {
        "name": "file_test",
        "tasks": [
            {"task_id": "x1", "category": "qa", "prompt": "Hello?", "reference": "Hi"},
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp = Path(f.name)
    try:
        spec = BenchmarkSpec.from_file(tmp)
        assert spec.name == "file_test"
        assert len(spec.tasks) == 1
    finally:
        tmp.unlink()


def test_spec_missing_task_id_raises():
    import pytest
    with pytest.raises((ValueError, KeyError)):
        BenchmarkSpec._from_dict({"tasks": [{"category": "qa", "prompt": "x", "reference": "y"}]})


def test_spec_save_and_reload():
    spec = BenchmarkSpec.builtin()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "out.json"
        spec.save(path)
        reloaded = BenchmarkSpec.from_file(path)
    assert reloaded.name == spec.name
    assert len(reloaded.tasks) == len(spec.tasks)


def test_spec_len():
    spec = BenchmarkSpec.builtin()
    assert len(spec) == 10


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

def test_runner_offline_basic():
    spec = BenchmarkSpec(name="t", tasks=[Task("q1", "qa", "What is 2+2?", "4")])
    runner = BenchmarkRunner(spec=spec)
    results = runner.run_offline(lambda _: "4")
    assert len(results) == 1
    assert results[0].exact_match == 1.0


def test_runner_offline_error_captured():
    spec = BenchmarkSpec(name="t", tasks=[Task("q1", "qa", "Prompt", "ref")])
    runner = BenchmarkRunner(spec=spec)

    def bad_model(_):
        raise RuntimeError("model exploded")

    results = runner.run_offline(bad_model)
    assert results[0].error == "model exploded"
    assert results[0].prediction == ""


def test_runner_accumulates_results():
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "x")
    runner.run_offline(lambda _: "y")
    assert len(runner.results) == 20  # 10 tasks × 2 runs


def test_runner_summarize_keys():
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "answer")
    summary = runner.summarize()
    assert "overall" in summary
    assert "by_category" in summary
    assert summary["overall"]["n"] == 10


def test_runner_summarize_empty():
    runner = BenchmarkRunner()
    assert runner.summarize() == {}


def test_runner_export_jsonl():
    runner = BenchmarkRunner(tasks=[Task("q1", "qa", "Hello?", "Hi")])
    runner.run_offline(lambda _: "Hi")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.jsonl"
        runner.export_jsonl(p)
        lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["task_id"] == "q1"
    assert "composite_score" in obj
    assert "task_checksum" in obj


def test_runner_export_csv():
    runner = BenchmarkRunner(tasks=[Task("q1", "qa", "Hello?", "Hi")])
    runner.run_offline(lambda _: "Hi")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.csv"
        runner.export_csv(p)
        text = p.read_text(encoding="utf-8")
    assert "task_id" in text
    assert "composite_score" in text


def test_runner_tasks_from_spec():
    spec = BenchmarkSpec.builtin()
    runner = BenchmarkRunner(spec=spec)
    assert runner.tasks is spec.tasks


def test_runner_tasks_list_compat():
    tasks = [Task("a", "qa", "P", "R")]
    runner = BenchmarkRunner(tasks=tasks)
    assert runner.tasks == tasks
