"""Tests for the llmbench package (JOSS-level coverage)."""
import json
import sys
import pathlib
import tempfile

# Ensure the src layout is on the path for development/CI runs
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import llmbench as lb
from llmbench.metrics import (
    bleu_1,
    composite_score,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)
from llmbench.runner import BenchmarkResult, BenchmarkRunner, _approx_tokens
from llmbench.spec import BenchmarkSpec, Task


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


def test_package_exports():
    assert hasattr(lb, "BenchmarkRunner")
    assert hasattr(lb, "BenchmarkResult")
    assert hasattr(lb, "BenchmarkSpec")
    assert hasattr(lb, "Task")
    assert hasattr(lb, "SAMPLE_TASKS")
    assert hasattr(lb, "rouge_l")
    assert hasattr(lb, "bleu_1")
    assert hasattr(lb, "f1_score")
    assert hasattr(lb, "exact_match")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_exact_match_identical():
    assert exact_match("Paris", "Paris") == 1.0


def test_exact_match_different():
    assert exact_match("London", "Paris") == 0.0


def test_exact_match_strips_whitespace():
    assert exact_match("  Paris  ", "Paris") == 1.0


def test_exact_match_case_sensitive():
    assert exact_match("paris", "Paris") == 0.0


def test_exact_match_normalised_ignores_case_and_punctuation():
    assert exact_match_normalised("Hello, World!", "hello world") == 1.0


def test_rouge_l_perfect():
    assert rouge_l("the cat sat", "the cat sat") == 1.0


def test_rouge_l_empty_prediction():
    assert rouge_l("", "reference") == 0.0


def test_rouge_l_empty_reference():
    assert rouge_l("prediction", "") == 0.0


def test_rouge_l_partial():
    score = rouge_l("a b c", "a b c d")
    assert 0.0 < score < 1.0


def test_bleu_1_perfect():
    score = bleu_1("the cat sat", "the cat sat")
    assert score > 0.9


def test_bleu_1_empty_prediction():
    assert bleu_1("", "reference") == 0.0


def test_bleu_1_no_overlap():
    assert bleu_1("xyz uvw", "abc def") == 0.0


def test_f1_perfect():
    assert f1_score("a b c", "a b c") == 1.0


def test_f1_no_overlap():
    assert f1_score("abc", "xyz") == 0.0


def test_f1_empty():
    assert f1_score("", "reference") == 0.0


def test_composite_score_all_ones():
    assert abs(composite_score(1.0, 1.0, 1.0, 1.0) - 1.0) < 1e-9


def test_composite_score_all_zeros():
    assert composite_score(0.0, 0.0, 0.0, 0.0) == 0.0


def test_composite_score_weights():
    # 0.40*1 + 0.30*0 + 0.20*0 + 0.10*0 = 0.40
    assert abs(composite_score(1.0, 0.0, 0.0, 0.0) - 0.40) < 1e-9


def test_approx_tokens():
    assert _approx_tokens("hello world") == 2
    assert _approx_tokens("") == 0


# ---------------------------------------------------------------------------
# Task and BenchmarkSpec
# ---------------------------------------------------------------------------


def test_task_to_dict():
    t = Task("id1", "qa", "Prompt?", "Answer")
    d = t.to_dict()
    assert d["task_id"] == "id1"
    assert d["category"] == "qa"
    assert d["prompt"] == "Prompt?"
    assert d["reference"] == "Answer"


def test_task_from_dict():
    t = Task.from_dict({"task_id": "t1", "category": "qa", "prompt": "Q", "reference": "A"})
    assert t.task_id == "t1"


def test_task_from_dict_missing_field():
    try:
        Task.from_dict({"task_id": "t1", "category": "qa"})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_benchmark_spec_from_list():
    spec = BenchmarkSpec.from_list(
        [{"task_id": "t1", "category": "qa", "prompt": "Q?", "reference": "A"}]
    )
    assert len(spec) == 1
    assert spec.tasks[0].task_id == "t1"


def test_benchmark_spec_empty_raises():
    try:
        BenchmarkSpec([])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_benchmark_spec_from_jsonl(tmp_path):
    p = tmp_path / "tasks.jsonl"
    p.write_text(
        '{"task_id":"t1","category":"qa","prompt":"Q?","reference":"A"}\n',
        encoding="utf-8",
    )
    spec = BenchmarkSpec.from_jsonl(p)
    assert len(spec) == 1
    assert spec.tasks[0].task_id == "t1"


def test_benchmark_spec_to_jsonl(tmp_path):
    spec = BenchmarkSpec.from_list(
        [{"task_id": "t1", "category": "qa", "prompt": "Q?", "reference": "A"}]
    )
    p = tmp_path / "out.jsonl"
    spec.to_jsonl(p)
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["task_id"] == "t1"


def test_benchmark_spec_repr():
    spec = BenchmarkSpec.from_list(
        [{"task_id": "t1", "category": "qa", "prompt": "Q?", "reference": "A"}]
    )
    assert "BenchmarkSpec" in repr(spec)


# ---------------------------------------------------------------------------
# BenchmarkRunner — offline
# ---------------------------------------------------------------------------


def test_runner_perfect_match():
    tasks = [Task("t1", "qa", "What is 2+2?", "4")]
    runner = BenchmarkRunner(tasks)
    results = runner.run_offline(lambda _: "4")
    assert len(results) == 1
    r = results[0]
    assert r.exact_match == 1.0
    assert r.rouge_l == 1.0
    assert r.sha256 != ""


def test_runner_sha256_deterministic():
    tasks = [Task("t1", "qa", "Q?", "A")]
    r1 = BenchmarkRunner(tasks).run_offline(lambda _: "A")[0]
    r2 = BenchmarkRunner(tasks).run_offline(lambda _: "A")[0]
    assert r1.sha256 == r2.sha256


def test_runner_sha256_changes_with_prediction():
    tasks = [Task("t1", "qa", "Q?", "A")]
    r1 = BenchmarkRunner(tasks).run_offline(lambda _: "A")[0]
    r2 = BenchmarkRunner(tasks).run_offline(lambda _: "B")[0]
    assert r1.sha256 != r2.sha256


def test_runner_error_handling():
    def bad_model(prompt):
        raise RuntimeError("model failure")

    tasks = [Task("t1", "qa", "Q?", "A")]
    results = BenchmarkRunner(tasks).run_offline(bad_model)
    r = results[0]
    assert r.error is not None
    assert "model failure" in r.error
    assert r.prediction == ""
    assert r.sha256 == ""


def test_runner_accumulates_results():
    tasks = [Task("t1", "qa", "Q?", "A"), Task("t2", "qa", "Q2?", "B")]
    runner = BenchmarkRunner(tasks)
    runner.run_offline(lambda _: "X")
    assert len(runner.results) == 2


def test_runner_progress_callback():
    calls = []
    tasks = [Task("t1", "qa", "Q?", "A"), Task("t2", "qa", "Q2?", "B")]
    runner = BenchmarkRunner(tasks)
    runner.run_offline(lambda _: "X", progress_callback=lambda d, t, r: calls.append((d, t)))
    assert calls == [(1, 2), (2, 2)]


def test_runner_with_spec():
    spec = BenchmarkSpec.from_list(
        [{"task_id": "t1", "category": "qa", "prompt": "Q?", "reference": "A"}]
    )
    runner = BenchmarkRunner(spec=spec)
    results = runner.run_offline(lambda _: "A")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# BenchmarkRunner — summarize
# ---------------------------------------------------------------------------


def test_summarize_empty():
    runner = BenchmarkRunner([])
    assert runner.summarize() == {}


def test_summarize_structure():
    tasks = [
        Task("t1", "qa", "Q?", "A"),
        Task("t2", "coding", "C?", "def f(): pass"),
    ]
    runner = BenchmarkRunner(tasks)
    runner.run_offline(lambda p: p[:1])
    s = runner.summarize()
    assert "overall" in s
    assert "by_category" in s
    assert s["overall"]["n"] == 2
    assert "qa" in s["by_category"]
    assert "coding" in s["by_category"]


def test_summarize_composite_is_float():
    tasks = [Task("t1", "qa", "Q?", "A")]
    runner = BenchmarkRunner(tasks)
    runner.run_offline(lambda _: "A")
    s = runner.summarize()
    assert isinstance(s["overall"]["composite"], float)


# ---------------------------------------------------------------------------
# BenchmarkRunner — export
# ---------------------------------------------------------------------------


def test_export_jsonl(tmp_path):
    tasks = [Task("t1", "qa", "Q?", "A")]
    runner = BenchmarkRunner(tasks)
    runner.run_offline(lambda _: "A")
    out = tmp_path / "results.jsonl"
    runner.export_jsonl(out)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["task_id"] == "t1"
    assert "composite_score" in obj
    assert "sha256" in obj


def test_export_csv(tmp_path):
    tasks = [Task("t1", "qa", "Q?", "A")]
    runner = BenchmarkRunner(tasks)
    runner.run_offline(lambda _: "A")
    out = tmp_path / "results.csv"
    runner.export_csv(out)
    content = out.read_text(encoding="utf-8")
    assert "task_id" in content
    assert "t1" in content


def test_export_markdown(tmp_path):
    tasks = [Task("t1", "qa", "Q?", "A")]
    runner = BenchmarkRunner(tasks)
    runner.run_offline(lambda _: "A")
    out = tmp_path / "report.md"
    runner.export_markdown(out)
    content = out.read_text(encoding="utf-8")
    assert "# llmbench Report" in content
    assert "ROUGE-L" in content


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


def test_result_to_dict_has_composite():
    r = BenchmarkResult(
        task_id="t1", category="qa", prompt="Q?", reference="A",
        prediction="A", latency_s=0.1,
        exact_match=1.0, exact_match_norm=1.0,
        rouge_l=1.0, bleu_1=1.0, f1=1.0, approx_tokens=1,
    )
    d = r.to_dict()
    assert "composite_score" in d
    assert d["composite_score"] == 1.0


# ---------------------------------------------------------------------------
# Sample tasks
# ---------------------------------------------------------------------------


def test_sample_tasks_non_empty():
    assert len(lb.SAMPLE_TASKS) >= 5


def test_sample_tasks_valid():
    for t in lb.SAMPLE_TASKS:
        assert t.task_id
        assert t.category
        assert t.prompt
        assert t.reference


# ---------------------------------------------------------------------------
# CLI (smoke tests without real API calls)
# ---------------------------------------------------------------------------


def test_cli_sample_command(capsys):
    from llmbench.cli import main
    rc = main(["sample"])
    assert rc == 0
    out = capsys.readouterr().out
    first = json.loads(out.split("\n")[0])
    assert "task_id" in first


def test_cli_no_command(capsys):
    from llmbench.cli import main
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    summary = json.loads(out)
    assert "overall" in summary


def test_cli_score_command(tmp_path, capsys):
    from llmbench.cli import main
    p = tmp_path / "preds.jsonl"
    p.write_text(
        '{"task_id":"t1","category":"qa","prompt":"Q?","reference":"A","prediction":"A"}\n',
        encoding="utf-8",
    )
    rc = main(["score", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Tasks: 1" in out


def test_cli_score_missing_file(capsys):
    from llmbench.cli import main
    rc = main(["score", "/nonexistent/file.jsonl"])
    assert rc == 1
