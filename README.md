# llmbench

[![Tests](https://github.com/vdeshmukh203/llmbench/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/llmbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

**llmbench** is a lightweight, stdlib-only Python framework for designing, running, and reporting reproducible benchmarks of large language models (LLMs). It provides:

- **Declarative task specifications** — load tasks from JSON, JSONL, or YAML files, or define them in Python.
- **Provider-agnostic inference** — run benchmarks against any callable Python function or any OpenAI-compatible REST API.
- **Standard metrics** — ROUGE-L, BLEU-1, token-level F1, exact match, and a configurable composite score.
- **Structured output** — results stored in JSONL with complete provenance (model ID, prompt, reference, scores, latency).
- **CLI and GUI** — a rich command-line interface and an optional Tkinter desktop GUI.
- **Zero external dependencies** — the core package requires only Python's standard library.

---

## Installation

```bash
pip install llmbench
```

For YAML task files, install the optional extra:

```bash
pip install "llmbench[yaml]"
```

---

## Quick start

### Python API

```python
from llmbench import BenchmarkRunner, BenchmarkSpec

# Run against a custom model function
runner = BenchmarkRunner()           # uses 10 built-in sample tasks
results = runner.run_offline(lambda prompt: "Paris")

summary = runner.summarize()
print(summary["overall"]["composite"])   # weighted composite score

runner.export_jsonl("results.jsonl")
runner.export_csv("results.csv")
```

### Command-line interface

```bash
# Print built-in sample tasks as JSONL
llmbench sample

# Score an existing predictions file
llmbench score predictions.jsonl

# Score with JSON output
llmbench score predictions.jsonl --format json

# Query an OpenAI-compatible API
llmbench run --api-key sk-... --model gpt-4o --output results.jsonl

# Load custom tasks from a file
llmbench run --api-key sk-... --tasks my_tasks.jsonl

# Launch the GUI
llmbench gui
```

### GUI

```bash
llmbench gui
# or directly:
llmbench-gui
```

The GUI lets you load task files, run offline or OpenAI benchmarks with a
progress bar, browse per-task detail, inspect summary statistics, and export
results — all without writing any code.

---

## Task file format

Tasks can be provided as a JSON array or JSONL file (one object per line):

```json
[
  {
    "task_id": "qa_01",
    "category": "qa",
    "prompt": "What is the capital of France?",
    "reference": "Paris"
  }
]
```

Optional YAML (requires `pip install pyyaml`):

```yaml
- task_id: qa_01
  category: qa
  prompt: What is the capital of France?
  reference: Paris
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| `exact_match` | 1.0 if prediction == reference (after stripping) |
| `exact_match_norm` | 1.0 if token sequences match after lowercasing |
| `rouge_l` | ROUGE-L F1 via longest common subsequence |
| `bleu_1` | Unigram BLEU with brevity penalty |
| `f1` | Token-level F1 using bag-of-words set overlap |
| `composite` | 0.40·ROUGE-L + 0.30·F1 + 0.20·EM-norm + 0.10·BLEU-1 |

---

## Project structure

```
src/llmbench/
├── __init__.py   # public API
├── metrics.py    # ROUGE-L, BLEU-1, F1, exact match
├── spec.py       # Task, BenchmarkSpec, SAMPLE_TASKS
├── runner.py     # BenchmarkResult, BenchmarkRunner
├── cli.py        # command-line interface
└── gui.py        # Tkinter desktop GUI
```

---

## Contributing

Bug reports and pull requests are welcome on [GitHub](https://github.com/vdeshmukh203/llmbench).

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you use llmbench in research, please cite:

```bibtex
@article{deshmukh2026llmbench,
  title   = {llmbench: A lightweight framework for reproducible benchmarking of large language models},
  author  = {Deshmukh, Vaibhav},
  journal = {Journal of Open Source Software},
  year    = {2026}
}
```
