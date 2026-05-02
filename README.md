# llmbench

[![CI](https://github.com/vdeshmukh203/llmbench/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/llmbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

**llmbench** is a lightweight, reproducible Python framework for benchmarking large language models (LLMs). It evaluates model outputs using ROUGE-L, BLEU-1, F1, and exact-match metrics, records SHA-256 provenance checksums, and produces structured JSONL/CSV reports—all with **zero runtime dependencies**.

---

## Features

- **Zero dependencies** — pure Python stdlib; no NumPy, no transformers, no NLTK
- **Five metrics** — ROUGE-L, BLEU-1, token-set F1, exact match, normalised exact match
- **SHA-256 checksums** — every task and result is fingerprinted for tamper detection
- **JSONL + CSV export** — outputs compatible with pandas, DuckDB, and standard tools
- **OpenAI-compatible API runner** — works with OpenAI, Azure, vLLM, Ollama, and any OpenAI-compatible endpoint
- **Offline runner** — test any Python callable as a model; no API key required
- **Declarative task files** — JSON (built-in) or YAML (requires `pyyaml`) task specs
- **Tkinter GUI** — point-and-click benchmark runs, results table, summary view

---

## Installation

```bash
# From source (recommended during development)
pip install -e .

# With YAML task file support
pip install -e ".[yaml]"
```

Python 3.8 or later is required.

---

## Quick Start

### Python API

```python
from llmbench import BenchmarkRunner, BenchmarkSpec, Task

# Use the built-in sample tasks
runner = BenchmarkRunner()

# Run against any callable (no API key needed)
results = runner.run_offline(lambda prompt: "Paris")

print(runner.summarize())
```

### Custom task file

```python
from llmbench import BenchmarkRunner, BenchmarkSpec

spec = BenchmarkSpec.from_file("tasks/qa_simple.json")
runner = BenchmarkRunner(spec=spec)
results = runner.run_offline(my_model_fn)
runner.export_jsonl("results.jsonl")
runner.export_csv("results.csv")
```

### Run against an OpenAI-compatible API

```python
import os
from llmbench import BenchmarkRunner

runner = BenchmarkRunner()
runner.run_openai(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",
)
print(runner.summarize())
```

### Use a custom endpoint (vLLM, Ollama, Azure, …)

```python
runner.run_openai(
    api_key="...",
    model="llama3",
    base_url="http://localhost:11434/v1",  # Ollama
)
```

---

## CLI

```bash
# Demo: run built-in tasks with echo model (no API key needed)
llmbench

# Run against an API
llmbench run --model gpt-4o-mini --output results.jsonl

# Supply an API key inline or via environment variable
llmbench run --api-key sk-... --model gpt-4o-mini

# Use a custom task file
llmbench run --tasks tasks/qa_simple.json --output qa_results.jsonl

# Score predictions from a JSONL file
llmbench score results.jsonl
llmbench score results.jsonl --format json

# Print built-in sample tasks
llmbench sample

# Launch the GUI
llmbench gui
```

---

## GUI

Launch the graphical interface:

```bash
llmbench gui
```

Or from Python:

```python
from llmbench.gui import launch_gui
launch_gui()
```

The GUI provides four tabs:

| Tab | Description |
|-----|-------------|
| **Tasks** | Browse loaded tasks; load a JSON/YAML file or use built-in samples |
| **Run** | Choose offline or API mode; configure model and credentials; run and monitor progress |
| **Results** | Sortable table of per-task scores (composite, ROUGE-L, F1, EM, BLEU-1, latency) |
| **Summary** | JSON summary with overall and per-category statistics |

Export results to JSONL or CSV from the **File** menu.

---

## Task File Format

Tasks are defined in JSON (or YAML with `pip install pyyaml`):

```json
{
  "name": "my_benchmark",
  "version": "1.0",
  "tasks": [
    {
      "task_id": "q1",
      "category": "qa",
      "prompt": "What is the capital of France?",
      "reference": "Paris"
    }
  ]
}
```

See [`tasks/qa_simple.json`](tasks/qa_simple.json) and [`tasks/coding_basic.json`](tasks/coding_basic.json) for ready-to-use examples.

---

## Output Format

Each result row (JSONL) includes:

| Field | Description |
|-------|-------------|
| `task_id` | Unique task identifier |
| `category` | Task category (e.g. `qa`, `coding`) |
| `prediction` | Model output |
| `reference` | Ground-truth reference |
| `rouge_l` | ROUGE-L F1 (0–1) |
| `bleu_1` | BLEU-1 with brevity penalty (0–1) |
| `f1` | Token-set F1 (0–1) |
| `exact_match` | Strict string equality (0 or 1) |
| `exact_match_norm` | Normalised exact match (0 or 1) |
| `composite_score` | `0.4×ROUGE-L + 0.3×F1 + 0.2×EM_norm + 0.1×BLEU-1` |
| `latency_s` | Wall-clock inference time in seconds |
| `task_checksum` | SHA-256 of `(task_id, prompt, reference)` |
| `run_timestamp` | ISO-8601 UTC timestamp |
| `error` | Error message if inference failed, else `null` |

---

## Reproducibility

Every task carries a SHA-256 checksum over `(task_id, prompt, reference)`. Every result carries a second SHA-256 over the prediction and all metric scores. Include these checksums in publications to allow exact verification of reported numbers.

---

## Metrics

| Metric | Formula |
|--------|---------|
| **Exact Match** | `1` iff `strip(pred) == strip(ref)` |
| **Exact Match (norm)** | `1` iff lowercased token sequences are equal |
| **ROUGE-L** | F1 based on longest common subsequence of tokens |
| **BLEU-1** | Clipped unigram precision × brevity penalty |
| **F1** | `2·|P∩R| / (|P|+|R|)` over token sets |
| **Composite** | `0.4·RL + 0.3·F1 + 0.2·EM_norm + 0.1·B1` |

All metrics are implemented in pure Python stdlib with no external dependencies.

---

## Development

```bash
pip install -e ".[yaml]"
pytest tests/ -v
```

---

## Citation

If you use llmbench in your research, please cite:

```bibtex
@article{deshmukh2026llmbench,
  title   = {llmbench: A lightweight framework for reproducible benchmarking of large language models},
  author  = {Deshmukh, Vaibhav},
  year    = {2026},
  journal = {Journal of Open Source Software}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
