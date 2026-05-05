# llmbench

**Lightweight reproducible LLM benchmarking framework**

`llmbench` evaluates language model outputs against reference answers using a suite of standard text-similarity metrics (ROUGE-L, BLEU-1, Exact Match, F1). It runs against any OpenAI-compatible API, or against a local callable, and exports structured results with SHA-256 provenance checksums for full reproducibility.

---

## Features

- **Four metrics** out of the box: ROUGE-L, BLEU-1, Exact Match (raw & normalised), and token-level F1
- **Weighted composite score** combining all four metrics
- **OpenAI-compatible API** support via stdlib `urllib` — no extra dependencies
- **Offline mode** for benchmarking local callables (any `str → str` function)
- **Declarative JSON spec** format for portable, versioned benchmark suites
- **SHA-256 provenance checksums** on every JSONL result row
- **JSONL + CSV export** for downstream analysis
- **Graphical interface** (tkinter, stdlib) via `llmbench gui` or `llmbench-gui`
- **Stdlib-only** — no mandatory third-party packages

---

## Installation

```bash
pip install llmbench
```

Or from source:

```bash
git clone https://github.com/vdeshmukh203/llmbench.git
cd llmbench
pip install -e .
```

---

## Quick start

### Python API

```python
import llmbench as lb

# Score a callable against the built-in 10-task suite
runner = lb.BenchmarkRunner()
results = runner.run_offline(lambda prompt: "42")  # replace with your model

summary = runner.summarize()
print(summary["overall"])
# {'n': 10, 'exact_match': 0.1, 'rouge_l': ..., 'composite': ..., ...}

# Export
runner.export_jsonl("results.jsonl")
runner.export_csv("results.csv")
```

### Individual metrics

```python
import llmbench as lb

lb.rouge_l("the cat sat on the mat", "the cat sat")       # 0.75
lb.bleu_1("the cat sat", "the cat sat on the mat")        # 0.8
lb.exact_match("Paris", "Paris")                          # 1.0
lb.exact_match_normalised("paris.", "Paris")              # 1.0
lb.f1_score("the quick brown fox", "the slow brown fox")  # 0.857...
```

### CLI

```bash
# Print built-in sample tasks
llmbench sample

# Run demo benchmark (offline, no API key)
llmbench

# Run against OpenAI API
llmbench run --api-key sk-... --model gpt-4o-mini --output results.jsonl

# Score a JSONL file of predictions
llmbench score predictions.jsonl
llmbench score predictions.jsonl --format json

# Launch GUI
llmbench gui
# or
llmbench-gui
```

### Benchmark spec file

Create `my_bench.json`:

```json
{
  "name": "my-evaluation",
  "description": "Custom QA benchmark",
  "model": "gpt-4o-mini",
  "max_tokens": 256,
  "temperature": 0.0,
  "tasks": [
    {"task_id": "q1", "category": "qa", "prompt": "Capital of France?", "reference": "Paris"},
    {"task_id": "q2", "category": "qa", "prompt": "2 + 2 = ?",          "reference": "4"}
  ]
}
```

```bash
llmbench run --spec my_bench.json --api-key sk-...
```

---

## Metrics

| Metric | Description |
|---|---|
| **Exact Match** | `1.0` if prediction equals reference after stripping whitespace |
| **Exact Match (norm.)** | Exact match after lowercasing and removing punctuation |
| **ROUGE-L** | F1 based on longest common subsequence of tokens |
| **BLEU-1** | Unigram precision with brevity penalty |
| **F1** | Token-level set F1 (precision × recall harmonic mean) |
| **Composite** | `0.40·ROUGE-L + 0.30·F1 + 0.20·EM-norm + 0.10·BLEU-1` |

---

## Reproducibility

Every row in JSONL output includes a `sha256` field — a SHA-256 checksum of the canonically serialised result. This allows detection of any post-hoc modification to result files.

---

## Graphical Interface

```bash
llmbench-gui   # or: llmbench gui
```

The GUI provides:
- **Run tab** — configure mode (demo / API), model, temperature, max tokens; track progress
- **Results tab** — per-task score table; click any row to inspect the prompt, prediction, and reference
- **Summary tab** — aggregate statistics by category
- File → **Load Spec** to import a JSON benchmark spec
- File → **Export CSV / JSONL** to save results

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Citation

If you use `llmbench` in academic work, please cite:

```bibtex
@article{llmbench2026,
  author  = {Deshmukh, Vaibhav},
  title   = {llmbench: A lightweight framework for reproducible benchmarking of large language models},
  journal = {Journal of Open Source Software},
  year    = {2026}
}
```

---

## License

MIT © Vaibhav Deshmukh
