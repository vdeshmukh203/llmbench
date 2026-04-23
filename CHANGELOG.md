# Changelog

## [Unreleased]
- Multi-turn conversation task type (#1)
- Statistical significance testing between benchmark runs (#2)
- Ollama local inference provider (#3)

## [0.1.0] - 2026-04-23
### Added
- YAML-driven benchmark task definitions
- Provider-agnostic runner: OpenAI, Anthropic, Google
- SHA-256 result hashing for reproducibility verification
- Exact match, BLEU, and custom metric support
- CLI: `llmbench run tasks/qa_simple.yaml`
- Python API: `BenchmarkRunner`, `BenchmarkSpec`
