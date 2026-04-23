---
title: 'llmbench: A lightweight framework for reproducible benchmarking of large language models'
tags:
  - Python
  - LLM
  - benchmarking
  - evaluation
  - reproducibility
authors:
  - name: Vaibhav Deshmukh
    orcid: 0000-0001-6745-7062
    affiliation: 1
affiliations:
  - name: Independent Researcher, Nagpur, India
    index: 1
date: 23 April 2026
bibliography: paper.bib
---

# Summary

`llmbench` is a lightweight Python framework for designing, running, and reporting reproducible benchmarks of large language models (LLMs). It provides a declarative YAML-based task specification format, a provider-agnostic inference client that supports OpenAI-compatible APIs and locally hosted models, deterministic seed management, and structured result storage in JSONL format with SHA-256 checksums for tamper detection [@nist2015sha]. Benchmark results are stored with complete provenance — model identifier, inference parameters, prompt templates, and timestamps — enabling exact reproduction of reported numbers.

# Statement of Need

LLM benchmarking lacks standardisation: researchers write ad hoc evaluation scripts that are rarely shared, making published results difficult to reproduce or compare [@gundersen2018state; @kung2023chatgpt]. Existing benchmarking frameworks are either tightly coupled to specific model providers or require complex infrastructure. `llmbench` occupies the lightweight end of the spectrum: it can be installed with pip, configured with a single YAML file, and run on any machine with API access to an LLM. It produces structured output compatible with common analysis tools while recording all information needed to reproduce the benchmark exactly [@pineau2021improving].

# Acknowledgements

The author used Claude (Anthropic) for drafting portions of this manuscript. All scientific claims and design decisions are the author's own.

# References
