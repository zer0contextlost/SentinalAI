# SentinalAI

Research codebase for AI-generated code detection via style fingerprinting.

## What This Is

An empirical study of whether stylometric features can reliably detect AI-generated
code. Includes a 65-feature extractor, corpus collection pipelines, and classifier
evaluation across two datasets — SemEval-2026 Task 13 (500K samples, 34 AI models)
and a novel paired Codeforces corpus (same problem, human vs. AI solution).

## Project Structure

```
features/          65-feature extractor (lexical + Python AST)
collector/         corpus collection — SemEval pull, Codeforces AI gen, GitHub human scrape
scripts/           feature matrix build, perplexity validation, classifier training/eval
models/            baseline and paired corpus classifier training
api/               inference endpoint (stub)
```

## Datasets

**SemEval-2026 Task 13** — `DaniilOr/SemEval-2026-Task13` on HuggingFace  
500K samples, 34 AI generators, Python/C++/Java

**Paired Codeforces corpus** — built by this repo  
96 competitive programming problems each solved by a human (pre-2022 GitHub) and an AI (deepseek-coder:6.7b).
Collected via `collector/scrape_codeforces.py` and `collector/fetch_github_human_solutions.py`.

## Setup

```bash
pip install -r requirements.txt
# Requires Ollama running at localhost:11434 for AI generation and perplexity scoring
# ollama pull deepseek-coder:6.7b
```

## Running

```bash
# Pull SemEval corpus
python collector/pull_semeval_dataset.py

# Build feature matrix
python scripts/build_feature_matrix.py

# Train baseline on SemEval
python models/train_baseline.py

# Build paired corpus features
python scripts/build_paired_features.py

# Train on paired corpus (leave-one-problem-out CV)
python scripts/train_paired_classifier.py

# Cross-dataset generalization tests
python scripts/test_generalization.py
python scripts/test_generalization_reverse.py
```
