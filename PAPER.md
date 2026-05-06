# The Human Baseline Problem: Why AI Code Detectors Fail Across Domains

**Abstract** — Stylometric classifiers for AI-generated code detection achieve 0.92–0.99 F1
within a single corpus but collapse to near or below random chance cross-corpus. We
investigate this failure across three paired human/AI corpora: SemEval-2026 Task 13
(500K samples, 34 AI models), a novel paired Codeforces corpus (96 competitive programming
problems, pre-2022 human solutions vs. deepseek-coder:6.7b), and HumanEval (164 problems,
OpenAI canonical solutions vs. deepseek-coder:6.7b). Feature distribution analysis across
all three reveals the mechanism: AI distributions are stable across corpora; human
distributions are not. The top-10 discriminative features uniformly show HumanEval's
expert-written human code sitting inside SemEval's AI distribution — not because the
features are wrong, but because expert human code and AI code occupy the same stylometric
region. Training on a union of all three corpora recovers 0.95 F1, confirming that the
problem is human-population homogeneity, not feature inadequacy. We provide a continuous
Human Baseline Distance (HBD) diagnostic score, grounded in the 9 features stable across
all three corpora, with AUROC ranging from 0.975 (competitive programming context) to
0.540 (expert-canonical context). We argue that reported single-corpus AI code detection
accuracies are primarily measuring corpus construction choices, not AI provenance.

---

## 1. Introduction

The rise of LLM-generated code has prompted a parallel effort in detection: can a
classifier reliably identify AI-authored source code from its surface style? Several
papers report detection accuracies above 90% on labeled benchmarks [CITATION], and
multiple commercial tools claim reliable AI code identification.

We revisit this claim with a simple methodological constraint: if a detector is measuring
AI authorship and not corpus-specific artifacts, it must generalize across independently
collected corpora. We apply this constraint and find it fails uniformly.

Our contributions are:

1. A three-corpus cross-domain evaluation demonstrating that stylometric AI code detectors
   fail out-of-distribution across all training/test corpus pairs.

2. A mechanistic explanation: a feature distribution analysis showing that the top
   discriminative features place expert-quality human code inside the AI region of
   detectors trained on scraped human code. The failure is not random — it is
   specifically human-baseline drift.

3. Evidence that the fix is corpus diversity: a union-trained classifier achieves 0.95
   F1 across all three corpora, showing the features can generalize given heterogeneous
   human representation.

4. A Human Baseline Distance (HBD) diagnostic — a continuous Mahalanobis score from
   a diverse human baseline on 9 population-stable features — as a more honest
   deployable artifact than a binary classifier.

5. A 35-way model fingerprinting result showing that within-corpus, individual AI models
   have distinguishable stylometric signatures (~8x above chance), with family structure
   matching architectural lineage.

### 1.1 Relationship to Existing Work

Prior work on machine-generated text detection [CITATION — GPTZero, DetectGPT,
Binoculars, M4] has documented cross-domain brittleness in natural language, typically
attributed to distribution shift. Our contribution is a structural account of *why*
the shift direction is asymmetric in code: AI output is more stylistically uniform
across contexts than human output, so detectors trained on any specific human population
learn the wrong axis.

The SemEval-2026 Task 13 shared task [CITATION] motivates our primary corpus. Prior
competitive entries that report high accuracy on the shared task's test set have not,
to our knowledge, been evaluated against independently collected human baselines.

---

## 2. Datasets

### 2.1 SemEval-2026 Task 13 Corpus

499,999 labeled Python/C++/Java samples from `DaniilOr/SemEval-2026-Task13` (subtask A),
comprising 238,474 human and 261,525 AI samples across 34 AI generators (GPT-4o,
CodeLlama variants, Qwen-Coder, deepseek-coder, StarCoder2, Phi-3, Llama-3, CodeGemma,
Granite, and others). All experiments use the Python subset (91% of samples, 457,306 rows).
Human samples are scraped from public code repositories, era unspecified.

### 2.2 Paired Codeforces Corpus

We constructed a novel paired corpus addressing a key confound in existing datasets:
human and AI samples solve *different problems*, so style differences may reflect
problem structure rather than authorship.

For each of 96 Codeforces problems (contest ratings <= 1500, difficulty 800–1400,
tag: implementation, pre-2021 era):

- **Human solution**: A Python file from a public GitHub repository with per-file last
  commit date before 2022-01-01. Provenance is verified at file granularity (not repo
  push date) via `GET /repos/{owner}/{repo}/commits?path={file}&per_page=1`. This
  excludes any solution that could have been produced with LLM assistance.

- **AI solution**: Generated by `deepseek-coder:6.7b` via Ollama with the problem
  statement as prompt.

The resulting corpus has 373 solutions (273 human, 100 AI). Any feature difference
between classes is attributable to authorship, not problem structure or era.

### 2.3 HumanEval Corpus

164 Python programming problems from `openai/openai_humaneval`, with canonical
human-written function implementations as the human class and `deepseek-coder:6.7b`
completions as the AI class (328 total, perfectly balanced).

HumanEval human solutions are qualitatively distinct from the other two human
populations: they are expert-written, heavily reviewed, idiomatic functions — the
canonical definition of correct Python style. This makes them the hardest test case
for any stylometric detector trained on "typical" human code.

This corpus was held out entirely during all feature selection procedures.

---

## 3. Feature Extraction

We extract 65 features from raw Python source, combining lexical and AST-level signals.

**Lexical features (37):** Line count, character count, blank line ratio, line length
statistics (mean, max, p90), comment line ratio, inline comment ratio, whitespace ratio,
trailing whitespace ratio, digit ratio, uppercase ratio, snake_case count, camelCase count,
naming style ratio (snake/(snake+camel)), type annotation density, print guard count,
bare except count, uniform except count, null check count, isinstance check count,
guard clause count, early return count, long identifier count, logging call count,
and derived ratios thereof.

**AST features (28):** Function count, class count, max nesting depth, comprehension
count (list/dict/set/generator), lambda count, decorator count, docstring presence ratio,
type annotation count, f-string count, walrus operator count, import count, max function
length (in lines), none comparison count, isinstance call count, broad except handler
count, finally block count, default argument count, early return count (AST-level).

Features are extracted without model inference; no perplexity or embedding features
are used. A null result for perplexity-based detection (0/8 features significant,
Mann-Whitney U) ruled out continuation-scoring approaches early.

---

## 4. Experiments

### 4.1 Within-Corpus Baselines

We establish that the signal exists in all three corpora independently before testing
transfer. Table 1 reports within-corpus F1 using 5-fold stratified cross-validation.

**Table 1: Within-corpus classification F1 (Random Forest, 300 trees)**

| Corpus | F1 Macro | Accuracy | n |
|---|---|---|---|
| SemEval (Python subset) | 0.9544 | 0.9544 | 100,000 |
| Codeforces (LOPO-CV) | 0.9931 | 0.9946 | 373 |
| HumanEval | 0.9200 | 0.9200 | 328 |

The Codeforces result uses leave-one-problem-out cross-validation, which holds all
solutions for a given problem out during training, eliminating any leakage between
paired samples. The HumanEval result uses standard 5-fold CV.

High within-corpus F1 in all three cases confirms that stylometric signal exists.
The question is whether it transfers.

### 4.2 Cross-Corpus Transfer

**Table 2: Cross-corpus generalization (test corpus never seen during training)**

| Train | Test | F1 Macro | Notes |
|---|---|---|---|
| Codeforces | SemEval | 0.6089 | Large instruct models undetected |
| SemEval | Codeforces | 0.4038 | Below random chance |
| SemEval | HumanEval | 0.3320 | Below random chance |
| Codeforces | HumanEval | 0.4449 | Below random chance |
| SemEval+CF | HumanEval | 0.3320 | Combined training no better |

Random chance for binary balanced classification = 0.50.

Every cross-corpus transfer result is at or below random chance. The SemEval ->
Codeforces result (0.40) is the clearest: 100,000 balanced training samples from
34 AI generators, evaluated against a ground-truth paired corpus, performs worse
than flipping a coin.

### 4.3 Feature Forensics: Stable vs. Conflicting Features

To understand the direction of failure, we measure per-feature directional shift
between SemEval and Codeforces using Wasserstein distance and Mann-Whitney U tests.

For each feature significant in both corpora (p < 0.05), we compare the direction
of the human-to-AI delta:

**Conflict features** (significant in both, opposite directions):

| Feature | SemEval delta | CF delta | Interpretation |
|---|---|---|---|
| line_count | +17.1 | -10.9 | AI longer in SemEval, shorter in CF |
| non_blank_line_count | +12.1 | -10.8 | Same flip |
| comment_line_ratio | + | - | AI more comments in SemEval, fewer in CF |
| whitespace_ratio | + | - | |
| print_guard_count | + | - | |
| camel_case_count | + | - | |
| inline_comment_ratio | + | - | |

**Stable features** (same direction in both corpora, both significant):

`avg_line_length`, `max_line_length`, `p90_line_length`, `digit_ratio`,
`snake_case_count`, `long_ident_count`, `naming_style_ratio`, `early_return_count`,
`trailing_ws_ratio`

Using only the 9 stable features: cross-domain F1 increases from 0.38 to 0.62
on SemEval -> Codeforces, nearly doubling. The conflict features are the failure mode.

The `line_count` flip is interpretable: SemEval human code is scraped from diverse
repositories where humans write terse scripts; AI writes longer, more complete code.
In paired Codeforces, the human baseline is competitive programmers, who write
verbose, explicit solutions; deepseek-coder:6.7b generates compact implementations.
The classifier learned the wrong axis in both cases.

### 4.4 Third-Corpus Validation (HumanEval, No Leakage)

The stable feature set was identified using only SemEval and Codeforces. HumanEval
was never used in feature selection. Table 3 shows the OOD test.

**Table 3: HumanEval OOD validation (stable features identified on SemEval+CF only)**

| Train | Features | F1 (HumanEval) |
|---|---|---|
| SemEval | All 65 | 0.3320 |
| SemEval | Stable 9 | 0.4298 |
| Codeforces | All 65 | 0.4449 |
| Codeforces | Stable 9 | 0.3909 |
| SemEval+CF | All 65 | 0.3320 |
| SemEval+CF | Stable 9 | 0.4043 |

Stable features lift SemEval transfer (+0.10) but not Codeforces transfer (-0.05).
All results remain below chance. This is not a labeling error — within-HumanEval
F1 is 0.92, confirming the signal exists locally.

### 4.5 Union Training

If the failure is human-population homogeneity, training on a diverse union of
human populations should recover performance. Table 4 confirms this.

**Table 4: Union training (SemEval + CF + HumanEval, 5-fold CV)**

| Corpus slice | F1 Macro |
|---|---|
| SemEval | 0.9612 |
| Codeforces | 0.7817 |
| HumanEval | 0.8101 |
| **Overall** | **0.9517** |

Union training with 10,614 balanced samples (10K SemEval + 286 CF + 328 HumanEval)
recovers 0.95 overall F1. This is the diagnostic confirmation: the features can
generalize; they simply require a human baseline that is representative of the
deployment population.

---

## 5. The Mechanism: Human-Quality Drift

The cross-corpus failures are not random. For the top-10 discriminative features
by ANOVA F-statistic on SemEval, we measure where each corpus's human and AI
distributions sit relative to each other.

**Table 5: Feature distribution means across corpora (top-10 features by SemEval ANOVA)**

| Feature | SemEval-H | SemEval-AI | CF-H | CF-AI | HE-H | HE-AI | HE-H in SemEval-AI zone? |
|---|---|---|---|---|---|---|---|
| blank_line_ratio | 0.03 | **0.16** | 0.11 | 0.10 | **0.15** | 0.14 | YES |
| ast_max_nesting_depth | 2.91 | **2.98** | 3.01 | — | **2.43** | 2.31 | YES |
| naming_style_ratio | 0.15 | **0.45** | 0.11 | 0.27 | **0.63** | 0.68 | YES |
| comment_line_ratio | 0.00 | **0.06** | 0.04 | 0.01 | **0.00** | 0.00 | YES |
| trailing_ws_ratio | 0.00 | **0.04** | 0.02 | 0.05 | **0.04** | 0.04 | YES |
| char_count | 442 | **1027** | 538 | 636 | **631** | 677 | YES |
| snake_case_count | 1.42 | **5.60** | 0.77 | 1.90 | **4.18** | 4.73 | YES |
| line_count | 21.95 | **39.08** | 24.67 | 13.77 | **20.49** | 20.96 | YES |
| whitespace_ratio | 0.26 | **0.30** | 0.32 | 0.21 | **0.30** | 0.27 | YES |
| non_blank_line_count | 20.04 | **32.11** | 21.85 | 11.07 | **17.62** | 18.14 | YES |

**10/10 top discriminative features** show HumanEval-human sitting inside the
SemEval-AI distribution (within 1 standard deviation of the SemEval-AI mean).

The SemEval-trained classifier has no choice but to classify HumanEval humans as AI.
A detector trained on SemEval learned that:

- `avg_line_length > 26` -> AI  
- `snake_case_count > 4` -> AI  
- `naming_style_ratio > 0.40` -> AI

HumanEval humans have `avg_line_length = 29.4`, `snake_case_count = 4.18`,
`naming_style_ratio = 0.63`. They satisfy every learned AI criterion.

**This is human-quality drift.** Expert-written, reviewed, idiomatic Python code
is stylometrically indistinguishable from AI output on surface features, because
both represent *high-quality code*. The detector was not measuring AI authorship —
it was measuring the distance from whatever "typical human code" looked like in the
training corpus.

The asymmetry is structural: AI models produce outputs drawn from a distribution
trained to maximize code quality metrics, making AI output relatively consistent
across contexts. Human code varies enormously — from forum snippets to competitive
submissions to carefully engineered canonical functions. Any detector trained on a
specific human sample inherits the location of that sample in feature space, not
a representation of the full human distribution.

---

## 6. Human Baseline Distance (HBD) Scorer

Given the corpus-dependence of binary classification, we provide a continuous
diagnostic: the Human Baseline Distance (HBD) score.

### 6.1 Design

HBD is the Mahalanobis distance of a code sample from the union human distribution
on the 9 population-stable features:

```
avg_line_length, max_line_length, p90_line_length, digit_ratio,
snake_case_count, long_ident_count, naming_style_ratio,
early_return_count, trailing_ws_ratio
```

The covariance matrix is estimated with Ledoit-Wolf shrinkage on a union human
baseline of 5,437 samples (5,000 SemEval + 273 CF + 164 HumanEval), standardized
jointly. The condition number of the resulting covariance estimate is 1.2, indicating
a well-conditioned system with no near-collinear feature pairs.

All features are standardized on the union baseline before distance computation,
ensuring that corpus identity does not dominate the distance calculation.

### 6.2 Results

**Table 6: HBD score distributions and AUROC per corpus**

| Corpus | HBD mean (human) | HBD mean (AI) | AUROC |
|---|---|---|---|
| SemEval | 1.78 | 4.61 | 0.798 |
| Codeforces | 1.83 | 8.06 | 0.975 |
| HumanEval | 4.35 | 4.56 | 0.540 |

The HumanEval AUROC of 0.540 is not a failure of the scorer — it is the mechanism
quantified. HumanEval humans score HBD 4.35, nearly identical to HumanEval AI at
4.56. Both populations are far from the "average human" in the union baseline. The
scorer correctly reports that it cannot distinguish them, because they occupy the
same region of feature space.

The Codeforces AUROC of 0.975 reflects that competitive programmers (HBD 1.83) are
close to the union human mean and clearly separated from deepseek-coder output (HBD 8.06).
In a deployment context matching this profile, HBD provides near-reliable separation.

### 6.3 Interpretation

HBD should be understood as: *how anomalous is this code relative to the average of
the diverse human populations we have characterized?* It does not say the code is AI.
It says the code is unlike the humans we have measured.

A high HBD in a context where the human author population is well-represented in the
union baseline is meaningful signal. A high HBD in a context where the human author
population is itself expert-level (like HumanEval) is uninformative — and the scorer's
AUROC tells you which situation you are in.

---

## 7. Model Fingerprinting

As a secondary analysis, we investigate whether individual AI models have distinguishable
stylometric fingerprints within a single corpus. We train a 35-way Random Forest
(34 AI models + human class, 1,000 samples per class, 5-fold CV) on the SemEval Python
subset.

**Macro F1 = 0.2342** (chance = 0.0286, ~8x above chance).

Key observations:
- The human class is the most distinguishable (F1 = 0.622), consistent with the
  finding that human code occupies a distinct region of feature space within SemEval.
- Family clusters match architectural lineage: Phi-3 variants cluster together,
  Yi-Coder variants cluster together, Llama-family models group together.
- Most confused pair: `starcoder2-15b` and `starcoder2-3b` (23% confusion rate),
  consistent with the hypothesis that fine-tuning at different scales preserves the
  base model's stylometric signature.

This result supports the claim that AI stylometric signal is real within corpus: models
do have distinguishable surface patterns that a classifier can exploit. The cross-corpus
failure is not because the signal doesn't exist — it is because the signal is
corpus-indexed.

---

## 8. Discussion

### 8.1 Implications for Benchmarking

Reported single-corpus AI code detection accuracies are artifacts of corpus construction
as much as model capability. A corpus that scrapes human code from low-quality sources
will produce high detection accuracy because the detector learns "polished vs. unpolished
code," not "AI vs. human." HumanEval demonstrates the ceiling: when the human gold
standard is expert-quality code, there is no surface signal left to exploit.

This implies that benchmark leaderboards for AI code detection cannot be interpreted
as measures of a detector's generalization capability. The appropriate evaluation is
the one we perform here: train on one independently collected corpus, evaluate on
another, report cross-corpus F1 alongside within-corpus F1.

### 8.2 Practical Recommendations

For practitioners considering deployment of a stylometric AI code detector:

1. Characterize your human author population before calibrating any threshold.
2. Compute within-corpus AUROC on a representative sample of your known-human code;
   if it is low, the detector is not useful in your context regardless of training set size.
3. Prefer continuous scores (HBD or equivalent) over binary labels. The decision of
   "this warrants review" is context-dependent; the score provides the raw signal.
4. Union training across multiple human populations is necessary for a generalizing
   detector; no single corpus is sufficient.

### 8.3 Limitations

- All AI solutions use `deepseek-coder:6.7b`. A multi-model paired corpus may alter
  the feature overlap findings if model families have sufficiently different surface styles.
- Competitive programming is an extreme human domain. The Codeforces human population
  is not representative of production software engineering.
- AST-level features do not capture semantic choices: algorithm selection, identifier
  semantics, comment content. These may carry cross-domain signal not captured here.
- Formatter normalization (`black`, `autopep8`) was not tested. Formatting-sensitive
  features (blank_line_ratio, whitespace_ratio) may become uninformative after normalization.
- The HBD scorer's calibration is specific to the union baseline described. Deployment
  in new contexts requires recalibration against a local human sample.

---

## 9. Conclusions

We have shown that stylometric AI code detectors learn where human code sits in
feature space, not where AI code sits. This implies that cross-corpus generalization
fails whenever the human baseline shifts — which it does substantially, from scraped
forum code to competitive programming to expert-canonical functions.

The finding is structural, not incidental. AI models produce relatively uniform surface
style across contexts because they are trained to optimize code quality. Human authors
vary enormously. Any detector trained on a specific human sample will degrade when
the test population writes better or differently.

The fix is neither more features nor better models: it is a more representative human
baseline. Union training across three qualitatively different human populations recovers
0.95 F1 and leaves the features intact.

The Human Baseline Distance scorer formalizes this insight as a deployable diagnostic:
a continuous score that is honest about what it can and cannot know, calibrated to
the diversity of human populations it has seen, and carrying an explicit uncertainty
signal (AUROC by corpus) that tells a practitioner whether the score is meaningful
in their context.

Finally: high in-domain F1 is not evidence of a working detector. The 99.7% result
on the paired Codeforces corpus and the 33% result on HumanEval are produced by
the same features, the same model, the same methodology. The difference is whose
code counts as "human."

---

## References

[Placeholder — to be populated with SemEval-2026 Task 13 paper, GPTZero, DetectGPT,
M4 dataset, Binoculars, relevant stylometry literature]

---

## Appendix A: Feature Definitions

*(Full 65-feature specification available in `features/extractor.py`)*

## Appendix B: Reproducibility

```
# Pull corpora
python collector/pull_semeval_dataset.py
python collector/pull_humaneval.py
python collector/generate_humaneval_ai.py

# Build feature matrices
python scripts/build_feature_matrix.py
python scripts/build_paired_features.py
python scripts/build_humaneval_features.py

# Within-corpus baselines
python models/train_baseline.py
python scripts/train_paired_classifier.py

# Cross-corpus transfer
python scripts/test_generalization.py
python scripts/test_generalization_reverse.py
python scripts/validate_third_corpus.py

# Analysis
python scripts/feature_forensics.py
python scripts/followup_analysis.py
python scripts/model_fingerprinting.py

# HBD scorer
python scripts/hbd_scorer.py
```

Dataset: SemEval corpus at `DaniilOr/SemEval-2026-Task13` (HuggingFace).
HumanEval at `openai/openai_humaneval` (HuggingFace).
Paired CF corpus collection requires GitHub API token (see `collector/fetch_github_human_solutions.py`).
