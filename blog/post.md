# The Best Human Coders Are Undetectable

*SentinalAI Research*

---

I spent several months building a system to detect AI-generated code. It works
great — until it doesn't. Here's what I actually found, and why I think most
"AI detector" accuracy numbers you've seen are measuring the wrong thing entirely.

---

## The Setup

There's a real research question here: can you tell AI-written code from human-written
code just by looking at surface patterns — line lengths, comment density, naming
style, nesting depth? No model inference, no embeddings, just old-school stylometry.

I built a 65-feature extractor (lexical + Python AST), collected three datasets, and
ran a proper cross-domain evaluation. Here's what the numbers look like:

### Within a single dataset: works great

| Dataset | What it is | F1 Score |
|---|---|---|
| SemEval-2026 | 500K samples, 34 AI models, scraped human code | **0.9544** |
| Paired Codeforces | Same competitive programming problem: human (pre-2022) vs. deepseek-coder | **0.9931** |
| HumanEval | OpenAI canonical functions vs. deepseek-coder | **0.9200** |

99% accuracy on competitive code. 95% on the big benchmark. Looking good.

### Cross-dataset: falls apart completely

| Train on | Test on | F1 Score | Notes |
|---|---|---|---|
| SemEval (500K samples, 34 models) | Paired Codeforces | **0.4038** | Worse than random |
| SemEval | HumanEval | **0.3320** | Worse than random |
| Codeforces | SemEval | 0.6089 | Partial |
| Codeforces | HumanEval | **0.4449** | Below random |

Random chance is 0.50. A classifier trained on 100,000 samples from 34 different
AI models performs *worse than flipping a coin* when you change the human baseline.

"Worse than random" isn't just "noisy" — it means the classifier is **confidently wrong**,
flipping its labels in a predictable direction. It's not failing to find a pattern; it
learned a real pattern that happens to be inverted in the new domain. That's a more
damning result than "doesn't generalize."

---

## Why? Start With the Features That Lie

Before looking at which features are reliable, here are the ones that actively betray you.
These "conflict features" are statistically significant in both training and test corpora —
but they point in **opposite directions**:

| Feature | SemEval: AI vs human | Codeforces: AI vs human |
|---|---|---|
| line_count | AI writes **+17 more** lines | AI writes **−11 fewer** lines |
| comment_line_ratio | AI comments **more** | AI comments **less** |
| whitespace_ratio | AI uses **more** whitespace | AI uses **less** whitespace |

Same feature. Opposite signal. In SemEval, the human baseline is scraped forum code —
terse, uncommented — so AI looks verbose by comparison. In Codeforces, the human
baseline is competitive programmers — verbose, explicit — so AI looks compact.
The classifier learns the *contrast between AI and whatever specific humans it saw*,
not anything stable about AI itself.

This is why cross-corpus transfer doesn't just degrade — it inverts.

---

## The Overlap: Why HumanEval Humans Look Like AI

I measured where each corpus's human and AI distributions sit in feature space.
For the top 10 most discriminative features, here's what I found:

| Feature | SemEval humans | SemEval AI | HumanEval humans | HumanEval AI |
|---|---|---|---|---|
| avg_line_length | 20.5 | **26.4** | **29.4** | 30.8 |
| blank_line_ratio | 0.03 | **0.16** | **0.15** | 0.14 |
| snake_case_count | 1.42 | **5.60** | **4.18** | 4.73 |
| naming_style_ratio | 0.15 | **0.45** | **0.63** | 0.68 |
| char_count | 442 | **1,027** | **631** | 677 |

Bold = in the "AI zone" for the SemEval-trained detector.

**HumanEval humans sit inside the SemEval-AI distribution on 10 out of 10 features.**

The detector trained on SemEval learned:
- `avg_line_length > 26` → AI
- `snake_case_count > 4` → AI
- `naming_style_ratio > 0.40` → AI

HumanEval humans have `avg_line_length = 29.4`, `snake_case_count = 4.18`,
`naming_style_ratio = 0.63`. They trigger every learned AI criterion — because
they're expert programmers writing polished, idiomatic Python. The classifier
never met a human who coded this well.

---

## The Actual Finding

This isn't random domain shift. It's **human-quality drift**:

> AI models produce relatively consistent surface style across contexts because they're
> trained to generate quality code. Human authors vary enormously — from rushed forum
> snippets to competitive submission to carefully crafted canonical functions.
> A detector trained on any single human population learns the location of *that
> population* in feature space, not anything fundamental about AI output.

---

## What Actually Works

### 1. Train on diverse humans

When I trained on all three corpora combined — 10,000 SemEval samples + 286
Codeforces + 328 HumanEval — accuracy recovered to **0.9517** across all three.
Same 65 features. Same Random Forest. Just a broader human baseline.

| Slice | F1 |
|---|---|
| SemEval | 0.9612 |
| Codeforces | 0.7817 |
| HumanEval | 0.8101 |
| **Overall** | **0.9517** |

The features were never the problem. The human sample was.

### 2. Use stable features only

9 of the 65 features point in the same direction across all three corpora:
`avg_line_length`, `max_line_length`, `p90_line_length`, `digit_ratio`,
`snake_case_count`, `long_ident_count`, `naming_style_ratio`,
`early_return_count`, `trailing_ws_ratio`.

Using only those 9 raises cross-corpus F1 from 0.38 to 0.62 on the
SemEval → Codeforces transfer. Not solved, but nearly doubled.

### 3. Scores, not labels

I built a **Human Baseline Distance (HBD) scorer** — a Mahalanobis distance from
the union human distribution on the 9 stable features. Instead of "AI: yes/no,"
it outputs a continuous score: how far is this code from the average of all the
humans I've characterized?

| Context | Human HBD | AI HBD | AUROC |
|---|---|---|---|
| Codeforces | 1.83 | 8.06 | **0.975** |
| SemEval | 1.78 | 4.61 | **0.798** |
| HumanEval | 4.35 | 4.56 | **0.540** |

The HumanEval AUROC of 0.540 is not a failure — it's the finding made honest.
Expert humans and AI land at HBD 4.35 and 4.56 respectively. They're in the same
region. The scorer says: "I can't tell these apart, and here's the number that
proves I know I can't."

---

## What This Means for AI Detectors in the Wild

If someone shows you an AI code detector with 95% accuracy, ask one question:

**Who wrote the human code in your training set?**

If the answer is "scraped GitHub repos" or "student submissions" or "Stack Overflow
answers" — and your deployment context involves a different kind of human writer —
that 95% number does not apply to you.

If you're being shown an AI-detection tool for hiring, academic integrity, or code
review — ask the vendor what the human population in their training set looks like,
and whether it matches who actually writes code in your context. If they can't answer,
the accuracy number is not predictive of anything in your deployment.

The detection is real. The signal is there. But it's calibrated to a specific human
population, and when that population shifts, the calibration inverts.

The honest version of AI code detection isn't "this was written by AI." It's:
"this code is X standard deviations from the average of the humans I've characterized.
Make of that what you will."

---

## Appendix: Model Fingerprinting

While I had the data, I also tested a separate question: can you identify *which* AI
model wrote a piece of code, from style alone?

- **Macro F1 = 0.2342** across 35 classes (34 AI models + human) vs random chance of 0.0286 — ~8x above chance
- Models cluster by architecture family — Phi-3 variants are nearly indistinguishable from each other, Yi-Coder variants similarly
- `starcoder2-15b` and `starcoder2-3b` are 23% confused — bigger model, same stylistic recipe
- **The human class is the most recognizable** (F1 = 0.622) within SemEval

This is a separate finding from the detection problem: within a single corpus,
individual models do have distinguishable fingerprints. The failure is cross-domain
transfer, not signal existence. More in [PAPER.md](../PAPER.md).

---

## The Code

Everything is open at [github.com/zer0contextlost/SentinalAI](https://github.com/zer0contextlost/SentinalAI):

- 65-feature extractor (`features/extractor.py`)
- Corpus collection scripts (SemEval, paired Codeforces, HumanEval)
- All training and evaluation scripts
- Trained HBD scorer (`models/hbd_scorer.pkl`)
- Full paper draft (`PAPER.md`)

The full research writeup is in `FINDINGS.md`. The academic paper draft is in `PAPER.md`.

---

*Want the same thing explained with pizza? [Read the pizza version →](explain-it-in-pizza)*
