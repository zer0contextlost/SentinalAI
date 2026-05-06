"""
Test perplexity extractor against a human and AI sample from the corpus.
Run after: ollama pull deepseek-coder:6.7b
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import pandas as pd
from features import perplexity as perp
from features.extractor import extract_features

MODEL = "deepseek-coder:6.7b"

if not perp.is_available(MODEL):
    print(f"Model {MODEL} not available. Run: ollama pull {MODEL}")
    sys.exit(1)

human = pd.read_parquet("corpus/human/semeval/A__train__shard000.parquet")
ai    = pd.read_parquet("corpus/ai/semeval/A__train__shard000.parquet")

# Pick Python samples for clean AST comparison
h_sample = human[human["language"] == "Python"].iloc[0]["code"]
a_sample = ai[ai["language"] == "Python"].iloc[0]["code"]

print("Testing perplexity extractor...\n")
print(f"Human sample ({len(h_sample)} chars):")
print(h_sample[:200], "..." if len(h_sample) > 200 else "")
print(f"\nAI sample ({len(a_sample)} chars):")
print(a_sample[:200], "..." if len(a_sample) > 200 else "")

print("\nExtracting features (this takes a few seconds per sample)...\n")

h_feat = perp.extract(h_sample, MODEL)
a_feat = perp.extract(a_sample, MODEL)

print(f"{'Feature':<30} {'Human':>12} {'AI':>12}  {'AI > Human':>12}")
print("-" * 70)
for k in h_feat:
    hv = h_feat[k]
    av = a_feat[k]
    if isinstance(hv, float):
        direction = "AI higher" if av > hv else "Human higher"
        print(f"{k:<30} {hv:>12.4f} {av:>12.4f}  {direction:>12}")
    else:
        print(f"{k:<30} {str(hv):>12} {str(av):>12}")

print("\nKey insight:")
print(f"  Perplexity  — Human: {h_feat['perp_perplexity']:.2f}  AI: {a_feat['perp_perplexity']:.2f}")
print(f"  Variance    — Human: {h_feat['perp_variance']:.4f}  AI: {a_feat['perp_variance']:.4f}")
print(f"  Surprising  — Human: {h_feat['perp_surprising_ratio']:.3f}  AI: {a_feat['perp_surprising_ratio']:.3f}")
