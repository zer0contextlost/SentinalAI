"""
Validate perplexity signal on a sample of 500 human + 500 AI Python samples.
Tests whether perplexity features distinguish AI from human code statistically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    from scipy import stats
except ImportError:
    print("Run: pip install pandas numpy scipy")
    sys.exit(1)

from features.perplexity import extract, is_available, DEFAULT_MODEL
from tqdm import tqdm

SAMPLE_N    = 200  # per class — sufficient for Mann-Whitney signal detection
CHECKPOINT  = Path(__file__).parent.parent / "corpus" / "processed" / "perplexity_checkpoint.parquet"
OUT         = Path(__file__).parent.parent / "corpus" / "processed" / "perplexity_sample.parquet"
SAVE_EVERY  = 10   # checkpoint after every N samples


def main():
    if not is_available(DEFAULT_MODEL):
        print(f"Model {DEFAULT_MODEL} not available. Run: ollama pull {DEFAULT_MODEL}")
        sys.exit(1)

    # Load checkpoint if exists
    done_rows = []
    done_count = {0: 0, 1: 0}
    if CHECKPOINT.exists():
        ckpt = pd.read_parquet(CHECKPOINT)
        done_rows = ckpt.to_dict("records")
        done_count = ckpt["label"].value_counts().to_dict()
        print(f"Resuming from checkpoint: {done_count.get(0,0)} human, {done_count.get(1,0)} AI scored")

    print("Loading corpus samples...")
    human = pd.read_parquet("corpus/human/semeval/A__train__shard000.parquet")
    ai    = pd.read_parquet("corpus/ai/semeval/A__train__shard000.parquet")

    human_py = human[human["language"] == "Python"].sample(SAMPLE_N, random_state=42)
    ai_py    = ai[ai["language"] == "Python"].sample(SAMPLE_N, random_state=42)

    print(f"Target: {SAMPLE_N} human + {SAMPLE_N} AI Python samples")
    print(f"Model: {DEFAULT_MODEL}\n")

    rows = list(done_rows)
    for label, df, label_name in [(0, human_py, "human"), (1, ai_py, "ai")]:
        already = done_count.get(label, 0)
        remaining = df.iloc[already:]
        if already >= SAMPLE_N:
            print(f"  {label_name}: already complete ({already})")
            continue

        for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc=label_name)):
            feats = extract(row["code"], DEFAULT_MODEL)
            feats["label"]     = label
            feats["generator"] = row.get("generator", "unknown")
            rows.append(feats)

            if (i + 1) % SAVE_EVERY == 0:
                pd.DataFrame(rows).to_parquet(CHECKPOINT, index=False)

    results = pd.DataFrame(rows)
    results.to_parquet(OUT, index=False)
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
    print(f"\nSaved -> {OUT}")

    # Statistical analysis
    print("\n" + "=" * 65)
    print("PERPLEXITY SIGNAL ANALYSIS")
    print("=" * 65)

    human_r = results[results["label"] == 0]
    ai_r    = results[results["label"] == 1]

    perp_features = [c for c in results.columns if c.startswith("perp_") and c != "perp_token_count"]

    print(f"\n{'Feature':<28} {'Human mean':>12} {'AI mean':>12} {'p-value':>10} {'Signal':>8}")
    print("-" * 75)

    signals = []
    for feat in perp_features:
        hv = human_r[feat].dropna()
        av = ai_r[feat].dropna()
        _, p = stats.mannwhitneyu(hv, av, alternative="two-sided")
        direction = "H>AI" if hv.mean() > av.mean() else "AI>H"
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"{feat:<28} {hv.mean():>12.4f} {av.mean():>12.4f} {p:>10.4f} {sig:>4} {direction:>4}")
        signals.append((feat, p, hv.mean() - av.mean()))

    significant = [(f, p, d) for f, p, d in signals if p < 0.05]
    print(f"\n{len(significant)}/{len(signals)} features statistically significant (p<0.05)")

    if significant:
        print("\nStrongest signals:")
        for feat, p, delta in sorted(significant, key=lambda x: abs(x[2]), reverse=True)[:5]:
            direction = "Human higher" if delta > 0 else "AI higher"
            print(f"  {feat:<30} delta={delta:+.4f}  p={p:.4e}  ({direction})")

if __name__ == "__main__":
    main()
