"""
Third-corpus validation: do stable features truly generalize?

Protocol (avoids leakage):
  1. Identify stable features using ONLY SemEval + Codeforces (same logic as feature_forensics.py)
  2. Train on SemEval, test on HumanEval — all features vs stable-only
  3. Train on Codeforces, test on HumanEval — all features vs stable-only
  4. Train on SemEval+CF combined, test on HumanEval — all vs stable-only

HumanEval was never seen during feature selection, so this is a clean OOD test.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score
    from scipy.stats import wasserstein_distance, mannwhitneyu
    import warnings
    warnings.filterwarnings("ignore")
except ImportError:
    print("Run: pip install scikit-learn pandas numpy scipy")
    sys.exit(1)

SEMEVAL_PATH  = Path(__file__).parent.parent / "corpus" / "processed" / "features.parquet"
PAIRED_PATH   = Path(__file__).parent.parent / "corpus" / "processed" / "paired_features.parquet"
HE_PATH       = Path(__file__).parent.parent / "corpus" / "processed" / "humaneval_features.parquet"
OUT_DIR       = Path(__file__).parent.parent / "models"
DROP_COLS     = {"label", "language", "generator", "problem_id", "task_id", "source_file"}


def load_corpora():
    sem = pd.read_parquet(SEMEVAL_PATH)
    sem_py = sem[sem["language"].str.lower() == "python"]
    paired = pd.read_parquet(PAIRED_PATH)
    he = pd.read_parquet(HE_PATH)

    # Shared feature columns across all three
    feat_cols = [c for c in sem_py.columns if c not in DROP_COLS]
    feat_cols = [c for c in feat_cols if c in paired.columns and c in he.columns]

    print(f"  SemEval Python rows:  {len(sem_py):,}")
    print(f"  Paired CF rows:       {len(paired):,}")
    print(f"  HumanEval rows:       {len(he):,}")
    print(f"  Shared features:      {len(feat_cols)}")
    return sem_py, paired, he, feat_cols


def identify_stable_features(sem, paired, feature_cols):
    """
    Identify stable features using ONLY SemEval and Codeforces.
    HumanEval is never touched here — no leakage.
    """
    sem_human = sem[sem["label"] == 0]
    sem_ai    = sem[sem["label"] == 1]
    cf_human  = paired[paired["label"] == 0]
    cf_ai     = paired[paired["label"] == 1]

    stable = []
    conflict = []

    for feat in feature_cols:
        sh = sem_human[feat].dropna()
        sa = sem_ai[feat].dropna()
        ch = cf_human[feat].dropna()
        ca = cf_ai[feat].dropna()

        if len(sh) < 10 or len(sa) < 10 or len(ch) < 10 or len(ca) < 10:
            continue

        _, p_sem = mannwhitneyu(sh, sa, alternative="two-sided")
        _, p_cf  = mannwhitneyu(ch, ca, alternative="two-sided")

        if p_sem >= 0.05 or p_cf >= 0.05:
            continue

        dir_sem = 1 if sa.mean() > sh.mean() else -1
        dir_cf  = 1 if ca.mean() > ch.mean() else -1

        if dir_sem == dir_cf:
            stable.append(feat)
        else:
            conflict.append(feat)

    return stable, conflict


def evaluate(train_df, test_df, feature_cols, label_name):
    available = [f for f in feature_cols if f in train_df.columns and f in test_df.columns]

    h = train_df[train_df["label"] == 0].sample(
        min(50000, (train_df["label"] == 0).sum()), random_state=42
    )
    a = train_df[train_df["label"] == 1].sample(
        min(50000, (train_df["label"] == 1).sum()), random_state=42
    )
    train = pd.concat([h, a])

    X_train = train[available].fillna(0).astype(float)
    y_train = train["label"].astype(int)
    X_test  = test_df[available].fillna(0).astype(float)
    y_test  = test_df["label"].astype(int)

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    return {
        "label":    label_name,
        "n_feats":  len(available),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "accuracy": accuracy_score(y_test, preds),
    }


def main():
    print("=" * 70)
    print("THIRD-CORPUS VALIDATION — HumanEval OOD TEST")
    print("=" * 70)

    OUT_DIR.mkdir(exist_ok=True)

    print("\nLoading corpora...")
    sem, paired, he, feature_cols = load_corpora()

    print("\nIdentifying stable features from SemEval + CF (HumanEval unseen)...")
    stable, conflict = identify_stable_features(sem, paired, feature_cols)
    print(f"  Stable features:   {len(stable)}")
    print(f"  Conflict features: {len(conflict)}")
    print(f"  Stable set: {stable}")

    # Combined SemEval + CF training set
    sem_for_combine = sem[["label"] + [c for c in feature_cols if c in sem.columns]].copy()
    cf_for_combine  = paired[["label"] + [c for c in feature_cols if c in paired.columns]].copy()
    combined = pd.concat([sem_for_combine, cf_for_combine], ignore_index=True)

    experiments = []

    print("\nRunning cross-domain evaluations -> HumanEval test set...")
    for train_name, train_df in [
        ("SemEval",        sem),
        ("Codeforces",     paired),
        ("SemEval+CF",     combined),
    ]:
        for feat_name, feats in [
            ("All features",           feature_cols),
            (f"Stable only ({len(stable)})", stable),
        ]:
            result = evaluate(train_df, he, feats, f"{train_name} | {feat_name}")
            experiments.append(result)
            print(f"  {result['label']:<45} F1={result['f1_macro']:.4f}  Acc={result['accuracy']:.4f}  n={result['n_feats']}")

    print(f"\n[Summary Table]")
    print(f"\n  {'Train set':<20} {'Features':<30} {'F1 macro':>10} {'Accuracy':>10}")
    print("  " + "-" * 74)
    for r in experiments:
        parts = r["label"].split(" | ")
        train = parts[0]
        feats = parts[1] if len(parts) > 1 else ""
        print(f"  {train:<20} {feats:<30} {r['f1_macro']:>10.4f} {r['accuracy']:>10.4f}")

    print(f"\n  Baseline random chance F1 ~= 0.50")

    # Save results
    results_df = pd.DataFrame(experiments)
    results_df.to_csv(OUT_DIR / "third_corpus_validation.csv", index=False)
    print(f"\n  Saved -> models/third_corpus_validation.csv")

    # Key finding
    all_sem  = next(r for r in experiments if r["label"] == "SemEval | All features")
    stab_sem = next(r for r in experiments if "SemEval" in r["label"] and "Stable" in r["label"])
    lift = stab_sem["f1_macro"] - all_sem["f1_macro"]
    print(f"\n  Stable-feature lift (SemEval train): {lift:+.4f}")
    print(f"  (SemEval+CF same direction shift as seen in SemEval/CF alone?)")

    print("\nDone.")


if __name__ == "__main__":
    main()
