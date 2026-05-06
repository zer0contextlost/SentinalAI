"""
Pivot #3: Feature forensics — why does cross-dataset generalization fail?

For each of the 65 features, measures:
  - Distribution shift between SemEval and Codeforces (for both AI and human)
  - Features that are important but point in OPPOSITE directions across corpora
  - Features that are stable (shift same direction in both corpora)
  - Retrains a classifier using only stable features, measures cross-domain F1
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

SEMEVAL_PATH = Path(__file__).parent.parent / "corpus" / "processed" / "features.parquet"
PAIRED_PATH  = Path(__file__).parent.parent / "corpus" / "processed" / "paired_features.parquet"
OUT_DIR      = Path(__file__).parent.parent / "models"
DROP_COLS    = {"label", "language", "generator", "problem_id", "source_file"}


def load_both():
    sem = pd.read_parquet(SEMEVAL_PATH)
    sem_py = sem[sem["language"].str.lower() == "python"]
    paired = pd.read_parquet(PAIRED_PATH)

    feature_cols = [c for c in sem_py.columns if c not in DROP_COLS]
    # Use only features present in both
    feature_cols = [c for c in feature_cols if c in paired.columns]

    print(f"  SemEval Python: {len(sem_py):,} rows")
    print(f"  Paired CF:      {len(paired):,} rows")
    print(f"  Shared features: {len(feature_cols)}")
    return sem_py, paired, feature_cols


def compute_direction(human_vals, ai_vals):
    """Returns +1 if AI > human, -1 if human > AI."""
    return 1 if ai_vals.mean() > human_vals.mean() else -1


def feature_shift_analysis(sem, paired, feature_cols):
    """
    For each feature, compute:
      - Direction in SemEval (AI > human or human > AI)
      - Direction in Paired CF
      - Whether directions agree (stable) or disagree (unstable)
      - Wasserstein distance of AI distribution between the two corpora
      - Mann-Whitney p-value in each corpus
    """
    sem_human = sem[sem["label"] == 0]
    sem_ai    = sem[sem["label"] == 1]
    cf_human  = paired[paired["label"] == 0]
    cf_ai     = paired[paired["label"] == 1]

    rows = []
    for feat in feature_cols:
        sh = sem_human[feat].dropna()
        sa = sem_ai[feat].dropna()
        ch = cf_human[feat].dropna()
        ca = cf_ai[feat].dropna()

        if len(sh) < 10 or len(sa) < 10 or len(ch) < 10 or len(ca) < 10:
            continue

        dir_sem    = compute_direction(sh, sa)
        dir_cf     = compute_direction(ch, ca)
        stable     = dir_sem == dir_cf

        # How much does the AI distribution shift between corpora?
        ai_shift   = wasserstein_distance(
            sa.sample(min(len(sa), 2000), random_state=42),
            ca.sample(min(len(ca), 2000), random_state=42)
        )
        # How much does human distribution shift?
        human_shift = wasserstein_distance(
            sh.sample(min(len(sh), 2000), random_state=42),
            ch.sample(min(len(ch), 2000), random_state=42)
        )

        _, p_sem = mannwhitneyu(sh, sa, alternative="two-sided")
        _, p_cf  = mannwhitneyu(ch, ca, alternative="two-sided")

        rows.append({
            "feature":      feat,
            "dir_semeval":  dir_sem,
            "dir_cf":       dir_cf,
            "stable":       stable,
            "ai_shift_W":   ai_shift,
            "human_shift_W": human_shift,
            "p_semeval":    p_sem,
            "p_cf":         p_cf,
            "sem_human_mean": sh.mean(),
            "sem_ai_mean":    sa.mean(),
            "cf_human_mean":  ch.mean(),
            "cf_ai_mean":     ca.mean(),
        })

    return pd.DataFrame(rows)


def print_shift_report(df):
    stable   = df[df["stable"]]
    unstable = df[~df["stable"]]

    sig_both  = df[(df["p_semeval"] < 0.05) & (df["p_cf"] < 0.05)]
    conflict  = sig_both[~sig_both["stable"]]
    agree     = sig_both[sig_both["stable"]]

    print(f"\n  Total features analyzed: {len(df)}")
    print(f"  Stable (same direction both corpora): {len(stable)}")
    print(f"  Unstable (flip direction):            {len(unstable)}")
    print(f"  Significant in BOTH corpora:          {len(sig_both)}")
    print(f"    -> agree on direction:               {len(agree)}")
    print(f"    -> CONFLICT (flip direction):        {len(conflict)}")

    print(f"\n[Features significant in both corpora but pointing OPPOSITE directions]")
    print(f"  These are the failure-mode features — they encode dataset-specific artifacts")
    print(f"\n  {'Feature':<35} {'SemEval (H->AI)':>16} {'CF (H->AI)':>12} {'AI shift W':>12}")
    print("  " + "-" * 80)
    for _, r in conflict.sort_values("ai_shift_W", ascending=False).iterrows():
        sem_delta = r["sem_ai_mean"] - r["sem_human_mean"]
        cf_delta  = r["cf_ai_mean"]  - r["cf_human_mean"]
        print(f"  {r['feature']:<35} {sem_delta:>+16.4f} {cf_delta:>+12.4f} {r['ai_shift_W']:>12.4f}")

    print(f"\n[Features stable across both corpora]")
    print(f"  These are candidates for a generalizing detector")
    print(f"\n  {'Feature':<35} {'SemEval (H->AI)':>16} {'CF (H->AI)':>12} {'p_sem':>10} {'p_cf':>10}")
    print("  " + "-" * 85)
    for _, r in agree.sort_values("p_cf").iterrows():
        sem_delta = r["sem_ai_mean"] - r["sem_human_mean"]
        cf_delta  = r["cf_ai_mean"]  - r["cf_human_mean"]
        print(f"  {r['feature']:<35} {sem_delta:>+16.4f} {cf_delta:>+12.4f} {r['p_semeval']:>10.4f} {r['p_cf']:>10.4f}")

    return conflict, agree


def stable_feature_classifier(sem, paired, stable_features, all_features):
    """Train on SemEval, test on paired — first with all features, then stable only."""
    if len(stable_features) == 0:
        print("\n  No stable features found — skipping retrain")
        return

    cf_for_stable = [f for f in stable_features if f in paired.columns]

    results = {}
    for label, feats in [("All 65 features", all_features), (f"Stable only ({len(cf_for_stable)} features)", cf_for_stable)]:
        available = [f for f in feats if f in sem.columns and f in paired.columns]

        # Balance SemEval training set
        h = sem[sem["label"] == 0].sample(min(50000, (sem["label"]==0).sum()), random_state=42)
        a = sem[sem["label"] == 1].sample(min(50000, (sem["label"]==1).sum()), random_state=42)
        train = pd.concat([h, a])
        X_train = train[available].fillna(0).astype(float)
        y_train = train["label"].astype(int)

        X_test = paired[available].fillna(0).astype(float)
        y_test = paired["label"].astype(int)

        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        f1  = f1_score(y_test, preds, average="macro")
        acc = accuracy_score(y_test, preds)
        results[label] = (f1, acc)

    print(f"\n[Cross-domain F1: SemEval-trained -> Paired CF test]")
    print(f"\n  {'Feature set':<40} {'F1 macro':>10} {'Accuracy':>10}")
    print("  " + "-" * 62)
    for label, (f1, acc) in results.items():
        print(f"  {label:<40} {f1:>10.4f} {acc:>10.4f}")
    print(f"\n  (Baseline random chance F1 ~= 0.50)")


def main():
    print("=" * 65)
    print("FEATURE FORENSICS — WHY CROSS-DOMAIN GENERALIZATION FAILS")
    print("=" * 65)

    OUT_DIR.mkdir(exist_ok=True)
    print("\nLoading corpora...")
    sem, paired, feature_cols = load_both()

    print("\nComputing feature distribution shifts...")
    shift_df = feature_shift_analysis(sem, paired, feature_cols)
    shift_df.to_csv(OUT_DIR / "feature_shift_analysis.csv", index=False)
    print(f"  Saved -> models/feature_shift_analysis.csv")

    conflict, agree = print_shift_report(shift_df)

    stable_feats = agree["feature"].tolist()
    stable_feature_classifier(sem, paired, stable_feats, feature_cols)

    print(f"\nDone.")
    print(f"  Conflicting features (failure mode): {len(conflict)}")
    print(f"  Stable features (robust candidates): {len(agree)}")


if __name__ == "__main__":
    main()
