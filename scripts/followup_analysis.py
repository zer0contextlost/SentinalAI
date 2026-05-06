"""
Follow-up analyses for Opus review:

1. Per-feature overlap table: for top-10 discriminative features,
   show where each corpus's human and AI distributions sit relative
   to each other. Does HumanEval-human land in SemEval-AI territory
   across most features?

2. Train-on-union experiment: train on SemEval + CF + HumanEval combined
   (stratified so HumanEval isn't overwhelmed), test via held-out splits.
   Does a diverse human baseline recover high F1?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.feature_selection import f_classif
    from scipy.stats import mannwhitneyu
    import warnings
    warnings.filterwarnings("ignore")
except ImportError:
    print("Run: pip install scikit-learn pandas numpy scipy")
    sys.exit(1)

SEMEVAL_PATH = Path(__file__).parent.parent / "corpus" / "processed" / "features.parquet"
PAIRED_PATH  = Path(__file__).parent.parent / "corpus" / "processed" / "paired_features.parquet"
HE_PATH      = Path(__file__).parent.parent / "corpus" / "processed" / "humaneval_features.parquet"
OUT_DIR      = Path(__file__).parent.parent / "models"
DROP_COLS    = {"label", "language", "generator", "problem_id", "task_id", "source_file"}


def load_all():
    sem = pd.read_parquet(SEMEVAL_PATH)
    sem_py = sem[sem["language"].str.lower() == "python"]
    cf = pd.read_parquet(PAIRED_PATH)
    he = pd.read_parquet(HE_PATH)

    feat_cols = [c for c in sem_py.columns if c not in DROP_COLS]
    feat_cols = [c for c in feat_cols if c in cf.columns and c in he.columns]
    return sem_py, cf, he, feat_cols


def top_features_by_anova(sem, feat_cols, n=10):
    """ANOVA on SemEval to find most discriminative features."""
    X = sem[feat_cols].fillna(0).astype(float)
    y = sem["label"].astype(int)
    f_stats, _ = f_classif(X, y)
    ranked = sorted(zip(feat_cols, f_stats), key=lambda x: -x[1])
    return [f for f, _ in ranked[:n]]


def overlap_table(sem, cf, he, top_feats):
    """
    For each feature, show mean ± std for each corpus×class combo.
    Key question: does HumanEval-human overlap with SemEval-AI?
    """
    sem_h = sem[sem["label"] == 0]
    sem_a = sem[sem["label"] == 1]
    cf_h  = cf[cf["label"] == 0]
    cf_a  = cf[cf["label"] == 1]
    he_h  = he[he["label"] == 0]
    he_a  = he[he["label"] == 1]

    print(f"\n{'Feature':<28} {'SemEval-H':>11} {'SemEval-AI':>11} {'CF-H':>11} {'CF-AI':>11} {'HE-H':>11} {'HE-AI':>11}  Overlap?")
    print("  " + "-" * 108)

    rows = []
    for feat in top_feats:
        sh_m = sem_h[feat].mean(); sh_s = sem_h[feat].std()
        sa_m = sem_a[feat].mean(); sa_s = sem_a[feat].std()
        ch_m = cf_h[feat].mean();  ch_s = cf_h[feat].std()
        ca_m = cf_a[feat].mean();  ca_s = cf_a[feat].std()
        hh_m = he_h[feat].mean();  hh_s = he_h[feat].std()
        ha_m = he_a[feat].mean();  ha_s = he_a[feat].std()

        # Does HumanEval-human sit within SemEval-AI range (mean +/- 1 std)?
        in_ai_zone = abs(hh_m - sa_m) < sa_s
        overlap_flag = "YES ***" if in_ai_zone else "no"

        print(f"  {feat:<28} {sh_m:>8.2f}    {sa_m:>8.2f}    {ch_m:>8.2f}    {ca_m:>8.2f}    {hh_m:>8.2f}    {ha_m:>8.2f}  {overlap_flag}")
        rows.append({
            "feature": feat,
            "sem_human_mean": sh_m, "sem_ai_mean": sa_m,
            "cf_human_mean": ch_m,  "cf_ai_mean": ca_m,
            "he_human_mean": hh_m,  "he_ai_mean": ha_m,
            "he_human_in_sem_ai_zone": in_ai_zone,
        })

    return pd.DataFrame(rows)


def union_experiment(sem, cf, he, feat_cols):
    """
    Train on SemEval + CF + HumanEval combined.
    Use stratified K-fold that keeps corpus proportions.
    Tests whether a diverse human baseline recovers F1.
    """
    # Balance each corpus independently before combining
    def balance(df, n=1000):
        h = df[df["label"] == 0].sample(min(n, (df["label"] == 0).sum()), random_state=42)
        a = df[df["label"] == 1].sample(min(n, (df["label"] == 1).sum()), random_state=42)
        return pd.concat([h, a])

    # Use equal-ish representation per corpus
    sem_bal = balance(sem, n=5000)   # cap SemEval so it doesn't drown others
    cf_bal  = balance(cf,  n=186)    # all of CF (it's small)
    he_bal  = balance(he,  n=164)    # all of HumanEval

    union = pd.concat([sem_bal, cf_bal, he_bal], ignore_index=True)

    available = [f for f in feat_cols if f in union.columns]
    X = union[available].fillna(0).astype(float)
    y = union["label"].astype(int)

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(rf, X, y, cv=skf, n_jobs=-1)

    f1  = f1_score(y, preds, average="macro")
    acc = accuracy_score(y, preds)

    print(f"\n  Union size: {len(union):,} ({len(sem_bal)} SemEval + {len(cf_bal)} CF + {len(he_bal)} HumanEval)")
    print(f"  F1 macro: {f1:.4f}   Accuracy: {acc:.4f}")

    # Also test within each corpus slice (subset of held-out preds)
    print(f"\n  Per-corpus breakdown (from union CV predictions):")
    sem_idx = range(0, len(sem_bal))
    cf_idx  = range(len(sem_bal), len(sem_bal) + len(cf_bal))
    he_idx  = range(len(sem_bal) + len(cf_bal), len(union))

    for name, idx in [("SemEval", sem_idx), ("Codeforces", cf_idx), ("HumanEval", he_idx)]:
        yi = y.iloc[list(idx)]
        pi = preds[list(idx)]
        fi = f1_score(yi, pi, average="macro")
        print(f"    {name:<15} F1={fi:.4f}  n={len(yi)}")

    return f1, acc


def main():
    print("=" * 70)
    print("FOLLOW-UP ANALYSES — FEATURE OVERLAP + UNION TRAINING")
    print("=" * 70)

    OUT_DIR.mkdir(exist_ok=True)
    print("\nLoading corpora...")
    sem, cf, he, feat_cols = load_all()
    print(f"  Features: {len(feat_cols)}")

    print("\n[1] Top-10 discriminative features (SemEval ANOVA) — overlap table")
    print("    Key: does HumanEval-human sit in SemEval-AI territory?")
    top_feats = top_features_by_anova(sem, feat_cols, n=10)

    overlap_df = overlap_table(sem, cf, he, top_feats)
    n_overlap = overlap_df["he_human_in_sem_ai_zone"].sum()
    print(f"\n  HumanEval-human in SemEval-AI zone: {n_overlap}/{len(overlap_df)} features")
    overlap_df.to_csv(OUT_DIR / "feature_overlap_table.csv", index=False)
    print(f"  Saved -> models/feature_overlap_table.csv")

    print("\n[2] Union training experiment (SemEval + CF + HumanEval, 5-fold CV)")
    print("    Prediction: diverse human baseline recovers high F1")
    f1_union, acc_union = union_experiment(sem, cf, he, feat_cols)

    print(f"\n[Summary]")
    print(f"  Overlap (HE-human in SemEval-AI zone): {n_overlap}/{len(top_feats)} top features")
    print(f"  Union F1 (diverse training):           {f1_union:.4f}")
    print(f"  Interpretation:")
    if n_overlap >= 6:
        print(f"    -> HumanEval-human broadly overlaps SemEval-AI territory")
        print(f"       Confirms: detector measures human baseline quality, not AI authorship")
    if f1_union > 0.75:
        print(f"    -> Union training recovers high F1: diverse human baseline fixes the problem")
    elif f1_union > 0.60:
        print(f"    -> Union training partially recovers F1: diverse training helps but not fully")
    else:
        print(f"    -> Union training does not recover F1: features cannot generalize even with diverse training")

    print("\nDone.")


if __name__ == "__main__":
    main()
