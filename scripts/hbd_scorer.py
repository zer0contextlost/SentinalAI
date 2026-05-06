"""
HBD (Human Baseline Distance) scorer.

Computes a continuous diagnostic score: how far a code sample sits from
the union human distribution on the 9 stable features. Higher score =
more anomalous relative to diverse human code.

This is NOT a binary AI detector. It measures distributional distance
from a known-diverse human baseline. Anomalous != AI-generated.

Reference: "The Human Baseline Problem" (SentinalAI research)

Stable features (directionally consistent across SemEval, CF, HumanEval):
  avg_line_length, max_line_length, p90_line_length, digit_ratio,
  snake_case_count, long_ident_count, naming_style_ratio,
  early_return_count, trailing_ws_ratio
"""
import sys
import json
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    from sklearn.covariance import LedoitWolf
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import warnings
    warnings.filterwarnings("ignore")
except ImportError:
    print("Run: pip install scikit-learn pandas numpy scipy")
    sys.exit(1)

SEMEVAL_PATH = Path(__file__).parent.parent / "corpus" / "processed" / "features.parquet"
PAIRED_PATH  = Path(__file__).parent.parent / "corpus" / "processed" / "paired_features.parquet"
HE_PATH      = Path(__file__).parent.parent / "corpus" / "processed" / "humaneval_features.parquet"
OUT_DIR      = Path(__file__).parent.parent / "models"
MODEL_PATH   = OUT_DIR / "hbd_scorer.pkl"

STABLE_FEATURES = [
    "avg_line_length",
    "max_line_length",
    "p90_line_length",
    "digit_ratio",
    "snake_case_count",
    "long_ident_count",
    "naming_style_ratio",
    "early_return_count",
    "trailing_ws_ratio",
]

DROP_COLS = {"label", "language", "generator", "problem_id", "task_id", "source_file"}


class HBDScorer:
    """
    Human Baseline Distance scorer.
    Fit on human samples from diverse corpora. Score any code sample.
    """

    def __init__(self, features=STABLE_FEATURES):
        self.features = features
        self.scaler = StandardScaler()
        self.cov_estimator = LedoitWolf()
        self.mean_ = None
        self.precision_ = None   # inverse covariance (Mahalanobis)
        self.condition_number_ = None

    def fit(self, X_human: pd.DataFrame):
        X = X_human[self.features].fillna(0).astype(float)
        # Standardize on union humans
        X_scaled = self.scaler.fit_transform(X)
        # Ledoit-Wolf shrinkage covariance
        self.cov_estimator.fit(X_scaled)
        cov = self.cov_estimator.covariance_
        self.precision_ = self.cov_estimator.precision_
        self.mean_ = np.zeros(len(self.features))  # after standardization, mean ~ 0
        eigvals = np.linalg.eigvalsh(cov)
        self.condition_number_ = eigvals.max() / eigvals.min()
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns Mahalanobis distance from union human distribution (lower = more human-like)."""
        Xs = self.scaler.transform(X[self.features].fillna(0).astype(float))
        diff = Xs - self.mean_
        # Mahalanobis: sqrt(diff @ precision @ diff.T) per sample
        scores = np.sqrt(np.einsum("ij,jk,ik->i", diff, self.precision_, diff))
        return scores

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "HBDScorer":
        with open(path, "rb") as f:
            return pickle.load(f)


def load_corpora():
    sem = pd.read_parquet(SEMEVAL_PATH)
    sem_py = sem[sem["language"].str.lower() == "python"]
    cf  = pd.read_parquet(PAIRED_PATH)
    he  = pd.read_parquet(HE_PATH)
    return sem_py, cf, he


def build_union_human(sem, cf, he):
    """Build diverse human baseline: sample from all three corpora."""
    sh = sem[sem["label"] == 0].sample(5000, random_state=42)
    ch = cf[cf["label"] == 0]
    hh = he[he["label"] == 0]
    return pd.concat([sh, ch, hh], ignore_index=True)


def evaluate_scorer(scorer, sem, cf, he):
    """
    Report HBD score distributions per corpus×class and AUROC.
    Honest check: are per-corpus human scores well-separated?
    """
    corpora = [
        ("SemEval",    sem[sem["label"]==0],   sem[sem["label"]==1]),
        ("Codeforces", cf[cf["label"]==0],     cf[cf["label"]==1]),
        ("HumanEval",  he[he["label"]==0],     he[he["label"]==1]),
    ]

    print(f"\n  {'Corpus':<14} {'Class':<8} {'Mean HBD':>10} {'Median':>8} {'Std':>8}")
    print("  " + "-" * 52)

    all_scores = []
    all_labels = []
    per_corpus_results = {}

    for name, hdf, adf in corpora:
        h_scores = scorer.score(hdf)
        a_scores = scorer.score(adf)

        print(f"  {name:<14} {'human':<8} {h_scores.mean():>10.3f} {np.median(h_scores):>8.3f} {h_scores.std():>8.3f}")
        print(f"  {name:<14} {'ai':<8} {a_scores.mean():>10.3f} {np.median(a_scores):>8.3f} {a_scores.std():>8.3f}")

        scores = np.concatenate([h_scores, a_scores])
        labels = np.concatenate([np.zeros(len(h_scores)), np.ones(len(a_scores))])
        auroc = roc_auc_score(labels, scores)
        per_corpus_results[name] = auroc
        print(f"  {name:<14} {'AUROC':<8} {auroc:>10.3f}")
        print()

        all_scores.append(scores)
        all_labels.append(labels)

    # Overall AUROC across all three
    overall_scores = np.concatenate(all_scores)
    overall_labels = np.concatenate(all_labels)
    overall_auroc = roc_auc_score(overall_labels, overall_scores)
    per_corpus_results["Overall"] = overall_auroc

    return per_corpus_results


def main():
    print("=" * 65)
    print("HBD SCORER — HUMAN BASELINE DISTANCE")
    print("=" * 65)

    OUT_DIR.mkdir(exist_ok=True)

    print("\nLoading corpora...")
    sem, cf, he = load_corpora()
    print(f"  SemEval Python: {len(sem):,}  CF: {len(cf)}  HumanEval: {len(he)}")

    print(f"\nFitting HBD scorer on union human baseline...")
    union_human = build_union_human(sem, cf, he)
    print(f"  Union human samples: {len(union_human):,}")

    scorer = HBDScorer()
    scorer.fit(union_human)

    print(f"  Covariance condition number: {scorer.condition_number_:.1f}")
    if scorer.condition_number_ > 100:
        print(f"  WARNING: high condition number — some features may be near-collinear")
    else:
        print(f"  (well-conditioned — Ledoit-Wolf shrinkage working)")

    print(f"\nFeature means after standardization (should be ~0):")
    X_sample = union_human[STABLE_FEATURES].fillna(0).astype(float)
    X_scaled = scorer.scaler.transform(X_sample)
    for feat, mean in zip(STABLE_FEATURES, X_scaled.mean(axis=0)):
        print(f"  {feat:<30} {mean:>+.4f}")

    print(f"\nEvaluating scorer — HBD distributions and AUROC per corpus:")
    print(f"  (AUROC: how well HBD separates human from AI within each corpus)")
    auroc_results = evaluate_scorer(scorer, sem, cf, he)

    # Save the scorer
    scorer.save(MODEL_PATH)
    print(f"Scorer saved -> {MODEL_PATH}")

    # Save per-corpus results
    results_df = pd.DataFrame([
        {"corpus": k, "auroc": v} for k, v in auroc_results.items()
    ])
    results_df.to_csv(OUT_DIR / "hbd_auroc.csv", index=False)
    print(f"AUROC results saved -> models/hbd_auroc.csv")

    # Demo: score a few samples
    print(f"\n[Demo] HBD scores on example samples from each corpus:")
    for name, df in [("SemEval-human", sem[sem["label"]==0].head(3)),
                     ("SemEval-AI",    sem[sem["label"]==1].head(3)),
                     ("CF-human",      cf[cf["label"]==0].head(3)),
                     ("CF-AI",         cf[cf["label"]==1].head(3)),
                     ("HumanEval-human", he[he["label"]==0].head(3)),
                     ("HumanEval-AI",    he[he["label"]==1].head(3))]:
        scores = scorer.score(df)
        print(f"  {name:<20} {scores}")

    print(f"\nDone.")
    print(f"  Use HBDScorer.load('{MODEL_PATH}') to score new code samples.")
    print(f"  Input: DataFrame with columns {STABLE_FEATURES}")
    print(f"  Output: Mahalanobis distance from union human baseline (no threshold — diagnostic only)")


if __name__ == "__main__":
    main()
