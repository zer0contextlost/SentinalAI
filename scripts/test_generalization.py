"""
Generalization test: train on paired Codeforces corpus (deepseek-coder only),
evaluate on SemEval corpus (34 different AI models).

This tests whether the signal is model-specific or generalizes to AI authorship broadly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    import warnings
    warnings.filterwarnings("ignore")
except ImportError:
    print("Run: pip install scikit-learn pandas numpy")
    sys.exit(1)

PAIRED_PATH  = Path(__file__).parent.parent / "corpus" / "processed" / "paired_features.parquet"
SEMEVAL_PATH = Path(__file__).parent.parent / "corpus" / "processed" / "features.parquet"
DROP_COLS    = {"label", "problem_id", "source_file", "language", "generator"}


def load_paired():
    df = pd.read_parquet(PAIRED_PATH)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].fillna(0).astype(float)
    y = df["label"].astype(int)
    print(f"  Train (paired CF): {len(df)} rows, {len(feature_cols)} features")
    print(f"  Label balance: {y.value_counts().to_dict()}")
    return X, y, feature_cols


def load_semeval(feature_cols, sample_n=5000):
    df = pd.read_parquet(SEMEVAL_PATH)
    # Match features — SemEval matrix was built before defensive features were added
    # Use intersection of available columns
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  Missing features in SemEval (will zero-fill): {len(missing)}")
        for col in missing:
            df[col] = 0.0

    # Sample evenly across generators for a fair test
    human = df[df["label"] == 0]
    ai    = df[df["label"] == 1]
    n_each = min(sample_n // 2, len(human), len(ai))
    sampled = pd.concat([
        human.sample(n_each, random_state=42),
        ai.sample(n_each, random_state=42),
    ])

    X = sampled[feature_cols].fillna(0).astype(float)
    y = sampled["label"].astype(int)

    print(f"  Test  (SemEval):   {len(sampled)} rows sampled")
    print(f"  Label balance: {y.value_counts().to_dict()}")
    if "generator" in sampled.columns:
        gen_counts = sampled[sampled["label"] == 1]["generator"].value_counts()
        print(f"  AI generators in test: {gen_counts.nunique()} models")
        print(f"  Top 5: {dict(list(gen_counts.items())[:5])}")
    return X, y


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n[{name}]")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    f1  = f1_score(y_test, preds, average="macro")
    acc = accuracy_score(y_test, preds)
    print(f"  F1 macro:  {f1:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["human", "ai"], digits=3))
    return model, f1


def per_generator_breakdown(model, X_test_df, y_test, semeval_sample, feature_cols):
    if "generator" not in semeval_sample.columns:
        return
    print("\n[Per-generator accuracy on AI samples]")
    print(f"  {'Generator':<35} {'N':>5} {'Accuracy':>10}")
    print("  " + "-" * 55)
    ai_mask = y_test == 1
    for gen in sorted(semeval_sample[semeval_sample["label"] == 1]["generator"].unique()):
        gen_mask = (semeval_sample["generator"] == gen).values & ai_mask.values
        if gen_mask.sum() == 0:
            continue
        X_gen = X_test_df[gen_mask]
        y_gen = y_test.values[gen_mask]
        preds = model.predict(X_gen)
        acc = accuracy_score(y_gen, preds)
        n = len(y_gen)
        bar = "#" * int(acc * 20)
        print(f"  {gen:<35} {n:>5} {acc:>10.3f}  {bar}")


def main():
    print("=" * 65)
    print("GENERALIZATION TEST")
    print("Train: paired Codeforces (deepseek-coder:6.7b only)")
    print("Test:  SemEval (34 AI models)")
    print("=" * 65)

    print("\nLoading training data...")
    X_train, y_train, feature_cols = load_paired()

    print("\nLoading test data...")
    # Load full SemEval for per-generator breakdown
    semeval_full = pd.read_parquet(SEMEVAL_PATH)
    available = [c for c in feature_cols if c in semeval_full.columns]
    missing   = [c for c in feature_cols if c not in semeval_full.columns]
    for col in missing:
        semeval_full[col] = 0.0

    human = semeval_full[semeval_full["label"] == 0]
    ai    = semeval_full[semeval_full["label"] == 1]
    n_each = min(2500, len(human), len(ai))
    sampled = pd.concat([
        human.sample(n_each, random_state=42),
        ai.sample(n_each, random_state=42),
    ])
    X_test = sampled[feature_cols].fillna(0).astype(float)
    y_test = sampled["label"].astype(int)

    print(f"  Test rows: {len(sampled)}")
    print(f"  Missing features zero-filled: {len(missing)}")
    if "generator" in sampled.columns:
        n_gen = sampled[sampled["label"] == 1]["generator"].nunique()
        print(f"  AI generators in test sample: {n_gen}")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }

    best_model, best_f1, best_name = None, 0, ""
    for name, model in models.items():
        m, f1 = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        if f1 > best_f1:
            best_model, best_f1, best_name = m, f1, name

    print(f"\nBest: {best_name} ({best_f1:.4f} F1)")

    # Per-generator breakdown with best model
    per_generator_breakdown(best_model, X_test, y_test, sampled, feature_cols)


if __name__ == "__main__":
    main()
