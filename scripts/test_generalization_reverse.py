"""
Reverse generalization test: train on SemEval (34 AI models, all languages),
evaluate on paired Codeforces corpus (deepseek-coder only, Python).

This tests whether a model trained on diverse AI outputs generalizes
to clean ground-truth paired data where era/domain are controlled.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    import warnings
    warnings.filterwarnings("ignore")
except ImportError:
    print("Run: pip install scikit-learn pandas numpy")
    sys.exit(1)

SEMEVAL_PATH = Path(__file__).parent.parent / "corpus" / "processed" / "features.parquet"
PAIRED_PATH  = Path(__file__).parent.parent / "corpus" / "processed" / "paired_features.parquet"
DROP_COLS    = {"label", "problem_id", "source_file", "language", "generator"}


def load_semeval(feature_cols=None):
    df = pd.read_parquet(SEMEVAL_PATH)

    # Python only — matches the paired corpus language
    py = df[df["language"].str.lower() == "python"]
    print(f"  SemEval Python rows: {len(py):,}  (of {len(df):,} total)")

    if feature_cols is None:
        feature_cols = [c for c in py.columns if c not in DROP_COLS]

    # Balance classes for training
    human = py[py["label"] == 0]
    ai    = py[py["label"] == 1]
    n = min(len(human), len(ai), 50_000)
    balanced = pd.concat([
        human.sample(n, random_state=42),
        ai.sample(n, random_state=42),
    ])

    X = balanced[feature_cols].fillna(0).astype(float)
    y = balanced["label"].astype(int)
    print(f"  Training on {len(balanced):,} balanced rows ({n:,} per class)")
    if "generator" in balanced.columns:
        n_gen = balanced[balanced["label"] == 1]["generator"].nunique()
        print(f"  AI generators in train: {n_gen}")
    return X, y, feature_cols


def load_paired(feature_cols):
    df = pd.read_parquet(PAIRED_PATH)
    # Use only features present in both datasets
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  Missing in paired (zero-fill): {len(missing)}")
        for col in missing:
            df[col] = 0.0

    X = df[feature_cols].fillna(0).astype(float)
    y = df["label"].astype(int)
    groups = df["problem_id"]
    print(f"  Paired CF test: {len(df)} rows, {groups.nunique()} problems")
    print(f"  Label balance: {y.value_counts().to_dict()}")
    return X, y, groups


def main():
    print("=" * 65)
    print("REVERSE GENERALIZATION TEST")
    print("Train: SemEval 34-model diverse corpus (Python only)")
    print("Test:  Paired Codeforces (same problem, human vs AI)")
    print("=" * 65)

    print("\nLoading SemEval training data...")
    X_train, y_train, feature_cols = load_semeval()

    print("\nLoading paired Codeforces test data...")
    X_test, y_test, groups = load_paired(feature_cols)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n[{name}]")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1  = f1_score(y_test, preds, average="macro")
        acc = accuracy_score(y_test, preds)
        print(f"  F1 macro:  {f1:.4f}")
        print(f"  Accuracy:  {acc:.4f}")
        print(classification_report(y_test, preds, target_names=["human", "ai"], digits=3))
        results[name] = (f1, model)

    best_name = max(results, key=lambda k: results[k][0])
    best_f1, best_model = results[best_name]
    print(f"\nBest: {best_name} ({best_f1:.4f} F1)")

    # Feature importance from RF
    rf_model = results["Random Forest"][1]
    importances = rf_model.feature_importances_
    imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    print("\n[Top features driving generalization]")
    print(f"  {'Feature':<35} {'Importance':>10}")
    print("  " + "-" * 47)
    for feat, score in imp[:15]:
        bar = "#" * int(score * 300)
        print(f"  {feat:<35} {score:>10.4f}  {bar}")


if __name__ == "__main__":
    main()
