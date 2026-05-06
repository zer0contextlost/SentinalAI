"""
Train classifier on the paired Codeforces corpus.
Uses leave-one-problem-out CV to avoid data leakage between paired solutions.
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

FEATURES_PATH = Path(__file__).parent.parent / "corpus" / "processed" / "paired_features.parquet"
DROP_COLS     = {"label", "problem_id", "source_file", "language", "generator"}


def load_data():
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].fillna(0).astype(float)
    y = df["label"].astype(int)
    groups = df["problem_id"]
    print(f"  {len(df)} rows, {len(feature_cols)} features")
    print(f"  Label balance: {y.value_counts().to_dict()}")
    print(f"  Problems: {groups.nunique()}")
    return X, y, groups, feature_cols


def leave_one_problem_out_cv(model_factory, X, y, groups):
    """CV where each fold holds out all solutions for one problem."""
    problem_ids = groups.unique()
    preds, trues = [], []

    for pid in problem_ids:
        test_mask  = groups == pid
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        model = model_factory()
        model.fit(X[train_mask], y[train_mask])
        p = model.predict(X[test_mask])
        preds.extend(p)
        trues.extend(y[test_mask])

    return np.array(trues), np.array(preds)


def evaluate(name, model_factory, X, y, groups):
    print(f"\n[{name}]")
    trues, preds = leave_one_problem_out_cv(model_factory, X, y, groups)
    f1  = f1_score(trues, preds, average="macro")
    acc = accuracy_score(trues, preds)
    print(f"  F1 macro  (LOPO-CV): {f1:.4f}")
    print(f"  Accuracy  (LOPO-CV): {acc:.4f}")
    print(classification_report(trues, preds, target_names=["human", "ai"], digits=3))
    return f1


def feature_importance(X, y, feature_cols):
    print("\n[Feature Importance — Random Forest, full dataset]")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1])

    print(f"\n  {'Feature':<35} {'Importance':>10}")
    print("  " + "-" * 47)
    for feat, score in imp[:20]:
        bar = "#" * int(score * 300)
        print(f"  {feat:<35} {score:>10.4f}  {bar}")

    # Save
    out = Path(__file__).parent.parent / "models" / "paired_feature_importance.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(imp, columns=["feature", "importance"]).to_csv(out, index=False)
    print(f"\n  Saved -> {out}")


def main():
    print("Loading paired feature matrix...")
    X, y, groups, feature_cols = load_data()

    models = {
        "Logistic Regression": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42
        ),
    }

    results = {}
    for name, factory in models.items():
        results[name] = evaluate(name, factory, X, y, groups)

    best = max(results, key=results.get)
    print(f"\nBest model: {best} ({results[best]:.4f} F1 LOPO-CV)")

    feature_importance(X, y, feature_cols)


if __name__ == "__main__":
    main()
