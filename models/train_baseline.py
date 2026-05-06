"""
Train baseline classifier on the 47-feature matrix.
Reports accuracy, F1, confusion matrix, and feature importance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    import warnings
    warnings.filterwarnings("ignore")
except ImportError:
    print("Run: pip install scikit-learn pandas numpy")
    sys.exit(1)

FEATURES_PATH = Path(__file__).parent.parent / "corpus" / "processed" / "features.parquet"
OUT_DIR       = Path(__file__).parent

LABEL_COL     = "label"
DROP_COLS     = {"label", "language", "generator"}


def load_data():
    print("Loading feature matrix...")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].fillna(0).astype(float)
    y = df[LABEL_COL].astype(int)

    print(f"  Features : {len(feature_cols)}")
    print(f"  Label balance: {y.value_counts().to_dict()}")
    return X, y, feature_cols


def evaluate(name, pipeline, X, y, cv=5):
    print(f"\n[{name}]")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    print(f"  F1 macro  (CV={cv}): {scores.mean():.4f} +/- {scores.std():.4f}")
    acc = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  Accuracy  (CV={cv}): {acc.mean():.4f} +/- {acc.std():.4f}")
    return scores.mean()


def feature_importance(X, y, feature_cols):
    print("\n[Feature Importance — Random Forest]")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1])
    print(f"\n  {'Feature':<35} {'Importance':>10}")
    print("  " + "-" * 47)
    for feat, score in imp[:20]:
        bar = "#" * int(score * 400)
        print(f"  {feat:<35} {score:>10.4f}  {bar}")

    # Save full importance list
    imp_df = pd.DataFrame(imp, columns=["feature", "importance"])
    imp_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print(f"\n  Full list saved -> models/feature_importance.csv")
    return rf


def main():
    X, y, feature_cols = load_data()

    # Sample for speed during CV (use full set for final)
    sample_size = min(100_000, len(X))
    idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
    Xs, ys = X.iloc[idx], y.iloc[idx]
    print(f"\nUsing {sample_size:,} sample for CV evaluation")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        results[name] = evaluate(name, model, Xs, ys)

    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name} ({results[best_name]:.4f} F1)")

    # Feature importance on full dataset
    feature_importance(X, y, feature_cols)

    print("\nDone.")


if __name__ == "__main__":
    main()
