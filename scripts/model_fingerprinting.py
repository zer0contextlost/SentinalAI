"""
Pivot #1: Per-model stylometric fingerprinting.
Trains a 35-way classifier (34 AI models + human) and produces:
  - Per-model F1 scores
  - Pairwise confusion heatmap (which models are indistinguishable?)
  - Hierarchical clustering dendrogram of model families
  - Top features that discriminate between models (ANOVA F-statistic)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.feature_selection import f_classif
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    import warnings
    warnings.filterwarnings("ignore")
except ImportError:
    print("Run: pip install scikit-learn pandas numpy scipy")
    sys.exit(1)

FEATURES_PATH = Path(__file__).parent.parent / "corpus" / "processed" / "features.parquet"
OUT_DIR       = Path(__file__).parent.parent / "models"
DROP_COLS     = {"label", "language", "generator"}

SAMPLE_PER_CLASS = 1000  # balance classes; 1K per model = ~35K total


def load_data():
    df = pd.read_parquet(FEATURES_PATH)
    py = df[df["language"].str.lower() == "python"].copy()
    py["class"] = py["generator"].fillna("human")

    feature_cols = [c for c in py.columns if c not in DROP_COLS and c != "class"]

    # Balance — sample up to SAMPLE_PER_CLASS per class
    frames = []
    for cls, grp in py.groupby("class"):
        n = min(len(grp), SAMPLE_PER_CLASS)
        frames.append(grp.sample(n, random_state=42))
    balanced = pd.concat(frames, ignore_index=True)

    X = balanced[feature_cols].fillna(0).astype(float)
    y = balanced["class"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"  Classes: {len(le.classes_)}")
    print(f"  Rows:    {len(balanced):,}")
    print(f"  Features: {len(feature_cols)}")
    return X, y_enc, y, le, feature_cols, balanced


def train_and_evaluate(X, y_enc, le):
    print("\nTraining 35-way RF classifier (5-fold CV)...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, max_depth=20)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(rf, X, y_enc, cv=skf, n_jobs=-1)

    report = classification_report(y_enc, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).T
    print("\n[Per-model F1 scores]")
    model_rows = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    model_rows = model_rows.sort_values("f1-score", ascending=False)
    print(f"\n  {'Class':<45} {'F1':>6} {'Precision':>10} {'Recall':>8} {'N':>6}")
    print("  " + "-" * 80)
    for cls, row in model_rows.iterrows():
        print(f"  {cls:<45} {row['f1-score']:>6.3f} {row['precision']:>10.3f} {row['recall']:>8.3f} {int(row['support']):>6}")

    overall_f1 = report["macro avg"]["f1-score"]
    print(f"\n  Macro F1: {overall_f1:.4f}  (chance = {1/len(le.classes_):.4f})")
    return y_pred, report_df, overall_f1


def build_confusion_analysis(y_enc, y_pred, le):
    print("\n[Confusion analysis — most commonly confused pairs]")
    cm = confusion_matrix(y_enc, y_pred)
    # Normalize by true class size
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    np.fill_diagonal(cm_norm, 0)  # zero out correct predictions for confusion analysis

    # Find top confused pairs
    confused_pairs = []
    n = len(le.classes_)
    for i in range(n):
        for j in range(n):
            if i != j and cm_norm[i, j] > 0.01:
                confused_pairs.append((cm_norm[i, j], le.classes_[i], le.classes_[j]))
    confused_pairs.sort(reverse=True)

    print(f"\n  {'True class':<40} {'Predicted as':<40} {'Rate':>6}")
    print("  " + "-" * 90)
    for rate, true_cls, pred_cls in confused_pairs[:20]:
        print(f"  {true_cls:<40} {pred_cls:<40} {rate:>6.3f}")

    # Save full confusion matrix
    cm_df = pd.DataFrame(cm_norm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(OUT_DIR / "confusion_matrix_normalized.csv")
    print(f"\n  Full confusion matrix saved -> models/confusion_matrix_normalized.csv")
    return cm_norm, cm_df


def cluster_models(cm_norm, le):
    print("\n[Hierarchical clustering of model families]")
    # Convert confusion to distance: models confused with each other are closer
    # Symmetrize: distance = 1 - max(confusion in either direction)
    n = len(le.classes_)
    dist = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i, j] = 0
            else:
                dist[i, j] = 1 - max(cm_norm[i, j], cm_norm[j, i])

    condensed = squareform(dist)
    Z = linkage(condensed, method="average")

    # Print text dendrogram
    print("\n  Model family tree (by stylistic similarity):")
    from scipy.cluster.hierarchy import to_tree
    tree = to_tree(Z)

    def print_tree(node, indent=0, labels=le.classes_):
        prefix = "  " + "  " * indent
        if node.is_leaf():
            print(f"{prefix}-- {labels[node.id]}")
        else:
            print(f"{prefix}+")
            print_tree(node.left, indent + 1, labels)
            print_tree(node.right, indent + 1, labels)

    print_tree(tree)

    # Save linkage for external plotting
    Z_df = pd.DataFrame(Z, columns=["idx1", "idx2", "distance", "count"])
    Z_df.to_csv(OUT_DIR / "model_clustering_linkage.csv", index=False)
    pd.Series(le.classes_).to_csv(OUT_DIR / "model_clustering_labels.csv", index=False)
    print(f"\n  Linkage matrix saved -> models/model_clustering_linkage.csv")
    return Z


def feature_discrimination(X, y_enc, feature_cols, le):
    print("\n[Top features discriminating between models — ANOVA F-statistic]")
    f_stats, p_vals = f_classif(X, y_enc)
    feat_importance = sorted(zip(feature_cols, f_stats, p_vals), key=lambda x: -x[1])

    print(f"\n  {'Feature':<35} {'F-stat':>10} {'p-value':>12} {'Sig':>5}")
    print("  " + "-" * 67)
    for feat, f, p in feat_importance[:20]:
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {feat:<35} {f:>10.1f} {p:>12.2e} {sig:>5}")

    pd.DataFrame(feat_importance, columns=["feature", "f_stat", "p_value"]).to_csv(
        OUT_DIR / "model_discrimination_features.csv", index=False
    )
    print(f"\n  Full list saved -> models/model_discrimination_features.csv")


def main():
    print("=" * 65)
    print("MODEL FINGERPRINTING — 35-WAY CLASSIFICATION")
    print("=" * 65)

    OUT_DIR.mkdir(exist_ok=True)
    print("\nLoading data...")
    X, y_enc, y_raw, le, feature_cols, balanced = load_data()

    y_pred, report_df, macro_f1 = train_and_evaluate(X, y_enc, le)
    cm_norm, cm_df = build_confusion_analysis(y_enc, y_pred, le)
    cluster_models(cm_norm, le)
    feature_discrimination(X, y_enc, feature_cols, le)

    print(f"\nDone. Macro F1 across 35 classes: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
