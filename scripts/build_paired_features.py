"""
Build feature matrix from the paired Codeforces corpus.
Each row is one solution; problem_id links human and AI rows.
Output: corpus/processed/paired_features.parquet
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
except ImportError:
    print("Run: pip install pandas pyarrow")
    sys.exit(1)

from features.extractor import extract_features

HUMAN_DIR = Path(__file__).parent.parent / "corpus" / "paired" / "codeforces" / "human"
AI_DIR    = Path(__file__).parent.parent / "corpus" / "paired" / "codeforces" / "ai"
OUT       = Path(__file__).parent.parent / "corpus" / "processed" / "paired_features.parquet"


def load_dir(directory: Path, label: int) -> list[dict]:
    rows = []
    for f in sorted(directory.glob("*.json")):
        d = json.loads(f.read_text())
        feats = extract_features(d["code"], language="python")
        feats["label"]      = label
        feats["problem_id"] = d["problem_id"]
        feats["source_file"] = f.name
        rows.append(feats)
    return rows


def main():
    print("Building paired feature matrix...")

    print("  Extracting human solution features...")
    human_rows = load_dir(HUMAN_DIR, label=0)
    print(f"    {len(human_rows)} human rows")

    print("  Extracting AI solution features...")
    ai_rows = load_dir(AI_DIR, label=1)
    print(f"    {len(ai_rows)} AI rows")

    df = pd.DataFrame(human_rows + ai_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"\nSaved {len(df)} rows, {len(df.columns)} columns -> {OUT}")
    print(f"Label balance: {df['label'].value_counts().to_dict()}")
    print(f"Paired problems: {df['problem_id'].nunique()}")


if __name__ == "__main__":
    main()
