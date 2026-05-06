"""
Build feature matrix from the HumanEval paired corpus.
Output: corpus/processed/humaneval_features.parquet
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

HUMAN_DIR = Path(__file__).parent.parent / "corpus" / "humaneval" / "human"
AI_DIR    = Path(__file__).parent.parent / "corpus" / "humaneval" / "ai"
OUT       = Path(__file__).parent.parent / "corpus" / "processed" / "humaneval_features.parquet"


def load_dir(directory: Path, label: int) -> list[dict]:
    rows = []
    for f in sorted(directory.glob("*.json")):
        d = json.loads(f.read_text())
        code = d.get("code", "")
        if not code.strip():
            continue
        feats = extract_features(code, language="python")
        feats["label"]    = label
        feats["task_id"]  = d.get("task_id", f.stem)
        feats["source_file"] = f.name
        rows.append(feats)
    return rows


def main():
    print("Building HumanEval feature matrix...")

    print("  Extracting human solution features...")
    human_rows = load_dir(HUMAN_DIR, label=0)
    print(f"    {len(human_rows)} human rows")

    print("  Extracting AI solution features...")
    ai_rows = load_dir(AI_DIR, label=1)
    print(f"    {len(ai_rows)} AI rows")

    if not human_rows and not ai_rows:
        print("No data found. Run pull_humaneval.py and generate_humaneval_ai.py first.")
        sys.exit(1)

    df = pd.DataFrame(human_rows + ai_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"\nSaved {len(df)} rows, {len(df.columns)} columns -> {OUT}")
    print(f"Label balance: {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
