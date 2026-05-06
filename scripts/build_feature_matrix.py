"""
Run feature extractor across the full corpus and save a labeled feature matrix.
Output: corpus/processed/features.parquet
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    from tqdm import tqdm
except ImportError:
    print("Missing deps. Run: pip install pandas pyarrow tqdm")
    sys.exit(1)

from features.extractor import extract_features

CORPUS = Path(__file__).parent.parent / "corpus"
OUT    = CORPUS / "processed" / "features.parquet"

SOURCES = [
    (CORPUS / "human" / "semeval", 0),
    (CORPUS / "ai"    / "semeval", 1),
]


def load_shards(directory: Path) -> pd.DataFrame | None:
    shards = sorted(directory.glob("*.parquet"))
    if not shards:
        return None
    return pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)


def process_batch(df: pd.DataFrame, label: int) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"label={label}", leave=False):
        code = row.get("code", "")
        lang = row.get("language", "python")
        if not isinstance(code, str) or not code.strip():
            continue
        feats = extract_features(code, lang)
        feats["label"]     = label
        feats["language"]  = lang
        feats["generator"] = row.get("generator", "unknown")
        rows.append(feats)
    return pd.DataFrame(rows)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    t0 = time.time()

    for directory, label in SOURCES:
        print(f"\nLoading {'human' if label == 0 else 'ai'} shards from {directory.name}/...")
        df = load_shards(directory)
        if df is None:
            print(f"  No shards found, skipping.")
            continue
        print(f"  {len(df):,} rows loaded.")
        result = process_batch(df, label)
        frames.append(result)
        print(f"  {len(result):,} rows processed.")

    if not frames:
        print("No data found.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(OUT, index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Feature matrix : {OUT}")
    print(f"Shape          : {combined.shape}")
    print(f"Label balance  : {combined['label'].value_counts().to_dict()}")
    print(f"Languages      : {combined['language'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
