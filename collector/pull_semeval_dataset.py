"""
Pull SemEval-2026 Task 13 dataset from HuggingFace and save as Parquet shards.

Dataset: DaniilOr/SemEval-2026-Task13
  - Subtask A: binary classification (label: 0=human, 1=ai)
  - Subtask B: model attribution (label: 0=human, 1-10=specific model)
  - Subtask C: mixed-source / human-edited AI code

Outputs to corpus/ai/semeval/ and corpus/human/semeval/
"""
import argparse
import sys
from pathlib import Path

try:
    from datasets import load_dataset
    import pandas as pd
except ImportError:
    print("Missing dependencies. Run: pip install datasets pandas pyarrow")
    sys.exit(1)

HF_DATASET = "DaniilOr/SemEval-2026-Task13"
ROOT = Path(__file__).parent.parent / "corpus"

SUBTASKS = ["A", "B", "C"]
SPLITS = ["train", "validation", "test"]

# Subtask A label mapping
SUBTASK_A_LABELS = {0: "human", 1: "ai"}

# Subtask B model attribution labels (from competition docs)
SUBTASK_B_LABELS = {
    0:  "human",
    1:  "deepseek",
    2:  "qwen",
    3:  "yi",
    4:  "starcoder",
    5:  "gemma",
    6:  "phi",
    7:  "llama",
    8:  "granite",
    9:  "mistral",
    10: "gpt",
}

SHARD_SIZE = 50_000


def save_parquet(df: pd.DataFrame, dest: Path, subtask: str, split: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    total = len(df)
    shards = max(1, total // SHARD_SIZE)
    for i, chunk in enumerate(
        [df.iloc[j * SHARD_SIZE:(j + 1) * SHARD_SIZE] for j in range(shards)]
        + ([df.iloc[shards * SHARD_SIZE:]] if total % SHARD_SIZE else [])
    ):
        if len(chunk) == 0:
            continue
        path = dest / f"{subtask}__{split}__shard{i:03d}.parquet"
        chunk.to_parquet(path, index=False)
        print(f"    saved {len(chunk):,} rows -> {path.name}")


def pull_subtask(subtask: str, splits: list[str]) -> None:
    print(f"\n[{subtask}]")
    for split in splits:
        print(f"  loading {split}...")
        try:
            ds = load_dataset(HF_DATASET, subtask, split=split)
        except Exception as e:
            print(f"  SKIP {split}: {e}")
            continue

        df = ds.to_pandas()
        print(f"  {len(df):,} rows, columns: {list(df.columns)}")

        # Normalize label column name
        label_col = next((c for c in df.columns if "label" in c.lower()), None)
        if label_col and label_col != "label":
            df = df.rename(columns={label_col: "label"})

        # Add human-readable label
        if subtask == "A" and "label" in df.columns:
            df["label_str"] = df["label"].map(SUBTASK_A_LABELS)
        elif subtask == "B" and "label" in df.columns:
            df["label_str"] = df["label"].map(SUBTASK_B_LABELS)

        # Route to corpus/human or corpus/ai based on label
        if subtask == "A" and "label" in df.columns:
            human_df = df[df["label"] == 0].copy()
            ai_df = df[df["label"] == 1].copy()
            if len(human_df):
                save_parquet(human_df, ROOT / "human" / "semeval", subtask, split)
            if len(ai_df):
                save_parquet(ai_df, ROOT / "ai" / "semeval", subtask, split)
        else:
            # Subtask B/C: save everything to ai/semeval (contains both, label distinguishes)
            save_parquet(df, ROOT / "ai" / "semeval", subtask, split)


def main():
    parser = argparse.ArgumentParser(description="Pull SemEval-2026 Task 13 from HuggingFace")
    parser.add_argument(
        "--subtasks", nargs="+", default=SUBTASKS,
        choices=["A", "B", "C"], help="Which subtasks to pull (default: all)"
    )
    parser.add_argument(
        "--splits", nargs="+", default=SPLITS,
        choices=SPLITS, help="Which splits to pull (default: all)"
    )
    args = parser.parse_args()

    print(f"Pulling {HF_DATASET}")
    print(f"Subtasks : {args.subtasks}")
    print(f"Splits   : {args.splits}")

    for subtask in args.subtasks:
        pull_subtask(subtask, args.splits)

    print("\nDone.")
    print(f"Human corpus -> {ROOT / 'human' / 'semeval'}")
    print(f"AI corpus    -> {ROOT / 'ai' / 'semeval'}")


if __name__ == "__main__":
    main()
