"""
Inspect the SentinalAI corpus — distribution across labels, languages, generators.
Reads all Parquet shards from corpus/human/semeval/ and corpus/ai/semeval/
"""
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Missing dependency. Run: pip install pandas pyarrow")
    sys.exit(1)

CORPUS = Path(__file__).parent.parent / "corpus"


def load_shards(directory: Path) -> pd.DataFrame | None:
    parquets = sorted(directory.glob("*.parquet"))
    if not parquets:
        return None
    return pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)


def divider(title: str) -> None:
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print(f"{'-' * 50}")


def show_counts(series: pd.Series, label: str, total: int) -> None:
    counts = series.value_counts().sort_values(ascending=False)
    print(f"\n  {label}:")
    for val, count in counts.items():
        bar = "#" * int(count / total * 40)
        print(f"    {str(val):<22} {count:>7,}  {count/total*100:5.1f}%  {bar}")


def inspect(df: pd.DataFrame, name: str) -> None:
    divider(f"{name}  ({len(df):,} rows)")

    total = len(df)
    print(f"\n  Columns : {list(df.columns)}")
    print(f"  Rows    : {total:,}")

    # Code length stats
    df["_code_len"] = df["code"].str.len()
    print(f"\n  Code length (chars):")
    print(f"    min    {df['_code_len'].min():>8,}")
    print(f"    median {int(df['_code_len'].median()):>8,}")
    print(f"    mean   {int(df['_code_len'].mean()):>8,}")
    print(f"    max    {df['_code_len'].max():>8,}")

    if "language" in df.columns:
        show_counts(df["language"], "language", total)

    if "generator" in df.columns:
        show_counts(df["generator"], "generator", total)

    if "label" in df.columns:
        show_counts(df["label"], "label (0=human 1=ai)", total)

    if "label_str" in df.columns:
        show_counts(df["label_str"], "label_str", total)

    df.drop(columns=["_code_len"], inplace=True)


def main():
    print("SentinalAI Corpus Inspection")

    human_dir = CORPUS / "human" / "semeval"
    ai_dir = CORPUS / "ai" / "semeval"

    human_df = load_shards(human_dir)
    ai_df = load_shards(ai_dir)

    if human_df is None and ai_df is None:
        print("No Parquet shards found. Run collector/pull_semeval_dataset.py first.")
        sys.exit(1)

    if human_df is not None:
        inspect(human_df, "HUMAN corpus")

    if ai_df is not None:
        inspect(ai_df, "AI corpus")

    if human_df is not None and ai_df is not None:
        divider("COMBINED SUMMARY")
        combined = pd.concat([human_df, ai_df], ignore_index=True)
        total = len(combined)
        print(f"\n  Total rows : {total:,}")
        print(f"  Human      : {len(human_df):,}  ({len(human_df)/total*100:.1f}%)")
        print(f"  AI         : {len(ai_df):,}  ({len(ai_df)/total*100:.1f}%)")

        if "language" in combined.columns:
            show_counts(combined["language"], "language (combined)", total)


if __name__ == "__main__":
    main()
