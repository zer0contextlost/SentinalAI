"""
Pull OpenAI HumanEval dataset from HuggingFace.
Saves each problem's canonical (human-written) solution as a JSON file.

Output: corpus/humaneval/human/
"""
import json
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Run: pip install datasets")
    sys.exit(1)

HF_DATASET = "openai/openai_humaneval"
OUT_DIR = Path(__file__).parent.parent / "corpus" / "humaneval" / "human"


def safe_task_id(task_id: str) -> str:
    return task_id.replace("/", "__")


def main():
    print(f"Pulling {HF_DATASET}...")
    ds = load_dataset(HF_DATASET, split="test")
    print(f"  {len(ds)} problems")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    for row in ds:
        task_id = row["task_id"]           # e.g. "HumanEval/0"
        prompt = row["prompt"]             # function signature + docstring
        solution = row["canonical_solution"]
        entry_point = row["entry_point"]

        # Full runnable function = prompt + solution
        full_code = prompt + solution

        record = {
            "task_id":    task_id,
            "label":      "human",
            "language":   "python",
            "source":     "humaneval_canonical",
            "entry_point": entry_point,
            "code":       full_code,
            "prompt":     prompt,
        }

        out = OUT_DIR / f"{safe_task_id(task_id)}.json"
        out.write_text(json.dumps(record, indent=2))
        saved += 1

    print(f"  Saved {saved} human solutions -> {OUT_DIR}")


if __name__ == "__main__":
    main()
