"""
Generate AI solutions for HumanEval problems using deepseek-coder via Ollama.
Reads human solutions from corpus/humaneval/human/ to get the problem prompts.
Output: corpus/humaneval/ai/

Requires Ollama running at localhost:11434 with deepseek-coder:6.7b pulled.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "deepseek-coder:6.7b"
HUMAN_DIR  = Path(__file__).parent.parent / "corpus" / "humaneval" / "human"
OUT_DIR    = Path(__file__).parent.parent / "corpus" / "humaneval" / "ai"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Complete the following Python function. "
    "Return ONLY the function implementation — no explanation, no markdown fences, "
    "no imports unless the function needs them. "
    "Start your response with the first line of the function body."
)


def ollama_generate(prompt: str) -> str:
    payload = json.dumps({
        "model":  MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"].strip()


def already_done(task_id: str) -> bool:
    safe = task_id.replace("/", "__")
    return (OUT_DIR / f"{safe}.json").exists()


def main():
    problems = sorted(HUMAN_DIR.glob("*.json"))
    print(f"HumanEval problems: {len(problems)}")
    print(f"Model: {MODEL}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    errors = 0

    for i, f in enumerate(problems):
        rec = json.loads(f.read_text())
        task_id = rec["task_id"]

        if already_done(task_id):
            print(f"  [{i+1}/{len(problems)}] {task_id}: skip (exists)")
            saved += 1
            continue

        print(f"  [{i+1}/{len(problems)}] {task_id}...", end=" ", flush=True)
        try:
            completion = ollama_generate(rec["prompt"])

            # Reconstruct full function: prompt + generated body
            full_code = rec["prompt"] + completion

            out_rec = {
                "task_id":    task_id,
                "label":      "ai",
                "language":   "python",
                "source":     "ollama",
                "model":      MODEL,
                "entry_point": rec["entry_point"],
                "code":       full_code,
                "prompt":     rec["prompt"],
            }
            safe = task_id.replace("/", "__")
            (OUT_DIR / f"{safe}.json").write_text(json.dumps(out_rec, indent=2))
            saved += 1
            print("ok")
        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")

        time.sleep(0.1)

    print(f"\nDone. Saved: {saved}  Errors: {errors}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
