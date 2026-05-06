"""
Generate labeled AI code samples via local Ollama models.
Outputs one JSON file per sample into corpus/ai/ollama/
"""
import json
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
DEST = Path(__file__).parent.parent / "corpus" / "ai" / "ollama"

MODELS = [
    "codellama",
    "deepseek-coder",
    "llama3",
]

PROMPTS = [
    ("python", "Write a Python function that reads a CSV file and returns a list of dicts."),
    ("python", "Write a Python class that implements a simple LRU cache."),
    ("python", "Write a Python function that recursively flattens a nested list."),
    ("javascript", "Write a JavaScript function that debounces another function."),
    ("javascript", "Write a JavaScript class that implements an event emitter."),
    ("java", "Write a Java method that performs binary search on a sorted int array."),
    ("java", "Write a Java class that implements a stack using a linked list."),
]


def generate(model: str, prompt: str) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        print(f"\nModel: {model}")
        for language, prompt in PROMPTS:
            print(f"  [{language}] {prompt[:60]}...")
            try:
                code = generate(model, prompt)
                sample = {
                    "id": str(uuid.uuid4()),
                    "label": "ai",
                    "language": language,
                    "model": model,
                    "prompt": prompt,
                    "code": code,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
                out = DEST / f"{model.replace(':', '_')}__{sample['id'][:8]}.json"
                out.write_text(json.dumps(sample, indent=2))
                print(f"  saved -> {out.name}")
            except Exception as e:
                print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
