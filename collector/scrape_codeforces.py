"""
Build a paired corpus from Codeforces:
  - Fetch problem statements
  - Collect human solutions (pre-2022, Python)
  - Generate AI solutions to the same problems via Ollama
  - Label both, store with problem_id linking them

Output: corpus/paired/codeforces/
"""
import html
import json
import re
import time
import uuid
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CF_API      = "https://codeforces.com/api"
CF_WEB      = "https://codeforces.com"
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "deepseek-coder:6.7b"
OUT_DIR     = Path(__file__).parent.parent / "corpus" / "paired" / "codeforces"

# Only collect human solutions before ChatGPT launch
HUMAN_BEFORE_EPOCH = 1669766400  # 2022-11-30 00:00 UTC
# Only use old contests (pre-2021) so human solutions definitely exist
MAX_CONTEST_ID     = 1500

TARGET_PROBLEMS       = 100
SOLUTIONS_PER_PROBLEM = 3

# Regex to extract source code from CF submission page
_CODE_RE = re.compile(
    r'<pre[^>]*id=["\']program-source-text["\'][^>]*>(.*?)</pre>',
    re.DOTALL,
)


def cf_get(method: str, **params) -> dict:
    url = f"{CF_API}/{method}?" + "&".join(f"{k}={v}" for k, v in params.items())
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())
    if data["status"] != "OK":
        raise RuntimeError(f"CF API error: {data}")
    time.sleep(0.5)
    return data["result"]


def fetch_submission_source(contest_id: int, submission_id: int) -> str | None:
    url = f"{CF_WEB}/contest/{contest_id}/submission/{submission_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            page = r.read().decode("utf-8", errors="replace")
        m = _CODE_RE.search(page)
        if m:
            return html.unescape(m.group(1)).strip()
    except Exception:
        pass
    return None


def get_problems(count: int = TARGET_PROBLEMS) -> list[dict]:
    print("Fetching problem list (old contests only)...")
    data = cf_get("problemset.problems", tags="implementation")
    problems = [
        p for p in data["problems"]
        if p.get("type") == "PROGRAMMING"
        and 800 <= p.get("rating", 0) <= 1400
        and p.get("contestId", 9999) <= MAX_CONTEST_ID
    ]
    print(f"  {len(problems)} suitable pre-2021 problems found, using {count}")
    return problems[:count]


def get_human_solutions(contest_id: int, problem_index: str, n: int = SOLUTIONS_PER_PROBLEM) -> list[dict]:
    try:
        submissions = cf_get("contest.status", contestId=contest_id, from_=1, count=1000)
    except Exception as e:
        print(f"    SKIP contest {contest_id}: {e}")
        return []

    candidates = [
        s for s in submissions
        if s.get("problem", {}).get("index") == problem_index
        and s.get("verdict") == "OK"
        and s.get("programmingLanguage", "").startswith("Python")
        and s.get("creationTimeSeconds", 0) < HUMAN_BEFORE_EPOCH
    ]

    solutions = []
    for sub in candidates[:n * 3]:  # try up to 3x needed in case scrape fails
        source = fetch_submission_source(contest_id, sub["id"])
        if not source:
            continue
        solutions.append({
            "submission_id": sub["id"],
            "author":        sub.get("author", {}).get("members", [{}])[0].get("handle", "unknown"),
            "language":      sub.get("programmingLanguage"),
            "created_at":    datetime.fromtimestamp(
                sub["creationTimeSeconds"], tz=timezone.utc
            ).isoformat(),
            "code": source,
        })
        time.sleep(0.5)  # polite scraping
        if len(solutions) >= n:
            break
    return solutions


def generate_ai_solution(problem: dict) -> str | None:
    name  = problem.get("name", "")
    tags  = ", ".join(problem.get("tags", []))
    rating = problem.get("rating", "unknown")
    prompt = (
        f"Solve the following competitive programming problem in Python.\n"
        f"Problem: {name}\n"
        f"Difficulty: {rating}\n"
        f"Tags: {tags}\n"
        f"Write a complete, working Python solution that reads from stdin and writes to stdout.\n"
        f"Code only, no explanation:\n"
    )
    try:
        payload = json.dumps({
            "model":   MODEL,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.2, "num_predict": 512},
        }).encode()
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["response"]
    except Exception as e:
        print(f"    AI generation failed: {e}")
        return None


def save(record: dict, subdir: str) -> None:
    dest = OUT_DIR / subdir
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{record['problem_id']}__{record['id']}.json"
    path.write_text(json.dumps(record, indent=2))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problems = get_problems()
    saved_pairs = 0

    # Build set of already-completed problem_ids
    done = {p.stem.split("__")[0] for p in (OUT_DIR / "ai").glob("*.json")} if (OUT_DIR / "ai").exists() else set()
    print(f"Already done: {len(done)} problems")

    for i, problem in enumerate(problems):
        contest_id = problem.get("contestId")
        index      = problem.get("index")
        name       = problem.get("name", "")
        problem_id = f"{contest_id}{index}"

        if problem_id in done:
            print(f"  skip {problem_id} (already done)")
            saved_pairs += 1
            continue

        print(f"\n[{i+1}/{len(problems)}] {problem_id}: {name}")

        # Generate AI solution
        ai_code = generate_ai_solution(problem)
        if not ai_code:
            continue

        ai_record = {
            "id":         str(uuid.uuid4())[:8],
            "problem_id": problem_id,
            "label":      "ai",
            "language":   "python",
            "model":      MODEL,
            "code":       ai_code,
            "problem":    {"name": name, "rating": problem.get("rating"), "tags": problem.get("tags", [])},
        }
        save(ai_record, "ai")
        print(f"  AI solution saved ({len(ai_code)} chars)")

        # Fetch human solutions
        humans = get_human_solutions(contest_id, index)
        for sol in humans:
            if sol["code"] is None:
                continue  # source unavailable via status API
            human_record = {
                "id":         str(uuid.uuid4())[:8],
                "problem_id": problem_id,
                "label":      "human",
                "language":   "python",
                "author":     sol["author"],
                "created_at": sol["created_at"],
                "code":       sol["code"],
                "problem":    {"name": name, "rating": problem.get("rating"), "tags": problem.get("tags", [])},
            }
            save(human_record, "human")

        saved_pairs += 1
        print(f"  Pair saved (total: {saved_pairs})")

    print(f"\nDone. {saved_pairs} AI solutions generated.")
    print(f"Note: Human solution source code requires separate download via CF submission API.")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
