"""
Fetch human Python solutions to Codeforces problems from GitHub.

For each problem ID in our AI corpus (e.g. "1300B"), searches GitHub for
Python files with matching names in repos last pushed before 2022-01-01.
This gives provably pre-AI human solutions to the exact same problems.

Requirements: GITHUB_TOKEN env var or set TOKEN below.
Output: corpus/paired/codeforces/human/
"""
import json
import os
import time
import uuid
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone

GITHUB_API   = "https://api.github.com"
TOKEN        = os.environ.get("GITHUB_TOKEN", "")
AI_DIR       = Path(__file__).parent.parent / "corpus" / "paired" / "codeforces" / "ai"
OUT_DIR      = Path(__file__).parent.parent / "corpus" / "paired" / "codeforces" / "human"

SOLUTIONS_PER_PROBLEM = 3
# Repos pushed after this date are excluded (post-ChatGPT era)
CUTOFF_DATE  = "2022-01-01"

MIN_CODE_LEN = 50    # skip trivially short files
MAX_CODE_LEN = 8000  # skip absurdly long files


def gh_get(path: str, params: dict = None) -> dict | list | None:
    url = f"{GITHUB_API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "SentinalAI-research",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            # Rate limit — wait and retry once
            print("    Rate limit hit, sleeping 60s...")
            time.sleep(60)
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        if e.code == 422:
            return None  # Search index doesn't have this file yet
        raise


def search_python_solutions(problem_id: str) -> list[dict]:
    """Search GitHub for Python files named like '1300B.py' or '1300b.py'."""
    contest_id = "".join(c for c in problem_id if c.isdigit())
    index      = "".join(c for c in problem_id if c.isalpha())

    candidates = []
    for name_variant in [f"{problem_id}.py", f"{problem_id.lower()}.py",
                         f"{contest_id}_{index}.py", f"{contest_id}_{index.lower()}.py"]:
        query = f"filename:{name_variant}"
        result = gh_get("/search/code", {"q": query, "per_page": 20})
        time.sleep(1.5)  # GitHub search: max 30/min authenticated
        if not result or "items" not in result:
            continue
        candidates.extend(result["items"])

    return candidates


def file_last_modified(repo_full_name: str, file_path: str) -> str | None:
    """Return the date (YYYY-MM-DD) of the most recent commit touching this file."""
    commits = gh_get(f"/repos/{repo_full_name}/commits", {"path": file_path, "per_page": 1})
    time.sleep(0.3)
    if not commits or not isinstance(commits, list):
        return None
    return commits[0]["commit"]["committer"]["date"][:10]


def fetch_file_content(item: dict) -> str | None:
    url = item.get("url")  # raw API URL with ?ref=...
    if not url:
        return None
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {TOKEN}",
        "Accept": "application/vnd.github.v3.raw",
        "User-Agent": "SentinalAI-research",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def load_ai_problems() -> list[dict]:
    problems = []
    for f in sorted(AI_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        problems.append({
            "problem_id": d["problem_id"],
            "name":       d["problem"]["name"],
            "rating":     d["problem"].get("rating"),
            "tags":       d["problem"].get("tags", []),
        })
    return problems


def already_done(problem_id: str) -> int:
    if not OUT_DIR.exists():
        return 0
    return len(list(OUT_DIR.glob(f"{problem_id}__*.json")))


def save(record: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{record['problem_id']}__{record['id']}.json"
    path.write_text(json.dumps(record, indent=2))


def main():
    problems = load_ai_problems()
    print(f"Problems to fetch human solutions for: {len(problems)}")
    print(f"Cutoff: repos pushed before {CUTOFF_DATE}\n")

    total_saved = 0

    for i, prob in enumerate(problems):
        pid = prob["problem_id"]
        have = already_done(pid)
        if have >= SOLUTIONS_PER_PROBLEM:
            print(f"  [{i+1}/{len(problems)}] {pid}: already have {have}, skip")
            total_saved += have
            continue

        need = SOLUTIONS_PER_PROBLEM - have
        print(f"[{i+1}/{len(problems)}] {pid}: {prob['name']}  (need {need} more)")

        candidates = search_python_solutions(pid)
        print(f"  GitHub hits: {len(candidates)}")

        saved = 0
        seen_repos = set()
        for item in candidates:
            if saved >= need:
                break

            repo  = item.get("repository", {}).get("full_name", "")
            fpath = item.get("path", "")
            key   = f"{repo}/{fpath}"
            if key in seen_repos:
                continue
            seen_repos.add(key)

            # Check file-level commit date (not repo push date)
            last_mod = file_last_modified(repo, fpath)
            if not last_mod or last_mod >= CUTOFF_DATE:
                continue

            code = fetch_file_content(item)
            if not code:
                continue
            code = code.strip()
            if not (MIN_CODE_LEN <= len(code) <= MAX_CODE_LEN):
                continue

            record = {
                "id":         str(uuid.uuid4())[:8],
                "problem_id": pid,
                "label":      "human",
                "language":   "python",
                "source":     "github",
                "repo":       repo,
                "file":       fpath,
                "file_last_modified": last_mod,
                "code":       code,
                "problem":    prob,
            }
            save(record)
            saved += 1
            print(f"  Saved: {repo}/{item.get('path','')}")

        total_saved += saved
        if saved == 0:
            print(f"  No qualifying solutions found")

    print(f"\nDone. Human solutions saved: {total_saved}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
