"""
Clone all human corpus repos from repos.json into corpus/human/<owner>__<repo>/
Skips repos already cloned.
"""
import json
import subprocess
import sys
from pathlib import Path

REPOS_FILE = Path(__file__).parent.parent / "corpus" / "human" / "repos.json"
DEST = Path(__file__).parent.parent / "corpus" / "human"


def clone(repo: str) -> None:
    owner, name = repo.split("/")
    dest = DEST / f"{owner}__{name}"
    if dest.exists():
        print(f"  skip {repo} (already cloned)")
        return
    url = f"https://github.com/{repo}.git"
    print(f"  cloning {repo} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"  done -> {dest.name}")


def main():
    repos = json.loads(REPOS_FILE.read_text())
    print(f"Cloning {len(repos)} human repos...")
    for entry in repos:
        clone(entry["repo"])
    print("Complete.")


if __name__ == "__main__":
    main()
