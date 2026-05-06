"""
Unified feature extractor. Combines lexical + language-specific AST features
into a flat dict ready for ML training.

Usage:
    from features.extractor import extract_features
    feats = extract_features(code="def foo(): pass", language="python")
    # -> dict[str, float]
"""
from typing import Any

from features import lexical, ast_python, perplexity


def extract_features(
    code: str,
    language: str = "python",
    use_perplexity: bool = False,
    perplexity_model: str = perplexity.DEFAULT_MODEL,
) -> dict[str, Any]:
    feats: dict[str, Any] = {}
    feats.update(lexical.extract(code, language))

    if language.lower() == "python":
        feats.update(ast_python.extract(code))

    if use_perplexity:
        feats.update(perplexity.extract(code, perplexity_model))

    return feats


def feature_names(language: str = "python", use_perplexity: bool = False) -> list[str]:
    return list(extract_features("x = 1\n", language, use_perplexity=use_perplexity).keys())


if __name__ == "__main__":
    import json, sys

    sample = sys.stdin.read() if not sys.stdin.isatty() else _DEMO
    lang = sys.argv[1] if len(sys.argv) > 1 else "python"
    feats = extract_features(sample, lang)
    print(json.dumps(feats, indent=2))


_DEMO = '''
import os
from typing import Optional

def read_file(path: str) -> Optional[str]:
    """Read a file and return its contents, or None on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (IOError, OSError) as e:
        print(f"Error reading {path}: {e}")
        return None

if __name__ == "__main__":
    content = read_file("example.txt")
    if content:
        print(content)
'''
