"""
Token-level perplexity features via Ollama.

Ollama returns logprobs for GENERATED tokens only (not prompt tokens).
Strategy: use the first 60% of the code as context, score the model's
continuation logprobs. AI code context produces more confident continuations
because the model strongly recognizes AI-style patterns.

Signal:
  LOW perplexity  (high confidence continuations) -> likely AI context
  HIGH perplexity (surprising continuations)      -> likely human context
  LOW variance    (uniform confidence)            -> likely AI context

Requires Ollama running at localhost:11434 with a code model loaded.
Recommended: deepseek-coder:6.7b (fits in 12GB VRAM, code-specific priors)
"""
import json
import math
import urllib.request
from typing import Any

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "deepseek-coder:6.7b"

# Use first 800 chars as context, generate 60 tokens to score
CONTEXT_CHARS = 800
NUM_PREDICT   = 60


def extract(code: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    code = code.strip()
    if not code:
        return _empty()

    context = code[:CONTEXT_CHARS]

    try:
        logprobs = _get_continuation_logprobs(context, model)
    except Exception:
        return _empty()

    lps = [e["logprob"] for e in logprobs if isinstance(e, dict) and e.get("logprob") is not None]
    if not lps:
        return _empty()

    n          = len(lps)
    mean_lp    = sum(lps) / n
    perplexity = math.exp(-mean_lp) if mean_lp < 0 else 1.0
    variance   = sum((lp - mean_lp) ** 2 for lp in lps) / n
    min_lp     = min(lps)
    max_lp     = max(lps)

    # Tokens model found highly surprising (log prob < -3, ~5% probability)
    surprising = sum(1 for lp in lps if lp < -3.0) / n
    # Tokens model predicted confidently (log prob > -0.5, ~60% probability)
    confident  = sum(1 for lp in lps if lp > -0.5) / n

    return {
        "perp_mean_logprob":     mean_lp,
        "perp_perplexity":       perplexity,
        "perp_variance":         variance,
        "perp_min_logprob":      min_lp,
        "perp_max_logprob":      max_lp,
        "perp_range":            max_lp - min_lp,
        "perp_surprising_ratio": surprising,
        "perp_confident_ratio":  confident,
        "perp_token_count":      n,
    }


def _get_continuation_logprobs(context: str, model: str) -> list:
    payload = json.dumps({
        "model":    model,
        "prompt":   context,
        "stream":   False,
        "logprobs": True,
        "options":  {"temperature": 0, "num_predict": NUM_PREDICT},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    return data.get("logprobs") or []


def _empty() -> dict[str, Any]:
    return {
        "perp_mean_logprob":     0.0,
        "perp_perplexity":       0.0,
        "perp_variance":         0.0,
        "perp_min_logprob":      0.0,
        "perp_max_logprob":      0.0,
        "perp_range":            0.0,
        "perp_surprising_ratio": 0.0,
        "perp_confident_ratio":  0.0,
        "perp_token_count":      0,
    }


def is_available(model: str = DEFAULT_MODEL) -> bool:
    try:
        payload = json.dumps({"model": model, "prompt": "x", "stream": False, "options": {"num_predict": 1}}).encode()
        req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False
