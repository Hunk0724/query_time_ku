"""Env-gated LLM/embedding cost+latency logger.

Appends one JSONL line per model call to the file named by env MEM0_COST_LOG.
When the env var is unset every function is a no-op (byte-identical behavior),
so instrumentation never affects results or fairness -- it only records.

Each line: {"stage", "model", "prompt_tokens", "completion_tokens", "latency_s"}.
Aggregate per-stage tokens / calls / time / est$ with summarize_cost.py.
"""
import json
import os
import time

# gpt-4o-mini + text-embedding-3-small USD per 1M tokens (update if pricing changes).
PRICE = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "text-embedding-3-small": {"in": 0.02, "out": 0.0},
}


def _path():
    return os.environ.get("MEM0_COST_LOG")


def log(stage, model, prompt_tokens=0, completion_tokens=0, latency_s=0.0):
    """Record one call. No-op when MEM0_COST_LOG is unset."""
    p = _path()
    if not p:
        return
    try:
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "stage": stage,
                "model": model,
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "latency_s": round(float(latency_s or 0.0), 4),
            }) + "\n")
    except Exception:
        pass  # logging must never break a run


def timed(stage, model, fn, *, usage_of=None):
    """Run fn(), time it, extract usage via usage_of(result), log, return result.

    usage_of(result) -> (prompt_tokens, completion_tokens) or None. When
    MEM0_COST_LOG is unset, fn() still runs but timing/usage extraction is
    skipped (zero overhead)."""
    if not _path():
        return fn()
    t0 = time.time()
    result = fn()
    dt = time.time() - t0
    pt = ct = 0
    if usage_of is not None:
        try:
            u = usage_of(result)
            if u:
                pt, ct = u
        except Exception:
            pass
    log(stage, model, pt, ct, dt)
    return result
