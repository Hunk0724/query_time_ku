"""Shared Langfuse + OpenAI setup. Imported by all instrumented scripts."""
from __future__ import annotations
import os
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
# 1. Load .env (manual loader — no python-dotenv dep)
# ────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
env_path = ROOT / ".env"
for line in env_path.read_text().splitlines():
    if line.strip() and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# ────────────────────────────────────────────────────────────────────────────
# 2. Langfuse client (used for trace metadata + scoring)
# ────────────────────────────────────────────────────────────────────────────
from langfuse import Langfuse, observe, get_client  # noqa: E402

# Initialize the SDK singleton (auto-reads LANGFUSE_* env vars).
# Larger flush_at / flush_interval = fewer batches but they're more reliable.
_lf = Langfuse(
    flush_at=200,         # flush when 200 events queued (default 15)
    flush_interval=2.0,   # or every 2 sec (default 0.5)
    tracing_enabled=True,
)

# Warm up the OTEL exporter so the FIRST batch doesn't race the first @observe call
_lf.auth_check()

# ────────────────────────────────────────────────────────────────────────────
# 3. Auto-instrumented OpenAI client (drop-in replacement)
#
#    Every client.chat.completions.create() call becomes a Langfuse
#    GENERATION span automatically. No further code changes needed.
# ────────────────────────────────────────────────────────────────────────────
from langfuse.openai import OpenAI  # noqa: E402

__all__ = ["OpenAI", "observe", "get_client", "_lf", "ROOT"]
