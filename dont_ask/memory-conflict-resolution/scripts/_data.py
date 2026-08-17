"""Shared data-loading + BM25-indexing pipeline.

Fully instrumented with Langfuse. Each step is a @observe span so the entire
setup pipeline shows up as one trace in Langfuse with rich detail per step.

Usage:
    from _data import setup_for_factconsolidation_mh
    data = setup_for_factconsolidation_mh()  # creates a "setup" trace in Langfuse
    # data is a dict with keys: bm25, fact_indices, fact_texts, questions, answers
"""
from __future__ import annotations
import re
import time
from typing import Any

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi

from _lf import observe, get_client


def tokenize(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


# ────────────────────────────────────────────────────────────────────────────
# Individual setup steps — each a Langfuse span
# ────────────────────────────────────────────────────────────────────────────
@observe(name="load_dataset", as_type=None)
def _load_fc_mh_262k_row() -> dict[str, Any]:
    """Load the HF dataset and pick the FC-MH 262K row."""
    t0 = time.time()
    ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision="main")
    row = [s for s in ds if s["metadata"]["source"] == "factconsolidation_mh_262k"][0]
    elapsed_ms = int((time.time() - t0) * 1000)
    ctx_chars = len(row["context"])
    n_questions = len(row["questions"])

    get_client().update_current_span(
        input={
            "dataset": "ai-hyz/MemoryAgentBench",
            "split": "Conflict_Resolution",
            "revision": "main",
            "source_filter": "factconsolidation_mh_262k",
        },
        output={
            "context_chars": ctx_chars,
            "n_questions": n_questions,
            "metadata_keys": sorted(row["metadata"].keys()),
            "elapsed_ms": elapsed_ms,
        },
    )
    return row


@observe(name="parse_facts", as_type=None)
def _parse_facts(context: str) -> list[tuple[int, str]]:
    """Regex-extract numbered facts from the context string.

    Each fact looks like 'N. <text>' separated by ' '. Pattern: find every
    '<num>. ' marker, take the text between consecutive markers."""
    t0 = time.time()
    pat = re.compile(r"(\d+)\.\s")
    matches = list(pat.finditer(context))
    facts: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        s = m.end()
        e = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        text = context[s:e].strip().rstrip(".")
        facts.append((idx, text))
    elapsed_ms = int((time.time() - t0) * 1000)

    get_client().update_current_span(
        input={"regex_pattern": r"(\d+)\.\s", "context_chars": len(context)},
        output={
            "n_raw_matches": len(matches),
            "n_facts_extracted": len(facts),
            "first_3_facts": [{"idx": i, "text": t[:80]} for i, t in facts[:3]],
            "last_3_facts": [{"idx": i, "text": t[:80]} for i, t in facts[-3:]],
            "elapsed_ms": elapsed_ms,
        },
    )
    return facts


@observe(name="dedupe_facts", as_type=None)
def _dedupe_facts(facts: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Keep only the first occurrence of each fact index.

    The regex sometimes matches the same fact boundary multiple times when
    facts overlap in sub-strings; this collapses duplicates."""
    t0 = time.time()
    seen: set[int] = set()
    unique: list[tuple[int, str]] = []
    for idx, text in facts:
        if idx not in seen:
            seen.add(idx); unique.append((idx, text))
    elapsed_ms = int((time.time() - t0) * 1000)

    indices = [i for i, _ in unique]
    get_client().update_current_span(
        input={"n_input_facts": len(facts)},
        output={
            "n_unique_facts": len(unique),
            "n_duplicates_removed": len(facts) - len(unique),
            "index_range": [min(indices), max(indices)] if indices else None,
            "largest_index_gap": max(
                (indices[i+1] - indices[i] for i in range(len(indices)-1)),
                default=0,
            ),
            "elapsed_ms": elapsed_ms,
        },
    )
    return unique


@observe(name="tokenize_corpus", as_type=None)
def _tokenize_corpus(facts: list[tuple[int, str]]) -> list[list[str]]:
    """Tokenize each fact for BM25 (lowercase + alphanumeric split)."""
    t0 = time.time()
    fact_texts = [text for _, text in facts]
    corpus_tokens = [tokenize(t) for t in fact_texts]
    elapsed_ms = int((time.time() - t0) * 1000)

    total_tokens = sum(len(toks) for toks in corpus_tokens)
    unique_tokens = len(set(t for toks in corpus_tokens for t in toks))
    avg_tokens_per_fact = total_tokens / max(len(corpus_tokens), 1)

    get_client().update_current_span(
        input={
            "n_facts": len(facts),
            "tokenizer": "lowercase + re.findall(r'[A-Za-z0-9]+', s)",
        },
        output={
            "total_tokens": total_tokens,
            "unique_tokens_vocab": unique_tokens,
            "avg_tokens_per_fact": round(avg_tokens_per_fact, 2),
            "elapsed_ms": elapsed_ms,
        },
    )
    return corpus_tokens


@observe(name="build_bm25_index", as_type=None)
def _build_bm25(corpus_tokens: list[list[str]]) -> BM25Okapi:
    """Construct the BM25Okapi index."""
    t0 = time.time()
    bm25 = BM25Okapi(corpus_tokens)
    elapsed_ms = int((time.time() - t0) * 1000)
    get_client().update_current_span(
        input={"n_documents": len(corpus_tokens), "implementation": "rank_bm25.BM25Okapi"},
        output={
            "elapsed_ms": elapsed_ms,
            "avg_doc_length": round(sum(len(d) for d in corpus_tokens) / max(len(corpus_tokens), 1), 2),
        },
    )
    return bm25


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator — one Langfuse trace covering the whole setup
# ────────────────────────────────────────────────────────────────────────────
@observe(name="setup")
def setup_for_factconsolidation_mh() -> dict[str, Any]:
    """Run the full setup pipeline. Creates one Langfuse 'setup' trace with 5 spans."""
    get_client().update_current_span(
        metadata={
            "system": "Data setup for FactConsolidation-MH 262K",
            "experiment": "setup",
            "dataset": "factconsolidation_mh_262k",
        },
    )

    row = _load_fc_mh_262k_row()
    raw_facts = _parse_facts(row["context"])
    unique_facts = _dedupe_facts(raw_facts)
    corpus_tokens = _tokenize_corpus(unique_facts)
    bm25 = _build_bm25(corpus_tokens)

    result = {
        "bm25": bm25,
        "fact_indices": [idx for idx, _ in unique_facts],
        "fact_texts": [text for _, text in unique_facts],
        "questions": row["questions"],
        "answers": row["answers"],
        "context_chars": len(row["context"]),
        "n_facts": len(unique_facts),
    }
    get_client().update_current_span(
        output={
            "n_facts": result["n_facts"],
            "n_questions": len(result["questions"]),
            "context_chars": result["context_chars"],
        },
    )
    return result
