"""Paper experiment: BM25 baseline vs SH-conflict pipeline on n=100 of factconsolidation_sh.

Usage:
  python scripts/13_paper_experiment.py --source factconsolidation_sh_262k

Runs one context length. Use the bash launcher to run all 4 in parallel.

Each question's BM25 baseline AND SH conflict pipeline run get their own Langfuse
trace, tagged with: experiment, competency, dataset (source), question_index,
context_length, ground_truth.

Output: poc_results/paper_sh_conflict_<source>.json
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from typing import Any

from datasets import load_dataset
from rank_bm25 import BM25Okapi

sys.path.insert(0, '.')
from _lf import OpenAI, observe, get_client, ROOT
from _pipeline import (
    tokenize, bm25_retrieve, evaluate_answer,
    _extract_candidates, _freshness_pick,
    run_bm25_baseline, MODEL, TOP_K,
)


def parse_facts(ctx: str) -> list[tuple[int, str]]:
    pat = re.compile(r"(\d+)\.\s")
    matches = list(pat.finditer(ctx))
    facts: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        s = m.end()
        e = matches[i + 1].start() if i + 1 < len(matches) else len(ctx)
        text = ctx[s:e].strip().rstrip(".")
        facts.append((idx, text))
    seen: set[int] = set()
    unique: list[tuple[int, str]] = []
    for idx, text in facts:
        if idx not in seen:
            seen.add(idx); unique.append((idx, text))
    return unique


@observe(name="sh_conflict_paper")
def run_sh_conflict(question: str, question_index: int, ground_truth: list[str],
                    bm25: BM25Okapi, fact_indices: list[int], fact_texts: list[str],
                    client: OpenAI, source: str) -> dict[str, Any]:
    """SH conflict pipeline: BM25 → LLM extract candidates → Python max(serial) freshness pick."""
    get_client().update_current_span(
        name=f"sh_conflict_paper_{source}_q{question_index + 1}",
        metadata={
            "experiment": "paper_sh_conflict_n100",
            "competency": "Conflict_Resolution",
            "dataset": source,
            "context_length": source.split("_")[-1],
            "question_index": question_index,
            "ground_truth": ground_truth,
        },
        input={"question": question},
    )

    retrieved = bm25_retrieve(bm25, question, fact_indices, fact_texts, TOP_K)
    candidates = _extract_candidates(client, question, retrieved)
    chosen = _freshness_pick(candidates)
    answer = chosen["answer_entity"] if chosen else "(no answer)"
    eval_result = evaluate_answer(answer, ground_truth)

    get_client().update_current_span(
        output={"answer": answer, "is_correct": eval_result["is_correct_subem"],
                "n_candidates": len(candidates),
                "chosen_serial": chosen.get("serial") if chosen else None},
    )
    if eval_result["is_correct_subem"]:
        get_client().score_current_trace(name="correctness", value=1.0)
    else:
        get_client().score_current_trace(name="correctness", value=0.0)

    return {
        "answer": answer, "is_correct": eval_result["is_correct_subem"],
        "n_candidates": len(candidates),
        "chosen_serial": chosen.get("serial") if chosen else None,
        "retrieved_serials": [r["fact_idx"] for r in retrieved],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True,
                   choices=["factconsolidation_sh_6k", "factconsolidation_sh_32k",
                            "factconsolidation_sh_64k", "factconsolidation_sh_262k"])
    args = p.parse_args()
    source = args.source
    length_label = source.split("_")[-1]

    print(f"[{length_label}] Loading dataset…")
    ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision="main")
    row = next(s for s in ds if s["metadata"]["source"] == source)
    ctx = row["context"]
    questions = row["questions"] if isinstance(row["questions"], list) else [row["questions"]]
    answers = row["answers"] if isinstance(row["answers"], list) else [row["answers"]]

    facts = parse_facts(ctx)
    fact_indices = [f[0] for f in facts]
    fact_texts = [f[1] for f in facts]
    bm25 = BM25Okapi([tokenize(t) for t in fact_texts])

    n_q = len(questions)
    print(f"[{length_label}] {n_q} questions × {len(facts)} facts. Starting…")

    client = OpenAI()
    results = []
    bm_correct = sh_correct = 0
    t_start = time.time()

    for q_idx in range(n_q):
        question = questions[q_idx]
        gt = answers[q_idx]
        gt_list = gt if isinstance(gt, list) else [gt]

        try:
            bm = run_bm25_baseline(
                question=question, question_index=q_idx, ground_truth=gt_list,
                bm25=bm25, fact_indices=fact_indices, fact_texts=fact_texts,
                client=client, temperature=0.7,
                dataset_name=source, competency="Conflict_Resolution",
            )
            bm_answer = bm["answer"]; bm_ok = bool(bm["is_correct"])
        except Exception as e:
            bm_answer = f"<error: {str(e)[:60]}>"; bm_ok = False

        try:
            sh = run_sh_conflict(question, q_idx, gt_list, bm25, fact_indices, fact_texts, client, source)
            sh_answer = sh["answer"]; sh_ok = bool(sh["is_correct"])
        except Exception as e:
            sh_answer = f"<error: {str(e)[:60]}>"; sh_ok = False; sh = {}

        if bm_ok: bm_correct += 1
        if sh_ok: sh_correct += 1
        results.append({
            "q_idx": q_idx, "question": question, "ground_truth": gt_list,
            "bm25_answer": bm_answer, "bm25_correct": bm_ok,
            "sh_answer": sh_answer, "sh_correct": sh_ok,
            "sh_n_candidates": sh.get("n_candidates"),
            "sh_chosen_serial": sh.get("chosen_serial"),
        })

        if (q_idx + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (q_idx + 1) / elapsed
            eta = (n_q - q_idx - 1) / rate
            print(f"[{length_label}] Q{q_idx+1}/{n_q} | BM25 {bm_correct}/{q_idx+1}={100*bm_correct/(q_idx+1):.1f}% "
                  f"SH {sh_correct}/{q_idx+1}={100*sh_correct/(q_idx+1):.1f}% "
                  f"| {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - t_start
    summary = {
        "source": source, "context_length": length_label, "n_questions": n_q,
        "bm25_correct": bm_correct, "bm25_accuracy": bm_correct / n_q,
        "sh_correct": sh_correct, "sh_accuracy": sh_correct / n_q,
        "elapsed_seconds": elapsed, "results": results,
    }

    out_path = ROOT / "poc_results" / f"paper_sh_conflict_{source}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[{length_label}] DONE. Saved → {out_path}")
    print(f"[{length_label}] BM25: {bm_correct}/{n_q} = {100*bm_correct/n_q:.1f}%   "
          f"SH conflict: {sh_correct}/{n_q} = {100*sh_correct/n_q:.1f}%   "
          f"({elapsed:.0f}s)")

    get_client().flush(); time.sleep(3)


if __name__ == "__main__":
    main()
