"""Ablation driver — supports three experiments via CLI args.

Usage:
  # Chunk-size ablation (4096 instead of fact-level)
  python scripts/14_ablations.py --source factconsolidation_sh_262k --chunk-strategy chunk4096

  # FC-MH with CAR pipeline (multi-hop conflict resolution)
  python scripts/14_ablations.py --source factconsolidation_mh_262k --task mh

  # gpt-4o backbone ablation
  PIPELINE_MODEL=gpt-4o python scripts/14_ablations.py --source factconsolidation_sh_262k

Output: poc_results/ablation_<task>_<chunk>_<model>_<source>.json
"""
from __future__ import annotations
import argparse
import json
import os
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
    run_bm25_baseline, run_car_v2,
    MODEL, TOP_K,
)


def parse_facts_numbered(ctx: str) -> list[tuple[int, str]]:
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


def chunk_4096(ctx: str) -> list[tuple[int, str]]:
    """Slide-window chunker, 4096 chars per chunk. Chunk index = serial."""
    chunks = []
    for i, start in enumerate(range(0, len(ctx), 4096)):
        chunk_text = ctx[start:start + 4096]
        chunks.append((i, chunk_text))
    return chunks


@observe(name="ablation_sh_conflict")
def run_sh_conflict_ablated(question: str, question_index: int, ground_truth: list[str],
                            bm25: BM25Okapi, fact_indices: list[int], fact_texts: list[str],
                            client: OpenAI, source: str, experiment_tag: str) -> dict[str, Any]:
    """Same SH conflict pipeline; works on either fact-level or chunk-level data."""
    get_client().update_current_span(
        name=f"{experiment_tag}_{source}_q{question_index + 1}",
        metadata={
            "experiment": experiment_tag,
            "competency": "Conflict_Resolution",
            "dataset": source,
            "context_length": source.split("_")[-1],
            "question_index": question_index,
            "ground_truth": ground_truth,
            "pipeline_model": MODEL,
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
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--task", default="sh", choices=["sh", "mh"],
                   help="sh = single-hop pipeline, mh = CAR multi-hop pipeline")
    p.add_argument("--chunk-strategy", default="fact",
                   choices=["fact", "chunk4096"],
                   help="fact = parse numbered facts, chunk4096 = 4096-char windows")
    args = p.parse_args()

    source = args.source
    task = args.task
    chunk_strategy = args.chunk_strategy
    length_label = source.split("_")[-1]

    # Build experiment tag — encodes all ablation dimensions
    experiment_tag = f"ablation_{task}_{chunk_strategy}_{MODEL.replace('-', '').replace('.', '')}"
    print(f"[{length_label}] experiment={experiment_tag} model={MODEL} chunking={chunk_strategy} task={task}")

    print(f"[{length_label}] Loading dataset…")
    competency = "Conflict_Resolution"
    ds = load_dataset("ai-hyz/MemoryAgentBench", split=competency, revision="main")
    row = next(s for s in ds if s["metadata"]["source"] == source)
    ctx = row["context"]
    questions = row["questions"] if isinstance(row["questions"], list) else [row["questions"]]
    answers = row["answers"] if isinstance(row["answers"], list) else [row["answers"]]

    # Chunk according to strategy
    if chunk_strategy == "fact":
        chunks = parse_facts_numbered(ctx)
    else:  # chunk4096
        chunks = chunk_4096(ctx)
    fact_indices = [c[0] for c in chunks]
    fact_texts = [c[1] for c in chunks]
    bm25 = BM25Okapi([tokenize(t) for t in fact_texts])

    n_q = len(questions)
    print(f"[{length_label}] {n_q} questions × {len(chunks)} chunks (strategy={chunk_strategy})")

    client = OpenAI()
    results = []
    correct = 0
    t_start = time.time()

    for q_idx in range(n_q):
        question = questions[q_idx]
        gt = answers[q_idx]
        gt_list = gt if isinstance(gt, list) else [gt]

        try:
            if task == "sh":
                r = run_sh_conflict_ablated(
                    question=question, question_index=q_idx, ground_truth=gt_list,
                    bm25=bm25, fact_indices=fact_indices, fact_texts=fact_texts,
                    client=client, source=source, experiment_tag=experiment_tag,
                )
            else:  # mh — CAR pipeline
                r = run_car_v2(
                    question=question, question_index=q_idx, ground_truth=gt_list,
                    bm25=bm25, fact_indices=fact_indices, fact_texts=fact_texts,
                    client=client, dataset_name=source, competency=competency,
                )
                # Normalize MH return key to match SH "is_correct"
                r["is_correct"] = r.get("is_correct", False)
                r["answer"] = r.get("final_answer", "(no answer)")
            answer = r["answer"]; ok = bool(r["is_correct"])
        except Exception as e:
            answer = f"<error: {str(e)[:60]}>"; ok = False; r = {}

        if ok: correct += 1
        results.append({
            "q_idx": q_idx, "question": question, "ground_truth": gt_list,
            "answer": answer, "correct": ok,
            "n_candidates": r.get("n_candidates"),
            "chosen_serial": r.get("chosen_serial"),
            "n_hops_planned": r.get("n_hops_planned"),
            "n_hops_executed": r.get("n_hops_executed"),
        })

        if (q_idx + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"[{length_label}] Q{q_idx+1}/{n_q} | correct {correct}/{q_idx+1}={100*correct/(q_idx+1):.1f}% "
                  f"| {elapsed:.0f}s elapsed")

    elapsed = time.time() - t_start
    summary = {
        "experiment": experiment_tag, "model": MODEL, "task": task,
        "chunk_strategy": chunk_strategy,
        "source": source, "context_length": length_label, "n_questions": n_q,
        "correct": correct, "accuracy": correct / n_q,
        "elapsed_seconds": elapsed, "results": results,
    }

    out_path = ROOT / "poc_results" / f"{experiment_tag}_{source}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[{length_label}] DONE. Saved → {out_path}")
    print(f"[{length_label}] {experiment_tag}: {correct}/{n_q} = {100*correct/n_q:.1f}%   ({elapsed:.0f}s)")

    get_client().flush(); time.sleep(3)


if __name__ == "__main__":
    main()
