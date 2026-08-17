"""Bridge: drive our mem0-based memory pipeline (ours / mem0(b) / vanilla / zep)
over the OFFICIAL LongMemEval data (longmemeval_s_ku.json), KU subset, and
emit a hypothesis jsonl for the OFFICIAL judge (origin_longmemeval evaluate_qa.py).

Why a bridge (not MemoryAgentBench's longmemeval packaging): MABench's
Accurate_Retrieval packaging does not preserve the multi-session knowledge-update
structure (old + new value both in the haystack). The official data does, so KU
is actually tested. We therefore feed the official haystack through OUR memory
method and judge with the official LLM-judge.

Design (confirmed with user):
  * sub_dataset = longmemeval_s_ku  -> longmemeval template (conversation memorize,
    concise QA query) + an ISOLATED qdrant store (agent.py suffixes by sub_dataset).
  * session-level ingestion: each haystack session is one memorize() unit, ingested
    in haystack order (haystack is timestamp-sorted) so ordinal == recency. The
    session date is embedded in-context (real timestamps). This is a benchmark
    adaptation only; the memory method itself is unchanged.
  * per-question isolation via user_id=context_{qid}_{sub_dataset} (vector search is
    user_id-filtered; subjects are user_id-namespaced) -> one store, 78 partitions.
  * the method (ours vs mem0(b) vs vanilla vs zep) is selected by ENV in the launch
    script (MEM0_ADD_MODE / MEM0_QUERY_MODE / caches), exactly like run_fc_sh.sh.

Answer = agent.send_message(wrapped_q, memorizing=False)["output"]; we skip EM and
let the official judge score. Output line: {"question_id":..., "hypothesis":...}.

Usage:
  python run_longmemeval_ku.py --agent_config <yaml> --data <longmemeval_*.json>
        --out <hyp.jsonl> [--sub_dataset longmemeval_s_ku] [--limit N] [--qtype knowledge-update]
"""
import argparse, json, os, sys, time

sys.path.insert(0, __import__("os").environ.get("REPO_ROOT") or str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import yaml
from agent import AgentWrapper
from utils.templates import get_template


def fmt_session(turns, date):
    """One haystack session -> a dated transcript string (the memorize context)."""
    lines = [f"Date: {date}"]
    for t in turns:
        role = str(t.get("role", "user")).capitalize()
        lines.append(f"{role}: {t.get('content','')}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sub_dataset", default="longmemeval_s_ku")
    ap.add_argument("--qtype", default="knowledge-update")
    ap.add_argument("--limit", type=int, default=0, help="0 = all matching questions")
    ap.add_argument("--gen_max", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0, help="this shard index (0-based)")
    ap.add_argument("--nshard", type=int, default=1, help="total shards (parallel processes)")
    ap.add_argument("--query-only", action="store_true",
                    help="Skip memorize phase (assume store already populated by prior run with same sub_dataset). "
                         "For ours (+P5) reuse of ours_no_p5's store to get P5 decision trace without re-ingest.")
    args = ap.parse_args()

    agent_config = yaml.safe_load(open(args.agent_config))
    dataset_config = {
        "dataset": "LongMemEval",
        "sub_dataset": args.sub_dataset,          # -> longmemeval template + isolated store
        "context_max_length": 200000,
        "generation_max_length": args.gen_max,
    }
    agent = AgentWrapper(agent_config, dataset_config, None)
    agent_name = agent_config["agent_name"]

    # longmemeval rag_agent query template (same inference template the benchmark uses
    # for this task family); _retrieval_query strips it back to the bare question.
    q_tmpl = get_template(args.sub_dataset, "query", agent_name)

    data = json.load(open(args.data))
    items = [d for d in data if d.get("question_type") == args.qtype]
    if args.limit:
        items = items[: args.limit]
    # shard by stable index (deterministic, no overlap, full coverage across shards)
    if args.nshard > 1:
        items = [d for i, d in enumerate(items) if i % args.nshard == args.shard]
    print(f"[lme] {len(items)} '{args.qtype}' questions (shard {args.shard}/{args.nshard}) "
          f"from {os.path.basename(args.data)} | agent={agent_name} | out={args.out}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):  # resume support
        for ln in open(args.out):
            try: done.add(json.loads(ln)["question_id"])
            except Exception: pass
    out_f = open(args.out, "a", encoding="utf-8")

    t0 = time.time()
    for i, d in enumerate(items):
        qid = d["question_id"]
        if qid in done:
            print(f"[lme] ({i+1}/{len(items)}) {qid} already done, skip", flush=True)
            continue
        cid = qid.replace("-", "_")  # context/user_id partition (filesystem/id-safe)
        sessions = d["haystack_sessions"]
        dates = d.get("haystack_dates", [""] * len(sessions))
        ts = time.time()
        if not args.query_only:
            for sess, date in zip(sessions, dates):
                agent.send_message(fmt_session(sess, date), memorizing=True, query_id=qid, context_id=cid)
        wrapped_q = q_tmpl.format(question=d["question"])
        if d.get("question_date"):
            wrapped_q = f"Today's date is {d['question_date']}.\n\n" + wrapped_q
        out = agent.send_message(wrapped_q, memorizing=False, query_id=qid, context_id=cid)
        hyp = out["output"] if isinstance(out, dict) else str(out)
        out_f.write(json.dumps({"question_id": qid, "hypothesis": hyp}, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"[lme] ({i+1}/{len(items)}) {qid} | {len(sessions)} sess | "
              f"{time.time()-ts:.0f}s | ans: {hyp[:80]!r}", flush=True)

    out_f.close()
    print(f"[lme] DONE {len(items)} questions in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
