# memory-conflict-resolution

Companion code, prompts, and per-question results for
**"Don't Ask the LLM to Track Freshness: A Deterministic Recipe for
Memory Conflict Resolution"** (arXiv preprint, 2026).

## What this is

A minimal recipe for memory conflict resolution on MemoryAgentBench's
FactConsolidation task:
1. **Retrieve** candidate facts with BM25.
2. **Extract** semantically matching candidates with an LLM (strict
   subject + predicate match; do not pick a "best").
3. **Pick** the candidate with the highest serial number using a
   deterministic Python `max(serial)`.

For multi-hop questions, the same primitive is applied per hop after
Self-Ask-style decomposition (the CAR pipeline).

Headline numbers (n=100 per cell × 4 context lengths; 95% Wilson CIs):

| Pipeline | Backbone | FC-SH avg | FC-MH avg |
|---|---|:-:|:-:|
| SH-conflict + Python `max(serial)` | gpt-4o-mini | **78.0 [73.7, 81.8]** | — |
| SH-conflict + Python `max(serial)` | gpt-4o | **94.8 [92.1, 96.5]** | — |
| CAR (per-hop deterministic freshness) | gpt-4o-mini | — | **30.2 [26.0, 34.9]** |

At matched-backbone, matched-262K comparisons against MAB Table 3,
this beats HippoRAG-v2 (54%) by +28 pp on FC-SH, beats GPT-4o
long-context (60%) by +33 pp at gpt-4o, and beats the best published
FC-MH result (7%) by +20 pp. See `PAPER.md` for full results and the
matched-setup comparison ablation.

## Layout

```
scripts/
  _data.py                  # dataset loading helpers
  _lf.py                    # Langfuse + OpenAI client setup
  _pipeline.py              # shared primitives (BM25, candidate extraction, freshness pick, CAR)
  13_paper_experiment.py    # headline Python-max pipeline + LLM-judgment baseline (FC-SH)
  14_ablations.py           # chunk-4096, gpt-4o, and FC-MH CAR ablations

poc_results/
  paper_sh_conflict_factconsolidation_sh_{6k,32k,64k,262k}.json
                            # headline (Python max) + LLM-judgment baseline, per-question
  ablation_sh_chunk4096_gpt4omini_factconsolidation_sh_{...}.json
  ablation_sh_fact_gpt4o_factconsolidation_sh_{...}.json
  ablation_mh_fact_gpt4omini_factconsolidation_mh_{...}.json

PAPER.md                    # arXiv preprint source (Markdown)
```

Each `poc_results/*.json` file contains per-question records with the
question text, ground-truth answer list, both pipeline answers,
SubEM correctness, retrieved-fact serials, extracted candidates, and
the chosen serial.

## Setup

```bash
git clone https://github.com/cvikasreddy/memory-conflict-resolution
cd memory-conflict-resolution
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY (required) and LANGFUSE_* keys (optional, for tracing)
```

Langfuse tracing is on by default. If you don't want to use Langfuse,
either set the `LANGFUSE_*` env vars to enable it, or edit
`scripts/_lf.py` to stub `observe` as a no-op decorator and skip the
`Langfuse(...)` initialization.

## One-command headline replication

```bash
python scripts/13_paper_experiment.py --source factconsolidation_sh_262k
```

Output: `poc_results/paper_sh_conflict_factconsolidation_sh_262k.json`
with `sh_accuracy` (Python max pipeline) and `bm25_accuracy`
(LLM-judgment baseline) plus full per-question records.

All four context lengths in parallel:

```bash
for L in 6k 32k 64k 262k; do
  python scripts/13_paper_experiment.py --source factconsolidation_sh_${L} &
done
wait
```

## Reproducing each cell of the master table

| Cell | Command |
|---|---|
| Headline + LLM-judgment baseline (FC-SH, gpt-4o-mini, fact-level) | `python scripts/13_paper_experiment.py --source factconsolidation_sh_<L>` |
| Ablation A (FC-SH, gpt-4o-mini, chunk-4096) | `python scripts/14_ablations.py --source factconsolidation_sh_<L> --chunk-strategy chunk4096` |
| Ablation B (FC-MH, gpt-4o-mini, fact-level, CAR) | `python scripts/14_ablations.py --source factconsolidation_mh_<L> --task mh` |
| Ablation C (FC-SH, gpt-4o, fact-level) | `PIPELINE_MODEL=gpt-4o python scripts/14_ablations.py --source factconsolidation_sh_<L>` |

`<L>` is one of `6k`, `32k`, `64k`, `262k`. `6k` is a diagnostic slice
constructed using MAB's MQUAKE edit-pair procedure and is not part of
the MAB main benchmark.

## Compute and cost

- **1600 evaluations** in the full master table (16 cells × 100 questions).
- **Cost**: ~$3 across the full benchmark, dominated by the four
  gpt-4o cells (~$2). gpt-4o-mini cells contribute ~$0.3 across the
  headline, LLM-judgment baseline, chunk-4096 ablation, and CAR
  multi-hop runs.
- **Wall-clock**: each cell ~5–10 min on a single OpenAI API endpoint;
  all 16 cells in parallel ~15 min total.

## What's in PAPER.md

The full preprint, including:
- §3 method (SH-conflict and CAR pipelines with pseudocode)
- §4 experimental setup
- §5 master results, comparison to 22 MAB-published systems, matched-setup
  pipeline comparison (§5.3, +10.8 pp), retrieval upper bound and
  method complementarity (§5.5)
- §6 discussion, precision/recall trade-off, limitations
- Appendix A: full prompts (CANDIDATE_PROMPT, DECOMP_PROMPT, MAB BM25 prompt)

## Citation

```bibtex
@article{reddy2026dontask,
  title={Don't Ask the LLM to Track Freshness: A Deterministic Recipe
         for Memory Conflict Resolution},
  author={Reddy, Vikas Challaram},
  journal={arXiv preprint},
  year={2026}
}
```

(Update with the actual arXiv ID once posted.)

## License

MIT. See `LICENSE`.
