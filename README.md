# A Query-Time Deterministic Memory Update Mechanism via Structural Matching

Code and reproduction instructions for the experiments in the thesis
**"A Query-Time Deterministic Memory Update Mechanism via Structural Matching for
Language Models."** The method proposes a memory framework that treats Knowledge
Update (KU) as a **query-time** problem: it identifies the same fact by
`(subject, predicate)` structural matching and picks the current version by
ingestion time, so the version decision never depends on an LLM.

We evaluate on two KU benchmarks with different text characteristics:
- **FC-SH** (FactConsolidation Single-Hop, from MemoryAgentBench) — *counterfactual* KU.
- **LME-KU** (knowledge-update subset of LongMemEval) — *personal* KU.

Every result table in the thesis can be reproduced with the commands in
[§3](#3-reproducing-each-table). All paths below are relative to the package root;
you clone/unzip and run directly — no nested paths, no per-user setup.

> **Attribution.** This package contains our original method plus code forked
> from / vendored from prior projects (MemoryAgentBench, mem0, LongMemEval, and
> the "Don't Ask" baseline). What is ours vs. third-party, with licenses and
> citations, is stated in [§7](#7-attribution); the vendored components keep their
> upstream license texts under `dont_ask/*/LICENSE`.

---

## 1. Setup

### 1.1 Environment

```bash
conda create -n query_time_ku python=3.10 -y && conda activate query_time_ku
pip install -r requirements.txt
```

### 1.2 API keys

Copy the template to `.env`, then fill in your keys (the run scripts load `.env`
automatically — no other key configuration is needed):

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-...   # required — extraction, answering, embeddings, and the LLM judge
ZEP_API_KEY=z_...       # only for the Zep baseline rows; get one at https://www.getzep.com/
```

The OpenAI key alone reproduces every table except the Zep rows — omit
`ZEP_API_KEY` if you are not running Zep. These single-name variables are read
directly; the `_A/_B/...`-suffixed forms exist only for optional parallel runs.

### 1.3 Datasets

The paper-relevant data subsets **ship in this repo** under `data/`, so
experiments run offline with no download step:

- `data/longmemeval/longmemeval_s_ku.json` — the 78 LongMemEval-S
  knowledge-update questions (LME-KU).
- `data/fc_sh/factconsolidation_sh_{6k,32k,64k,262k}.json` — one file per context
  length (each run loads the file for its length, matching the four dataset
  configs). `utils/eval_data_utils.py` loads the local file automatically; if it
  is absent it falls back to the Hugging Face dataset `ai-hyz/MemoryAgentBench`.

Both are exact filters of the official public releases:

- **LME-KU** = the 78 `knowledge-update` questions of **LongMemEval-S (cleaned)** —
  official source: HF dataset `xiaowu0162/longmemeval-cleaned`
  (`longmemeval_s_cleaned.json`, 500 questions; MIT, Wu et al. 2024; repo *LongMemEval*).
- **FC-SH** = the FactConsolidation Single-Hop rows (6k/32k/64k/262k) of the
  `Conflict_Resolution` split of **MemoryAgentBench** — official source: HF dataset
  `ai-hyz/MemoryAgentBench` (MIT, Hu et al. 2026; official repo *MemoryAgentBench*).

To regenerate them from these official sources:

```bash
python scripts/build_data_subsets.py   # pulls the two HF datasets above (see script header)
```

### 1.4 Local backbones (only for `tab:gemma3` / `tab:crossfamily`)

The open-weight tables run each backbone through [Ollama](https://ollama.com); the
OpenAI-backbone tables need no local model. Pull the model, then select it with
`MODEL_TAG` + `MEM0_TRIPLE_MODEL` (see §3):

```bash
ollama pull gemma3:4b     # tags: gemma3:{1b,4b,12b,27b}, llama3.1:8b, qwen2.5:7b,
                          #       mistral:7b  (all Q4_K_M);  gemma2:9b is Q4_0
                          #       (its default tag) — do not force Q4_K_M.  temp 0.
```

Things to know before you run local backbones:

- **Per-backbone extraction is automatic.** `MEM0_TRIPLE_MODEL=<ollama-tag>` makes
  that backbone do its own fact/triple extraction + grouping; caches are keyed per
  backbone under `analysis/results/p1_caches__<backbone>/` (shipped).
- **Embeddings stay on OpenAI** (`text-embedding-3-small`), so `OPENAI_API_KEY` is
  needed even for a "local" run — only the chat backbone is swapped.
- **Weak-model defaults are baked in; do not lower them.** `OLLAMA_NUM_CTX=8192` (at
  4096 the top-100 grouping prompt overflows → empty groups), `OLLAMA_NUM_PREDICT=2048`
  (bounds weak-model rambling), and a uniform per-call 300 s timeout + `max_tokens`
  cap so a backbone that fails a step degrades to a declared fallback instead of
  hanging or crashing. All are env-overridable but the defaults are the paper setup.
- **One model at a time.** Ollama serves serially, so backbones run one after another
  (`ollama stop` between). The paper ran these on a single ~119 GB unified-memory box
  (27B ≈ 20 GB VRAM); 27B is slow enough that only the 6k cell was run.
- **Version sensitivity.** The paper's local numbers were produced with **Ollama 0.13**
  and the quantizations above (gemma2:9b Q4_0, the rest Q4_K_M); a newer Ollama build
  or a different quant can move a cell by a few points, within the ±3-point tolerance
  (§3). Local backbones are 6k-only (see `tab:gemma3` / `tab:crossfamily`).

### 1.5 Sanity check

```bash
bash scripts/run_fc_sh.sh 6k ours_struct_llm_matching
# expected: overall SubEM around 94/100. Confirms the pipeline is wired correctly.
```

---

## 2. Method names

The `<method>` argument uses names consistent with the paper. Mapping to paper rows:

| Paper row | `<method>` argument |
|:--|:--|
| Ours (Struct + LLM-Fallback) — main method | `ours_struct_llm_matching` |
| Ours (Struct-Only) | `ours_struct_only` |
| Ours (LLM-Identity-Only) | `ours_llm_matching_only` |
| Mem0 Vanilla | `mem0_vanilla` |
| Mem0 + Fact Extraction | `mem0_with_ours_extraction` |
| Vanilla-RAG (FC-SH) | `vanilla_rag` |
| Vanilla-RAG (2-stage, LME-KU) | `vanilla_rag_2stage` |
| Zep | `scripts/run_zep_fc.sh` (dedicated entry) |
| LCA (full context) | `scripts/run_lca_fc.sh` (dedicated entry) |
| Don't Ask | `scripts/maxserial_theircode.py` (FC-SH) / `dont_ask` (LME-KU) |

**Backbone selection** (default `gpt-4o-mini`): prefix with `MODEL_TAG=<dir>
MEM0_TRIPLE_MODEL=<model>`. `MODEL_TAG` names a directory under
`configs/agent_conf/RAG_Agents/` (`gpt-4o-mini`, `gpt-5.4-mini`, `gemma3-1b`,
`gemma3-4b`, `gemma3-12b`, `gemma3-27b`, `llama3.1-8b`, `qwen2.5-7b`, `gemma2-9b`,
`mistral-7b`). `MEM0_TRIPLE_MODEL` is the model string the backend expects — the same
name for GPT (`gpt-4o-mini`), but the **Ollama tag with a colon** for local backbones
(dir `gemma3-4b` → tag `gemma3:4b`; see §1.4).

**Metrics.** FC-SH uses the **official MemoryAgentBench substring exact match (SubEM)**
metric (the `substring_exact_match` field), 100 queries per length; each run prints
`SubEM x/100` at the end — the paper-table number (SubEM, not strict exact match).
LME-KU uses the official LongMemEval LLM autoevaluator (`gpt-4o-mini` judge), 78 questions.

**Shared settings** (all methods, both benchmarks): chunk size 512, vector retrieval
top-100 (Zep top-10, its official setting), temperature 0, single deterministic run.
FC-SH's KU error analysis uses the `has_pair` subset — the queries with a genuine
old+new pair — with denominators 74/65/66/77 of 100 for 6k/32k/64k/262k.

**Experiment setup — run order.** The variants
`mem0_with_ours_extraction`, `ours_struct_only`, `ours_llm_matching_only`, and
`vanilla_rag` reuse the extraction cache produced by `ours_struct_llm_matching`
(so the write stage is held fixed across methods). For a given length/backbone,
run `ours_struct_llm_matching` first, then the others.

---

## 3. Reproducing each table

Assumes `conda activate query_time_ku` and a populated `.env`.

### Table `tab:fcsh_main` — FC-SH SubEM, gpt-4o-mini, four context lengths

For each `L` in `6k 32k 64k 262k`:

```bash
bash scripts/run_fc_sh.sh $L ours_struct_llm_matching      # Ours
bash scripts/run_fc_sh.sh $L mem0_with_ours_extraction     # Mem0 + Fact Extraction
bash scripts/run_fc_sh.sh $L vanilla_rag                   # Vanilla-RAG
bash scripts/run_fc_sh.sh $L mem0_vanilla                  # Mem0 Vanilla
bash scripts/run_zep_fc.sh $L                              # Zep
bash scripts/run_lca_fc.sh $L                              # LCA
PIPELINE_MODEL=gpt-4o-mini python scripts/maxserial_theircode.py --length $L   # Don't Ask
```

Expected averages (AVG column): Ours 92.5, Don't Ask 85.0, Vanilla-RAG 84.0,
LCA 67.5, Zep 66.8, Mem0 + Fact Extraction 54.8, Mem0 Vanilla 20.8.

### Analysis tables (`tab:retrieval_len`, `tab:errormode_gtold`, `tab:retrieval_config`)

No new runs — these read the FC-SH outputs already produced above (plus the shipped
`analysis/results/sh_<L>_mquake_analysis.json` has_pair labels, and for
`tab:retrieval_config` the self-contained `analysis/recall_audit_6k/` bundle). The
analysis scripts read your fresh `outputs/` by default; to reproduce the paper
numbers from the shipped reference without rerunning, prefix
`OUTPUT_ROOT=reference_outputs`.

```bash
python analysis/fcsh_retrieval_ceiling.py                    # tab:retrieval_len (gt_new reaches top-100)
python analysis/errormode_gtold.py                           # tab:errormode_gtold (% of failures = gt_old)
python analysis/recall_audit_6k/pool_recall_4config_6k.py    # tab:retrieval_config (8 backbones)
```

### Table `tab:ablation_components` — component ablation, gpt-4o-mini

```bash
for L in 6k 32k 64k 262k; do
  bash scripts/run_fc_sh.sh $L ours_struct_llm_matching    # Struct + LLM-Fallback
  bash scripts/run_fc_sh.sh $L ours_struct_only            # Struct-Only
  bash scripts/run_fc_sh.sh $L ours_llm_matching_only      # LLM-Identity-Only
done
```

Expected: Struct + LLM-Fallback 94/91/94/91, Struct-Only 91/87/92/86,
LLM-Identity-Only 97/91/91/87.

### Table `tab:gemma3` — FC-SH 6k, gemma3 {1B, 4B, 12B, 27B}

Per-backbone extraction — each backbone extracts with itself. **Two names differ
and must not be swapped:** `MODEL_TAG` is the config-dir name (hyphen, e.g.
`gemma3-4b`); `MEM0_TRIPLE_MODEL` / `PIPELINE_MODEL` are the **Ollama model tag**
(colon, e.g. `gemma3:4b`) sent straight to Ollama. Run one backbone at a time
(Ollama serves one model). Shown for gemma3-4b:

```bash
ollama pull gemma3:4b
export MODEL_TAG=gemma3-4b MEM0_TRIPLE_MODEL=gemma3:4b     # dir=hyphen, ollama tag=colon
bash scripts/run_fc_sh.sh 6k ours_struct_llm_matching     # Ours; run first (builds the bank)
for m in ours_struct_only mem0_with_ours_extraction vanilla_rag mem0_vanilla; do
  bash scripts/run_fc_sh.sh 6k $m
done
# Don't Ask must point at THIS backbone's own P1 bank (else it silently uses gpt-4o-mini's):
PIPELINE_MODEL=gemma3:4b OLLAMA_CHAT_URL=http://localhost:11434/v1 \
  MAXSERIAL_BANK_TAG=$MODEL_TAG \
  MAXSERIAL_BANK_CACHE=analysis/results/p1_caches__${MODEL_TAG}/extraction_cache_p1_6k.json \
  python scripts/maxserial_theircode.py --length 6k
```

Repeat for the other three, pairing `MODEL_TAG` / Ollama tag:
`gemma3-1b`/`gemma3:1b`, `gemma3-12b`/`gemma3:12b`, `gemma3-27b`/`gemma3:27b`.

Expected values: see `tab:gemma3`. The `Δ (vs. Best Baseline)` row is the main
method minus the strongest baseline in each column. The **Zep** row on local
backbones is shipped as reference (`reference_outputs/$B-zep/`) but is *not* re-run
by these commands: Zep's answer client in this package targets OpenAI/Azure, and its
graph is backbone-independent, so only the answer stage differs across backbones.

### Table `tab:crossfamily` — FC-SH 6k, {Llama3.1-8B, Qwen2.5-7B, Gemma2-9B, Mistral-7B}

Same Ollama setup as `tab:gemma3`. This family's table has no Mem0 Vanilla and no
LLM-Identity-Only row, and only the matching configs ship — do **not** run
`mem0_vanilla` or `ours_llm_matching_only` here (their yamls do not exist for these
backbones). For each `MODEL_TAG` in `llama3.1-8b qwen2.5-7b gemma2-9b mistral-7b`
(pull the matching tag first):

```bash
export MODEL_TAG=<B> MEM0_TRIPLE_MODEL=<ollama-tag>    # dir=hyphen, ollama tag=colon
bash scripts/run_fc_sh.sh 6k ours_struct_llm_matching  # Ours; run first
for m in ours_struct_only mem0_with_ours_extraction vanilla_rag; do
  bash scripts/run_fc_sh.sh 6k $m
done
PIPELINE_MODEL=<ollama-tag> OLLAMA_CHAT_URL=http://localhost:11434/v1 \
  MAXSERIAL_BANK_TAG=$MODEL_TAG \
  MAXSERIAL_BANK_CACHE=analysis/results/p1_caches__${MODEL_TAG}/extraction_cache_p1_6k.json \
  python scripts/maxserial_theircode.py --length 6k    # Don't Ask via Ollama
```

Pair `MODEL_TAG` / Ollama tag: `llama3.1-8b`/`llama3.1:8b`, `qwen2.5-7b`/`qwen2.5:7b`,
`gemma2-9b`/`gemma2:9b`, `mistral-7b`/`mistral:7b`. Expected values: see
`tab:crossfamily`. As with `tab:gemma3`, the **Zep** row is reference-only
(`reference_outputs/<B>-zep/`).

### Table `tab:strongbackbone` — FC-SH 6k, gpt-4o-mini / gpt-5.4-mini

The gpt-4o-mini column is the 6k column of `tab:fcsh_main`. For gpt-5.4-mini:

```bash
export MODEL_TAG=gpt-5.4-mini MEM0_TRIPLE_MODEL=gpt-5.4-mini
bash scripts/run_fc_sh.sh 6k ours_struct_llm_matching
for m in mem0_with_ours_extraction vanilla_rag; do bash scripts/run_fc_sh.sh 6k $m; done
bash scripts/run_zep_fc.sh 6k                            # gpt-5.4-mini has a Zep config
PIPELINE_MODEL=gpt-5.4-mini python scripts/maxserial_theircode.py --length 6k
```

Expected: Ours 98, Vanilla-RAG 98, Don't Ask 96, Zep 93. (This table has no Mem0
Vanilla row; `mem0_vanilla` is not run here.)

### Table `tab:lme_main` — LME-KU accuracy, gpt-4o-mini, N=78

Third argument `1` runs the official judge after generation. Run the main method
first (the reuse-based methods depend on its store):

```bash
bash scripts/run_lme_ku.sh ours_struct_llm_matching 0 1   # Ours (main)
bash scripts/run_lme_ku.sh dont_ask 0 1
bash scripts/run_lme_ku.sh vanilla_rag_2stage 0 1
bash scripts/run_lme_ku.sh mem0_with_ours_extraction 0 1
bash scripts/run_lme_ku.sh mem0_vanilla 0 1
```

Expected (correct/78): Ours 55 (70.5), Don't Ask 59 (75.6), Vanilla-RAG 54 (69.2),
Mem0 + Fact Extraction 47 (60.3), Mem0 Vanilla 53 (67.9).

Setup notes: **Vanilla-RAG on LME uses the two-stage variant** (`vanilla_rag_2stage`).
LME's native answer template carries no serial rule, so a single-call version would
be the only method given a recency hint in its prompt — unfair; the two-stage split
(stage 1 the LLM picks winners from the ordinal-prefixed top-100, stage 2 answers
with the shared template) is the fair, canonical number. **Zep is excluded** from
LME-KU: its free-plan per-graph cap (~128k tokens) is below LME's ~128k average
context, so `graph.add` cannot ingest a full question (see §6). The judge is the
official LongMemEval autoevaluator with the `gpt-4o-mini` (2024-07-18) model.

Each run ends with `Accuracy: 0.xxxx` (× 78 = correct count); your run writes the
hyp + eval-results to `analysis/results/lme_hyps/`. The reference copies under
`reference_outputs/lme_hyps/*.eval-results-gpt-4o-mini` let you verify the counts
without rerunning (count `"label": true`).

### Table `tab:lme_gap` — derived

No runs. `vs. Don't Ask` gap = FC-SH AVG minus LME-KU:
`+7.5` (92.5 − 85.0) and `−5.1` (70.5 − 75.6).

### Checking your runs against the paper

The thesis's own run products are shipped compressed as `reference_outputs.tar.gz` —
extract it first into a `reference_outputs/` folder (see §5). Your runs write to
`outputs/` (FC-SH) and `analysis/results/lme_hyps/` (LME-KU) and
never touch `reference_outputs/`. To diff a fresh run against the reference (SubEM for
FC-SH, accuracy for LME-KU; PASS within ±3 points):

```bash
python scripts/compare_to_reference.py                      # scan outputs/ + lme_hyps
python scripts/compare_to_reference.py --fc outputs/.../<result>.json
python scripts/compare_to_reference.py --lme <hyp>.jsonl.eval-results-gpt-4o-mini
```

**Score with SubEM, not strict EM.** The paper's FC-SH numbers (`tab:fcsh_main`,
`tab:gemma3`, `tab:crossfamily`) are **substring exact match (SubEM)** — the
`substring_exact_match` field, which `compare_to_reference.py` reads. Do **not**
count the `exact_match` field instead: it is strict EM and runs well below SubEM for
verbose backbones that wrap the answer in prose (e.g. llama3.1-8b Struct-Only is
`exact_match` 17 but SubEM 81; mistral-7b Ours is 32 vs 66; gemma3-27b Ours 94 vs 99).
Use the shipped comparator (or the `substring_exact_match` field) and the numbers
match the tables.

---

## 4. Repository layout

```
.
├── README.md                 # this file — setup + how to reproduce each table
├── requirements.txt          # pinned dependencies (§1.1)
├── .env.example              # API-key template; copy to .env (§1.2)
├── agent.py main.py conversation_creator.py initialization.py   # benchmark run engine
├── mem0/                     # patched mem0 — our write/extraction changes live here
├── methods/                  # our method: phase0 (write-time) + phase2 (query-time KU), Zep, retriever
├── utils/  configs/  llm_based_eval/   # engine support: data loaders, agent/dataset yamls, LME judge
├── scripts/                  # run + analysis entry scripts, flat (§3)
├── dont_ask/                 # vendored Don't Ask authors' code (MIT)
├── data/                     # shipped FC-SH + LME-KU data subsets — runs work offline (§1.3)
├── analysis/                 # analysis scripts + results/ (per-backbone extraction caches, has_pair labels)
├── reference_outputs.tar.gz  # this thesis's experiment results, gzip'd — extract to use (§5)
└── outputs/                  # YOUR fresh run products — absent at clone, created on first run
```

## 5. Shipped reference and analysis inputs

`reference_outputs/` is **this thesis's own experiment results** — the exact run
products behind every table, kept as the reference your fresh runs are compared against.
To keep the repo Windows-friendly (the deep result paths would otherwise exceed the
260-character limit), it is committed **compressed** as `reference_outputs.tar.gz`
(≈68 MB). The archive holds the result folders directly, so **extract it into a
`reference_outputs/` folder before use:**

```bash
mkdir -p reference_outputs
tar -xf reference_outputs.tar.gz -C reference_outputs    # → reference_outputs/... (≈422 MB)
```

Double-clicking the archive on Windows/macOS works too — it creates a single
`reference_outputs/` folder with the results inside. Once extracted, prefix any analysis
script with `OUTPUT_ROOT=reference_outputs` to reproduce the paper numbers without
rerunning (§3, *Checking your runs*, covers the fresh-vs-reference diff). `analysis/results/`
ships the pipeline inputs a run reads or regenerates: `p1_caches*/` (held-fixed
per-backbone extraction caches), `sh_<L>_mquake_analysis.json` (has_pair labels for
the error-mode / retrieval tables), and `phase0/`.

## 6. Notes

- **Fresh vs reference:** your runs write to `outputs/` (created on first run) and
  `analysis/results/lme_hyps/`; the shipped reference is `reference_outputs/`, which
  your runs never touch. Datasets ship in-repo (§1.3); the only thing regenerated on
  a run is the Qdrant vector store, rebuilt from the shipped data on first use.
- **Running Zep — read before the Zep rows.** Zep is a hosted asynchronous service,
  and its free-plan behaviour is the most common reason a Zep run *looks* broken (an
  empty retrieval or an anomalously low score is usually operational, not a code bug):
  - *Async ingestion — do not hard-cut the wait.* After upload the graph is not
    immediately queryable; the runner sleeps, then probes until the episode count
    stabilises. `ZEP_INITIAL_WAIT` (default 360s; 1200s at 262k) and `ZEP_MAX_PROBES`
    (8; 20 at 262k) bound it — roughly ≤46 min at ≤64k, ≤120 min at 262k. An empty
    retrieval almost always means the graph was queried before it finished indexing.
  - *Free-plan caps (the usual culprit).* On an unpaid key, each ingested message is
    truncated to 2400 characters (`methods/zep.py`) — long chunks lose facts — and
    `get_user_context` is rate-limited to ~5/min (429s; the runner backs off per
    `Retry-After`, so runs crawl). Mitigate by using a **paid plan**; or rotate
    several keys (`ZEP_API_KEY_A/B/C`, selected via `RUN_ZEP_KEY_NAME`) across runs;
    or self-host **Graphiti**, the OSS backend behind Zep.
  - *Re-query without re-ingesting.* The graph persists on Zep by id, so once built
    you can re-run queries against it without paying the ingestion + wait again.
  - Retrieval is top-10 per scope (`ZEP_TOP_K`, default 10), per Zep's official setup.
- **Parallel runs (optional speedup):** `run_lme_ku.sh` accepts `SHARD`/`NSHARD`
  to split the 78 questions across processes; not required for correctness.
- **Run-to-run variation:** even at temperature 0, server-side numerical
  nondeterminism can move a fresh run ±2–3 points from the reference (within
  `compare_to_reference.py`'s tolerance). The destructive baselines (Mem0 Vanilla,
  Mem0 + Fact Extraction) vary more, since their write-time LLM ADD/UPDATE/DELETE
  decision is stochastic; the append-only and deterministic methods reproduce tightly.

## 7. Attribution

The table below states origin and license for each part. Vendored/forked
components retain their upstream copyright notices, preserved per MIT/Apache-2.0:
MemoryAgentBench © 2026 Yuanzhe Hu (MIT); mem0 (Apache-2.0, `mem0/` is a modified
copy); "Don't Ask" © 2026 Vikas Challaram Reddy (MIT, full text at
`dont_ask/memory-conflict-resolution/LICENSE`); LongMemEval © 2024 Di Wu (MIT).

| Component | Origin | License |
|:--|:--|:--|
| `methods/phase0_*`, `methods/phase2_query.py`, our `agent.py` handlers, `scripts/` | **Ours (this work)** | MIT (proposed) |
| Benchmark engine: `agent.py`, `main.py`, `conversation_creator.py`, `utils/`, `configs/`, `methods/{zep,embedding_retriever}.py` | forked & modified from **MemoryAgentBench** (Hu et al., ICLR 2026, arXiv:2507.05257) | MIT |
| `mem0/` | patched copy of **mem0** (mem0ai) | Apache-2.0 |
| `dont_ask/memory-conflict-resolution/` (Don't Ask) | vendored verbatim (Reddy & Challaram, 2026, arXiv:2606.01435) | MIT |
| `llm_based_eval/evaluate_qa_official.py` | vendored from **LongMemEval** (Wu et al., ICLR 2025, arXiv:2410.10813) | MIT |
| Zep baseline | hosted service via `zep-cloud` SDK (Rasmussen et al., 2025, arXiv:2501.13956) | — |

**"Don't Ask" — naming and version note.** *Don't Ask* is the short name **we** use for
this baseline; it is the work's original title, later retitled *Reliable Post-Retrieval
Assembly for Agent Memory: Separating Evidence Extraction from Policy Execution* (COLM
2026 Lifelong-Agent Workshop). The vendored pipeline is pinned at upstream commit
`b6b92b4`; its candidate-extraction prompt (`scripts/_pipeline.py` `CANDIDATE_PROMPT`)
and default deterministic max-serial pick are **byte-identical** in the later release —
the only upstream additions are an o-series temperature guard (a no-op for the
non-reasoning backbones used here) and an optional LLM-picker ablation not on the default
path — so our Don't Ask results are stable across upstream versions.
