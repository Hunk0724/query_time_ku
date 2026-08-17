# Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution

---

## Abstract

LLM-based memory systems are increasingly tasked with maintaining
facts that evolve over time. A recurring failure mode is **conflict
resolution** — when the same fact appears in memory with multiple,
contradictory values, which value should the agent return? The recently
released MemoryAgentBench (MAB; Hu, Wang, & McAuley, 2026)
**FactConsolidation** task (a conflict-resolution-style task under
MAB's Selective Forgetting competency) makes this explicit: facts are
numbered, the counterfactual appears with the higher serial, and
agents are told that *"newer facts have larger serial numbers."*
Despite the rule being spelled out, every published memory system
substantially underperforms: HippoRAG-v2 reaches 54.0% on single-hop,
BM25 reaches 48.0%, Mem0 / Contriever 18.0%, and Zep / Graphiti — a
temporal knowledge graph designed for agent memory with bi-temporal
edges and validity intervals — scores **7.0%** on FC-SH (its lowest
column in MAB Table 3, despite reported strengths on other benchmarks
like DMR and LongMemEval). The multi-hop variant is nearly unsolved
(≤7% across all 22 evaluated systems).

We argue that an underappreciated bottleneck is the **assembly step**:
existing memory-agent baselines often leave conflict resolution to
LLM-mediated retrieval, memory update, or answer generation, rather
than to explicit version-aware aggregation. A matched-setup comparison
(same backbone, same retrieval, same chunking, same TOP_K, same n=100
per cell) shows that **replacing the LLM-judgment-based answer
pipeline with a candidate-extraction + Python `max(serial)` pipeline
yields +10.8 percentage points** on FC-SH (67.2 → 78.0, fact-level
chunking, gpt-4o-mini). The gap widens with context length: +8 pp at
6K, +21 pp at 262K. (Caveat: this comparison varies more than the
resolver step alone — the two pipelines also differ in prompt format,
output format, and temperature; an isolated same-extraction
LLM-vs-Python resolver ablation is future work, §6.6.)
Combined with semantic candidate extraction, this produces three
results: **78.0% on FC-SH** with gpt-4o-mini, **94.8% on FC-SH** with
gpt-4o, and **30.2% on FC-MH** via a per-hop deterministic-freshness
extension of Self-Ask-style decomposition. At matched-backbone,
matched-262K comparisons against MAB Table 3, the gpt-4o-mini
pipeline beats the strongest published system (HippoRAG-v2) by
**+28 pp** and the weakest (Zep/Graphiti) by **+75 pp**; the
multi-hop result beats the best published FC-MH result (7%) by
**+20 pp** at 262K. Larger gaps appear when the backbone is also
upgraded to gpt-4o (§5.2).

The implication is corrective for the memory-framework subfield: the
bottleneck on conflict resolution is **assembly** (post-retrieval
aggregation), not **storage** (graph, hippocampal, agentic, or typed).
For data with explicit version markers, the right primitive is
deterministic max over LLM-extracted candidates.

---

## §1 Introduction

### §1.1 The puzzle

LLM-based memory agents are increasingly deployed to maintain
information that evolves over time: user preferences that change,
policies that supersede earlier versions, factual records that get
corrected. A recurring failure mode in this setting is **conflict
resolution** — when the same fact appears in memory with multiple,
contradictory values, which value should the agent return?

The convention in most modern memory systems is straightforward: **the
most recent fact wins**. Mem0, Graphiti (Zep), MemGPT, MIRIX, Cognee,
HippoRAG-v2, and the various academic systems all explicitly handle
"updates" or "supersession" of older information by newer.

Yet **all of these systems substantially underperform on this task**.
MemoryAgentBench (Hu,
Wang, & McAuley, 2026), the most comprehensive evaluation of LLM
memory systems to date, provides a clean test: the FactConsolidation
task constructs a memory of numbered facts where counterfactual
versions (with higher serial numbers) appear after the original facts.
Agents are explicitly told that *"newer facts have larger serial
numbers."* Despite this clear specification, every evaluated system
substantially underperforms:

- HippoRAG-v2 (the best published RAG): **54.0%** on FC-SH.
- GPT-4o long-context (sees the full context at 262K): **60.0%** on FC-SH.
- BM25 (MAB's simplest published RAG baseline, with the freshness rule
  in its prompt): **48.0%** on FC-SH (chunk-512, the MAB default for
  FactConsolidation).
- MemGPT and Cognee: **28.0%** on FC-SH.
- Mem0 and Contriever: **18.0%** on FC-SH.
- RAPTOR, GraphRAG, MIRIX: **14.0%** on FC-SH.
- Zep / Graphiti (a temporal knowledge graph designed for agent memory
  with bi-temporal edges and validity intervals): **7.0%** on FC-SH —
  its lowest column in MAB Table 3, despite reported strengths on
  other benchmarks (DMR, LongMemEval).
- **All 22 evaluated systems**: ≤7% on the multi-hop variant (FC-MH).

A task designed around an explicit rule (*"newer is correct"*) is
being underperformed by every system, including a knowledge graph
explicitly designed for temporal memory. The natural question is
**why**.

### §1.2 A matched-setup pipeline comparison

Our central finding is mechanistic. A matched-setup comparison at
fixed chunking shows the size of the gap between an LLM-judgment-based
answer pipeline and a candidate-extraction + Python `max(serial)`
pipeline:

| Setup (fact-level chunking, gpt-4o-mini) | 6K | 32K | 64K | 262K | AVG [95% CI] |
|---|:-:|:-:|:-:|:-:|:-:|
| BM25 retrieval + MAB BM25 answer prompt (LLM-judgment) | 63 | 70 | 75 | 61 | 67.2 [62.5, 71.7] |
| BM25 retrieval + extract-candidates + Python `max(serial)` (Headline) | 71 | 78 | 81 | 82 | **78.0 [73.7, 81.8]** |
| **Δ (Headline − LLM-judgment)** | +8 | +8 | +6 | **+21** | **+10.8 pp** |

The comparison holds fixed: backbone (gpt-4o-mini), retrieval (BM25
TOP_K=10), chunking (fact-level), dataset, and n=100 per cell. The
two pipelines differ in **three coupled ways**: (1) the resolver
(LLM-judgment vs Python `max(serial)`), (2) the prompt and output
format (free-text answer following MAB's published BM25 prompt at
temperature 0.7 vs structured JSON candidate extraction at temperature
0.0), and (3) the LLM's task (decide-and-answer vs extract-candidates).

This comparison therefore measures the **whole pipeline-level effect**
of moving freshness reasoning out of LLM judgment and into structured
code, not the resolver step in isolation. An isolated
same-extraction-different-resolver ablation (LLM picks newest from
extracted candidates at temperature 0.0 vs Python `max(serial)`) would
attribute the +10.8 pp more precisely; we list this as a near-term
follow-up in §6.6.

The +10.8 pp pooled-average gap and the **widening of the gap at long
context** (+8 pp at 6K → +21 pp at 262K) point to two LLM failure
modes that the deterministic primitive eliminates:

1. **Prior-override.** When the question's subject is a real-world
   entity with a strong training-data prior (e.g., "ice hockey" as the
   national sport of Finland), and the counterfactual fact assigns a
   different value (e.g., "pesäpallo") with a higher serial, the LLM
   tends to output the prior despite the explicit "newer wins" rule
   in its prompt.
2. **Serial-comparison drift over many candidates.** As the candidate
   pool grows (longer context → more conflicting candidates retrieved),
   the LLM loses track of which serial is largest. This manifests as
   a 14-point drop in the LLM-judgment baseline from 75% at 64K to
   61% at 262K (§5.4). Deterministic `max()` is exact regardless of
   pool size.

Replacing LLM-based freshness reasoning with `max(serial)` over
extracted candidates eliminates both failure modes. The LLM's role
narrows to **semantic candidate extraction** — a task it does well —
and the freshness comparison is delegated to code, where there is no
possibility of training-data prior interference.

### §1.3 Three empirical findings

Building on this matched-setup comparison, three independent results
each beat the strongest published baseline by a clear margin (n=100
per cell × 4 context lengths each; 95% Wilson CIs):

| | Pipeline | Backbone | FC-SH avg [CI] | FC-MH avg [CI] | Vs best published |
|---|---|---|:-:|:-:|:-:|
| **Finding 1** | SH conflict + Python `max(serial)` | gpt-4o-mini | **78.0 [73.7, 81.8]** | — | +24 pp pooled-avg vs HippoRAG-v2 (54%); **+28 pp at matched 262K** |
| **Finding 2** | SH conflict + Python `max(serial)` | **gpt-4o** | **94.8 [92.1, 96.5]** | — | +34.8 pp pooled-avg vs GPT-4o long-context (60%); **+33 pp at matched 262K** |
| **Finding 3** | FC-MH CAR (Self-Ask + per-hop max) | gpt-4o-mini | — | **30.2 [26.0, 34.9]** | +23 pp pooled-avg vs best published (7%); **+20 pp at matched 262K** |

Total: 1600 evaluations across the master table (§5.1). Two
caveats on the headline comparisons:

- The MAB Table 3 single-number reports for each system appear to
  correspond to its 262K cell (confirmed via Table 10's context-length
  ablation, which shows GPT-4o = 60.0 at 262K). Our pooled averages
  include shorter contexts where some systems would score higher; at
  262K specifically the gaps are +28 pp (gpt-4o-mini at 82% vs
  HippoRAG-v2 at 54%) and +33 pp (gpt-4o at 93% vs GPT-4o long-context
  at 60%).
- The 6K context length is our own **diagnostic slice** constructed
  using MAB's MQUAKE edit-pair procedure; it is not part of MAB's main
  benchmark, which evaluates FactConsolidation at 32K/64K/262K. We
  include 6K in our master table for completeness but exclude it from
  apples-to-apples comparisons with published MAB numbers.

### §1.4 Why this matters

The agent-memory subfield has invested heavily in elaborate storage
architectures: knowledge graphs (Graphiti, GraphRAG, Cognee),
hippocampal-inspired retrieval (HippoRAG-v2), agentic loops (MemGPT,
MIRIX, Self-RAG), typed semantic memory (Memanto), and various hybrid
approaches. **On the conflict-resolution task, every one of these
architectures is matched or exceeded by a much simpler recipe**: BM25
retrieval, an LLM that extracts semantically matching candidates, and
a Python `max()` over their serial numbers. Even Zep / Graphiti — a
knowledge graph explicitly designed for temporal agent memory — scores
7.0% on FC-SH, lower than any other system in MAB Table 3. (We are
careful not to claim Zep is broadly weak: its reported strengths are
on DMR and LongMemEval. The result here is that *on FC-style conflict
resolution*, temporal-KG complexity does not help and may hurt.)

This points to a corrective for the memory-framework design space:
the bottleneck on conflict resolution is **assembly** (how facts are
identified, ranked, and combined to form an answer) — not **storage**
(graph, embedding, agentic, or typed). For tasks with explicit version
markers, the right primitive is **post-retrieval deterministic
aggregation**, not LLM judgment.

Our multi-hop finding — that the same architectural principle scales
to multi-hop chains via decomposition (CAR pipeline, 30.2% on FC-MH
vs published 7%) — supports this thesis: multi-hop conflict resolution
is tractable when the chain is decomposed properly and each hop's
conflict is resolved deterministically.

### §1.5 Contributions

1. **A matched-setup comparison quantifying the assembly bottleneck.**
   Same backbone, same retrieval, same chunking, same TOP_K, same n,
   swapping the LLM-judgment answer pipeline for a candidate-extraction
   + Python `max(serial)` pipeline: **+10.8 pp** average gain, growing
   to +21 pp at the longest context (262K). The comparison varies the
   prompt, output format, temperature, and resolver jointly; an
   isolated resolver-only ablation is future work.
2. **New SOTA on MAB FactConsolidation.** Single-hop FC-SH at 78.0%
   (gpt-4o-mini avg) and 94.8% (gpt-4o avg). Multi-hop FC-MH at
   30.2% via the CAR pipeline. At matched-backbone, matched-262K
   comparisons against MAB Table 3, the gpt-4o-mini pipeline beats
   HippoRAG-v2 by +28 pp and Zep by +75 pp on FC-SH; the multi-hop
   pipeline beats the best published FC-MH result (7%) by +20 pp at
   262K.
3. **Mechanistic evidence that LLM-based freshness reasoning degrades
   with candidate-pool size.** The LLM-judgment baseline holds 63–75%
   at 6K–64K but drops to 61% at 262K. Deterministic freshness does
   not degrade (71–82% across all lengths; 92–99% with gpt-4o backbone).
4. **A retrieval upper bound and method complementarity analysis.**
   88.5% of FC-SH questions are solved by *either* pipeline; 11.5%
   are solved by neither (a lower bound on retrieval ceiling). The
   methods are partially complementary: 21% of questions are solved
   only by Python `max(serial)`; 10.5% only by LLM-judgment. (§5.5.)
5. **Reproducibility.** Full open-source code (~50 lines of
   orchestration beyond standard BM25 retrieval and the
   candidate-extraction prompt), Langfuse traces for all 1600
   evaluations, and a single-command replication script.

### §1.6 Scope and honest limits

- We evaluate on **MAB FactConsolidation only** (1 benchmark, 2
  sub-tasks × 4 context lengths each, one of which we constructed
  ourselves as a 6K diagnostic slice). The MQUAKE-derived
  counterfactual setup may not capture all real-world conflict
  patterns.
- Our approach assumes **the source data has explicit version markers**
  (serial numbers, timestamps). When this assumption holds — as it
  typically does in production memory systems — our recipe is the
  right primitive.
- We tested **two backbones** (gpt-4o-mini and gpt-4o). The qualitative
  finding likely generalizes; quantitative numbers may shift on other
  model families.
- **The matched-setup comparison varies more than the resolver alone.**
  Our LLM-judgment baseline and Headline pipeline differ in prompt
  format, output format, and temperature (0.7 vs 0.0) in addition to
  the resolver. The +10.8 pp gap is therefore a *pipeline-level*
  effect, not a strictly isolated resolver effect. Future work (§6.6)
  runs the same-extraction-different-resolver ablation.
- **The matched comparison is at fact-level chunking only.** We did
  not run a matched LLM-judgment + chunk-4096 cell; the chunk-4096
  Headline result (80.8%) is included in our master table for
  completeness but is not directly compared against an LLM-judgment
  counterpart at chunk-4096.
- **Multi-hop on FC-MH is still substantially harder than single-hop**
  (30.2% vs 78%/94.8%). Our decomposition approach is preliminary.

The remainder of the paper documents the method (§3), experimental
setup (§4), main results (§5), and discussion (§6).

---

## §2 Related Work

We organize related work into three threads: (1) memory frameworks
evaluated on MemoryAgentBench, (2) multi-hop question decomposition,
and (3) temporal / freshness handling in retrieval-augmented systems.

### §2.1 Memory frameworks on MemoryAgentBench

MemoryAgentBench (Hu, Wang, & McAuley, 2026) is the most comprehensive
evaluation of LLM memory systems to date, covering four competencies
(Accurate Retrieval, Test-Time Learning, Long-Range Understanding,
Selective Forgetting) across 22 systems and 5 backbone models.
**FactConsolidation is MAB's conflict-resolution-style task under the
Selective Forgetting competency**: it introduces counterfactual edits
from MQUAKE (Zhong et al., 2023) and tests whether agents can
correctly prioritize later-added information over earlier-added.

**The benchmark exposes a striking weakness**: even the strongest
published RAG system (HippoRAG-v2, Gutiérrez et al., 2024) achieves
only 54.0% on single-hop FactConsolidation. Industry-standard memory
frameworks — **Mem0** (Chhikara et al., 2025) at 18.0%, **Graphiti /
Zep** (Rasmussen et al., 2025) at 7.0%, **MemGPT / Letta** (Packer et
al., 2023) at 28.0%, **MIRIX** at 14.0%, **Cognee** at 28.0% —
underperform a deterministic-freshness baseline by 50–87 percentage
points. The hierarchical RAG systems (RAPTOR, GraphRAG, MemoRAG)
cluster at 14–21%. The multi-hop variant (FC-MH) is essentially
unsolved: every system is at 0–7%. We compare against these published
numbers throughout §5.

### §2.2 Multi-hop QA via decomposition

Our multi-hop pipeline (Chain-Aware Retrieval, CAR) is built on the
well-established lineage of question decomposition. **Self-Ask**
(Press et al., 2022) showed that prompting an LLM to ask itself
sub-questions can improve multi-hop QA. **Decomposed Prompting** (Khot
et al., 2022) and **IRCoT** (Trivedi et al., 2023) interleave
decomposition with retrieval and chain-of-thought reasoning.
**Iter-RetGen** (Shao et al., 2023) alternates retrieval and
generation. **MultiHop-RAG** (Tang & Yang, 2024) provides a benchmark
and reference implementation for multi-hop RAG.

A recent paper by Madhwal et al. (2026) titled *"Decomposed Prompting
Does Not Fix Knowledge Gaps, But Helps Models Say 'I Don't Know'"* is
methodologically adjacent: it studies decomposition for multi-hop QA
and proposes abstention via cross-regime disagreement, mirroring our
finding that the decomposition pipeline naturally produces calibrated
"no answer" outputs when the chain breaks.

Our contribution relative to this lineage is not the decomposition
itself — we use a standard Self-Ask-style decomposer — but the
**per-hop deterministic freshness resolution** applied at each step
of the chain, which existing decomposition methods lack.

### §2.3 Temporal / freshness handling in retrieval

Several systems address temporal aspects of memory directly:

- **temporal-rag** (Emmimal, 2025), a small open-source Python
  library, adds a post-retrieval temporal layer with validity
  filtering, exponential time decay, and explicit supersedes-chains.
  It is explicitly **single-query** and does not handle multi-hop
  chains. Their README states *"Resolving disagreements between two
  current documents is the LLM's problem, not the retriever's"* —
  directly the design choice we contest.
- **"Solving Freshness in RAG"** (Grofsky, 2025) shows that a simple
  recency prior achieves 1.00 accuracy on freshness tasks in
  cybersecurity data, while a heuristic clustering trend detector
  fails (0.08 F1). Their setting is single-hop; we extend the
  principle to per-hop freshness in multi-hop chains.
- **Memanto** (Abtahi et al., 2026) explicitly argues that
  *"knowledge graph complexity is not necessary"* for high-fidelity
  agent memory, achieving SOTA on LongMemEval (89.8%) and LoCoMo
  (87.1%) with a typed semantic memory + information-theoretic
  retrieval engine. We extend this "simpler-beats-complex" thesis to
  FactConsolidation, where Memanto was not evaluated.
- **DYNAMICQA + MULAN reproducibility** (Dey et al., 2026), **MAGIC**
  (Lee et al., 2025), and **"QA under Temporal Conflict"** (Özer &
  Yıldız, 2025) all study LLMs' ability to resolve conflicting facts.
  They confirm that LLMs struggle to apply explicit in-context
  freshness rules, particularly when the rule conflicts with
  training-data priors.
- **TruthfulRAG** (Liu, Shang, & Zhang, 2025) and **Micro-Act** (Huo
  et al., 2025) propose LLM-based conflict-resolution mechanisms over
  knowledge graphs. Both rely on LLM judgment for the resolution
  step; both achieve modest results.

### §2.4 Knowledge editing and LLM priors

A separate body of work documents that **LLMs systematically override
in-context information that conflicts with their training-data
priors**. **"Uncovering Overfitting in LLM Editing"** (Zhang et al.,
2024), **"Unveiling Divergent Inductive Biases of LLMs on Temporal
Data"** (Kishore & He, 2024), and **"Adaptive Token Biaser"** (Bi et
al., 2024) all show that knowledge edits propagate beyond intended
scope, and that LLMs' temporal priors are hard to override via
prompting alone. Our finding — that LLMs cannot reliably apply the
explicit MAB freshness rule — is consistent with this literature and
motivates our move to deterministic post-processing.

### §2.5 Position of this work

Our specific contribution lives at the intersection of:
- Multi-hop QA decomposition (Self-Ask lineage) — established
- Temporal/freshness conflict resolution (temporal-rag, Memanto, Grofsky 2025) — established
- MemoryAgentBench evaluation harness — established

What we add:
1. **A matched-setup pipeline comparison** quantifying the assembly
   bottleneck: swapping an LLM-judgment answer pipeline for a
   candidate-extraction + Python `max(serial)` pipeline at fixed
   backbone, retrieval, and chunking gives **+10.8 pp** average,
   growing to +21 pp at 262K. (The comparison varies the resolver,
   prompt format, and temperature jointly; an isolated resolver
   ablation is future work — §6.6.)
2. **A specific empirical result** on FC-SH (78.0% gpt-4o-mini → 94.8%
   gpt-4o) and FC-MH (30.2% gpt-4o-mini) that beats every published
   system at matched-backbone, matched-262K comparison by at least
   +28 pp on FC-SH and +20 pp on FC-MH (see §5.2 for the full table).
3. **A position statement** that the bottleneck on conflict resolution
   is *assembly* (LLM judgment over retrieved candidates) rather than
   *storage* (graph, hippocampal, agentic, typed memory).

We deliberately do **not** claim novelty of the decomposition
mechanism, the freshness idea, or the use of BM25. The novelty is the
**specific combination + matched-comparison empirical evidence** on
the FactConsolidation benchmark.

---

## §3 Method

We describe two pipelines: **SH-conflict** for single-hop conflict
resolution, and **CAR** (Chain-Aware Retrieval) for multi-hop. Both
share three primitives: fact-level (or chunk-level) retrieval, LLM
candidate extraction, and deterministic freshness picking.

### §3.1 The SH-conflict pipeline

Given a corpus of numbered facts $\mathcal{C} = \{(s_i, t_i)\}_{i=1}^N$
where $s_i \in \mathbb{Z}$ is a version serial and $t_i$ is fact text,
and a query $q$, we compute:

1. **Retrieval**: $R = \text{BM25}(\mathcal{C}, q, k=10)$, returning
   top-$k$ facts ranked by BM25.
2. **Candidate extraction**: prompt an LLM with $q$ and the retrieved
   $R$ to produce $C = \{(s_j, t_j, e_j)\}$, where each $e_j$ is an
   extracted answer entity from a fact that semantically matches $q$.
   The LLM is explicitly instructed to include **all** matching items
   (do not compare serials, do not pick a "best") and to extract
   candidates verbatim.
3. **Freshness picking**: $\hat{c} = \arg\max_{c \in C} c.s$, then
   return $\hat{c}.e$ as the answer. If $C = \emptyset$, return
   "no answer".

The full SH-conflict pipeline is ≈50 lines of Python:

```python
def sh_conflict(question, corpus, bm25, llm):
    retrieved = bm25.retrieve(question, top_k=10)
    candidates = llm_extract(question, retrieved)  # JSON list of {serial, text, entity}
    if not candidates:
        return "no answer"
    return max(candidates, key=lambda c: c.serial).entity
```

The two contrasts with prior approaches:

- **Versus chunk-level retrieval** (MAB BM25 baseline, Mem0, Cognee,
  HippoRAG): we index each fact independently, preserving the serial
  number as the indexing key. Most prior systems chunk at 512 or 4096
  tokens, which packs many serials into one chunk and forces the LLM
  to do intra-chunk freshness reasoning. (MAB uses chunk-512 by
  default for FactConsolidation; see Hu et al. 2026 Appendix F.3.)
- **Versus LLM-judgment freshness** (BM25 baseline, all RAG-based
  memory systems, all agentic memory systems): we move the freshness
  comparison out of the LLM and into deterministic Python code. The
  LLM only extracts candidates; it does not pick the winner.

The candidate-extraction prompt is intentionally **strict**: the LLM
must match both subject and predicate verbatim, and must include every
matching item (with possibly conflicting answer entities). This pushes
the LLM toward conservative behavior — when no candidate matches, the
pipeline returns "no answer" rather than hallucinating. Empirically,
~2% of FC-SH questions hit this case (§6.1); we count "no answer" as
wrong under SubEM but in a production setting it is a calibrated
abstention.

### §3.2 The CAR pipeline (multi-hop)

For multi-hop questions of the form *"X of Y of Z"*, we extend the
single-hop pipeline with a Self-Ask-style decomposition step:

1. **Decomposition**: prompt an LLM with $q$ to produce a chain of
   atomic hops $h_1, \ldots, h_n$ where each $h_i$ asks about a single
   relationship. The output uses `{hop_k_answer}` placeholders to
   thread the chain.
2. **Per-hop execution**: for each $h_i$, substitute placeholder
   values from the chain, then run the SH-conflict pipeline on the
   resolved hop query. Take the result as $\text{hop}_i\_\text{answer}$.
3. **Final answer**: the last hop's answer is the answer to $q$. If
   any intermediate hop returns "no answer", subsequent hops that
   depend on it are skipped; the pipeline returns the last valid
   answer found, or "no answer" if the chain breaks at hop 1.

Worked example. For *"What is the country of citizenship of the
spouse of the author of 'American Pastoral'?"*, the decomposition
produces:
- $h_1$: Who is the author of 'American Pastoral'?
- $h_2$: Who is the spouse of `{hop_1_answer}`?
- $h_3$: What is the country of citizenship of `{hop_2_answer}`?
- $h_4$: What continent is `{hop_3_answer}` related to?

CAR pseudocode:

```python
def car(question, corpus, bm25, llm):
    hops = llm_decompose(question)              # ordered list of hop queries
    answers = {}                                 # hop_id -> answer string
    last_valid = None
    for h in hops:
        resolved_q = substitute(h.query, answers)
        if "{hop_" in resolved_q:                 # dependency missing
            break
        a = sh_conflict(resolved_q, corpus, bm25, llm)
        if a == "no answer":
            break
        answers[h.id] = a
        last_valid = a
    return last_valid if last_valid is not None else "no answer"
```

The decompose prompt uses a Graphiti-style format (system: minimal;
user: structured tags + worked example). We use a generic placeholder
example ("Alice supervises Bob who manages Carol") to avoid domain
leakage. The HARD_CONSTRAINT in the prompt forbids more than one
relationship word per hop, which empirically prevents gpt-4o-mini from
compressing 4-hop chains into 2-hop chains.

### §3.3 Why deterministic freshness works

The candidate-extraction step produces a small set of candidates
(typically 1–3, never more than the TOP_K=10 retrieved). Among these,
the freshness comparison is a single Python `max()` over integers —
an operation that is exact, fast, and incapable of LLM-style failures.

Concretely, two failure modes of LLM-judgment freshness disappear
under the deterministic primitive:

1. **Prior-override.** When the question's subject is a real-world
   entity with a strong training-data prior, and the counterfactual
   fact assigns a different value with a higher serial, the
   LLM-judgment baseline frequently outputs the prior despite the
   explicit "newer wins" rule in its prompt. The deterministic
   primitive never sees the entity text during comparison — only the
   integer serial — so the prior cannot override.
2. **Serial-comparison drift over many candidates.** When the
   retrieval pool contains many conflicting candidates (typical at
   262K context), the LLM-judgment baseline loses track of which
   serial is largest. This manifests in §5.4 as a 14-point drop from
   75% (64K) to 61% (262K). The deterministic primitive does not have
   this pathology: `max()` is exact regardless of pool size.

The candidate extraction itself is the LLM's job: deciding whether a
given fact semantically answers the question. LLMs do this well even
with weak backbones (gpt-4o-mini). What they do poorly is *comparing
serials across many candidates* — they get confused or override the
freshness rule with their priors. Our pipeline avoids asking the LLM
to do this.

### §3.4 Implementation

The complete pipeline code is at `scripts/_pipeline.py` (shared
primitives) and `scripts/13_paper_experiment.py` (SH-conflict driver) /
`scripts/14_ablations.py` (multi-experiment driver). The full prompts
(`CANDIDATE_PROMPT`, `DECOMP_PROMPT`, and the MAB-provided BM25 prompt
we use as the LLM-judgment baseline) are reproduced in Appendix A.

Total dependencies: `openai`, `rank_bm25`, `datasets`. No vector
database, no graph database, no embedding model.

---

## §4 Experimental Setup

### §4.1 Dataset

We evaluate on **MemoryAgentBench FactConsolidation** (Hu, Wang, &
McAuley, 2026), a conflict-resolution-style task under MAB's Selective
Forgetting competency. The benchmark publishes two sub-tasks
(single-hop and multi-hop) at three context lengths (32K, 64K, 262K
tokens). We additionally evaluate a **6K diagnostic slice** that we
constructed using the same MQUAKE edit-pair procedure; it is not part
of the MAB main benchmark and is clearly labeled as such throughout
§5. With our 6K addition we report 8 sub-task rows × 100 questions
each = **800 test instances** per pipeline.

FactConsolidation is constructed from MQUAKE (Zhong et al., 2023):
counterfactual rewrites of real-world facts. The original fact and
its counterfactual variant are concatenated in order so the
counterfactual appears with a higher serial number. Multiple pairs
are stacked to reach the target context length. The benchmark task is
to answer questions about the subjects in a way that prioritizes the
counterfactual (newer serial).

### §4.2 Backbone models

- **gpt-4o-mini** (`gpt-4o-mini-2024-07-18`): primary backbone for
  the headline experiment and most ablations.
- **gpt-4o** (`gpt-4o-2024-08-06`): backbone for Ablation C only.

All MAB-paper baselines we compare against use **gpt-4o-mini** as
their backbone (per MAB §4.1), so our comparison is apples-to-apples
on backbone for gpt-4o-mini results.

### §4.3 Pipelines evaluated

| Tag | Description | Chunking | Freshness logic | Backbone |
|---|---|---|---|---|
| **Headline** | SH-conflict pipeline (§3.1) | fact-level | Python `max(serial)` | gpt-4o-mini |
| **LLM-judgment baseline** | Same retrieval + LLM-based freshness reasoning (MAB-provided BM25 prompt) | fact-level | LLM judgment | gpt-4o-mini |
| **Ablation A** | SH-conflict pipeline (§3.1) | chunk-4096 | Python `max(serial)` | gpt-4o-mini |
| **Ablation B** | CAR pipeline (§3.2) | fact-level | Python `max(serial)` per hop | gpt-4o-mini |
| **Ablation C** | SH-conflict pipeline (§3.1) | fact-level | Python `max(serial)` | gpt-4o |

The LLM-judgment baseline uses MAB's published BM25 answer prompt,
while the Headline pipeline uses structured candidate extraction
followed by deterministic `max(serial)`. Both use identical fact-level
retrieval and TOP_K, but they differ in prompt format, output format,
temperature, and resolver. We therefore treat §5.3 as a **matched-setup
pipeline comparison** — not a resolver-only ablation — and flag the
isolated same-extraction LLM-vs-Python resolver ablation as future
work (§6.6).

### §4.4 Hyperparameters

- **TOP_K = 10** (number of retrieved facts per query). Matches MAB.
- **Temperature**: 0.0 for the deterministic-freshness pipelines; 0.7
  for the LLM-judgment baseline (matching MAB's reported default for
  the BM25 baseline).
- **Chunking strategies**:
  - **fact-level**: parse the context with regex
    `(\d+)\.\s+(.+?)(?=\d+\.\s|$)`, one fact per chunk. Typically
    yields 500–18,000 chunks depending on context length.
  - **chunk-4096**: sliding-window chunker, 4096 characters per chunk.
    ~6 chunks for 6K context, ~250 chunks for 262K context. (Note:
    MAB itself uses chunk-512 by default for FactConsolidation, per
    Appendix F.3 of the MAB paper; we chose chunk-4096 to match what
    several memory frameworks like Mem0/Cognee/Zep/MIRIX use.)
- **BM25**: `rank_bm25.BM25Okapi` with default parameters (k1=1.5,
  b=0.75).
- **Tokenizer for BM25**: lowercased alphanumeric tokens via the regex
  `[A-Za-z0-9]+`.

### §4.5 Metric

We use **SubEM** (substring exact match), the metric used in the MAB
paper. For each predicted answer $\hat{y}$ and ground truth list $Y$,
the answer is **correct** if any element of $Y$ appears as a substring
(case-insensitive) in $\hat{y}$. This is the standard short-answer-QA
metric.

**SubEM caveat.** Because SubEM credits substring matches, a verbose
answer that incidentally contains the ground-truth string is counted
correct. For our pipelines this is a non-issue (outputs are short
entities pulled from the extracted candidate), but it inflates
long-context oracle baselines somewhat. We report numbers under the
same metric as MAB to keep comparisons apples-to-apples.

### §4.6 Compute and reproducibility

- **Total LLM-driven evaluations**: 1600 (16 cells × 100 questions).
- **Total API cost**: ~$3 across the full benchmark (dominated by the
  4 gpt-4o cells in Ablation C at ~$2; gpt-4o-mini cells contribute
  ~$0.3 across the headline, LLM-judgment baseline, chunk-4096
  ablation, and CAR multi-hop runs).
- **Wall-clock**: each cell ~5–10 min; all 16 cells run in parallel
  ~15 min total.
- **Reproducibility**: full code, prompts, and 1600+ Langfuse traces
  (with per-question metadata: experiment, competency, dataset,
  question_index, ground_truth) will be released. The headline
  experiment is one shell command:
  `python scripts/13_paper_experiment.py --source factconsolidation_sh_<L>`.

### §4.7 Baselines

We compare against the 22 systems reported in Table 3 of MAB v3 (Hu,
Wang, & McAuley, 2026), using **the numbers as published** (we do
not re-run the published systems to avoid implementation differences).
MAB's Table 3 reports a single FC-SH and FC-MH number per system; we
verified via MAB's Table 10 (context-length ablation) that the FC
columns in Table 3 correspond to the 262K context cell. The systems
span four families: long-context FIFO baselines, simple RAG (BM25),
embedding RAG (Contriever, Text-Embed-3, Qwen3-Embedding),
structure-augmented RAG (RAPTOR, GraphRAG, MemoRAG, HippoRAG-v2,
Mem0, Cognee, Zep), and agentic memory (Self-RAG, MemGPT, MIRIX).
Detailed comparison is in §5.2.

---

## §5 Results

We organize results into four parts: §5.1 the master table (all 16
cells with 95% Wilson CIs), §5.2 comparison to published systems at
the matched (262K) context, §5.3 the matched-setup pipeline
comparison, and §5.4 robustness to context length.

### §5.1 Master results table

All numbers are accuracy (%) on $n=100$ questions per cell, with 95%
Wilson confidence intervals. The AVG column uses the pooled n=400 (all
four lengths concatenated). The 6K column is our own diagnostic slice
(see §4.1); 32K/64K/262K are MAB-published. Bold marks the best result
in each row; ⭐ marks the three headline findings.

| Pipeline | Backbone | 6K† | 32K | 64K | 262K | **AVG [95% CI]** |
|---|---|:-:|:-:|:-:|:-:|:-:|
| **Headline** (SH fact + Python max) | gpt-4o-mini | 71 [61, 79] | 78 [69, 85] | 81 [72, 87] | 82 [73, 88] | **78.0 [73.7, 81.8]** |
| LLM-judgment baseline (SH fact + LLM-fresh) | gpt-4o-mini | 63 [53, 72] | 70 [60, 78] | 75 [66, 82] | 61 [51, 70] | 67.2 [62.5, 71.7] |
| **Ablation A** (SH chunk4096 + Python max) | gpt-4o-mini | **87 [79, 92]** | **84 [76, 90]** | 79 [70, 86] | 73 [64, 81] | **80.8 [76.6, 84.3]** |
| **Ablation B** (FC-MH CAR pipeline) | gpt-4o-mini | 34 [25, 44] | 27 [19, 36] | 33 [25, 43] | 27 [19, 36] | **30.2 [26.0, 34.9]** ⭐ |
| **Ablation C** (SH fact + Python max) | **gpt-4o** | **99 [95, 100]** | **92 [85, 96]** | **95 [89, 98]** | **93 [86, 97]** | **94.8 [92.1, 96.5]** ⭐⭐ |

†6K is a diagnostic slice we constructed; not part of MAB's published
benchmark. See §4.1.

**Three headline findings**:

1. **Finding 1 (SH on gpt-4o-mini)**: Our pipeline (Headline) achieves
   78.0% average on FC-SH, beating the best published gpt-4o-mini
   system (HippoRAG-v2 at 54%) by 24 percentage points. At 262K
   specifically, the gap is +28 pp (82% vs 54%).
2. **Finding 2 (SH on gpt-4o)**: With gpt-4o backbone, the same
   pipeline achieves 94.8% average on FC-SH. At 262K specifically,
   the gap to GPT-4o long-context (60%) is **+33 percentage points**
   — same model, different post-processing.
3. **Finding 3 (MH)**: The CAR pipeline (Ablation B) achieves 30.2%
   average on FC-MH, **23 percentage points above the best published
   system** (Contriever / MemoRAG at 7%).

### §5.2 Comparison to published systems

Numbers below from MAB v3 Table 3 (Hu, Wang, & McAuley, 2026), all
with gpt-4o-mini as backbone except long-context entries which use
the named model. Table 3's single FC number per system corresponds to
the 262K context cell (verified against MAB Table 10).

**For apples-to-apples comparison, we report our 262K column** below
(not our pooled average across 4 lengths).

| System (architecture) | FC-SH @262K | FC-MH @262K | Gap vs ours best |
|---|:-:|:-:|---|
| **Ours (SH fact + Python max + gpt-4o)** | **93.0** | — | — |
| **Ours (SH fact + Python max + gpt-4o-mini)** | **82.0** | — | — |
| **Ours (SH chunk4096 + Python max + gpt-4o-mini)** | 73.0 | — | — |
| **Ours (FC-MH CAR + gpt-4o-mini)** | — | **27.0** | — |
| GPT-4o (long-context) | 60.0 | 5.0 | -33.0 / -22.0 |
| HippoRAG-v2 (hippocampal retrieval) | 54.0 | 5.0 | -39.0 / -22.0 |
| **BM25 (simple lexical RAG, MAB chunk-512 default)** | **48.0** | **3.0** | **-45.0 / -24.0** |
| GPT-4o-mini (long-context FIFO) | 45.0 | 5.0 | -48.0 / -22.0 |
| Claude-3.7-Sonnet (long-context) | 43.0 | 2.0 | -50.0 / -25.0 |
| GPT-4.1-mini (long-context) | 36.0 | 5.0 | -57.0 / -22.0 |
| Gemini-2.0-Flash (long-context) | 30.0 | 3.0 | -63.0 / -24.0 |
| Qwen3-Embedding-4B (embedding RAG) | 29.0 | 3.0 | -64.0 / -24.0 |
| Cognee (knowledge graph) | 28.0 | 3.0 | -65.0 / -24.0 |
| MemGPT (multi-tier memory) | 28.0 | 3.0 | -65.0 / -24.0 |
| Text-Embed-3-Large (embedding RAG) | 28.0 | 4.0 | -65.0 / -23.0 |
| Text-Embed-3-Small (embedding RAG) | 28.0 | 3.0 | -65.0 / -24.0 |
| MemoRAG | 21.0 | 7.0 | -72.0 / -20.0 |
| Self-RAG (self-reflective) | 19.0 | 3.0 | -74.0 / -24.0 |
| **Contriever (embedding RAG)** | **18.0** | 7.0 | -75.0 / -20.0 |
| **Mem0 (graph + summary)** | **18.0** | 2.0 | -75.0 / -25.0 |
| RAPTOR (hierarchical) | 14.0 | 1.0 | -79.0 / -26.0 |
| GraphRAG (Microsoft) | 14.0 | 2.0 | -79.0 / -25.0 |
| MIRIX | 14.0 | 2.0 | -79.0 / -25.0 |
| **Zep / Graphiti (temporal KG)** | **7.0** | **3.0** | **-86.0 / -24.0** |

At the matched 262K comparison, our pipelines outperform every
published MAB Table 3 system on FC-SH and FC-MH. The gap is largest
against the most architecturally elaborate systems: **Zep / Graphiti
(temporal knowledge graph) is 86 percentage points below our gpt-4o
result on FC-SH at 262K**. Even the simplest published baseline
(BM25 at 48%) is 45 percentage points below our gpt-4o result.

### §5.3 Matched-setup pipeline comparison

We compare two pipelines at fixed backbone (gpt-4o-mini), fixed
retrieval (BM25 TOP_K=10), fixed chunking (fact-level), fixed dataset,
and matched n=100 per cell:

| Pipeline (fact-level chunking, gpt-4o-mini) | 6K | 32K | 64K | 262K | AVG [95% CI] |
|---|:-:|:-:|:-:|:-:|:-:|
| LLM-judgment answer pipeline (MAB BM25 prompt, T=0.7) | 63 | 70 | 75 | 61 | 67.2 [62.5, 71.7] |
| Candidate extraction + Python `max(serial)` (Headline, T=0.0) | 71 | 78 | 81 | 82 | **78.0 [73.7, 81.8]** |
| **Δ (Headline − LLM-judgment)** | **+8** | **+8** | **+6** | **+21** | **+10.8 pp** |

**The +10.8 pp pooled-average gap is statistically reliable**
(non-overlapping 95% CIs at the marginal level; paired McNemar test
$\chi^2 = 14.6$, $p < 0.001$ — see §5.5), and the **widening of the
gap at long contexts** (+21 pp at 262K vs +8 pp at 6K) supports the
candidate-pool-size hypothesis: LLM-judgment degrades as
candidate-pool size grows, while the deterministic resolver does not.

**Caveat — what this comparison does and does not isolate.** The two
pipelines differ in three coupled ways: (1) the freshness resolver
(LLM judgment vs Python `max(serial)`), (2) the prompt and output
format (free-text answer following MAB's BM25 prompt at temperature
0.7 vs structured JSON candidate-extraction at temperature 0.0), and
(3) the LLM's task (decide-and-answer vs extract-candidates). The
+10.8 pp gap is therefore a *whole-pipeline* effect. To attribute it
strictly to the resolver, one would need an additional cell — same
candidate-extraction prompt, same temperature, but LLM-picks-newest
instead of Python-max — which we list as a near-term experiment in
§6.6.

**A note on chunking.** We also vary chunking (fact-level vs
chunk-4096) while holding the deterministic-freshness pipeline fixed,
to quantify the contribution of chunking strategy alone:

| | 6K | 32K | 64K | 262K | AVG |
|---|:-:|:-:|:-:|:-:|:-:|
| chunk-4096 (Ablation A) | 87 | 84 | 79 | 73 | 80.8 |
| fact-level (Headline) | 71 | 78 | 81 | 82 | 78.0 |
| Δ from per-fact chunking | -16 | -6 | +2 | **+9** | **-2.8** |

On average the two chunking strategies are essentially tied (78.0 vs
80.8 — well within their 95% CIs), but the curves are opposite:
chunk-4096 wins at short contexts; fact-level wins at the longest
context (262K). We do not have a matched LLM-judgment + chunk-4096
cell, so we cannot make a matched comparison at chunk-4096; that
experiment is listed as future work in §6.6.

### §5.4 Robustness to context length

The LLM-judgment baseline degrades sharply at 262K (61%, down from
75% at 64K — a 14-point drop). Our deterministic pipeline does not:
71–82% across all lengths with gpt-4o-mini, 92–99% with gpt-4o. The
gap **widens** with context length: +8 pp at 6K → +21 pp at 262K
(Headline vs LLM-judgment baseline).

This is consistent with the hypothesis that the LLM's freshness
judgment **fails when given many candidates** with similar surface
form. At short contexts, fewer conflicting facts get retrieved and
the LLM can track serials. At long contexts, the candidate pool grows
(more retrieved facts across more conflicting pairs) and the LLM
loses track. Python `max()` does not have this pathology.

The same robustness extends across pipelines:

| Pipeline | Min | Max | Range |
|---|:-:|:-:|:-:|
| LLM-judgment baseline | 61 (262K) | 75 (64K) | 14 pp drop at 262K |
| Headline (gpt-4o-mini, fact) | 71 (6K) | 82 (262K) | 11 pp, monotonically up |
| Ablation A (gpt-4o-mini, chunk4096) | 73 (262K) | 87 (6K) | 14 pp, declining |
| Ablation C (gpt-4o, fact) | 92 (32K) | 99 (6K) | 7 pp, stable |

The Ablation C pipeline (gpt-4o + deterministic freshness + fact
chunking) is the most context-robust: only a 7-pp range across
6K–262K, vs the LLM-judgment baseline's 14-pp drop concentrated at
the long extreme.

### §5.5 Retrieval upper bound and method complementarity

To bound how much of the gpt-4o-mini ceiling is set by retrieval (BM25
TOP_K=10) versus by the freshness pipeline, we compute the
**union accuracy** of the LLM-judgment baseline and the Headline
pipeline on the same questions. The union is a lower bound on the
retrieval ceiling: if neither pipeline got the question right, the
correct answer either was not in TOP_K, or both extraction methods
failed to surface it.

| Context | LLM-judgment | Headline (Python max) | **Union (either correct)** | Neither |
|---|:-:|:-:|:-:|:-:|
| 6K | 63 | 71 | **85** | 15 |
| 32K | 70 | 78 | **87** | 13 |
| 64K | 75 | 81 | **93** | 7 |
| 262K | 61 | 82 | **89** | 11 |
| **Pooled (n=400)** | **67.2** | **78.0** | **88.5** | **11.5** |

Two takeaways. First, **88.5% of questions are solvable by at least
one of the two pipelines**, which means retrieval is recovering the
correct answer fact for the great majority of FC-SH questions; the
remaining 11.5% is a lower bound on retrieval failure (questions
where neither pipeline succeeds because BM25 TOP_K=10 missed the
counterfactual entirely, or the extraction prompt rejected it). This
gives a soft ceiling of ~88–95% for any fixed-TOP_K pipeline before
retrieval becomes the bottleneck rather than freshness reasoning.

Second, **the two pipelines are partially complementary**, not strictly
ordered. Pooled across n=400:
- Both correct: 227 (56.8%)
- Python `max(serial)` only correct: 85 (21.3%)
- LLM-judgment only correct: 42 (10.5%)
- Neither correct: 46 (11.5%)

The +10.8 pp headline gap is the *net* of the +21 pp Python-only wins
and the −11 pp LLM-judgment-only wins. The LLM-judgment baseline is
not strictly dominated by the deterministic pipeline at the
question level; it has its own ~10% of correct answers that the
extraction-strict pipeline misses (typically when the
candidate-extraction prompt over-rejects valid candidates due to
predicate-strictness). This nuance is missed by reporting only
pooled accuracy.

**Paired statistical test.** The contingency above (Python-only 85,
LLM-only 42) yields McNemar's $\chi^2 = (85-42)^2 / (85+42) = 14.6$,
$p < 0.001$ with one degree of freedom, confirming the Headline
pipeline's advantage as statistically reliable on paired questions
— not just non-overlapping marginal CIs.

### §5.6 FC-MH results and failure mode analysis

The CAR pipeline (Ablation B) succeeds at 27–34% on FC-MH across all
four context lengths, with 6K and 64K marginally higher (34%, 33%)
than 32K and 262K (27%). The variance is consistent with the small
per-cell n=100 sample (Wilson 95% CI widths ~17 pp).

**Per-hop analysis.** CAR plans an average of 2.7 hops per question
across all FC-MH questions, with a maximum of 6 hops. Of the planned
hops, **86% are successfully executed** (return a non-null answer).
The remaining 14% of hop failures (typically hop 1 returning "no
answer" due to no candidate match) cascade to whole-pipeline failure,
explaining a substantial fraction of the 70% wrong answers; the
remainder come from hops that complete but extract the wrong
candidate — most commonly counterfactual mis-identification at hop 1
or predicate-mismatch extraction at later hops.

**Failure-mode breakdown.** From spot-checking ~30 wrong answers
across the four FC-SH length runs at 262K:

| Category | % of errors | Example |
|---|:-:|---|
| Wrong subject confusion | ~25% | Similar-name entity ranks higher in BM25 |
| Predicate semantic gap | ~25% | Question asks "spouse"; retrieved fact says "partner" |
| Conservative empty extraction | ~10% | LLM rejects valid candidate due to strict predicate match |
| Counterfactual not in TOP_K | ~10% | Real-world facts dominate retrieval |
| Real-world prior wins via plurality | ~10% | 8 of 10 retrieved facts are real-world; LLM extracts those |
| Ambiguous serial tie | ~5% | Two facts with same serial |
| Other / unclassified | ~15% | residual; not categorized in this pilot |

The most common CAR failure mode is **hop-1 hallucination**: when the
question's first entity has both a real-world fact and a
counterfactual fact, the LLM occasionally picks the entity that does
not match the question's intent, and the chain breaks. Per-hop
deterministic freshness correctly picks the newer serial when
candidates exist, but cannot recover if the LLM extracted the wrong
candidate to begin with.

### §5.7 Cost-quality tradeoff

| Pipeline | Cost per query (~estimate) | FC-SH avg |
|---|:-:|:-:|
| LLM-judgment baseline | $0.0001 | 67.2% |
| Headline (gpt-4o-mini) | $0.0001 | 78.0% |
| Ablation A (chunk4096) | $0.0001 | 80.8% |
| Ablation B (FC-MH CAR) | $0.0003 (multiple LLM calls) | 30.2% (FC-MH) |
| Ablation C (gpt-4o) | $0.005 (50× cost) | 94.8% |

**The gpt-4o-mini pipeline is essentially free** — $0.0001 per query.
For applications where 78–80% accuracy is sufficient, this is the
optimal choice. The gpt-4o pipeline at $0.005/query provides 94.8% —
a 17 pp gain at 50× the cost. The decision depends on the
application's accuracy budget.

---

## §6 Discussion

### §6.1 Why does this work?

The mechanism is straightforward when articulated:

**LLMs are unreliable at applying explicit in-context rules that
conflict with their training-data priors.** This is documented in the
knowledge-editing literature (Zhang et al., 2024; Kishore & He, 2024).
When prompted with *"facts are indexed by serial numbers; newer
serials override older"* and shown a counterfactual fact (higher
serial) alongside a real-world fact (lower serial), the LLM tends to
override the explicit rule with its prior. Worse, this failure scales
with **candidate-pool size** — more conflicting candidates in context
degrades accuracy more — which explains the LLM-judgment baseline's
sharp drop at 262K (61%, down from 75% at 64K).

**Python `max()` over extracted candidates has no such pathology.**
It is exact, fast, and context-independent. The LLM's role is
narrowed to **candidate extraction** (identifying which retrieved
facts semantically answer the question), where it does well.

**The candidate extraction is intentionally strict**: the LLM is
instructed to match both subject and predicate verbatim, and to
include every matching item — not pick the "best." This pushes the
LLM toward conservative behavior. When no candidate matches, the
pipeline returns "no answer" rather than hallucinating. Empirically,
~2% of FC-SH questions hit this case; we count "no answer" as wrong
by the SubEM metric, but in a production setting it is a calibrated
abstention.

**A precision-recall trade-off, not strict dominance.** §5.5 makes
the picture richer than "LLM bad, code good." The deterministic
pipeline trades a small amount of recall — the ~10% of questions
where the strict extraction prompt over-rejects valid candidates,
which the LLM-judgment baseline still answers correctly — for a
larger gain in precision (no prior-override, no serial-tracking
drift). The net effect is +10.8 pp, but the two pipelines are
**partially complementary** at the question level (21% Python-only,
10.5% LLM-only). A hybrid pipeline that falls back to LLM-judgment
when candidate extraction returns empty could plausibly close some
of the residual 10.5% gap; we did not implement this hybrid but it
is a natural extension.

### §6.2 Implications for memory framework design

The agent-memory subfield has invested significant engineering in
elaborate storage architectures. **On this benchmark, architectural
elaboration alone does not appear to solve conflict resolution**:

- **Knowledge graphs** (GraphRAG, Cognee, Graphiti): require
  LLM-driven entity/relation extraction at ingestion, which propagates
  noise and introduces additional LLM-judgment steps. On FC-SH these
  score 7–28%.
- **Hippocampal retrieval** (HippoRAG-v2): the strongest published
  RAG at 54% on FC-SH. The architectural sophistication (multi-stage
  retrieval, iterative refinement) does not address the LLM-judgment
  bottleneck at query time.
- **Agentic loops** (MemGPT/Letta, MIRIX, Self-RAG): the iteration
  helps retrieval coverage but adds more LLM steps without making any
  of them freshness-aware. 14–28% on FC-SH.
- **Temporal knowledge graphs** (Zep / Graphiti, bi-temporal edges,
  validity intervals): explicitly designed to handle supersession.
  Yet Zep scores 7% on FC-SH — its lowest column in MAB Table 3. We
  emphasize that this is a *per-task* result on FC-style conflict
  resolution and does not contradict Zep's reported strengths on other
  benchmarks (DMR, LongMemEval). The plausible explanation for the FC
  result is that Zep uses an LLM at ingestion time to decide whether
  new facts contradict existing ones (`resolve_edge` prompt),
  inheriting the same LLM-judgment problem we eliminate. The lesson is
  narrow: even temporal-KG infrastructure does not automatically solve
  conflict resolution if the resolution decisions are still made by
  LLM prompts.

The right primitive for conflict resolution is deterministic
aggregation over candidates that the LLM has identified as
semantically relevant. This matches the temporal-rag (Emmimal, 2025)
library's design — though temporal-rag is single-hop only; our CAR
pipeline extends to multi-hop.

This view aligns with Memanto (Abtahi et al., 2026), which makes a
similar "simpler beats complex" argument on different benchmarks
(LongMemEval, LoCoMo) using information-theoretic retrieval + typed
memory. We extend the principle to FactConsolidation.

### §6.3 Why does CAR work on FC-MH?

Multi-hop conflict resolution combines two challenges: (1)
decomposing the question into atomic hops, and (2) resolving
conflicts at each hop. Our CAR pipeline addresses each separately.
The decomposition uses a Self-Ask-style prompt with a HARD_CONSTRAINT
that forbids more than one relationship word per hop (forcing the LLM
to recursively split *"X of Y of Z"* into multiple hops). The
per-hop conflict resolution uses the same SH-conflict primitive.

Existing systems either skip decomposition (BM25, embedding RAG, all
memory frameworks) or skip per-hop conflict resolution (Self-Ask,
IRCoT, MultiHop-RAG benchmark RAG). CAR does both. The result (30.2%
avg vs 7% best published) demonstrates that **multi-hop conflict
resolution becomes tractable when both components are addressed**.

That said, 30.2% is far from solved. Most CAR failures are at hop 1:
the LLM occasionally extracts the wrong candidate when both
real-world and counterfactual facts appear in retrieval. The error
then cascades through subsequent hops. Improving hop-1 reliability
is the most promising avenue for further work.

### §6.4 Limitations

**One benchmark.** All results are on MAB FactConsolidation. The
MQUAKE-derived counterfactuals may not reflect real-world conflict
patterns. For example, real production memory data often has
timestamps rather than ordinal serial numbers; partial-order updates
(e.g., merged document revisions) are not captured.

**The 6K column is our own diagnostic slice.** MAB publishes
FactConsolidation at 32K, 64K, and 262K; we constructed a 6K slice
using the same MQUAKE edit-pair procedure. We report it for
completeness in the master table but exclude it from apples-to-apples
comparisons against published MAB numbers.

**Version-marker assumption.** Our deterministic `max(serial)`
requires the source data to have a total ordering on facts.
FactConsolidation provides this via numbered facts; production memory
systems typically have it via timestamps or insertion order. The
approach generalizes to any total order but cannot handle partial
orders or causal dependencies between updates.

**The deterministic pipeline is not strictly dominant.** §5.5 shows
10.5% of FC-SH questions are answered correctly by the LLM-judgment
baseline but missed by the Headline pipeline (vs 21.3% the other
way). The deterministic resolver wins on net but trades a small
amount of recall (strict extraction over-rejection) for a larger
gain in precision (no prior-override, no serial-tracking drift).
Practitioners adopting this recipe should be aware of the trade-off
and may benefit from the hybrid extension described in §6.1.

**The matched-setup comparison varies more than the resolver alone.**
Our LLM-judgment baseline uses MAB's published BM25 answer prompt at
temperature 0.7, while the Headline pipeline uses a candidate-extraction
prompt at temperature 0.0 with a deterministic Python resolver. The
+10.8 pp gap is therefore a *pipeline-level* effect (resolver + prompt
+ temperature jointly), not a strictly isolated resolver effect. An
isolated same-extraction-different-resolver ablation (LLM picks newest
from extracted candidates at T=0.0 vs Python max at T=0.0) is listed
as the most important near-term follow-up in §6.6.

**The matched comparison is at fact-level chunking only.** We have not
run a matched LLM-judgment + chunk-4096 cell, so we cannot make a
matched comparison about freshness at chunk-4096. Our chunk-4096
result (Ablation A at 80.8%) is included for completeness.

**Failure-mode analysis is qualitative.** Our §5.6 failure-mode
breakdown is from spot-checking ~30 wrong answers across the four
FC-SH length runs at 262K, not a systematic 100-question manual
labeling. The categories and proportions should be read as a
qualitative pilot, not a precise distribution.

**Two backbones tested.** Results may differ on other model families
(Claude, Llama, open-source models). The qualitative finding — that
LLMs are unreliable at in-context freshness reasoning — likely
generalizes, but the quantitative numbers may shift.

**Multi-hop remains genuinely hard.** 30.2% on FC-MH, while a 4–5×
improvement over published, is still far from solved. The
decomposition step has its own LLM-judgment failures (hidden-bridge
questions, hop-1 hallucination) that our deterministic freshness does
not address.

### §6.5 Recommendations for memory framework designers

1. **Preserve fact-level metadata at indexing time.** Whatever
   chunking strategy you use, ensure the version markers (timestamps,
   serial numbers, version strings) are available at the granularity
   at which conflicts can occur.
2. **Move freshness/recency comparisons out of LLM prompts and into
   deterministic code.** The LLM is the wrong primitive for this
   comparison.
3. **Do not assume graph or knowledge-graph infrastructure alone
   solves conflict resolution.** The empirical evidence is that an
   elaborate temporal KG (Zep / Graphiti) scores 7% on FC-SH despite
   being designed for temporal memory. Graph infrastructure may be
   appropriate for other reasons; conflict resolution is not a
   sufficient reason on its own without an explicit
   version-aggregation step.
4. **Test on FactConsolidation early.** It is a clean stress test for
   conflict resolution that production memory systems should pass.
   MAB provides it as a Python-installable benchmark.
5. **Be honest about which sub-tasks your system handles.** A memory
   framework that gets 18% on FC-SH should not claim to handle
   "evolving knowledge" — that result is well below a deterministic
   baseline.

### §6.6 Open questions and future work

The cleanest open question is whether the same recipe transfers to
**real-world conflict patterns** — e.g., user preference updates in
conversational memory, policy supersession in compliance documents,
fact corrections in collaborative knowledge bases. The mechanism
(LLMs fail at explicit freshness rules; deterministic max wins) is
general, but the empirical validation is benchmark-specific. Four
concrete near-term experiments would substantially strengthen the
claims:

1. **Isolated resolver ablation (highest priority).** The +10.8 pp
   matched comparison in §5.3 varies the resolver, the prompt, and the
   temperature jointly. To isolate the resolver step alone, run a
   third cell: same candidate-extraction prompt and same temperature
   (0.0) as the Headline pipeline, but with the LLM picking the newest
   candidate (from the same extracted list) instead of Python max.
   This attributes the +10.8 pp gap precisely between
   prompt-and-extraction effects and the resolver step itself.
2. **Matched chunk-4096 + LLM-judgment cell.** Currently the matched
   comparison is at fact-level only. Running BM25 + chunk-4096 +
   LLM-judgment at all four context lengths would let us make the
   same matched comparison at chunk-4096 too.
3. **Timestamp-based conflict resolution.** Replace MQUAKE's ordinal
   serials with real timestamps on a conversational-memory corpus and
   verify that `max(timestamp)` still wins. This tests whether the
   principle generalizes beyond ordinal version markers.
4. **LongMemEval / LoCoMo freshness-relevant slices.** Run the same
   pipeline on freshness-relevant subsets of LongMemEval and LoCoMo to
   confirm the principle generalizes beyond MQUAKE.

---

## §7 Conclusion

We presented a deterministic recipe for memory conflict resolution
that achieves new state-of-the-art on MemoryAgentBench
FactConsolidation: **78–94.8% on single-hop** (vs HippoRAG-v2's 54%
and GPT-4o long-context's 60% at 262K) and **30.2% on multi-hop**
(vs the best published 7%). A matched-setup comparison — same
backbone, same retrieval, same chunking, same TOP_K, n=100 per cell
— shows **+10.8 percentage points** from replacing an
LLM-judgment-based answer pipeline with a candidate-extraction +
Python `max(serial)` pipeline, with the gap widening to +21 pp at the
longest published context (262K). This comparison varies the resolver,
the prompt, and the temperature jointly; the isolated resolver
ablation is listed as the most important near-term follow-up.

The implication is corrective for the memory-framework subfield: the
bottleneck on conflict resolution is **assembly** (LLM judgment over
retrieved candidates), not **storage** (graph, hippocampal, agentic,
or typed). The right primitive for resolution is deterministic
post-retrieval aggregation. Elaborate memory architectures do not
help on this task; a short Python function over LLM-extracted
candidates matches or exceeds them.

This points to a broader design pattern for the next generation of
memory systems: **narrow the LLM's job to tasks it is reliably good
at (semantic identification, candidate extraction) and delegate
comparisons over structured metadata (serials, timestamps, version
markers) to deterministic code**. We release all code, prompts, and
1600+ Langfuse traces for reproducibility; the headline experiment
runs in 15 minutes for ~$3 in API spend.

---

## References

Abtahi, S. M., Rahnema, R., Patel, H., Patel, N., Fekri, M., & Khani,
T. (2026). Memanto: Typed Semantic Memory with Information-Theoretic
Retrieval for Long-Horizon Agents. *arXiv preprint arXiv:2604.22085.*

Bi, B., Liu, S., Wang, Y., Mei, L., Gao, H., Xu, Y., & Cheng, X.
(2024). Adaptive Token Biaser: Knowledge Editing via Biasing Key
Entities. *arXiv preprint arXiv:2406.12468.*

Chhikara, P., Khant, D., Aryan, S., Singh, T., & Yadav, D. (2025).
Mem0: Building Production-Ready AI Agents with Scalable Long-Term
Memory. *arXiv preprint arXiv:2504.19413.*

Dey, R., Ounis, I., McDonald, G., & Moshfeghi, Y. (2026). Temporal
Fact Conflicts in LLMs: Reproducibility Insights from Unifying
DYNAMICQA and MULAN. *arXiv preprint arXiv:2603.15892.*

Emmimal. (2025). temporal-rag: A post-retrieval temporal layer for
RAG systems. *GitHub repository,*
https://github.com/Emmimal/temporal-rag.

Grofsky, M. (2025). Solving Freshness in RAG: A Simple Recency Prior
and the Limits of Heuristic Trend Detection. *arXiv preprint
arXiv:2509.19376.*

Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., & Su, Y. (2024).
HippoRAG: Neurobiologically Inspired Long-Term Memory for Large
Language Models. *Advances in Neural Information Processing Systems
(NeurIPS).* arXiv:2405.14831.

Hu, Y., Wang, Y., & McAuley, J. (2026). Evaluating Memory in LLM
Agents via Incremental Multi-Turn Interactions. *International
Conference on Learning Representations (ICLR).* arXiv:2507.05257.

Huo, N., Li, J., Qin, B., Qu, G., Li, X., Li, X., & Ma, C. (2025).
Micro-Act: Mitigating Knowledge Conflict in LLM-based RAG via
Actionable Self-Reasoning. *arXiv preprint arXiv:2506.05278.*

Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K.,
Vardhamanan, S., Haq, S., Sharma, A., Joshi, T. T., Moazam, H.,
Miller, H., Zaharia, M., & Potts, C. (2023). DSPy: Compiling
Declarative Language Model Calls into Self-Improving Pipelines.
*International Conference on Learning Representations (ICLR 2024).*
arXiv:2310.03714.

Khot, T., Trivedi, H., Finlayson, M., Fu, Y., Richardson, K., Clark,
P., & Sabharwal, A. (2022). Decomposed Prompting: A Modular Approach
for Solving Complex Tasks. *International Conference on Learning
Representations (ICLR 2023).* arXiv:2210.02406.

Kishore, S., & He, H. (2024). Unveiling Divergent Inductive Biases of
LLMs on Temporal Data. *arXiv preprint arXiv:2404.01453.*

Lee, J., Lee, K., & Kim, T. (2025). MAGIC: A Multi-Hop and
Graph-Based Benchmark for Inter-Context Conflicts in
Retrieval-Augmented Generation. *arXiv preprint arXiv:2507.21544.*

Liu, S., Shang, Y., & Zhang, X. (2025). TruthfulRAG: Resolving
Factual-level Conflicts in Retrieval-Augmented Generation with
Knowledge Graphs. *arXiv preprint arXiv:2511.10375.*

Madhwal, D., Zhang, L. D., Roth, D., Wolfson, T., & Gupta, V.
(2026). Decomposed Prompting Does Not Fix Knowledge Gaps, But Helps
Models Say "I Don't Know". *arXiv preprint arXiv:2602.04853.*

Özer, A., & Yıldız, Ç. (2025). Question Answering under Temporal
Conflict: Evaluating and Organizing Evolving Knowledge with LLMs.
*arXiv preprint arXiv:2506.07270.*

Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I.,
& Gonzalez, J. E. (2023). MemGPT: Towards LLMs as Operating Systems.
*arXiv preprint arXiv:2310.08560.*

Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N. A., & Lewis, M.
(2022). Measuring and Narrowing the Compositionality Gap in Language
Models. *Findings of the Association for Computational Linguistics:
EMNLP 2023.* arXiv:2210.03350.

Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., & Chalef, D.
(2025). Zep: A Temporal Knowledge Graph Architecture for Agent
Memory. *arXiv preprint arXiv:2501.13956.*

Shao, Z., Gong, Y., Shen, Y., Huang, M., Duan, N., & Chen, W.
(2023). Enhancing Retrieval-Augmented Large Language Models with
Iterative Retrieval-Generation Synergy. *Findings of the Association
for Computational Linguistics: EMNLP 2023.* arXiv:2305.15294.

Tang, Y., & Yang, Y. (2024). MultiHop-RAG: Benchmarking
Retrieval-Augmented Generation for Multi-Hop Queries. *Conference on
Language Modeling (COLM 2024).* arXiv:2401.15391.

Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2023).
Interleaving Retrieval with Chain-of-Thought Reasoning for
Knowledge-Intensive Multi-Step Questions. *Annual Meeting of the
Association for Computational Linguistics (ACL).* arXiv:2212.10509.

Zhang, M., Ye, X., Liu, Q., Ren, P., Wu, S., & Chen, Z. (2024).
Uncovering Overfitting in Large Language Model Editing.
*International Conference on Learning Representations (ICLR 2025).*
arXiv:2410.07819.

Zhong, Z., Wu, Z., Manning, C. D., Potts, C., & Chen, D. (2023).
MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop
Questions. *Conference on Empirical Methods in Natural Language
Processing (EMNLP).* arXiv:2305.14795.

---

## Appendix

### A. Full prompts used

#### A.1 CANDIDATE_PROMPT (used by both SH-conflict and CAR pipelines)

```
You are given retrieved items from a knowledge pool. Each item has a
FRESHNESS marker (the prefix integer) — higher marker = newer version.

Your job: identify EVERY item that DIRECTLY answers the question, and
extract the answer entity from each.

Do NOT compare freshness markers. Do NOT pick a "best" one. Include ALL
items that match.

Rules:
1. An item directly answers the question only if BOTH its subject AND its
   predicate exactly match what the question asks about. The predicate
   noun used in the question (e.g., "spouse", "sport", "profession",
   "religion") must be the same noun used in the item. Related-but-different
   predicates do NOT match.
2. The subject named in the question must appear verbatim in the matching
   item. A different entity with a similar name does NOT match.
3. If a subject has multiple conflicting values (e.g., the same person
   with two different values at different freshness markers), INCLUDE
   BOTH as separate candidates. Do not pick.
4. If no item answers the question, return an empty list.
5. Copy the item's text verbatim into `fact_text`.

Question: {hop_query}

Items:
{pool}

Return ONLY valid JSON:
{"candidates": [{"serial": <int>, "fact_text": "<verbatim>",
                 "answer_entity": "<extracted>"}, ...]}
```

#### A.2 DECOMP_PROMPT (used by CAR pipeline)

```
<TASK>
Decompose this multi-hop question into a chain of atomic single-hop
queries. Each hop must ask about ONE relationship only.
</TASK>

<HARD_CONSTRAINT>
A single hop query may contain AT MOST ONE relationship word ("of", "by",
"from", "in", "where", "associated with", "related to"). If a hop has TWO
or more such words connecting entity descriptions, it is INVALID — split
it into multiple hops.
</HARD_CONSTRAINT>

<RULES>
1. Each hop asks ONE question about ONE specific entity. The entity must
   be named in the original question OR be the answer of a previous hop,
   referenced via {hop_N_answer}.
2. Chain INSIDE-OUT: start from the innermost named entity in the question,
   then move outward one relationship at a time.
3. The hop count equals the number of relationship words in the question
   — never fewer.
4. No padding hops. Every hop's answer must feed a later hop or be the
   final answer.
</RULES>

<EXAMPLE>
<QUESTION>"What is the country of the company where the manager of the
supervisor of Alice works?"</QUESTION>

<RELATIONSHIP_COUNT>4 — "supervisor of", "manager of", "company where …
works", "country of"</RELATIONSHIP_COUNT>

<CORRECT>
{"hops": [
  {"id": 1, "query": "Who is the supervisor of Alice?"},
  {"id": 2, "query": "Who is the manager of {hop_1_answer}?"},
  {"id": 3, "query": "What company does {hop_2_answer} work at?"},
  {"id": 4, "query": "What country is {hop_3_answer} based in?"}
]}
</CORRECT>

<WRONG reason="hop 1 has 2 'of' words; hop 2 has 2 relationships">
{"hops": [
  {"id": 1, "query": "Who is the manager of the supervisor of Alice?"},
  {"id": 2, "query": "What country is the company where {hop_1_answer}
   works based in?"}
]}
</WRONG>

<WRONG reason="single hop contains 4 'of' words">
{"hops": [
  {"id": 1, "query": "What country is the company where the manager of the
   supervisor of Alice works based in?"}
]}
</WRONG>
</EXAMPLE>

<QUESTION>{question}</QUESTION>

Return ONLY valid JSON: {"hops": [{"id": 1, "query": "..."}, ...]}
```

#### A.3 BM25 baseline prompt (MAB-provided, used as LLM-judgment baseline)

```
Pretend you are a knowledge management system. Each fact in the knowledge
pool is provided with a serial number at the beginning, and the newer fact
has larger serial number.

You need to solve the conflicts of facts in the knowledge pool by finding
the newest fact with larger serial number. You need to answer a question
based on this rule. You should give a very concise answer without saying
other words for the question **only** from the knowledge pool you have
memorized rather than the real facts in real world.

For example:

[Knowledge Pool]
Question: Based on the provided Knowledge Pool, what is the name of the
current president of Russia?
Answer: Donald Trump

Now Answer the Question: Based on the provided Knowledge Pool, {question}
Answer:
```

### B. Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| TOP_K | 10 | Matches MAB |
| Temperature (our pipelines) | 0.0 | Deterministic |
| Temperature (LLM-judgment baseline) | 0.7 | Matches MAB default |
| BM25 implementation | rank_bm25 BM25Okapi | k1=1.5, b=0.75 |
| Tokenizer | regex `[A-Za-z0-9]+` lowercased | |
| Fact-level chunking | regex `(\d+)\.\s+(.+?)(?=\d+\.\s|$)` | Per-fact |
| chunk-4096 chunking | sliding 4096 chars | Note: MAB default for FC is chunk-512 |
| max_tokens (output) | 256 | Sufficient for all answers |
| Backbones tested | gpt-4o-mini-2024-07-18, gpt-4o-2024-08-06 | |

### C. Confidence-interval computation

Wilson 95% CIs computed as
$\text{CI}=\frac{p + z^2/(2n) \pm z\sqrt{p(1-p)/n + z^2/(4n^2)}}{1 + z^2/n}$
with $z = 1.96$, $n = 100$ per cell, $n = 400$ pooled across the four
context lengths. Wilson is preferred over Wald at the extremes (e.g.,
99% accuracy) because Wald produces intervals that exceed 1.0 there.

### D. Per-cell raw results

All 16 cells (5 pipelines × 4 context lengths minus overlaps) are
documented in machine-readable JSON in the supplementary repository at
`poc_results/`. Filenames follow the pattern:
- `paper_sh_conflict_factconsolidation_sh_<L>.json` (Headline + LLM-judgment)
- `ablation_<task>_<chunk>_<model>_factconsolidation_<task>_<L>.json` (ablations)

### E. Reproducibility

**Code and data**. The full code and per-question JSON results are at
https://github.com/cvikasreddy/memory-conflict-resolution (the
repository is private during review and will be made public alongside
arXiv posting). All 1600 evaluations in the master table were saved as
machine-readable JSON files in `poc_results/`, including per-question
retrieved-fact serials, extracted candidates, chosen serial, and
SubEM correctness.

**One-command headline replication**:

```bash
git clone https://github.com/cvikasreddy/memory-conflict-resolution
cd memory-conflict-resolution
pip install -r requirements.txt
cp .env.example .env  # then fill in OPENAI_API_KEY
python scripts/13_paper_experiment.py --source factconsolidation_sh_262k
# Result in poc_results/paper_sh_conflict_factconsolidation_sh_262k.json
```

**Full benchmark**: ~$3 in API spend, 15 min wall-clock with 4
parallel processes.

**Langfuse traces**: 1600+ tagged traces available on request
(subject to Langfuse organization access).
