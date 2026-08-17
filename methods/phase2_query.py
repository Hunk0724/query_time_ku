"""Phase 2 query-time pipeline: conditional structural routing + structural
temporal resolve + LLM dynamic grouping (Option-B: clusters-only output).

Builds on Phase 0 storage (payload carries triple + ordinal). Shares the write
side with Phase 0; differs only at query time. Gated by env MEM0_QUERY_MODE
(see agent.py): "structural" = Phase 0 resolve; "phase2" = this module.

Resolution philosophy (decoupled):
  - LLM judges fact IDENTITY only (which retrieved memories are the same fact).
  - Deterministic argmax(ordinal) picks the latest version. LLM never picks recency.
  - Option B: keep everything by default; only DROP the stale members of clusters
    the LLM is confident about (false split is safe, false merge hides values).

The grouping prompt applies a single-valued / functional-property guard,
subject-disambiguation, and clusters-only output (no timestamps).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict

from methods.phase0_triple_extractor import _llm_chat

logger = logging.getLogger(__name__)


# ── LLM dynamic-grouping prompt ─────────────────────────────────────────────
GROUPING_PROMPT = """\
You are analyzing a list of memory entries retrieved for a user query. Your ONLY
task is to find CLUSTERS of entries that are the SAME fact recorded in different
versions. You do NOT answer the query, summarize, or decide which entry is newer.

# THE TEST (apply strictly)
Cluster two entries ONLY IF they are competing answers to the EXACT SAME question
about the EXACT SAME specific entity — same specific subject AND same property —
where only the value (answer) differs across versions.

# HARD RULES — when NOT to cluster (these dominate)
1. DIFFERENT SPECIFIC ENTITY = DIFFERENT FACT. If two entries are about different
   named entities (two different people, places, things, or organizations), they
   are DIFFERENT facts. Do NOT cluster them — even if they share the same wording,
   the same attribute type, or the SAME value. (A shared answer like many
   different people each being "a citizen of the USA" is NOT one fact.)
2. DIFFERENT PROPERTY = DIFFERENT FACT. Same entity but different property
   (its capital vs its head of state; the user's city vs the user's job) are
   DIFFERENT facts. Do NOT cluster them.
3. MULTI-VALUED PROPERTY = COEXIST. If the property naturally holds several
   values at once (hobbies, friends, languages spoken, places visited, skills,
   things liked), the values COEXIST — they are NOT versions. Do NOT cluster.

# Clustering is RARE
In a typical list, most or ALL entries are unrelated and form NO clusters. Expect
zero or very few clusters. Cluster only when you are CONFIDENT the entries are the
same single-valued fact about the same specific entity. When unsure, do not
cluster — unclustered entries are simply kept (a false merge hides a real value;
a false split is harmless).

# Identity only — never recency
Cluster purely by fact identity. Do NOT judge which entry is newer or correct —
a separate deterministic step handles that.

# Output — clusters only
Output ONLY the clusters you are confident about, each with 2+ members. Entries
not placed in a cluster MUST NOT be listed (they are kept automatically). If you
find no clusters, output {"groups": []}.

Return JSON exactly:
{
  "groups": [
    {"fact": "<the specific entity + property>", "memory_ids": ["<id>","<id>"], "reasoning": "<one sentence>"}
  ]
}

# Examples (span personal and world-knowledge; note the negatives)

Query: Where does the user live now?
[a] The user used to live in Taipei.
[b] The user now lives in Tainan.
[c] The user enjoys hiking.
Output: {"groups": [{"fact": "the user's current city of residence", "memory_ids": ["a","b"], "reasoning": "Same person, single-valued residence; values are versions."}]}

Query: Who is the CEO of Acme Corp?
[d] Acme Corp's CEO is Jane Doe.
[e] Acme Corp's CEO is now John Smith.
[f] Acme Corp is headquartered in Berlin.
Output: {"groups": [{"fact": "Acme Corp's current CEO", "memory_ids": ["d","e"], "reasoning": "Same org, single-valued CEO; f is a different property."}]}

Query: Who is allergic to peanuts?
[g] The user is allergic to peanuts.
[h] The user's son is allergic to peanuts.
Output: {"groups": []}   (different subjects: the user vs the son)

Query: Where were these people born?
[i] Marie Curie was born in Warsaw.
[j] Frederic Chopin was born in Warsaw.
Output: {"groups": []}   (different people; a shared value is not one fact)

Query: Tell me about the user.
[k] The user's current city is Tokyo.
[l] The user's current employer is Acme.
Output: {"groups": []}   (same user, different properties)

Query: What languages does the user speak?
[m] The user speaks French.
[n] The user speaks Japanese.
Output: {"groups": []}   (multi-valued: languages coexist)

# Now analyze:
Query: {query}
Memory entries:
{entries}
"""


# ── item accessors (mem0 search-result dict shape) ──────────────────────────
def _id(it):
    return str(it["id"])


def _text(it):
    return it.get("memory") or (it.get("metadata") or {}).get("data", "")


def _ordinal(it):
    return (it.get("metadata") or {}).get("ordinal", -1)


def _sp(it):
    t = (it.get("metadata") or {}).get("triple")
    return (t["subject_id"], t["predicate_norm"]) if t else None


def _subject(it):
    """Subject of the memory: triple.subject_id if present, else the write-time
    subject_fallback (triple-null items). May be None for genuinely subject-less
    memories."""
    md = it.get("metadata") or {}
    t = md.get("triple")
    return t["subject_id"] if t else md.get("subject_fallback")


def _subject_consistent(members):
    """Guard: reject an LLM cluster whose members span >=2 distinct KNOWN
    subjects (different specific entities = different facts). Members with an
    unknown (None) subject do not block — those are the genuinely LLM-only cases.

    Ablation: env MEM0_SUBJECT_GUARD_OFF=1 bypasses the guard entirely
    (accepts every LLM-proposed cluster). Isolates the guard's contribution
    to final EM. Default behavior unchanged when env unset.
    """
    if os.environ.get("MEM0_SUBJECT_GUARD_OFF") == "1":
        return True
    subs = {s for s in (_subject(m) for m in members) if s}
    return len(subs) <= 1


def _drop_older(members):
    """Ids of members strictly below the max ordinal (keep ALL at max -> tie-safe).
    Returns (drop_ids, count_at_max)."""
    mx = max(_ordinal(m) for m in members)
    drop = {_id(m) for m in members if _ordinal(m) < mx}
    n_at_max = sum(1 for m in members if _ordinal(m) == mx)
    return drop, n_at_max


# ── Stage 2: conditional structural routing ─────────────────────────────────
def conditional_structural_routing(candidates):
    """Split into structural_pool ((S,P) with >=2 competitors) and dynamic_pool
    (no-triple items + triple items whose (S,P) is a singleton in candidates).

    Ablation: env MEM0_STRUCTURAL_SKIP=1 forces ALL candidates into dynamic_pool
    (structural_pool empty). Isolates P3 LLM identity grouping's contribution
    from (S,P) structural grouping: pure-LLM identity resolution over top-k.
    Compare with pure-structural (ours_struct) for apple-to-apple structural
    vs LLM identity comparison.
    """
    if os.environ.get("MEM0_STRUCTURAL_SKIP") == "1":
        return {}, list(candidates)
    sp_map = defaultdict(list)
    no_triple = []
    for it in candidates:
        sp = _sp(it)
        (no_triple if sp is None else sp_map[sp]).append(it)
    structural_pool, singleton_triple = {}, []
    for sp, mems in sp_map.items():
        if len(mems) >= 2:
            structural_pool[sp] = mems
        else:
            singleton_triple.extend(mems)
    return structural_pool, no_triple + singleton_triple


# ── Multi-valued guard: only single-valued (functional) (S,P) groups may be
# temporally resolved (newest replaces older). Multi-valued relations COEXIST
# (a new value ADDS, not replaces) -> never drop. The arity is a STABLE property
# of the predicate, so one cached LLM judgment per predicate_norm (a SIMPLE,
# general, reusable question — fits the small-model regime). Empirically needed:
# subject="user::*" + generic predicates ("has been using", "is training for")
# otherwise false-merge distinct multi-valued facts.
PREDICATE_ARITY_PROMPT = """You decide whether a relation about one subject is SINGLE-VALUED or MULTI-VALUED, to know whether a newer value REPLACES an older one or COEXISTS with it.

- SINGLE-VALUED: the subject has at most ONE current value; the listed values are competing alternatives where the newest is the truth (e.g. current city, current employer, an amount, a personal-best time, a count/total like "four restaurants").
- MULTI-VALUED: the subject holds MANY of these at once; the listed values are DIFFERENT things that coexist, not versions (e.g. brands/products used, hobbies, languages spoken, places visited, individual restaurants, people known, things trained for).

Relation: "{predicate}"
Subject: {subject}
Values seen for it: {objects}

Return JSON exactly: {"arity": "single"}  or  {"arity": "multi"}"""


def _obj(it):
    t = (it.get("metadata") or {}).get("triple")
    return (t.get("object_text") if t else None) or _text(it)


def _predicate_is_multivalued(predicate, subject, objects, *, model=None,
                              ollama_url=None, cache_path=None):
    """One cached LLM judgment per predicate_norm: multi-valued (coexist) vs
    single-valued (functional, newest replaces older). Cached by predicate_norm.
    Defaults to multi (keep all = safe, no false drop) on error/uncertainty."""
    model = model or os.environ.get("MEM0_TRIPLE_MODEL", "gpt-4o-mini")
    ollama_url = ollama_url or os.environ.get("MEM0_TRIPLE_OLLAMA_URL", "http://localhost:11434")
    cache_path = cache_path or os.environ.get("MEM0_PREDICATE_CACHE")
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception as e:
            logger.error(f"[predicate-cache] read failed: {e}")
    if predicate in cache:
        return cache[predicate]
    prompt = (PREDICATE_ARITY_PROMPT.replace("{predicate}", predicate or "")
              .replace("{subject}", subject or "?")
              .replace("{objects}", "; ".join(str(o) for o in objects[:8])))
    try:
        raw = _llm_chat(prompt, model, ollama_url) or "{}"
        arity = (json.loads(raw).get("arity") or "").strip().lower()
        multi = not arity.startswith("single")  # "single" -> functional; else multi
    except Exception as e:
        logger.error(f"[predicate-arity] call failed for {predicate!r}: {e}")
        multi = True  # safe default: coexist (no false drop)
    cache[predicate] = multi
    if cache_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            tmp = cache_path + ".tmp"
            json.dump(cache, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(tmp, cache_path)
        except Exception as e:
            logger.error(f"[predicate-cache] write failed: {e}")
    return multi


# ── Stage 3: structural temporal resolve (tie -> escalate to LLM) ────────────
def structural_resolve(structural_pool):
    """Per (S,P) group: only SINGLE-VALUED predicates resolve (unique max ordinal
    -> drop older; tie -> escalate to LLM). MULTI-VALUED predicates COEXIST (kept,
    never dropped) -- guarded by a cached per-predicate arity judgment."""
    drop = set()
    escalated = []
    for (subj_id, pred), mems in structural_pool.items():
        if _predicate_is_multivalued(pred, subj_id, [_obj(m) for m in mems]):
            continue  # multi-valued -> COEXIST, keep all
        d, n_at_max = _drop_older(mems)
        if n_at_max == 1:
            drop |= d
        else:
            escalated.append(mems)
    return drop, escalated


# ── Stage 4: LLM dynamic grouping (clusters-only) ───────────────────────────
def _grouping_cache_key(query, pool):
    body = query + "\x1f" + "\n".join(sorted(_text(it) for it in pool))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def llm_dynamic_grouping(pool, query, *, model=None, ollama_url=None, cache_path=None):
    """Cluster the pool by fact identity; return drop_ids (stale members of each
    confident >=2 cluster). Content-keyed frozen cache (ingestion-independent)."""
    if len(pool) < 2:
        return set()
    model = model or os.environ.get("MEM0_TRIPLE_MODEL", "gpt-4o-mini")
    ollama_url = ollama_url or os.environ.get("MEM0_TRIPLE_OLLAMA_URL", "http://localhost:11434")
    cache_path = cache_path or os.environ.get("MEM0_GROUPING_CACHE")

    cache, key = {}, _grouping_cache_key(query, pool)
    if cache_path and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception as e:
            logger.error(f"[phase2-cache] read failed: {e}")

    groups_contents = cache.get(key)  # list[list[content]] or None
    if groups_contents is None:
        id2text = {_id(it): _text(it) for it in pool}
        entries = "\n".join(f"[{_id(it)}] {_text(it)}" for it in pool)
        prompt = GROUPING_PROMPT.replace("{query}", query).replace("{entries}", entries)
        groups_contents = []
        try:
            data = json.loads(_llm_chat(prompt, model, ollama_url))
            for g in data.get("groups", []):
                ids = [str(x) for x in g.get("memory_ids", []) if str(x) in id2text]
                if len(ids) >= 2:
                    groups_contents.append([id2text[i] for i in ids])
        except Exception as e:
            logger.error(f"[phase2 grouping] call/parse failed: {e}")
            groups_contents = []
        cache[key] = groups_contents
        if cache_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                tmp = cache_path + ".tmp"
                json.dump(cache, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
                os.replace(tmp, cache_path)
            except Exception as e:
                logger.error(f"[phase2-cache] write failed: {e}")

    # Map content-clusters back to items in THIS pool, resolve, collect drops.
    text2items = defaultdict(list)
    for it in pool:
        text2items[_text(it)].append(it)
    drop = set()
    for gtexts in groups_contents:
        members = [m for t in gtexts for m in text2items.get(t, [])]
        if len(members) < 2:
            continue
        # Structural guard: reject clusters spanning different specific subjects
        # (kills the dominant false-merge mode; uses triple/subject_fallback).
        if not _subject_consistent(members):
            logger.info("[phase2 guard] rejected cross-subject cluster: "
                        f"{[_subject(m) for m in members]}")
            continue
        d, _ = _drop_older(members)
        drop |= d
    return drop


# ── Identity clusters (LLM): same-fact clusters from the dynamic pool ────────
# Like llm_dynamic_grouping but RETURNS clusters (member lists) for the
# downstream conflict-type stage instead of resolving here. Subject guard rejects
# cross-subject false-merges. Content-keyed cache (MEM0_GROUPING_CACHE).
def llm_identity_clusters(pool, query, *, model=None, ollama_url=None, cache_path=None):
    if len(pool) < 2:
        return []
    os.environ["MEM0_COST_STAGE"] = "group_p3"  # cost-log stage tag
    model = model or os.environ.get("MEM0_TRIPLE_MODEL", "gpt-4o-mini")
    ollama_url = ollama_url or os.environ.get("MEM0_TRIPLE_OLLAMA_URL", "http://localhost:11434")
    cache_path = cache_path or os.environ.get("MEM0_GROUPING_CACHE")
    cache, key = {}, _grouping_cache_key(query, pool)
    if cache_path and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception as e:
            logger.error(f"[phase2-cache] read failed: {e}")
    groups_contents = cache.get(key)
    if groups_contents is None:
        id2text = {_id(it): _text(it) for it in pool}
        entries = "\n".join(f"[{_id(it)}] {_text(it)}" for it in pool)
        prompt = GROUPING_PROMPT.replace("{query}", query).replace("{entries}", entries)
        groups_contents = []
        try:
            data = json.loads(_llm_chat(prompt, model, ollama_url))
            for g in data.get("groups", []):
                ids = [str(x) for x in g.get("memory_ids", []) if str(x) in id2text]
                if len(ids) >= 2:
                    groups_contents.append([id2text[i] for i in ids])
        except Exception as e:
            logger.error(f"[phase2 grouping] call/parse failed: {e}")
            groups_contents = []
        cache[key] = groups_contents
        if cache_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                tmp = cache_path + ".tmp"
                json.dump(cache, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
                os.replace(tmp, cache_path)
            except Exception as e:
                logger.error(f"[phase2-cache] write failed: {e}")
    text2items = defaultdict(list)
    for it in pool:
        text2items[_text(it)].append(it)
    clusters = []
    for gtexts in groups_contents:
        members = [m for t in gtexts for m in text2items.get(t, [])]
        if len(members) >= 2 and _subject_consistent(members):
            clusters.append(members)
    return clusters


# ── Conflict-type classification (per candidate group, query-aware) ──────────
# 3-way taxonomy adapted from Cattan et al. 2025 ("DRAGged into CONFLICTS"),
# restricted to memory-accumulation of user assertions:
#   NO_CONFLICT   = surface variants of one value  -> argmax tau (any newest)
#   FRESHNESS     = mutually exclusive at any time  -> argmax tau (newest, drop older)
#   COMPLEMENTARY = mutually compatible coexisting   -> keep all
# Per-GROUP + QUERY-AWARE: fixes the per-predicate arity's context-dependence
# (the specific values + the query disambiguate, not the abstract predicate).
CONFLICT_TYPE_PROMPT = """You analyze a group of candidate memory entries that share the same subject and relation but have different recorded values. Determine which conflict type applies, so the downstream resolution step knows whether to select the most recent value or keep all of them.

Apply ONE of the following categories (adapted from Cattan et al., 2025, CONFLICTS taxonomy, restricted to memory accumulation scenarios where the source is user assertions):

(1) NO_CONFLICT — The candidates record the SAME value expressed in different surface forms (e.g., "Tokyo" vs "Tokyo"; "350,000" vs "$350k"; "PhD" vs "doctorate"). Minor variations in granularity or phrasing without semantic difference.

(2) FRESHNESS — The candidates are mutually exclusive at any single point in time. A single subject cannot simultaneously hold all listed values; the newer value supersedes the older. Examples: current city, current employer, current count/total, personal best, mortgage amount, marital status.

(3) COMPLEMENTARY — The candidates are mutually compatible. The same subject can reasonably hold all listed values simultaneously, without contradiction. The values are accumulating instances, not competing answers. Examples: brands tried, languages spoken, places visited, hobbies, restaurants visited.

DECISION TEST (apply strictly, in order):
  Step 1: Are these surface variants of one value? YES -> NO_CONFLICT.
  Step 2: Could the subject reasonably hold ALL listed values AT THE SAME TIME, right now? YES -> COMPLEMENTARY.  NO -> FRESHNESS.

When uncertain between FRESHNESS and COMPLEMENTARY, default to COMPLEMENTARY (false-merge into Freshness destroys information; false-split into Complementary is recoverable downstream).

Query: {query}
Subject: {subject}
Relation: {predicate}
Candidate values (with timestamps):
{candidates_with_timestamps}

Return JSON exactly: {"type": "no_conflict" | "freshness" | "complementary", "reasoning": "<one sentence>"}"""


def _classify_conflict_type(group, query, *, model=None, ollama_url=None, cache_path=None):
    """Per-group conflict type in {no_conflict, freshness, complementary}.
    Cached by (query + group member texts). Safe default = complementary."""
    os.environ["MEM0_COST_STAGE"] = "conflict_p5"  # cost-log stage tag
    model = model or os.environ.get("MEM0_TRIPLE_MODEL", "gpt-4o-mini")
    ollama_url = ollama_url or os.environ.get("MEM0_TRIPLE_OLLAMA_URL", "http://localhost:11434")
    cache_path = cache_path or os.environ.get("MEM0_CONFLICT_CACHE")
    key = hashlib.sha256(((query or "") + "\x1f" + "\n".join(sorted(_text(m) for m in group))).encode("utf-8")).hexdigest()
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception as e:
            logger.error(f"[conflict-cache] read failed: {e}")
    if key in cache:
        return cache[key]
    subj = next((_subject(m) for m in group if _subject(m)), "?")
    pred = next((_sp(m)[1] for m in group if _sp(m)), "?")
    cand = "\n".join(f"- {_obj(m)}  (tau={_ordinal(m)})" for m in group)
    prompt = (CONFLICT_TYPE_PROMPT.replace("{query}", query or "").replace("{subject}", subj or "?")
              .replace("{predicate}", pred or "?").replace("{candidates_with_timestamps}", cand))
    try:
        ctype = (json.loads(_llm_chat(prompt, model, ollama_url)).get("type") or "").strip().lower()
        if ctype not in ("no_conflict", "freshness", "complementary"):
            ctype = "complementary"
    except Exception as e:
        logger.error(f"[conflict-type] failed: {e}")
        ctype = "complementary"  # safe default: keep all
    cache[key] = ctype
    if cache_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            tmp = cache_path + ".tmp"
            json.dump(cache, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(tmp, cache_path)
        except Exception as e:
            logger.error(f"[conflict-cache] write failed: {e}")
    return ctype


# ── Orchestrator ────────────────────────────────────────────────────────────
def phase2_resolve(candidates, query):
    """Identity grouping -> unified candidate groups -> per-group conflict type
    -> resolve: FRESHNESS = argmax tau (drop older); NO_CONFLICT /
    COMPLEMENTARY = keep all. Returns retained items in original order.

    Only FRESHNESS (a genuine same-fact recency conflict) drops the older
    version. NO_CONFLICT means the grouped facts are not actually contradictory
    (e.g. different attributes) -> nothing to resolve, keep all. COMPLEMENTARY
    means coexisting values of a multi-valued attribute -> also keep all.

    Ablation: env MEM0_P5_SKIP=1 forces argmax on every group regardless of
    conflict-type (i.e. treats every same-identity group as FRESHNESS). This
    isolates the LLM identity grouping (P3) contribution from the LLM
    conflict-type classifier (P5). In FC world-fact + counter-factual scenarios
    where ~all conflicts are FRESHNESS by construction, this ablation should
    reveal whether P5 is neutral or actively harmful vs the P3-only baseline.
    """
    structural_pool, dynamic_pool = conditional_structural_routing(candidates)
    clusters = llm_identity_clusters(dynamic_pool, query)
    groups = list(structural_pool.values()) + clusters
    p5_skip = os.environ.get("MEM0_P5_SKIP") == "1"
    drop = set()
    for g in groups:
        if len(g) < 2:
            continue
        if not p5_skip and _classify_conflict_type(g, query) != "freshness":
            continue  # no_conflict / complementary -> keep all
        d, _ = _drop_older(g)  # freshness (or forced) -> newest wins
        drop |= d
    return [it for it in candidates if _id(it) not in drop]
