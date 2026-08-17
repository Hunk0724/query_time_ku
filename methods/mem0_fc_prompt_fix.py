"""L1 fact-extraction prompt fix for mem0 on FC (Fact Consolidation) tasks.

Problem:
    Upstream mem0's FACT_RETRIEVAL_PROMPT contains two few-shot examples that
    explicitly instruct the LLM to reject inputs resembling FC's declarative
    factual statements:

        Input: Hi.
        Output: {"facts" : []}

        Input: There are branches in trees.
        Output: {"facts" : []}

    Strict prompt-following LLMs (e.g. gemini-3.1-flash-lite) generalize this
    rejection to *all* FC inputs ("Thomas Kyd was born in London" → {facts: []}).
    Result: mem0 ingests 0 facts, EM = 0%.

Fix (L1):
    Surgically remove only those two rejection few-shots via regex. All other
    prompt elements (Personal Information Organizer framing, types of info to
    remember, extraction guidelines, language preservation) are preserved.

Disclosure for paper:
    This is a minimal modification to repair an input-modality mismatch in
    mem0's default prompt, not prompt engineering. With this fix mem0
    produces fact lists comparable in volume to what GPT-4o-mini extracts
    under the out-of-the-box prompt.
"""
from __future__ import annotations

import re
from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT


def make_l1_modified_prompt() -> str:
    """Return mem0's FACT_RETRIEVAL_PROMPT with two rejection few-shots removed.

    Raises RuntimeError if the pattern doesn't match (upstream prompt changed).
    """
    pattern = re.compile(
        r'Input: Hi\.\s*\nOutput: \{"facts" : \[\]\}\s*\n\s*\n'
        r'Input: There are branches in trees\.\s*\nOutput: \{"facts" : \[\]\}\s*\n\s*\n',
        re.MULTILINE,
    )
    modified = pattern.sub('', FACT_RETRIEVAL_PROMPT)
    if modified == FACT_RETRIEVAL_PROMPT:
        raise RuntimeError(
            "Could not locate the rejection few-shots in FACT_RETRIEVAL_PROMPT. "
            "Upstream mem0 prompt may have changed; verify in mem0/configs/prompts.py."
        )
    return modified


# ── L2: knowledge-extraction reframe ────────────────────────────────────────
# Why L1 is insufficient:
#   L1 only deletes 2 few-shots, but the prompt's CORE framing is a "Personal
#   Information Organizer" that extracts "facts and preferences ABOUT THE USER"
#   (personal preferences / details / plans …) and is told "if you do not find
#   anything relevant ... return an empty list". FC facts are general WORLD
#   KNOWLEDGE ("Steve Jobs was born in San Francisco"), not personal info, so a
#   strict LLM legitimately judges "no personal info here" → {"facts": []}.
#   At 6k this rarely triggered; at 32k it dropped 7-13 whole chunks (hundreds of
#   facts), making any downstream component comparison unfair.
#
# Fix (L2): replace the personal-assistant framing with a knowledge-extractor
#   framing that extracts EVERY stated fact. Recovers ~100% extraction on the
#   32k chunks L1 returned empty for (chunks 0, 10, 18, 28, 49).
#   Extraction is NOT our object of study (we study candidate-retrieval + update
#   decision); L2 + a frozen extraction cache make extraction a complete,
#   reproducible, SH/MH-identical precondition so the downstream components can
#   be compared fairly. Disclosed as such in the paper.

L2_KNOWLEDGE_PROMPT = """You are a knowledge extractor. The input is a knowledge base: a numbered list of factual statements. Extract EVERY factual statement as a separate atomic fact for storage in memory.

Rules:
- Extract ALL facts stated in the input — one per item, in the same order. Do not skip, merge, or summarize any.
- Drop the leading serial number; keep the factual statement itself.
- These are general world-knowledge facts (not personal information). Extract them regardless of topic.
- NEVER return an empty list when the input contains numbered facts.
- Do not add facts that are not stated in the input.

Return the result in JSON with a key "facts" whose value is a list of strings, as shown:

Input: Here is a list of facts:
0. Thomas Kyd was born in the city of London.
1. The chairperson of Fatah is Mahmoud Abbas.
Output: {"facts": ["Thomas Kyd was born in the city of London", "The chairperson of Fatah is Mahmoud Abbas"]}
"""


def make_l2_knowledge_prompt() -> str:
    """Return the L2 knowledge-extraction prompt (reliable on FC general facts)."""
    return L2_KNOWLEDGE_PROMPT


# ── Broadened-native: ONE unified front-end for FC + LongMemEval + any KU ─────
# Why:
#   L2 above is FC-format-OVERFITTED: it assumes the input is a numbered fact
#   list ("drop the leading serial number", "NEVER return an empty list"),
#   selectivity OFF. It maximizes FC recall but does NOT transfer to real
#   conversational KU (LongMemEval) -- it would over-extract greetings/questions
#   and assumes list formatting. Native FACT_RETRIEVAL is the opposite failure:
#   selective + conversation-robust, but PERSONAL-ONLY, so it drops FC's general
#   world facts ("no personal info" -> empty).
#
#   Broadened-native = native's selectivity + conversation-robustness, with the
#   DOMAIN widened from "personal info" to "any fact the source ASSERTS as true
#   (personal AND world knowledge)", the numbered-list coupling removed (one
#   List-inputs rule handles lists AND conversations), and a Faithfulness rule
#   added (transcribe counterfactual/updated values as-stated -- non-destructive
#   at extraction, mirroring our conservative write).
#
# Validated (chunk=512, FC 6k): recall ~100% (the apparent 96.7% is a
#   matcher artifact -- the extractor normalizes MQuAKE typos univeristy/origianl
#   so fuzzy match misses them); conflict pairs preserved (both versions kept);
#   conversational selectivity holds (greetings/questions dropped, personal +
#   world facts kept). Use this as the SHARED front-end for the internal ablation
#   (vanilla-write / structural / +LLM). For the CROSS-SYSTEM baseline, run mem0
#   with NO prompt flag (true native FACT_RETRIEVAL) to show its honest failure.
def make_broadened_native_prompt() -> str:
    """Return the broadened-native extraction prompt (one front-end, any KU task)."""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""You are a Memory Organizer, specialized in accurately recording facts, memories, and preferences so they can be retrieved and kept up to date over time. Your role is to extract, from the given input, every distinct piece of information that the user or source ASSERTS as true, and organize them into distinct, atomic facts. You record BOTH personal information about the user AND general / world-knowledge facts stated in the input.

Types of Information to Record:
1. Personal Preferences: likes, dislikes, and specific preferences (food, products, activities, entertainment).
2. Personal Details: names, relationships, important dates, and other significant personal information.
3. Plans and Intentions: upcoming events, trips, goals, and plans the user shares.
4. Activity and Service Preferences: dining, travel, hobbies, and other service preferences.
5. Health and Wellness: dietary restrictions, fitness routines, and other wellness information.
6. Professional Details: job titles, work habits, career goals, and other professional information.
7. Miscellaneous Personal Details: favorite books, movies, brands, and other details the user shares.
8. General / World-Knowledge Facts and Updates: factual statements the source asserts about entities -- people, places, organizations, works, events -- and their attributes (e.g. birthplace, position held, location, office holder, creator), INCLUDING new or revised values that update an earlier fact.

Guidelines:
- Faithfulness: transcribe each asserted fact EXACTLY as stated, even if it contradicts common knowledge or an earlier statement. Do NOT fact-check, correct, or judge truth. Conflicting or updated values are expected -- record them as given.
- Atomicity: record one fact per statement. Split a sentence that asserts multiple facts into separate atomic facts.
- Selectivity: record only information the source ASSERTS. Do NOT record greetings, questions, requests, or meta-dialogue. If the input asserts no facts, return an empty list.
- List inputs: if the input is a list (numbered or bulleted), treat each item as a separate input and extract its fact(s); ignore the list markers and serial numbers.
- Do not add facts that are not stated in the input.

Here are some examples:

Input: Hi.
Output: {{"facts": []}}

Input: Can you recommend a restaurant in San Francisco?
Output: {{"facts": []}}

Input: Hi, my name is John and I am a software engineer.
Output: {{"facts": ["Name is John", "Is a software engineer"]}}

Input: My favourite movies are Inception and Interstellar.
Output: {{"facts": ["Favourite movies are Inception and Interstellar"]}}

Input: Thomas Kyd was born in the city of London. The chairperson of Fatah is Mahmoud Abbas.
Output: {{"facts": ["Thomas Kyd was born in the city of London", "The chairperson of Fatah is Mahmoud Abbas"]}}

Input: 0. The capital of France is Berlin. 1. Amy Winehouse died in the city of Camden Town.
Output: {{"facts": ["The capital of France is Berlin", "Amy Winehouse died in the city of Camden Town"]}}

Input: By the way, I moved to Boston last month, and the CEO of Acme is now Dana Lee.
Output: {{"facts": ["Moved to Boston last month", "The CEO of Acme is now Dana Lee"]}}

Remember:
- Today's date is {today}.
- Detect the language of the input and record the facts in the same language.
- Record facts only from the user and assistant content, not from any system messages.
- Return the response as JSON with a single key "facts" whose value is a list of strings.
"""


# ── Unified extractor (the converged design) ────────────────────────────────
# Anchored on native FACT_RETRIEVAL_PROMPT, with four DE-OVERFITTED changes
# (none presuppose the input format):
#   (1) de-restrict domain (personal -> any fact the USER asserts, self OR world;
#       no FC-shaped attribute enumeration);
#   (2) faithfulness (transcribe counterfactual/updated values, never fact-check);
#   (3) keep native's selectivity (greetings/questions -> empty);
#   (4) SOURCE-based specificity: record facts the USER asserts; the assistant's
#       turns are context only, never a fact source. (Relies on the role labels
#       present in the text -- reliable + backbone-independent -- NOT on the LLM
#       judging what it already knows.)
# Removed vs broadened-native: the numbered-list rule and the world-knowledge
# attribute enumeration (both overfit). "Faithful storage" is part of our
# conservative-write commitment, not an orthogonal front-end.
# Validated: FC 6k chunks recall ~100% (extracted==gold);
# LongMemEval KU -> every assistant chunk yields 0 facts (advice dropped), user
# answer facts retained (25:50 / four / suburbs), counts back to native level.
# See writing_draft/method_pipeline_and_prompts.md (P1) and ku_taxonomy_and_scope_zh.md §6.
def make_unified_extractor_prompt() -> str:
    """Return the unified extractor prompt (one front-end for FC + dialogue KU)."""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""You are a Memory Organizer. From a conversation between a user and an assistant, extract the distinct facts that the USER asserts as true -- about themselves or about the world -- and record them as separate, atomic facts for later retrieval. The memory you build holds what the user has told the system; it is authoritative over the assistant's own knowledge.

Use the whole conversation -- both speakers -- to UNDERSTAND what the user means: resolve references ("it", "there", "that") and read the assistant's replies as context that clarifies the user's statements. But RECORD facts ONLY from what the USER asserts. The assistant's turns are context for understanding, NEVER a source of facts to store.

What to record (from the user's statements):
- Any fact the user states as true: personal information, preferences, plans, relationships, situations, AND general / world facts the user asserts (including values that differ from common knowledge).
- Record one fact per statement; split a sentence asserting several facts into separate atomic facts.

Faithfulness:
- Transcribe each asserted fact exactly as the user states it, even if it contradicts common knowledge or an earlier statement. Do NOT fact-check, correct, or judge truth. Conflicting or updated values are expected -- record them as given.

What NOT to record (selectivity):
- Greetings, questions, requests, and small talk that assert no fact.
- Anything the ASSISTANT contributes on its own -- advice, recommendations, instructions, explanations, or general knowledge. (As stated above, the assistant is context only, never a fact source. If the user later restates something as their own, record the user's statement.)
- If the user asserts no fact, return an empty list.

Return JSON with a single key "facts" whose value is a list of strings. Detect the input language and record the facts in that language.

Examples:
Input:
user: Hi there!
assistant: Hello! How can I help?
Output: {{"facts": []}}

Input:
user: My name is John and I just moved to Boston.
Output: {{"facts": ["Name is John", "Moved to Boston"]}}

Input:
user: Can you suggest a good 5K training plan?
assistant: Sure -- try interval sprints of 20-30 seconds with 1-2 min recovery.
Output: {{"facts": []}}

Input:
user: By the way, the CEO of Acme is now Dana Lee, and the capital of Australia is Sydney.
Output: {{"facts": ["The CEO of Acme is now Dana Lee", "The capital of Australia is Sydney"]}}

Remember today's date is {today}.
"""


if __name__ == "__main__":
    mod = make_l1_modified_prompt()
    orig_len = len(FACT_RETRIEVAL_PROMPT)
    mod_len = len(mod)
    print(f"Original prompt: {orig_len} chars")
    print(f"L1 modified:     {mod_len} chars ({orig_len - mod_len} removed)")
    print(f"L2 knowledge:    {len(L2_KNOWLEDGE_PROMPT)} chars (reframed)")
