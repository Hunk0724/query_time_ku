import concurrent
import hashlib
import json
import logging
import os
import re
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict

import pytz
from pydantic import ValidationError

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.enums import MemoryType
from mem0.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_conflict_classification_messages,
    get_update_memory_messages,
)
from mem0.memory.base import MemoryBase
from mem0.memory.setup import setup_config
from mem0.memory.storage import SQLiteManager
from mem0.memory.telemetry import capture_event
from mem0.memory.utils import (
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    remove_code_blocks,
)
from mem0.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory

# Setup user config
setup_config()

logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version

        self.enable_graph = False

        # U5 conflict-resolution (Phase 1): env-gated, off by default (vanilla
        # path stays bit-identical). Per-user_id ordinal = chunk ingestion order;
        # every new fact in the same _add_to_vector_store() call shares it.
        self._u5_mode = os.environ.get("MEM0_UPDATE_MODE") == "u5_classification"
        self._u5_ordinal = {}

        # Phase 0 (0615): structural conservative-ADD mode. env-gated, off by
        # default (vanilla path stays bit-identical). Disables the update-decision
        # LLM call entirely; every fact is ADDed, a (s,p,o) triple is extracted
        # (local model, batched, cached) and committed to a parallel (S,P)
        # inverted index (see methods/phase0_triple_extractor.py).
        self._phase0_mode = os.environ.get("MEM0_ADD_MODE") == "phase0_structural"
        self._phase0_ordinal = {}
        self._phase0_conf_threshold = float(os.environ.get("MEM0_TRIPLE_CONF_THRESHOLD", "0.5"))
        self._sp_index = {}  # str("subj_id\x1fpred_norm") -> list[memory_id], ts DESC
        self._sp_index_path = os.environ.get("MEM0_SP_INDEX_PATH")
        if self._sp_index_path and os.path.exists(self._sp_index_path):
            try:
                with open(self._sp_index_path, encoding="utf-8") as _f:
                    self._sp_index = json.load(_f)
            except Exception as _e:
                logging.error(f"[phase0] sp-index load failed: {_e}")

        if self.config.graph_store.config:
            from mem0.memory.graph_memory import MemoryGraph

            self.graph = MemoryGraph(self.config)
            self.enable_graph = True

        capture_event("mem0.init", self)

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def add(
        self,
        messages,
        user_id=None,
        agent_id=None,
        run_id=None,
        metadata=None,
        filters=None,
        infer=True,
        memory_type=None,
        prompt=None,
    ):
        """
        Create a new memory.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            filters (dict, optional): Filters to apply to the search. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
            result: dict of affected events with each dict has the following key:
              'memories': affected memories
              'graph': affected graph memories

              'memories' and 'graph' is a dict, each with following subkeys:
                'add': added memory
                'update': updated memory
                'delete': deleted memory


        """
        if metadata is None:
            metadata = {}

        filters = filters or {}
        if user_id:
            filters["user_id"] = metadata["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = metadata["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = metadata["run_id"] = run_id

        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=metadata, prompt=prompt)
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self._add_to_vector_store, messages, metadata, filters, infer)
            future2 = executor.submit(self._add_to_graph, messages, filters)

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            graph_result = future2.result()

        if self.api_version == "v1.0":
            warnings.warn(
                "The current add API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return vector_store_result

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        if not infer:
            returned_memories = []
            for message in messages:
                if message["role"] != "system":
                    message_embeddings = self.embedding_model.embed(message["content"], "add")
                    memory_id = self._create_memory(message["content"], message_embeddings, metadata)
                    returned_memories.append({"id": memory_id, "memory": message["content"], "event": "ADD"})
            return returned_memories

        parsed_messages = parse_messages(messages)

        # --- Extraction cache (env-gated): reuse a frozen, pre-validated complete
        # extraction so SH/MH and re-runs ingest IDENTICAL facts, making extraction
        # a reproducible precondition (not a per-run variable) for the downstream
        # component study. Keyed on the numbered-fact lines only (timestamp-
        # independent). No-op when MEM0_EXTRACTION_CACHE is unset. ---
        # Self-populating: a MISS runs fresh extraction and writes the result
        # back under _ckey, so pointing MEM0_EXTRACTION_CACHE at a fresh path
        # builds a reusable cache for later tests/analysis; pointing it at a
        # frozen complete cache always HITs and never writes (byte-identical to
        # before). Keyed on the numbered-fact lines only (timestamp-independent).
        new_retrieved_facts = None
        response = None  # ensure defined even on extraction-cache hit (guards the
                         # ext-log raw_response reference when cached facts are empty)
        _ext_cache_path = os.environ.get("MEM0_EXTRACTION_CACHE")
        _ckey = None
        if _ext_cache_path:
            _fact_lines = "\n".join(re.findall(r"^\s*\d+\.\s.*$", parsed_messages, re.MULTILINE))
            _ckey = hashlib.sha256(_fact_lines.encode("utf-8")).hexdigest()
            if os.path.exists(_ext_cache_path):
                try:
                    with open(_ext_cache_path, encoding="utf-8") as _cf:
                        _cache = json.load(_cf)
                    if _ckey in _cache:
                        new_retrieved_facts = _cache[_ckey]
                except Exception as _e:
                    logging.error(f"[ext-cache] read failed: {_e}")

        if new_retrieved_facts is None:
            if self.custom_fact_extraction_prompt:
                system_prompt = self.custom_fact_extraction_prompt
                user_prompt = f"Input:\n{parsed_messages}"
            else:
                system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)

            os.environ["MEM0_COST_STAGE"] = "extract_p1"  # cost-log stage tag
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            try:
                response = remove_code_blocks(response)
                new_retrieved_facts = json.loads(response)["facts"]
            except Exception as e:
                logging.error(f"Error in new_retrieved_facts: {e}")
                new_retrieved_facts = []

            # Write-back on miss (single-process sequential run -> plain RMW is safe).
            if _ext_cache_path and _ckey is not None:
                try:
                    _cache = {}
                    if os.path.exists(_ext_cache_path):
                        with open(_ext_cache_path, encoding="utf-8") as _cf:
                            _cache = json.load(_cf)
                    _cache[_ckey] = new_retrieved_facts
                    os.makedirs(os.path.dirname(_ext_cache_path) or ".", exist_ok=True)
                    with open(_ext_cache_path, "w", encoding="utf-8") as _cf:
                        json.dump(_cache, _cf, ensure_ascii=False)
                except Exception as _e:
                    logging.error(f"[ext-cache] write-back failed: {_e}")

        # --- Component-1 extraction instrumentation (env-gated, baseline-safe) ---
        # Records the raw per-chunk extraction output so we can attribute
        # "GT conflict fact never extracted" failures. No-op when env unset.
        _ext_log_dir = os.environ.get("MEM0_CAND_LOG_DIR")
        if _ext_log_dir:
            try:
                os.makedirs(_ext_log_dir, exist_ok=True)
                with open(os.path.join(_ext_log_dir, "extraction.jsonl"), "a", encoding="utf-8") as _f:
                    _f.write(json.dumps({
                        "user_id": (filters or {}).get("user_id", "unknown"),
                        "n_facts": len(new_retrieved_facts),
                        "facts": new_retrieved_facts,
                        "custom_extraction_prompt": bool(self.custom_fact_extraction_prompt),
                        # keep raw response only on extraction failure (else facts == parsed raw)
                        "raw_response": response if len(new_retrieved_facts) == 0 else None,
                    }, ensure_ascii=False) + "\n")
            except Exception as _e:
                logging.error(f"[ext-log] failed: {_e}")

        # === Phase 0 branch: conservative ADD + (S,P) structural indexing.
        # Shares the SAME extraction (L2 + frozen cache) as vanilla; skips the
        # candidate retrieval + update-decision LLM call entirely. ===
        if self._phase0_mode:
            _uid = (filters or {}).get("user_id", "default")
            # Fact-level ordinal: per-uid GLOBAL per-fact counter (not per-chunk).
            # base = counter before this chunk; each fact gets base + its
            # extraction index; advance by #facts written. Strictly monotonic
            # across chunks (no K sizing / overflow), so query-side max()=newest
            # needs ZERO change while intra-chunk same-(S,P) ties disappear.
            base_ordinal = self._phase0_ordinal.get(_uid, 0)
            _ret = self._add_phase0_structural(
                new_retrieved_facts=new_retrieved_facts,
                metadata=metadata,
                filters=filters,
                chunk_ordinal=base_ordinal,
            )
            self._phase0_ordinal[_uid] = base_ordinal + len(new_retrieved_facts)
            return _ret

        retrieved_old_memory = []
        new_message_embeddings = {}
        # --- Component-2 candidate-pool instrumentation (env-gated, baseline-safe) ---
        # When MEM0_CAND_LOG_DIR is unset this is a no-op and behaviour is
        # byte-identical to upstream. See docs/0603_current_research_main_evidence/.
        _cand_log_dir = os.environ.get("MEM0_CAND_LOG_DIR")
        _cand_dbg = [] if _cand_log_dir else None
        # U5: capture each candidate's stored ordinal (gated; does not alter the
        # vanilla retrieved_old_memory structure -> vanilla prompt stays verbatim).
        _u5_ord_by_uuid = {} if self._u5_mode else None
        # Batch-embed all new facts in one API call (was one round-trip per fact,
        # the write-time bottleneck at scale). Byte-identical vectors -> the
        # downstream search / update-decision is unchanged.
        _new_embs = self.embedding_model.embed_batch(new_retrieved_facts, "add")
        for _ei, new_mem in enumerate(new_retrieved_facts):
            messages_embeddings = _new_embs[_ei]
            new_message_embeddings[new_mem] = messages_embeddings
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                limit=5,
                filters=filters,
            )
            if _cand_dbg is not None:
                _cand_dbg.append({
                    "new_fact": new_mem,
                    "top5": [
                        {"rank": _i, "id": str(mem.id),
                         "score": getattr(mem, "score", None),
                         "text": mem.payload["data"]}
                        for _i, mem in enumerate(existing_memories, 1)
                    ],
                })
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload["data"]})
                if _u5_ord_by_uuid is not None:
                    _u5_ord_by_uuid[mem.id] = mem.payload.get("ordinal", -1)
        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logging.info(f"Total existing memories: {len(retrieved_old_memory)}")
        if _cand_dbg is not None:
            try:
                os.makedirs(_cand_log_dir, exist_ok=True)
                _uid = (filters or {}).get("user_id", "unknown")
                with open(os.path.join(_cand_log_dir, "candidate_pool.jsonl"), "a", encoding="utf-8") as _f:
                    _f.write(json.dumps({
                        "user_id": _uid,
                        "n_new_facts": len(new_retrieved_facts),
                        "pool_size_after_dedup": len(retrieved_old_memory),
                        "per_fact_candidates": _cand_dbg,
                    }, ensure_ascii=False) + "\n")
            except Exception as _e:
                logging.error(f"[cand-log] failed: {_e}")

        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        # === U5 branch: classification-only update (shares the SAME extraction,
        # candidate pool, single LLM call, and event-shaped return as vanilla;
        # only the LLM's job + ordinal visibility differ). ===
        if self._u5_mode:
            _uid = (filters or {}).get("user_id", "default")
            ordinal = self._u5_ordinal.get(_uid, 0)
            self._u5_ordinal[_uid] = ordinal + 1
            return self._update_u5(
                new_retrieved_facts=new_retrieved_facts,
                retrieved_old_memory=retrieved_old_memory,
                temp_uuid_mapping=temp_uuid_mapping,
                ord_by_uuid=_u5_ord_by_uuid or {},
                new_message_embeddings=new_message_embeddings,
                metadata=metadata,
                filters=filters,
                chunk_ordinal=ordinal,
            )

        function_calling_prompt = get_update_memory_messages(
            retrieved_old_memory, new_retrieved_facts, self.custom_update_memory_prompt
        )

        try:
            new_memories_with_actions = self.llm.generate_response(
                messages=[{"role": "user", "content": function_calling_prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logging.error(f"Error in new_memories_with_actions: {e}")
            new_memories_with_actions = []

        _t3_raw = new_memories_with_actions  # capture raw update response before parse (T3)
        try:
            new_memories_with_actions = remove_code_blocks(new_memories_with_actions)
            new_memories_with_actions = json.loads(new_memories_with_actions)
        except Exception as e:
            logging.error(f"Invalid JSON response: {e}")
            new_memories_with_actions = []

        # --- Component-3 update-decision instrumentation (env-gated, baseline-safe) ---
        # Captures the full update prompt + raw LLM response + which referenced
        # ids are hallucinated (not in candidate pool 0..N-1 — the exact condition
        # that makes the apply loop silently drop the action). No-op when unset.
        _upd_log_dir = os.environ.get("MEM0_CAND_LOG_DIR")
        if _upd_log_dir:
            try:
                _valid_ids = set(temp_uuid_mapping.keys())  # "0".."N-1"
                _parsed = new_memories_with_actions.get("memory", []) if isinstance(new_memories_with_actions, dict) else []
                _refs = [str(r.get("id")) for r in _parsed if r.get("event") in ("UPDATE", "DELETE")]
                _halluc = [i for i in _refs if i not in _valid_ids]
                _counts = {}
                for r in _parsed:
                    _counts[r.get("event")] = _counts.get(r.get("event"), 0) + 1
                os.makedirs(_upd_log_dir, exist_ok=True)
                with open(os.path.join(_upd_log_dir, "update_decision.jsonl"), "a", encoding="utf-8") as _f:
                    _f.write(json.dumps({
                        "user_id": (filters or {}).get("user_id", "unknown"),
                        "n_candidates": len(temp_uuid_mapping),
                        "n_new_facts": len(new_retrieved_facts),
                        "update_prompt": function_calling_prompt,
                        "raw_response": _t3_raw,
                        "parsed_actions": _parsed,
                        "event_counts": _counts,
                        "referenced_ids": _refs,
                        "hallucinated_ids": _halluc,
                        "id_to_uuid": temp_uuid_mapping,
                    }, ensure_ascii=False) + "\n")
            except Exception as _e:
                logging.error(f"[upd-log] failed: {_e}")

        returned_memories = []
        try:
            for resp in new_memories_with_actions.get("memory", []):
                logging.info(resp)
                try:
                    if not resp.get("text"):
                        logging.info("Skipping memory entry because of empty `text` field.")
                        continue
                    elif resp.get("event") == "ADD":
                        memory_id = self._create_memory(
                            data=resp.get("text"),
                            existing_embeddings=new_message_embeddings,
                            metadata=metadata,
                        )
                        returned_memories.append(
                            {
                                "id": memory_id,
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                            }
                        )
                    elif resp.get("event") == "UPDATE":
                        self._update_memory(
                            memory_id=temp_uuid_mapping[resp["id"]],
                            data=resp.get("text"),
                            existing_embeddings=new_message_embeddings,
                            metadata=metadata,
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif resp.get("event") == "DELETE":
                        self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                            }
                        )
                    elif resp.get("event") == "NONE":
                        logging.info("NOOP for Memory.")
                except Exception as e:
                    logging.error(f"Error in new_memories_with_actions: {e}")
        except Exception as e:
            logging.error(f"Error in new_memories_with_actions: {e}")

        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": list(filters.keys())},
        )

        return returned_memories

    def _update_u5(
        self,
        new_retrieved_facts,
        retrieved_old_memory,
        temp_uuid_mapping,
        ord_by_uuid,
        new_message_embeddings,
        metadata,
        filters,
        chunk_ordinal,
    ):
        """U5 Phase-1 update: LLM classifies relations, Python maps to ops.

        Shares the exact same candidate pool / single LLM call as vanilla. The
        only differences vs DEFAULT_UPDATE_MEMORY_PROMPT: the LLM outputs a
        relation per (new, existing) pair instead of an operation, and each fact
        carries an ordinal (ingestion order) so the LLM has a temporal signal.

        Phase 1 is destructive and reuses vanilla execution primitives
        (_create_memory / _delete_memory). No soft supersession / status fields.
        """
        # Build LLM input (ordinal is the only added signal).
        new_facts = [
            {"new_id": f"n{j}", "text": txt, "ordinal": chunk_ordinal}
            for j, txt in enumerate(new_retrieved_facts)
        ]
        existing_entries = [
            {
                "existing_id": item["id"],  # int-string id "0".."N-1"
                "text": item["text"],
                "ordinal": ord_by_uuid.get(temp_uuid_mapping.get(item["id"]), -1),
            }
            for item in retrieved_old_memory
        ]

        classifications = []
        raw_response = None
        if existing_entries and new_facts:
            system_prompt, user_prompt = get_conflict_classification_messages(
                new_facts, existing_entries
            )
            os.environ["MEM0_COST_STAGE"] = "update_decision"  # cost-log stage tag (vanilla)
            try:
                raw_response = self.llm.generate_response(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                classifications = json.loads(remove_code_blocks(raw_response)).get(
                    "classifications", []
                )
            except Exception as e:
                # Failsafe: malformed/no classification -> information-preserving
                # default (ADD every new fact, delete nothing).
                logging.error(f"[u5] classification failed: {e}")
                classifications = []

        # Group classifications by new fact id.
        by_new = {}
        for c in classifications:
            by_new.setdefault(c.get("new_id"), []).append(c)

        returned_memories = []
        deleted_uuids = set()
        mapped_ops = []

        for j, fact_text in enumerate(new_retrieved_facts):
            cls = by_new.get(f"n{j}", [])
            supersede_targets = []  # int-string ids
            merged_text = None
            has_duplicate = False
            for c in cls:
                rel = c.get("relation")
                eid = str(c.get("existing_id"))
                if rel in ("SUPERSEDE", "ENRICHMENT") and eid in temp_uuid_mapping:
                    supersede_targets.append(eid)
                    if rel == "ENRICHMENT" and c.get("merged_text"):
                        merged_text = c["merged_text"]
                elif rel == "DUPLICATE":
                    has_duplicate = True

            # Deterministic mapping (destructive).
            if supersede_targets:
                add_text = merged_text or fact_text
                action = "ENRICHMENT" if merged_text else "SUPERSEDE"
            elif has_duplicate:
                add_text = None  # DUPLICATE -> drop new fact
                action = "DUPLICATE"
            else:
                add_text = fact_text  # COEXIST / NO_RELATION / UNCERTAIN / no-candidate
                action = "ADD"

            # Execute: physical DELETE of superseded targets first, then ADD new.
            superseded_uuids = []
            for eid in supersede_targets:
                tgt_uuid = temp_uuid_mapping.get(eid)
                if tgt_uuid and tgt_uuid not in deleted_uuids:
                    try:
                        self._delete_memory(memory_id=tgt_uuid)
                        deleted_uuids.add(tgt_uuid)
                        superseded_uuids.append(tgt_uuid)
                        returned_memories.append(
                            {"id": tgt_uuid, "memory": None, "event": "DELETE"}
                        )
                    except Exception as e:
                        logging.error(f"[u5] delete failed for {tgt_uuid}: {e}")

            new_id = None
            if add_text is not None:
                md = {**(metadata or {}), "ordinal": chunk_ordinal}
                new_id = self._create_memory(
                    data=add_text,
                    existing_embeddings=new_message_embeddings,
                    metadata=md,
                )
                returned_memories.append(
                    {"id": new_id, "memory": add_text, "event": "ADD", "ordinal": chunk_ordinal}
                )

            mapped_ops.append(
                {
                    "new_id": f"n{j}",
                    "new_fact": fact_text,
                    "action": action,
                    "added_id": new_id,
                    "added_text": add_text,
                    "superseded_ids": superseded_uuids,
                }
            )

        # Decision log (gated by the existing instrumentation env var; off by
        # default -> no always-on audit log).
        _u5_log_dir = os.environ.get("MEM0_CAND_LOG_DIR")
        if _u5_log_dir:
            try:
                os.makedirs(_u5_log_dir, exist_ok=True)
                with open(os.path.join(_u5_log_dir, "u5_decision.jsonl"), "a", encoding="utf-8") as _f:
                    _f.write(json.dumps({
                        "user_id": (filters or {}).get("user_id", "unknown"),
                        "chunk_ordinal": chunk_ordinal,
                        "n_new_facts": len(new_retrieved_facts),
                        "n_candidates": len(existing_entries),
                        "new_facts": new_facts,
                        "existing_entries": existing_entries,
                        "raw_response": raw_response,
                        "classifications": classifications,
                        "mapped_ops": mapped_ops,
                    }, ensure_ascii=False) + "\n")
            except Exception as _e:
                logging.error(f"[u5-log] failed: {_e}")

        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": list(filters.keys())},
        )
        return returned_memories

    def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = self.graph.add(data, filters)

        return added_entities

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        filters = {key: memory.payload[key] for key in ["user_id", "agent_id", "run_id"] if memory.payload.get(key)}

        # Prepare base memory item
        memory_item = MemoryItem(
            id=memory.id,
            memory=memory.payload["data"],
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump(exclude={"score"})

        # Add metadata if there are additional keys
        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }
        additional_metadata = {k: v for k, v in memory.payload.items() if k not in excluded_keys}
        if additional_metadata:
            memory_item["metadata"] = additional_metadata

        result = {**memory_item, **filters}

        return result

    def get_all(self, user_id=None, agent_id=None, run_id=None, limit=100):
        """
        List all memories.

        Returns:
            list: List of all memories.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        capture_event("mem0.get_all", self, {"limit": limit, "keys": list(filters.keys())})

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, filters, limit)
            future_graph_entities = executor.submit(self.graph.get_all, filters, limit) if self.enable_graph else None

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories, "relations": graph_entities}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return all_memories
        else:
            return {"results": all_memories}

    def _get_all_from_vector_store(self, filters, limit):
        memories = self.vector_store.list(filters=filters, limit=limit)

        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }
        all_memories = [
            {
                **MemoryItem(
                    id=mem.id,
                    memory=mem.payload["data"],
                    hash=mem.payload.get("hash"),
                    created_at=mem.payload.get("created_at"),
                    updated_at=mem.payload.get("updated_at"),
                ).model_dump(exclude={"score"}),
                **{key: mem.payload[key] for key in ["user_id", "agent_id", "run_id"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories[0]
        ]
        return all_memories

    def search(self, query, user_id=None, agent_id=None, run_id=None, limit=100, filters=None):
        """
        Search for memories.

        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: List of search results.
        """
        filters = filters or {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        capture_event(
            "mem0.search",
            self,
            {"limit": limit, "version": self.api_version, "keys": list(filters.keys())},
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._search_vector_store, query, filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.search, query, filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            original_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return original_memories
        else:
            return {"results": original_memories}

    def _search_vector_store(self, query, filters, limit):
        embeddings = self.embedding_model.embed(query, "search")
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)

        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }

        original_memories = [
            {
                **MemoryItem(
                    id=mem.id,
                    memory=mem.payload["data"],
                    hash=mem.payload.get("hash"),
                    created_at=mem.payload.get("created_at"),
                    updated_at=mem.payload.get("updated_at"),
                    score=mem.score,
                ).model_dump(),
                **{key: mem.payload[key] for key in ["user_id", "agent_id", "run_id"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories
        ]

        return original_memories

    def update(self, memory_id, data):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (dict): Data to update the memory with.

        Returns:
            dict: Updated memory.
        """
        capture_event("mem0.update", self, {"memory_id": memory_id})

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id})
        self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        capture_event("mem0.delete_all", self, {"keys": list(filters.keys())})
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id})
        return self.db.get_history(memory_id)

    def _add_phase0_structural(self, new_retrieved_facts, metadata, filters, chunk_ordinal):
        """Phase 0 commit: pure ADD + (s,p,o) triple + (S,P) inverted index.

        No candidate retrieval, no update-decision LLM call, no overwrite/delete.
        Triple extraction is a local, batched, cached, per-fact call (M2). The
        triple is stored in the payload regardless of confidence; the confidence
        threshold only gates membership in the (S,P) inverted index. Three-state
        accounting (triple_ok / triple_low_conf / triple_null) keeps F1 clean.
        """
        import sys

        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from methods.phase0_triple_extractor import (
            extract_subjects_batch,
            extract_triples_batch,
            normalize_predicate,
            normalize_subject,
        )

        _uid = (filters or {}).get("user_id", "default")
        triples = extract_triples_batch(new_retrieved_facts)

        # Subject fallback: triple-null memories still get a subject tag so the
        # query-time subject-match guard applies universally (generalization).
        _null_idx = [i for i, t in enumerate(triples) if t is None]
        _subj_fb = {}
        if _null_idx:
            _subs = extract_subjects_batch([new_retrieved_facts[i] for i in _null_idx])
            for i, s in zip(_null_idx, _subs):
                if s:
                    _subj_fb[i] = normalize_subject(s, user_id=_uid)

        # Batch-embed all facts of this chunk in one API call instead of one
        # round-trip per fact (the write-time latency bottleneck at scale).
        # Vectors are byte-identical to per-fact embed(); results unchanged.
        _fact_embs = self.embedding_model.embed_batch(new_retrieved_facts, "add")

        returned_memories = []
        n_ok = n_low = n_null = 0
        for _fi, (fact, triple) in enumerate(zip(new_retrieved_facts, triples)):
            embeddings = _fact_embs[_fi]
            fact_ordinal = chunk_ordinal + _fi  # fact-level: base + extraction order
            md = {**(metadata or {}), "ordinal": fact_ordinal}

            sp_key = None
            if triple is None:
                n_null += 1
                md["triple"] = None
                if _fi in _subj_fb:
                    md["subject_fallback"] = _subj_fb[_fi]
            else:
                s_id = normalize_subject(triple["subject"], user_id=_uid)
                p_norm = normalize_predicate(triple["predicate"])
                md["triple"] = {
                    "subject_id": s_id,
                    "predicate_norm": p_norm,
                    "object_text": triple["object"],
                    "confidence": triple["confidence"],
                    "subject_raw": triple["subject"],
                    "predicate_raw": triple["predicate"],
                }
                # No confidence gate: every non-null triple enters the (S,P)
                # index (gpt-4o-mini confidence is near-constant ~0.95, so the
                # gate was a no-op; dropped for simplicity). confidence is still
                # recorded in the payload for analysis.
                n_ok += 1
                sp_key = f"{s_id}\x1f{p_norm}"

            memory_id = self._create_memory(fact, {fact: embeddings}, md)

            # (S,P) inverted index commit (M3): all non-null triples.
            if sp_key is not None:
                bucket = self._sp_index.setdefault(sp_key, [])
                bucket.append(memory_id)
                bucket.sort(
                    key=lambda mid, _o=fact_ordinal: _o, reverse=True
                )  # no-op sort (constant key); resolution uses payload 'ordinal' max()

            returned_memories.append(
                {"id": memory_id, "memory": fact, "event": "ADD", "ordinal": fact_ordinal}
            )

        # Persist (S,P) index (JSON; tuple keys flattened to "subj\x1fpred").
        if self._sp_index_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self._sp_index_path)), exist_ok=True)
                _tmp = self._sp_index_path + ".tmp"
                with open(_tmp, "w", encoding="utf-8") as _f:
                    json.dump(self._sp_index, _f, ensure_ascii=False)
                os.replace(_tmp, self._sp_index_path)
            except Exception as _e:
                logging.error(f"[phase0] sp-index persist failed: {_e}")

        # Three-state extraction accounting (gated by the shared instrumentation
        # env var; off by default).
        _log_dir = os.environ.get("MEM0_CAND_LOG_DIR")
        if _log_dir:
            try:
                os.makedirs(_log_dir, exist_ok=True)
                with open(os.path.join(_log_dir, "phase0_triples.jsonl"), "a", encoding="utf-8") as _f:
                    _f.write(json.dumps({
                        "user_id": _uid,
                        "chunk_ordinal": chunk_ordinal,
                        "n_facts": len(new_retrieved_facts),
                        "triple_ok": n_ok,
                        "triple_low_conf": n_low,
                        "triple_null": n_null,
                        "facts": new_retrieved_facts,
                        "triples": triples,
                    }, ensure_ascii=False) + "\n")
            except Exception as _e:
                logging.error(f"[phase0] triple-log failed: {_e}")

        capture_event(
            "mem0.add", self, {"version": self.api_version, "keys": list((filters or {}).keys())}
        )
        return returned_memories

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logging.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        self.db.add_history(memory_id, None, data, "ADD", created_at=metadata["created_at"])
        capture_event("mem0._create_memory", self, {"memory_id": memory_id})
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        # Generate embeddings for the summary
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        # Create the memory
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id})

        # Return results in the same format as add()
        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data")

        new_metadata = metadata or {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        if "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")
        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")
        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
        )
        capture_event("mem0._update_memory", self, {"memory_id": memory_id})
        return memory_id

    def _delete_memory(self, memory_id):
        logging.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload["data"]
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(memory_id, prev_value, None, "DELETE", is_deleted=1)
        capture_event("mem0._delete_memory", self, {"memory_id": memory_id})
        return memory_id

    def reset(self):
        """
        Reset the memory store.
        """
        logger.warning("Resetting all memories")
        self.vector_store.delete_col()
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.db.reset()
        capture_event("mem0.reset", self)

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")
