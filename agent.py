import os
import json
import torch
import tiktoken
from openai import OpenAI
from utils.templates import get_template
from utils.eval_data_utils import (
    format_chat,
)
import re
import time

from langchain_core.documents import Document
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig


class AgentWrapper:
    """
    A wrapper class for different types of memory agents including:
    - Long context agents (GPT, Claude, Gemini)
    - Letta agents
    - Mem0 agents  
    - Cognee agents
    - RAG agents (various implementations)
    """

    def __init__(self, agent_config, dataset_config, load_agent_from):
        """
        Initialize the agent wrapper with specified configuration.
        
        Args:
            agent_config: Configuration dictionary for the agent
            dataset_config: Configuration dictionary for the dataset
            load_agent_from: Optional path to load existing agent state from
        """
        # Basic agent configuration
        self.agent_name = agent_config['agent_name']
        self.sub_dataset = dataset_config['sub_dataset']
        self.context_max_length = dataset_config['context_max_length']
        self.dataset = dataset_config['dataset']
        
        # Output and storage configuration
        self.output_dir = agent_config['output_dir']
        self.agent_save_to_folder = load_agent_from
        
        # Context and token limits
        self.input_length_limit = (agent_config['input_length_limit'] - 
                                 agent_config['buffer_length'] - 
                                 dataset_config['generation_max_length'])
        
        # Model configuration
        self.model = agent_config['model']
        self.max_tokens = dataset_config['generation_max_length']
        self.temperature = agent_config.get('temperature', 0.0)
        
        # Initialize tokenizer (default to gpt-4o-mini for non-gpt models)
        model_for_tokenizer = self.model if "gpt-4o" in self.model else "gpt-4o-mini"
        self.tokenizer = tiktoken.encoding_for_model(model_for_tokenizer)
        
        # Initialize agent based on type
        self._initialize_agent_by_type(agent_config, dataset_config)

    def _initialize_agent_by_type(self, agent_config, dataset_config):
        """Initialize the specific agent type based on agent name."""
        
        if 'Long_context_agent' in self.agent_name:
            self._initialize_long_context_agent()
        elif self._is_agent_type("letta"):
            self._initialize_letta_agent(agent_config, dataset_config)
        elif self._is_agent_type("mem0"):
            self._initialize_mem0_agent(agent_config, dataset_config)
        elif self._is_agent_type("cognee"):
            self._initialize_cognee_agent(agent_config, dataset_config)
        elif self._is_agent_type("zep"):
            self._initialize_zep_agent(agent_config)
        elif self._is_agent_type("rag"):
            self._initialize_rag_agent(agent_config, dataset_config)
        else:
            raise NotImplementedError(f"Agent type not supported: {self.agent_name}")

    def _is_agent_type(self, agent_type):
        """Check if the current agent is of a specific type."""
        return agent_type in self.agent_name

    def _create_oai_client(self):
        """Create an OpenAI-compatible client. Uses Azure OpenAI if env vars are set.

        Environment variables for Azure:
          - AZURE_OPENAI_ENDPOINT
          - AZURE_OPENAI_API_VERSION (optional; default provided by SDK or pinned elsewhere)
          - AZURE_OPENAI_API_KEY

        When using Azure, ensure self.model is the deployment name.
        """
        try:
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            if azure_endpoint:
                # Lazy import to avoid requiring Azure class when not used
                from openai import AzureOpenAI
                return AzureOpenAI(
                    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=azure_endpoint,
                )
        except Exception:
            pass
        return OpenAI(max_retries=20)

    def _create_standard_response(self, output, input_tokens, output_tokens, memory_time, query_time):
        """Create standardized response dictionary."""
        return {
            "output": output,
            "input_len": input_tokens,
            "output_len": output_tokens,
            "memory_construction_time": memory_time,
            "query_time_len": query_time,
        }

    def _initialize_long_context_agent(self):
        """Initialize long context agent with appropriate client."""
        self.context = ''
        
        if "gpt" in self.model or "o4" in self.model:
            self.client = self._create_oai_client()
        elif "claude" in self.model:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=os.environ.get('Anthropic_API_KEY'),
            )
        elif "gemini" in self.model:
            from google import genai
            if os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').lower() == 'true':
                self.client = genai.Client(
                    vertexai=True,
                    project=os.environ.get('GOOGLE_CLOUD_PROJECT'),
                    location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1'),
                )
            else:
                self.client = genai.Client(api_key=os.environ.get('Google_API_KEY'))
        else:
            raise NotImplementedError(f"Model not supported for long context agent: {self.model}")

    def _initialize_letta_agent(self, agent_config, dataset_config):
        """Initialize Letta agent with proper configuration."""
        if "api" not in agent_config['agent_name']:
            from letta import create_client, LLMConfig, EmbeddingConfig, BasicBlockMemory

            self.chunk_size = agent_config['agent_chunk_size']
            self.letta_mode = agent_config['letta_mode']
            
            self.client = create_client()
            self.client.set_default_llm_config(LLMConfig.default_config(agent_config['model']))             
            self.agent_start_time = time.time()
            
            # Configure embedding
            if agent_config['text_embedding'] == 'text-embedding-3-small':
                self.client.set_default_embedding_config(EmbeddingConfig(
                    embedding_model="text-embedding-3-small",
                    embedding_endpoint_type="openai",
                    embedding_endpoint="https://api.openai.com/v1",
                    embedding_dim=1536,
                    embedding_chunk_size=self.chunk_size * 2,
                ))
            else:
                self.client.set_default_embedding_config(
                    EmbeddingConfig.default_config(agent_config['text_embedding'])
                )

            # Load system prompt
            system_path = agent_config['system_path']
            with open(system_path, 'r') as f:
                self.system = f.read()

            # Load or create agent
            if os.path.exists(self.agent_save_to_folder):
                self.load_agent()
            else:
                human_block = self.client.create_block(
                    label='human', 
                    value='User is sharing the contents they are reading recently.', 
                    limit=2000000
                )
                persona_block = self.client.create_block(
                    label='persona', 
                    value='You are a helpful assistant that can help memorize details in the conversation.', 
                    limit=2000000
                )
                memory = BasicBlockMemory(blocks=[human_block, persona_block])
                self.agent_state = self.client.create_agent(
                    name='mm_agent',
                    memory=memory,
                    system=self.system
                )
        ## use the letta api to create the agent
        else:
            from letta_client import Letta, CreateBlock
            
            self.chunk_size = agent_config['agent_chunk_size']
            self.letta_mode = agent_config['letta_mode']
            self.agent_start_time = time.time()
            
            
            self.client = Letta(token=os.environ.get('Letta_API_KEY'))
            self.agent_state = self.client.agents.create(
            memory_blocks=[
                CreateBlock(
                    label="human",
                    limit=2000000,
                    value="User is sharing the contents they are reading recently."
                ),
                CreateBlock(
                    label="persona",
                    limit=2000000,
                    value="You are a helpful assistant that can help memorize details in the conversation."
                )
            ],
            model=f"openai/{agent_config['model']}",
            embedding=f"openai/{agent_config['text_embedding']}"
        )

            
            
    def _initialize_mem0_agent(self, agent_config, dataset_config):
        """Initialize Mem0/Mem0g agent.

        Two layers of LLM/embedding here:
          1. mem0 internal LLM (fact extraction + update decision) — set via
             yaml `mem0_config.llm` / `mem0_config.embedder` and passed into
             `Memory(MemoryConfig(**mem0_config))`. If `mem0_config.graph_store`
             is present, mem0g (graph-augmented) mode is enabled automatically.
          2. Answer-time LLM (generates final response given retrieved
             memories) — chosen by `self.model`, may be OpenAI/Gemini.

        Without `mem0_config`, falls back to `Memory()` benchmark default
        (OpenAI gpt-4o-mini + text-embedding-3-small + Qdrant), requiring
        OPENAI_API_KEY.

        Example yaml (mem0g + Vertex Gemini):
            mem0_config:
              llm:
                provider: vertexai
                config: {model: gemini-2.5-flash-lite, temperature: 0.7}
              embedder:
                provider: vertexai
                config: {model: text-embedding-004, embedding_dims: 768}
              graph_store:
                provider: neo4j
                config: {url: neo4j://localhost:7687, username: neo4j, password: mem0gpassword}
        """
        from mem0.memory.main import Memory
        from mem0.configs.base import MemoryConfig
        from mem0.utils.factory import LlmFactory, EmbedderFactory

        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.context = ''

        # Answer-time client (separate from mem0 internal LLM)
        self.client = self._create_answer_client()

        # mem0 init
        mem0_config_dict = agent_config.get('mem0_config') or {}

        # Make qdrant path + collection sub_dataset-specific so parallel runs
        # of mem0/mem0g across different (task, ctx) combos don't collide.
        if mem0_config_dict.get('vector_store', {}).get('provider') == 'qdrant':
            vs_cfg = mem0_config_dict['vector_store'].setdefault('config', {})
            base_coll = vs_cfg.get('collection_name', 'mem0_default')
            base_path = vs_cfg.get('path', '/tmp/qdrant_mem0')
            suffix = self.sub_dataset.replace('-', '_')  # filesystem-safe
            vs_cfg['collection_name'] = f"{base_coll}__{suffix}"
            vs_cfg['path'] = f"{base_path}__{suffix}"

        # Same isolation for mem0's history.db (SQLite) — without this, parallel
        # mem0/mem0g processes hit "attempt to write a readonly database" because
        # the default ~/.mem0/history.db is shared, and SQLite locks one writer
        # at a time. Symptom: all update-phase ADD/UPDATE silently fail.
        # Suffix includes agent_name short fingerprint so different mem0/mem0g
        # variants (e.g. as-is vs prompt-aware) can run on the same sub_dataset
        # in parallel without state collision.
        import os as _os, re as _re
        sub_ds_suffix = self.sub_dataset.replace('-', '_')
        agent_fp = _re.sub(r'[^A-Za-z0-9]+', '_', self.agent_name)[:48].strip('_')
        sqlite_suffix = f"{sub_ds_suffix}__{agent_fp}"
        history_dir = _os.path.expanduser("~/.mem0")
        _os.makedirs(history_dir, exist_ok=True)
        mem0_config_dict['history_db_path'] = _os.path.join(
            history_dir, f"history__{sqlite_suffix}.db"
        )

        # Surface prompt-aware-graph flag from yaml mem0_config to instance
        # attribute. Consumed at inference time in _handle_mem0_agent to
        # verbalize graph relations into the system prompt (restoring the
        # upstream mem0 cookbook pattern; see docs/baseline_methods/
        # mem0g_reproducibility.md). When false (default), behaves as MABench
        # original (vector-only inference).
        self.mem0_prompt_aware_graph = bool(
            mem0_config_dict.pop('prompt_aware_graph', False)
        )

        # Redirect mem0's Vertex providers through methods/ wrappers when yaml asks for
        # vertexai (ADC-based). Upstream mem0 vertexai requires a service account JSON;
        # our wrappers use google.genai SDK with ADC, matching how we call Gemini LLM
        # in long-context agent. See methods/mem0_vertex_{gemini_llm,adc_embedder}.py.
        llm_cfg = mem0_config_dict.get('llm', {})
        emb_cfg = mem0_config_dict.get('embedder', {})
        if llm_cfg.get('provider') == 'vertexai' or emb_cfg.get('provider') == 'vertexai':
            import sys as _sys
            from pathlib import Path as _Path
            _METHODS = str(_Path(__file__).parent / "methods")
            if _METHODS not in _sys.path:
                _sys.path.insert(0, _METHODS)
            mem0_config_dict = json.loads(json.dumps(mem0_config_dict))  # deep copy

        if llm_cfg.get('provider') == 'vertexai':
            LlmFactory.provider_to_class["gemini"] = "mem0_vertex_gemini_llm.VertexGeminiLLM"
            mem0_config_dict['llm']['provider'] = 'gemini'   # whitelist requires this
            print("[mem0] LlmFactory monkey-patched: gemini → VertexGeminiLLM (ADC)")

        if emb_cfg.get('provider') == 'vertexai':
            EmbedderFactory.provider_to_class["vertexai"] = \
                "mem0_vertex_adc_embedder.VertexADCEmbedding"
            print("[mem0] EmbedderFactory monkey-patched: vertexai → VertexADCEmbedding (ADC)")

        # Vendored mem0 internal inconsistency: mem0/memory/main.py calls
        # EmbedderFactory.create(provider, config, vector_config) (3 args) but
        # mem0/memory/graph_memory.py calls EmbedderFactory.create(provider, config)
        # (2 args). Wrap .create so vector_config is genuinely optional, otherwise
        # mem0g raises TypeError on init.
        if mem0_config_dict.get('graph_store'):
            _orig_create = EmbedderFactory.create.__func__
            def _patched_create(cls, provider_name, config, vector_config=None):
                return _orig_create(cls, provider_name, config, vector_config)
            EmbedderFactory.create = classmethod(_patched_create)
            print("[mem0g] EmbedderFactory.create wrapped: vector_config defaulted to None")

            # Vendored mem0/memory/graph_memory.py inlines entity_type and
            # relationship names directly into Cypher (e.g. MERGE (n:{source_type})).
            # When the LLM extracts a label like "country/empire" Neo4j rejects it
            # with a SyntaxError.
            #
            # We wrap LLM-extracted labels in backticks (Neo4j-standard escape) so
            # the original semantic is preserved (e.g. `country/empire` stays a
            # distinct label from `country_empire`). All non-trivial wraps are
            # logged to outputs/.../mem0g_label_audit.jsonl for post-hoc audit.
            from mem0.memory import graph_memory as _gm_module
            from pathlib import Path as _Path2

            def _wrap_neo4j_label(s):
                if not s:
                    s = "Unknown"
                s = str(s)
                # Escape internal backticks per Neo4j rules: ` → ``
                escaped = s.replace("`", "``")
                return f"`{escaped}`"

            # Audit log path (under same retrieval dir as ingestion log)
            _AUDIT_DIR = (_Path2(__file__).parent / "outputs" / "rag_retrieved" /
                          self.agent_name / f"k_{self.retrieve_num}" /
                          self.sub_dataset / f"chunksize_{self.chunk_size}")
            _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
            _AUDIT_FILE = _AUDIT_DIR / "mem0g_label_audit.jsonl"

            def _audit(context, name, original, wrapped):
                """Log any non-trivial label wrap (i.e. original needed escaping)."""
                # original needed backticks if it contains chars outside [A-Za-z0-9_]
                # or starts with a digit
                if not original:
                    return
                import re as __re
                needs_wrap = bool(__re.search(r'[^A-Za-z0-9_]', str(original))) or \
                             (str(original) and str(original)[0].isdigit())
                if not needs_wrap:
                    return
                with open(_AUDIT_FILE, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "context": context, "name": name,
                        "original": original, "wrapped": wrapped,
                    }, ensure_ascii=False) + "\n")

            _orig_add_entities = _gm_module.MemoryGraph._add_entities

            def _patched_add_entities(self, to_be_added, user_id, entity_type_map):
                wrapped_map = {}
                for k, v in entity_type_map.items():
                    w = _wrap_neo4j_label(v)
                    wrapped_map[k] = w
                    _audit("node_label", k, v, w)
                wrapped_added = []
                for item in to_be_added:
                    orig_rel = item.get("relationship", "RELATED")
                    w_rel = _wrap_neo4j_label(orig_rel)
                    _audit("relationship", item.get("source", "?"), orig_rel, w_rel)
                    wrapped_added.append({**item, "relationship": w_rel})
                return _orig_add_entities(self, wrapped_added, user_id, wrapped_map)

            _gm_module.MemoryGraph._add_entities = _patched_add_entities
            print(f"[mem0g] MemoryGraph._add_entities patched: backtick wrap + audit → {_AUDIT_FILE.name}")

            # Same for _delete_entities (uses {relationship} in Cypher)
            _orig_delete_entities = _gm_module.MemoryGraph._delete_entities

            def _patched_delete_entities(self, to_be_deleted, user_id):
                wrapped = []
                for item in to_be_deleted:
                    orig_rel = item.get("relationship", "RELATED")
                    w_rel = _wrap_neo4j_label(orig_rel)
                    _audit("relationship_delete", item.get("source", "?"), orig_rel, w_rel)
                    wrapped.append({**item, "relationship": w_rel})
                return _orig_delete_entities(self, wrapped, user_id)

            _gm_module.MemoryGraph._delete_entities = _patched_delete_entities

            # Third vendored mem0g bug: _remove_spaces_from_entities does
            # `item["relationship"]` directly. When the LLM occasionally returns
            # an entity dict missing the relationship key, mem0g KeyError-crashes
            # mid-ingest. We patch to use .get() with sensible defaults.
            _orig_remove_spaces = _gm_module.MemoryGraph._remove_spaces_from_entities

            def _patched_remove_spaces(self, entity_list):
                clean = []
                for item in entity_list:
                    # Skip malformed entries entirely (none of source/destination/relationship)
                    if not isinstance(item, dict):
                        continue
                    safe = {
                        "source": str(item.get("source", "unknown")).lower().replace(" ", "_"),
                        "relationship": str(item.get("relationship", "RELATED")).lower().replace(" ", "_"),
                        "destination": str(item.get("destination", "unknown")).lower().replace(" ", "_"),
                    }
                    # carry forward any other keys mem0g may attach
                    for k, v in item.items():
                        if k not in safe:
                            safe[k] = v
                    clean.append(safe)
                return clean

            _gm_module.MemoryGraph._remove_spaces_from_entities = _patched_remove_spaces
            print("[mem0g] _remove_spaces_from_entities patched: tolerate missing 'relationship' key")

        # L1 prompt fix: remove two rejection few-shots that cause mem0 to drop
        # all FC inputs under strict-prompt-following LLMs (e.g. gemini-3.1-flash-lite).
        # See methods/mem0_fc_prompt_fix.py and Phase 0 README.
        if mem0_config_dict.get('use_l1_fc_prompt'):
            import sys as _sys2
            from pathlib import Path as _Path2
            _METHODS = str(_Path2(__file__).parent / "methods")
            if _METHODS not in _sys2.path:
                _sys2.path.insert(0, _METHODS)
            from mem0_fc_prompt_fix import make_l1_modified_prompt
            mem0_config_dict['custom_fact_extraction_prompt'] = make_l1_modified_prompt()
            mem0_config_dict.pop('use_l1_fc_prompt', None)
            print("[mem0] L1 FC prompt fix applied: removed 2 rejection few-shots (95 chars)")

        # L2 knowledge-extraction prompt: L1 is insufficient (its core
        # personal-assistant framing still drops whole chunks of general-knowledge
        # facts at 32k). L2 reframes extraction as knowledge extraction and is the
        # reliable fallback when the frozen extraction cache (MEM0_EXTRACTION_CACHE)
        # misses. See methods/mem0_fc_prompt_fix.py make_l2_knowledge_prompt.
        if mem0_config_dict.get('use_l2_fc_prompt'):
            import sys as _sys3
            from pathlib import Path as _Path3
            _METHODS = str(_Path3(__file__).parent / "methods")
            if _METHODS not in _sys3.path:
                _sys3.path.insert(0, _METHODS)
            from mem0_fc_prompt_fix import make_l2_knowledge_prompt
            mem0_config_dict['custom_fact_extraction_prompt'] = make_l2_knowledge_prompt()
            mem0_config_dict.pop('use_l2_fc_prompt', None)
            print("[mem0] L2 knowledge-extraction prompt applied (reliable FC extraction)")

        # Broadened-native: ONE unified front-end (native selectivity + widened
        # domain to any asserted fact incl. world knowledge + faithfulness rule),
        # for FC AND LongMemEval AND any KU task. Replaces the FC-overfit L2 as
        # the SHARED extraction in the internal ablation. NOTE: for the honest
        # cross-system baseline, set NO prompt flag -> mem0 runs true native
        # FACT_RETRIEVAL (which fails on FC's general facts). See
        # methods/mem0_fc_prompt_fix.make_broadened_native_prompt.
        if mem0_config_dict.get('use_broadened_native_prompt'):
            import sys as _sys4
            from pathlib import Path as _Path4
            _METHODS = str(_Path4(__file__).parent / "methods")
            if _METHODS not in _sys4.path:
                _sys4.path.insert(0, _METHODS)
            from mem0_fc_prompt_fix import make_broadened_native_prompt
            mem0_config_dict['custom_fact_extraction_prompt'] = make_broadened_native_prompt()
            mem0_config_dict.pop('use_broadened_native_prompt', None)
            print("[mem0] broadened-native extraction prompt applied (unified FC/LongMemEval front-end)")

        # Unified extractor (the converged design): the single front-end for FC +
        # dialogue KU. native + de-restrict domain + faithfulness + native
        # selectivity + SOURCE-based specificity (record user-asserted facts;
        # assistant turns are context only). One config across all datasets.
        # See methods/mem0_fc_prompt_fix.make_unified_extractor_prompt.
        if mem0_config_dict.get('use_unified_extractor'):
            import sys as _sys5
            from pathlib import Path as _Path5
            _METHODS = str(_Path5(__file__).parent / "methods")
            if _METHODS not in _sys5.path:
                _sys5.path.insert(0, _METHODS)
            from mem0_fc_prompt_fix import make_unified_extractor_prompt
            mem0_config_dict['custom_fact_extraction_prompt'] = make_unified_extractor_prompt()
            mem0_config_dict.pop('use_unified_extractor', None)
            print("[mem0] unified extractor prompt applied (source-based; one front-end for FC + dialogue KU)")

        if mem0_config_dict:
            self.memory = Memory(MemoryConfig(**mem0_config_dict))
        else:
            self.memory = Memory()  # legacy benchmark default (needs OPENAI_API_KEY)
        self.mem0_graph_enabled = bool(mem0_config_dict.get('graph_store'))
        if self.mem0_graph_enabled:
            print(f"[mem0g] Graph store enabled: {mem0_config_dict['graph_store'].get('provider', '?')}")

        self.agent_start_time = time.time()

    def _create_answer_client(self):
        """Build the LLM client used to answer questions given retrieved memories.

        Distinct from mem0's internal LLM (configured via mem0_config.llm).
        Routes by self.model:
          - 'gemini-*'                 → Vertex AI (if GOOGLE_GENAI_USE_VERTEXAI=true) or AI Studio
          - 'gemma*' / 'ollama:*' / 'llama3*' / 'qwen*' / 'mistral*'  → Ollama (local)
          - else                       → OpenAI / Azure (existing _create_oai_client path)
        Sets self.client_type ∈ {'gemini','ollama','openai'} for downstream dispatch in
        _answer_with_client.
        """
        if 'gemini' in self.model.lower():
            from google import genai
            self.client_type = 'gemini'
            if os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').lower() == 'true':
                return genai.Client(
                    vertexai=True,
                    project=os.environ.get('GOOGLE_CLOUD_PROJECT'),
                    location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1'),
                )
            return genai.Client(
                api_key=os.environ.get('Google_API_KEY') or os.environ.get('GEMINI_API_KEY')
            )
        # Local Ollama (weak-model regime). Triggered by model name prefix.
        _m = self.model.lower()
        if any(_m.startswith(p) for p in ('gemma', 'ollama:', 'llama3', 'qwen', 'mistral')):
            from ollama import Client as _OllamaClient
            self.client_type = 'ollama'
            return _OllamaClient(host=os.environ.get('MEM0_TRIPLE_OLLAMA_URL', 'http://localhost:11434'))
        self.client_type = 'openai'
        return self._create_oai_client()

    def _answer_with_client(self, messages, max_tokens=None, temperature=None):
        """Unified answer call. Returns (text, prompt_tokens, completion_tokens).

        OpenAI: standard chat.completions.create.
        Gemini: concatenates system + user into one prompt, uses generate_content.
        Ollama: native /api/chat with num_ctx + num_predict options.
        Caller is responsible for OpenAI-style messages list with role+content.
        """
        if getattr(self, 'client_type', 'openai') == 'ollama':
            # Ollama native chat. Token counts come from prompt_eval_count / eval_count.
            resp = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': temperature if temperature is not None else self.temperature,
                    'num_predict': max_tokens or self.max_tokens,
                    'num_ctx': int(os.environ.get('OLLAMA_NUM_CTX', '8192')),
                },
            )
            text = resp['message']['content'] if isinstance(resp, dict) else resp.message.content
            pt = (resp.get('prompt_eval_count') if isinstance(resp, dict) else getattr(resp, 'prompt_eval_count', 0)) or 0
            ct = (resp.get('eval_count') if isinstance(resp, dict) else getattr(resp, 'eval_count', 0)) or 0
            return text, pt, ct

        if getattr(self, 'client_type', 'openai') == 'gemini':
            system = next((m['content'] for m in messages if m['role'] == 'system'), '')
            user = next((m['content'] for m in messages if m['role'] == 'user'), '')
            prompt = (system + "\n\n" + user) if system else user

            from google.genai import types as genai_types
            cfg = genai_types.GenerateContentConfig(
                max_output_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                thinking_config=genai_types.ThinkingConfig(thinking_level="minimal"),
            )
            # Retry handled by _generate_with_gemini_retry if available, else inline.
            resp = self.client.models.generate_content(
                model=self.model, contents=prompt, config=cfg,
            )
            text = resp.text or ''
            try:
                usage = resp.usage_metadata
                pt = getattr(usage, 'prompt_token_count', 0) or 0
                ct = getattr(usage, 'candidates_token_count', 0) or 0
            except Exception:
                pt = ct = 0
            return text, pt, ct

        # OpenAI / Azure. GPT-5 family + o-series reasoning models rejected
        # `max_tokens` since March 2026 (HTTP 400 "unsupported_parameter");
        # require `max_completion_tokens`. Route by model-name prefix so
        # older-model callers stay byte-identical.
        _mn = str(self.model or "").lower()
        _needs_mc = any(_mn.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
        _tok_val = max_tokens or self.max_tokens
        _create_kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
        )
        if _tok_val is not None:
            _create_kwargs["max_completion_tokens" if _needs_mc else "max_tokens"] = _tok_val
        response = self.client.chat.completions.create(**_create_kwargs)
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    def _initialize_cognee_agent(self, agent_config, dataset_config):
        """Initialize Cognee agent with knowledge graph configuration."""
        self.context = ''
        self.chunks = []
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.agent_start_time = time.time()
        self.cognee_dir = './cognee/.cognee_system/databases/cognee.lancedb'
    
    def _initialize_zep_agent(self, agent_config):
        # from zep_cloud.client import AsyncZep
        # self.client = AsyncZep(api_key=os.getenv("ZEP_API_KEY"), base_url="https://api.development.getzep.com/api/v2")
        from zep_cloud import Zep
        from methods.zep import OpenAIAgent
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.context_id = -1

        self.client = Zep(api_key=os.getenv("ZEP_API_KEY"))
        # Use Azure OpenAI if env vars are set, otherwise fall back to OpenAI
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint:
            self.oai_client = OpenAIAgent(model=self.model, source="azure", api_dict={"endpoint":azure_endpoint, "api_version":os.environ.get("AZURE_OPENAI_API_VERSION"), "api_key":os.environ.get("AZURE_OPENAI_API_KEY")}, temperature=self.temperature)
        else:
            self.oai_client = OpenAIAgent(model=self.model, source="openai", api_dict={}, temperature=self.temperature)
        self.agent_start_time = time.time()

    def _initialize_rag_agent(self, agent_config, dataset_config):
        """Initialize RAG agent with retrieval configuration."""
        self.context = ''
        self.chunks = []
        self.raw_chunks = []  # raw content without memorize template wrapper, for stable HippoRAG hash_ids
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = dataset_config['chunk_size']
        self.context_len = 0
        self.context_id = -1

    def send_message(self, message, memorizing=False, query_id=None, context_id=None):
        """
        Send a message to the agent for either memorization or querying.
        
        Args:
            message: The message content (context for memorization, query for answering)
            memorizing: Whether to memorize the message (True) or answer it (False)
            query_id: Unique identifier for the query
            context_id: Unique identifier for the context
            
        Returns:
            dict or str: Agent response with metadata (for queries) or confirmation (for memorization)
        """
        # Route to appropriate agent handler based on agent type
        if 'Long_context_agent' in self.agent_name:
            return self._handle_long_context_agent(message, memorizing)
        elif any(self._is_agent_type(agent_type) for agent_type in ["letta", "cognee", "mem0", "zep"]):
            return self._handle_memory_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("rag"):
            return self._handle_rag_agent(message, memorizing, query_id, context_id)
        else:
            raise NotImplementedError(f"Agent type not supported: {self.agent_name}")

    def _handle_long_context_agent(self, message, memorizing):
        """Handle message processing for long context agents."""
        if memorizing:
            # Add message to context memory
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"
        else:
            # Process query with context
            return self._query_long_context_agent(message)

    def _query_long_context_agent(self, message):
        """Process a query for long context agents."""
        # Get appropriate tokenizer
        try:
            tokenizer = tiktoken.encoding_for_model(self.model)
        except:
            tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        
        # Handle context truncation for non-long context models
        buffer_length = 50000
        if self.input_length_limit <= self.context_max_length + buffer_length:
            self._truncate_context_if_needed(tokenizer)
                
        # Format message with context and system prompt
        full_message = self.context + "\n" + message
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        formatted_message = format_chat(message=full_message, system_message=system_message)
        
        # Query the model
        start_time = time.time()
        
        if "gpt" in self.model: 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_message,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return self._format_openai_response(response, start_time)
            
        elif "o4" in self.model:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_message,
            )
            return self._format_openai_response(response, start_time)
            
        elif "claude" in self.model:
            return self._query_claude(full_message, system_message, start_time)
            
        elif "gemini" in self.model:
            return self._query_gemini(formatted_message, start_time)
            
        else:
            raise NotImplementedError(f"Model not supported: {self.model}")

    def _truncate_context_if_needed(self, tokenizer):
        """Truncate context if it exceeds limits."""
        # Truncate context if it exceeds the context_max_length
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.context_max_length:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.context_max_length:])
        
        # Truncate if context exceeds the input_length_limit
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.input_length_limit:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.input_length_limit:])

    def _format_openai_response(self, response, start_time):
        """Format OpenAI API response into standard output format."""
        return self._create_standard_response(
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            0,
            time.time() - start_time
        )

    def _query_claude(self, message, system_message, start_time):
        """Query Claude model with proper formatting."""
        formatted_message = format_chat(message=message, system_message=system_message, include_system=False)
        response = self.client.messages.create(
            model=self.model,
            messages=formatted_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return self._create_standard_response(
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
            0,
            time.time() - start_time
        )

    def _query_gemini(self, formatted_message, start_time):
        """Query Gemini model with retry-with-backoff on 429/503."""
        from google.genai import types
        from google.genai.errors import ClientError, ServerError
        import re

        config = types.GenerateContentConfig(
            system_instruction=formatted_message[0]["content"],
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        )

        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=formatted_message[1]["content"],
                    config=config,
                )
                break
            except (ClientError, ServerError) as e:
                code = getattr(e, 'code', None)
                if code not in (429, 503):
                    raise
                if attempt == max_retries - 1:
                    raise
                msg = str(e)
                m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
                delay = float(m.group(1)) + 2 if m else min(2 ** attempt * 5, 60)
                print(f"[gemini retry] {code} (attempt {attempt + 1}/{max_retries}), sleeping {delay:.1f}s", flush=True)
                time.sleep(delay)

        text = response.text if response.text is not None else ""
        return self._create_standard_response(
            text,
            response.usage_metadata.prompt_token_count,
            response.usage_metadata.candidates_token_count,
            0,
            time.time() - start_time
        )
        
    def _handle_memory_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for memory-based agents (Letta, Cognee, Mem0)."""
        if self._is_agent_type("letta"):
            return self._handle_letta_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("cognee"):
            return self._handle_cognee_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("mem0"):
            return self._handle_mem0_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("zep"):
            return self._handle_zep_agent(message, memorizing, query_id, context_id)
        else:
            raise NotImplementedError(f"Memory agent type not supported: {self.agent_name}")

    def _handle_letta_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Letta agents."""
        # Format message based on context
        if memorizing:
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
        else:
            formatted_message = message
        
        # Handle memory construction time for queries
        memory_construction_time = 0 if memorizing else time.time() - self.agent_start_time
        
        # Reload agent for queries
        if not memorizing:
            if os.path.exists(self.agent_save_to_folder):
                self.load_agent()
            else:
                print(f"\n\nAgent {self.agent_name} not found in {self.agent_save_to_folder}\n\n")
        
        # Process based on Letta mode
        response = self._process_letta_message(formatted_message, memorizing, query_id, context_id)
        
        if memorizing:
            return "Memorized"
        
        # Create response for queries
        tokenizer = self.tokenizer
        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        output = self._create_standard_response(
            response,
            len(tokenizer.encode(message, disallowed_special=())),
            len(tokenizer.encode(response, disallowed_special=())),
            memory_construction_time,
            query_time_len
        )
        self.agent_start_time = time.time()  # Reset time
        return output
    
    def _process_letta_message(self, formatted_message, memorizing, query_id, context_id):
        """Process message with Letta client based on mode."""
        from letta_client import Letta, MessageCreate
        
        try:
            if self.letta_mode == 'insert':
                if memorizing:
                    self.client.server.passage_manager.insert_passage(
                        agent_state=self.agent_state,
                        agent_id=self.agent_state.id,
                        text=formatted_message,
                        actor=self.client.user,
                    )
                    # import ipdb; ipdb.set_trace()
                    return "Memorized"
                else:
                    response = self.client.send_message(
                        agent_id=self.agent_state.id,
                        message=formatted_message,
                        role='user')
                    ## save response.messages to a file / for debugging as JSON     
                    return json.loads(response.messages[-3].tool_call.arguments)['message']
            
            elif self.letta_mode == 'chat':
                response = self.client.send_message(
                    agent_id=self.agent_state.id,
                    message=formatted_message,
                    role='user')
                
                if memorizing:
                    return "Memorized"
                else:
                    ## save response.messages to a file / for debugging as JSON    
                    return json.loads(response.messages[-3].tool_call.arguments)['message']
            elif self.letta_mode == 'api':
                response = self.client.agents.messages.create(
                    agent_id=self.agent_state.id,
                    messages=[
                        MessageCreate(
                            role="user",
                            content=formatted_message,
                        ),
                    ],
                )
                print(f"\n\n\nresponse: {response}\n\n\n")
                return response.messages[-1].content
        except Exception as e:
            print(f"\n\n\nerror: {e}\n\n\n")
            return "Error"

    def _handle_cognee_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Cognee agents."""
        import cognee
        import asyncio
        dataset_name = f'default_dataset_{self.sub_dataset}_context_{context_id}'
        
        if memorizing:
            # Add context to Cognee knowledge base
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            
            # Add text to cognee and generate knowledge graph
            asyncio.run(cognee.add(formatted_message, dataset_name=dataset_name))
            asyncio.run(cognee.cognify(datasets=[dataset_name], chunk_size=self.chunk_size))

            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"
        else:                    
            # Query the knowledge graph
            memory_construction_time = time.time() - self.agent_start_time
            searched_results = asyncio.run(cognee.search(
                query_text=message, 
                top_k=self.retrieve_num, 
                datasets=[dataset_name]
            ))
                    
            # Format results
            total_results = ("".join([f"{result}\n" for result in searched_results]) 
                           if searched_results else "No results found.")
            
            # Return formatted output
            tokenizer = self.tokenizer
            query_time_len = time.time() - self.agent_start_time - memory_construction_time
            output = self._create_standard_response(
                total_results,
                len(tokenizer.encode(self.context, disallowed_special=())),
                len(tokenizer.encode(total_results, disallowed_special=())),
                memory_construction_time,
                query_time_len
            )
            self.agent_start_time = time.time()  # Reset time
            return output

    def _retrieval_query(self, message):
        """Recover the bare user question from MABench's qa-wrapped query for
        RETRIEVAL embedding only.

        MABench wraps every question in dataset-specific instruction boilerplate
        via the 'query' template (e.g. FactConsolidation prepends a long
        "Pretend you are a knowledge management system ... find the newest fact
        ... Now Answer the Question: Based on the provided Knowledge Pool, " and
        appends " Answer:"; see utils/templates.py). Embedding that WHOLE wrapped
        string for similarity search lets the shared boilerplate dominate the
        vector, diluting the actual question and pushing the genuinely relevant
        memory out of top-k (diagnosed in experiment_results.md §5.6: FC-SH 64k
        has_pair GT_new recall@100 = 79% wrapped vs 100% raw). The correct mem0
        retrieval contract is to embed the REAL question against each memory
        unit, so here we strip the known template prefix/suffix to recover it.
        The full wrapped `message` is still used downstream for inference, so the
        "find the newest fact" instruction is preserved.
        """
        try:
            tmpl = get_template(self.sub_dataset, 'query', self.agent_name)
        except Exception:
            return message
        if not isinstance(tmpl, str) or '{question}' not in tmpl:
            return message
        prefix, suffix = tmpl.split('{question}', 1)
        q = message
        if prefix and q.startswith(prefix):
            q = q[len(prefix):]
        if suffix and q.endswith(suffix):
            q = q[:len(q) - len(suffix)]
        q = q.strip()
        return q or message

    def _handle_mem0_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Mem0 agents."""
        user_id = f'context_{context_id}_{self.sub_dataset}'
        if memorizing:
            system_message = get_template(self.sub_dataset, 'system', self.agent_name)
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))

            memory_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": formatted_message},
                {"role": "assistant", "content": "I'll make sure to add the content into the memory."}
            ]

            print(f"[mem0 debug] memorize ctx={context_id} qid={query_id} user_id={user_id}")
            print(f"[mem0 debug] memory_messages[1].content len={len(memory_messages[1]['content'])} chars")
            print(f"[mem0 debug] memory_messages[1].content head: {memory_messages[1]['content'][:300]!r}")
            vector_results = self.memory.add(memory_messages, user_id=user_id)
            print(f"\n\n\nvector_results: {vector_results}\n\n\n")

            # Save ingestion results (extracted facts)
            ingest_save_dir = f"./outputs/rag_retrieved/{self.agent_name}/k_{self.retrieve_num}/{self.sub_dataset}/chunksize_{self.chunk_size}"
            os.makedirs(ingest_save_dir, exist_ok=True)
            ingest_path = os.path.join(ingest_save_dir, f"ingestion_context_{context_id}.jsonl")
            with open(ingest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"query_id": query_id, "vector_results": vector_results}, ensure_ascii=False) + "\n")

            return "Memorized"
        else:
            # Retrieve relevant memories and generate response
            memory_construction_time = time.time() - self.agent_start_time
            retrieval_query = self._retrieval_query(message)
            if retrieval_query != message:
                print(f"[mem0 retrieval] embedding bare question (len {len(retrieval_query)}) "
                      f"instead of wrapped query (len {len(message)}); head: {retrieval_query[:120]!r}")
            relevant_memories = self.memory.search(query=retrieval_query, user_id=user_id, limit=self.retrieve_num)
            print(f"\n\n\nrelevant_memories: {relevant_memories}\n\n\n")

            _results = relevant_memories["results"]
            # Query-time memory resolution, gated by MEM0_QUERY_MODE (0615):
            #   unset / "none"  -> raw top-k (vanilla / "conservative store, no
            #                      resolve" ablation); byte-identical to upstream.
            #   "structural"    -> Phase 0: (S,P) group + temporal argmax.
            #   "phase2"        -> Phase 2: conditional routing + structural resolve
            #                      + LLM dynamic grouping.
            # Both operate on the SAME top-k retrieval and feed the SAME inference
            # prompt; only memory selection differs.
            _qmode = os.environ.get("MEM0_QUERY_MODE")
            if _qmode is None and os.environ.get("MEM0_ADD_MODE") == "phase0_structural":
                _qmode = "structural"  # backward-compat for early phase0 runs
            _final = None  # default; structural / else / exception all keep None
            _q_llm_recency = (_qmode == "q_llm_recency")  # controls llm_messages construction below
            _dont_ask = (_qmode == "dont_ask")  # Don't Ask: mode produces the answer itself, skip answer-gen call
            _dont_ask_result = None  # (answer, in_tok, out_tok, messages) when dont_ask succeeds
            try:
                if _qmode == "structural":
                    from methods.phase0_query import assemble_context, group_and_resolve
                    _id2item = {str(e["id"]): e for e in _results}
                    _resolved, _ungrouped = group_and_resolve(list(_id2item.keys()), _id2item)
                    memories_str = assemble_context(_resolved, _ungrouped)
                elif _qmode == "phase2":
                    from methods.phase2_query import phase2_resolve
                    _final = phase2_resolve(_results, message)
                    memories_str = "\n".join(f"- {e['memory']}" for e in _final)
                elif _qmode == "q_llm_recency":
                    # Q-llm-recency baseline: naive fact-level RAG + LLM does recency
                    # judgment from ordinal-prefixed facts. Directly tests C1
                    # ("LLM 全程不參與 recency 裁決"): swap deterministic argmax(ord)
                    # for LLM inference over the same store.
                    # Design:
                    #  - top-K = 10 (recall saturated per §M-4 audit; avoid weak-backbone
                    #    attention degradation on long prompt)
                    #  - Format "<ordinal>. <fact>" matching raw FC dataset serial
                    #    number convention referenced by factconsolidation.rag_agent
                    #    template ("newer fact has larger serial number")
                    #  - Message structure follows origin _handle_embedding_rag: memories
                    #    + rag_agent template with question go together in USER turn;
                    #    SYSTEM turn = factconsolidation.system
                    _K = int(os.environ.get("MEM0_Q_LLM_RECENCY_TOPK", "10"))
                    memories_str = "\n".join(
                        f"{(e.get('metadata') or {}).get('ordinal', -1)}. {e['memory']}"
                        for e in _results[:_K]
                    )
                elif _qmode == "q_llm_recency_two_stage":
                    # Two-stage Q-llm-recency: separates recency
                    # filtering from answer generation so Stage 2 stays
                    # byte-identical to the standard else-branch (ours main /
                    # vanilla / b) — enabling fair comparison on datasets whose
                    # native rag_agent lacks a recency instruction (e.g. LME).
                    # Stage 1 prompt mirrors FC-SH `factconsolidation.rag_agent`
                    # (recency rule + "solve conflicts by finding newest fact
                    # with larger serial") but re-targets output from "Answer:"
                    # to "Selected serial:" so downstream code can filter to
                    # winners. Fallback (0 valid ordinals returned): use full
                    # top-K — fallback rate is a natural diagnostic of Q-llm-
                    # recency's LLM judgment reliability on the dataset.
                    _K = int(os.environ.get("MEM0_Q_LLM_RECENCY_TOPK", "100"))
                    _topK = _results[:_K]
                    _numbered = "\n".join(
                        f"{(e.get('metadata') or {}).get('ordinal', -1)}. {e['memory']}"
                        for e in _topK
                    )
                    _s1_sys = (
                        "You are a helpful assistant that can read the context "
                        "and memorize it for future retrieval."
                    )
                    _s1_user = (
                        f"{_numbered}\n\n"
                        "Pretend you are a knowledge management system. Each "
                        "fact in the knowledge pool above is provided with a "
                        "serial number at the beginning, and the newer fact "
                        "has larger serial number.\n"
                        "You need to solve the conflicts of facts in the "
                        "knowledge pool by finding the newest fact with larger "
                        "serial number. You need to identify the winning fact "
                        "based on this rule **only** from the knowledge pool "
                        "you have memorized rather than the real facts in real "
                        "world.\n\n"
                        "For example:\n\n"
                        "[Knowledge Pool]\n"
                        "1. The name of the current president of Russia is Vladimir Putin.\n"
                        "5. The name of the current president of Russia is Donald Trump.\n\n"
                        "Question: Based on the provided Knowledge Pool, what "
                        "is the name of the current president of Russia?\n"
                        "Selected serial: 5\n\n"
                        f"Now identify the serial for the Question: Based on "
                        f"the provided Knowledge Pool, {retrieval_query}\n"
                        "Selected serial:"
                    )
                    _s1_messages = [
                        {"role": "system", "content": _s1_sys},
                        {"role": "user", "content": _s1_user},
                    ]
                    _s1_resp, _s1_in_tok, _s1_out_tok = self._answer_with_client(_s1_messages)
                    _retrieved_ords = {(e.get("metadata") or {}).get("ordinal", -1) for e in _topK}
                    _ord_to_item = {(e.get("metadata") or {}).get("ordinal", -1): e for e in _topK}
                    _raw_ords = [int(x) for x in re.findall(r"\d+", _s1_resp or "")]
                    _selected_ords = []
                    _seen_ords = set()
                    for _o in _raw_ords:
                        if _o in _retrieved_ords and _o not in _seen_ords:
                            _selected_ords.append(_o)
                            _seen_ords.add(_o)
                    _fallback = not _selected_ords
                    if _fallback:
                        _winners = _topK
                        print(f"[q_llm_recency_2s] Stage 1 returned 0 valid "
                              f"ordinals; fallback to full top-{_K}. "
                              f"raw_resp={(_s1_resp or '')[:120]!r}")
                    else:
                        _winners = [_ord_to_item[_o] for _o in _selected_ords]
                    memories_str = "\n".join(f"- {e['memory']}" for e in _winners)
                    _s1_log_dir = os.environ.get("MEM0_CAND_LOG_DIR")
                    if _s1_log_dir:
                        try:
                            os.makedirs(_s1_log_dir, exist_ok=True)
                            with open(os.path.join(_s1_log_dir, "q_llm_recency_2s_stage1.jsonl"),
                                      "a", encoding="utf-8") as _f:
                                _f.write(json.dumps({
                                    "query_id": query_id,
                                    "context_id": context_id,
                                    "n_retrieved": len(_topK),
                                    "n_selected": len(_winners),
                                    "selected_ords": _selected_ords,
                                    "fallback": _fallback,
                                    "raw_resp": _s1_resp,
                                    "s1_in_tok": _s1_in_tok,
                                    "s1_out_tok": _s1_out_tok,
                                }, ensure_ascii=False) + "\n")
                        except Exception as _e:
                            print(f"[q_llm_recency_2s log] failed: {_e}")
                elif _qmode == "dont_ask":
                    # Don't Ask (Reddy & Challaram, 2026) — faithful port to LME-KU.
                    # Reference mechanism (maxserial): ONE LLM candidate-extraction
                    # over an ordinal-numbered top-K pool, then DETERMINISTIC
                    # max(serial) selects the freshest matching candidate; that
                    # candidate's answer_entity IS the final answer — there is NO
                    # answer-generation LLM call. Recency signal = per-user_id mem0
                    # `ordinal` (chronological: later session -> larger ordinal),
                    # i.e. the SAME bank ours uses (this branch runs query-only over
                    # ours_no_p5's populated store). Mechanism unchanged from the
                    # reference; only the extraction prompt is reworded for LME's
                    # first-person conversational personal facts (vs the original FC
                    # (S,P) factoid prompt that required a verbatim named subject).
                    _K = int(os.environ.get("MEM0_DONT_ASK_TOPK", "100"))
                    _topK = _results[:_K]
                    _numbered = "\n".join(
                        f"{(e.get('metadata') or {}).get('ordinal', -1)}. {e['memory']}"
                        for e in _topK
                    )
                    _ord_set = {(e.get("metadata") or {}).get("ordinal", -1) for e in _topK}
                    _da_tmpl = (
                        "You are given retrieved items from a user's personal memory. "
                        "Each item has a FRESHNESS marker (the prefix integer) — a "
                        "higher marker means the item was recorded more recently.\n\n"
                        "Your job: identify EVERY item that DIRECTLY answers the "
                        "question, and extract the answer from each such item.\n\n"
                        "Do NOT compare freshness markers. Do NOT pick a \"best\" one. "
                        "Include ALL items that answer the question.\n\n"
                        "Rules:\n"
                        "1. An item answers the question only if it states the specific "
                        "piece of information the question asks about (e.g. the user's "
                        "current city, job, or preference). A related-but-different "
                        "topic does NOT count.\n"
                        "2. Most items describe the user (\"User ...\"); treat the user "
                        "as the subject unless the question names someone else.\n"
                        "3. If the user has multiple conflicting values recorded at "
                        "different freshness markers, INCLUDE BOTH as separate "
                        "candidates. Do not pick.\n"
                        "4. If no item answers the question, return an empty list.\n"
                        "5. Copy the item's text verbatim into `fact_text`, and put the "
                        "concise answer to the question into `answer_entity`.\n\n"
                        "Question: {question}\n\n"
                        "Items:\n{pool}\n\n"
                        "Return ONLY valid JSON: {{\"candidates\": [{{\"serial\": <int>, "
                        "\"fact_text\": \"<verbatim>\", \"answer_entity\": "
                        "\"<answer>\"}}, ...]}}"
                    )
                    _da_sys = "You are a careful information-extraction system that outputs only JSON."
                    _da_user = _da_tmpl.format(question=retrieval_query, pool=_numbered)
                    _da_messages = [
                        {"role": "system", "content": _da_sys},
                        {"role": "user", "content": _da_user},
                    ]
                    _da_resp, _da_in_tok, _da_out_tok = self._answer_with_client(_da_messages)
                    # Parse candidates (robust: direct json, then first {...} block).
                    _cands = []
                    try:
                        _parsed = json.loads(_da_resp)
                        _cands = _parsed.get("candidates", []) if isinstance(_parsed, dict) else []
                    except Exception:
                        _m = re.search(r"\{.*\}", _da_resp or "", re.S)
                        if _m:
                            try:
                                _cands = (json.loads(_m.group(0)) or {}).get("candidates", [])
                            except Exception:
                                _cands = []
                    # Deterministic freshness pick: max serial among extracted
                    # candidates whose serial is actually in the retrieved pool.
                    _valid = []
                    for _c in (_cands or []):
                        try:
                            _s = int(_c.get("serial"))
                        except (TypeError, ValueError):
                            continue
                        if _s in _ord_set:
                            _valid.append((_s, str(_c.get("answer_entity", "")).strip()))
                    if _valid:
                        _chosen = max(_valid, key=lambda t: t[0])
                        _da_answer = _chosen[1] or "no answer"
                        _chosen_serial = _chosen[0]
                    else:
                        _da_answer = "no answer"
                        _chosen_serial = None
                    memories_str = _numbered
                    _dont_ask_result = (_da_answer, _da_in_tok, _da_out_tok, _da_messages)
                    _da_log_dir = os.environ.get("MEM0_CAND_LOG_DIR")
                    if _da_log_dir:
                        try:
                            os.makedirs(_da_log_dir, exist_ok=True)
                            with open(os.path.join(_da_log_dir, "dont_ask_extract.jsonl"),
                                      "a", encoding="utf-8") as _f:
                                _f.write(json.dumps({
                                    "query_id": query_id,
                                    "context_id": context_id,
                                    "n_retrieved": len(_topK),
                                    "n_candidates": len(_valid),
                                    "chosen_serial": _chosen_serial,
                                    "answer": _da_answer,
                                    "raw_resp": _da_resp,
                                    "in_tok": _da_in_tok,
                                    "out_tok": _da_out_tok,
                                }, ensure_ascii=False) + "\n")
                        except Exception as _e:
                            print(f"[dont_ask log] failed: {_e}")
                else:
                    memories_str = "\n".join(f"- {entry['memory']}" for entry in _results)
            except Exception as _e:
                print(f"[query-resolve {_qmode}] failed, raw fallback: {_e}")
                memories_str = "\n".join(f"- {entry['memory']}" for entry in _results)

            # Mem0g-prompt-aware variant: verbalize graph relations into the
            # system prompt, following the upstream mem0 cookbook pattern
            # ("Choose Vector vs Graph Memory" — Expected behavior: graph
            # memory returns the direct answer plus the relationship chain).
            # MABench-as-is path (mem0_prompt_aware_graph=False) keeps the
            # original vector-only prompt.
            if _dont_ask and _dont_ask_result is not None:
                # Don't Ask: the answer was already produced by candidate
                # extraction + deterministic max(serial); there is NO
                # answer-generation call (faithful to the reference pipeline).
                response_text, prompt_tokens, completion_tokens, llm_messages = _dont_ask_result
                system_prompt = llm_messages[0]["content"]
            elif _q_llm_recency:
                # Route via origin factconsolidation.rag_agent template + short system
                # message (per _handle_embedding_rag pattern). Memories + query template
                # go in USER turn; SYSTEM turn is the short factconsolidation.system.
                # Uses `retrieval_query` (stripped bare question) to avoid template
                # nesting when the incoming `message` already carries an outer wrapper.
                # Template dataset override: for cross-dataset canonical prompt
                # (e.g. LME run wants factconsolidation's recency-aware rag_agent
                # rather than LME's chat-history rag_agent, so the "LLM does
                # recency" arm gets the same recency instruction across datasets).
                _tmpl_ds = os.environ.get("MEM0_Q_LLM_RECENCY_TEMPLATE_DS") or self.sub_dataset
                _sys = get_template(_tmpl_ds, 'system', self.agent_name)
                _q_tmpl = get_template(_tmpl_ds, 'query', 'rag_agent')
                _wrapped_q = _q_tmpl.format(question=retrieval_query)
                _user_content = memories_str + "\n" + _wrapped_q
                llm_messages = [
                    {"role": "system", "content": _sys},
                    {"role": "user", "content": _user_content},
                ]
                # For introspection/logging parity with the standard branch below
                system_prompt = _sys
            else:
                # Mem0g-prompt-aware variant: verbalize graph relations into the
                # system prompt, following the upstream mem0 cookbook pattern
                # ("Choose Vector vs Graph Memory" — Expected behavior: graph
                # memory returns the direct answer plus the relationship chain).
                # MABench-as-is path (mem0_prompt_aware_graph=False) keeps the
                # original vector-only prompt.
                if getattr(self, "mem0_prompt_aware_graph", False):
                    rels = relevant_memories.get("relations") or []
                    rels_str = "\n".join(
                        f"- {r.get('source','?')} --[{r.get('relationship','?')}]--> "
                        f"{r.get('destination','?')}"
                        for r in rels
                    )
                    system_prompt = (
                        "You are a helpful AI. Answer the question based on the facts "
                        "and the relationship graph below.\n"
                        f"Facts:\n{memories_str}\n\n"
                        f"Relationships:\n{rels_str}\n"
                    )
                else:
                    # Generate assistant response (routed via _answer_with_client to support
                    # both OpenAI and Vertex/AI-Studio Gemini)
                    system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\n{memories_str}\n"
                llm_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message + "\n\nCurrent Time: " + time.strftime("%Y-%m-%d %H:%M:%S")}
                ]
            if not (_dont_ask and _dont_ask_result is not None):
                response_text, prompt_tokens, completion_tokens = self._answer_with_client(llm_messages)

            memory_retrieval_length = len(self.tokenizer.encode(memories_str, disallowed_special=()))
            query_time_len = time.time() - self.agent_start_time - memory_construction_time
            print(f"\nmemory_length: {memory_retrieval_length}\n")

            output = self._create_standard_response(
                response_text,
                prompt_tokens + memory_retrieval_length,
                completion_tokens,
                memory_construction_time,
                query_time_len
            )
            self.agent_start_time = time.time()  # Reset time

            # Save retrieved memories and full response.
            # For mem0g (graph enabled), relevant_memories also contains 'relations'.
            save_dir = f"./outputs/rag_retrieved/{self.agent_name}/k_{self.retrieve_num}/{self.sub_dataset}/chunksize_{self.chunk_size}/query_{query_id}_context_{context_id}.json"
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            with open(save_dir, "w", encoding="utf-8") as f:
                json.dump({
                    "retrieved_memories": relevant_memories.get("results", []),
                    "retrieved_relations": relevant_memories.get("relations", []),  # mem0g only
                    "resolved_pool": _final if _final is not None else None,  # post KU-resolution pool (M2 metric)
                    "memories_str": memories_str,
                    "system_prompt": system_prompt,
                    "user_message": llm_messages[1]["content"],
                    "response": response_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }, f, ensure_ascii=False, indent=2)

            return output
    
    # Zep
    def _handle_zep_agent(self, message, memorizing, query_id, context_id):
        """Handle Zep processing."""
        import inspect
        from zep_cloud import Message
        from methods.zep import compose_search_context, llm_response, get_retrieval_query, construct_messages

        # ── enhanced Zep logging ──
        # Log dir gated by ZEP_LOG_DIR env. When set, capture:
        #   graph_add.jsonl         : each graph.add call + response
        #   add_context.jsonl       : each thread.add_messages with return_context=True
        #   search_results.jsonl    : each graph.search full return payload per scope
        # No-op when unset. See project_zep_timeout_wait_cloud memory.
        _zep_log_dir = os.environ.get("ZEP_LOG_DIR")

        def _zep_to_dict(obj):
            """Convert Zep SDK object to JSON-serializable dict."""
            if obj is None:
                return None
            if hasattr(obj, "model_dump"):
                try:
                    return obj.model_dump(mode="json")
                except Exception:
                    pass
            if hasattr(obj, "dict"):
                try:
                    return obj.dict()
                except Exception:
                    pass
            if isinstance(obj, list):
                return [_zep_to_dict(x) for x in obj]
            return str(obj)

        def _zep_log(name, payload):
            if not _zep_log_dir:
                return
            try:
                os.makedirs(_zep_log_dir, exist_ok=True)
                with open(os.path.join(_zep_log_dir, f"{name}.jsonl"), "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception as _e:
                logging.error(f"[zep-log {name}] failed: {_e}")

        # user id / session id / oai client
        user_id = f'user_{context_id}_{self.sub_dataset}'
        graph_id = f'graph_{context_id}_{self.sub_dataset}'
        thread_id = f'thread_{context_id}_{self.sub_dataset}'
                
        # check the context id for user and session creation
        if self.context_id != context_id and memorizing:
            # Idempotent creation: ignore "already exists" errors
            for op_name, op in [
                ("user.add", lambda: self.client.user.add(user_id=user_id)),
                ("thread.create", lambda: self.client.thread.create(thread_id=thread_id, user_id=user_id)),
                ("graph.create", lambda: self.client.graph.create(graph_id=graph_id)),
            ]:
                try:
                    op()
                except Exception as e:
                    if "already exists" in str(e).lower() or "400" in str(e):
                        print(f"  {op_name} skipped (already exists): {user_id}/{graph_id}")
                    else:
                        raise
            self.context_id = context_id
        else:
            pass
            
        # Query-only mode (ablation-friendly): skip ingest when graph
        # is already populated on Zep cloud from a prior run. Also skips the
        # async-wait block below since graph is assumed stable. Env-gated:
        #   ZEP_QUERY_ONLY=1  → skip graph.add + thread.add_messages + initial wait
        # No-op when unset.
        _zep_query_only = os.environ.get("ZEP_QUERY_ONLY") == "1"

        if memorizing:
            if _zep_query_only:
                return "Memorized (query-only skip)"
            # graph add
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            content = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            _t_ga = time.time()
            graph_add_resp = self.client.graph.add(
                graph_id=graph_id,
                type="text",
                data=content[:9998]
            )
            _zep_log("graph_add", {
                "context_id": context_id, "graph_id": graph_id,
                "content_chars": len(content),
                "content_truncated": len(content) > 9998,
                "wall_s": round(time.time() - _t_ga, 3),
                "response": _zep_to_dict(graph_add_resp),
            })

            # # thread add
            messages = construct_messages(content, user_id)
            _t_ta = time.time()
            add_msgs_resp = self.client.thread.add_messages(
                thread_id=thread_id, messages=messages, return_context=True
            )
            _zep_log("add_context", {
                "context_id": context_id, "thread_id": thread_id,
                "n_messages": len(messages),
                "wall_s": round(time.time() - _t_ta, 3),
                "context": _zep_to_dict(getattr(add_msgs_resp, "context", None)),
                "response": _zep_to_dict(add_msgs_resp),
            })
            return "Memorized"
        else:
            # Wait for Zep async processing on first query for this context.
            # Probe-based waiter with EPISODE STABILITY CHECK: sleep 360s initial
            # then poll episode count; when count is stable across two consecutive
            # probes -> graph fully ingested. Up to 8 probes = ~40 min extra.
            # v2 fix: previous edge-count probe with limit=3 passed silently at 32k
            # because Zep returned first 3 edges even when only ~10/60 chunks were
            # processed. Episode count (== graph.add calls) is a direct progress signal.
            if _zep_query_only:
                # Skip wait entirely — graph assumed populated + stable from prior run
                self._zep_waited_for_context = context_id
                print(f"\n[zep-query-only] skipping wait for context {context_id} (assumed cached)")

            if not getattr(self, '_zep_waited_for_context', None) == context_id:
                # Length-adaptive wait: 262k needs much longer for Zep async graph build.
                # 32k/64k default: 360s + up to 8×300s probes = ~46min max.
                # 262k: 1200s + up to 20×300s probes = ~120min max.
                # Override via ZEP_INITIAL_WAIT / ZEP_MAX_PROBES env.
                _is_262k = "262k" in self.sub_dataset
                initial_wait = int(os.environ.get("ZEP_INITIAL_WAIT",
                                                  1200 if _is_262k else 360))
                max_probes = int(os.environ.get("ZEP_MAX_PROBES",
                                                20 if _is_262k else 8))
                print(f"\nWaiting {initial_wait}s for Zep async processing of context {context_id} (max_probes={max_probes})...")
                time.sleep(initial_wait)
                elapsed = initial_wait
                last_n_episodes = -1
                for probe in range(max_probes):  # up to max_probes after initial wait
                    try:
                        # Zep's graph.search caps `limit` at 50 (400 otherwise); 50
                        # episodes is plenty for the stability check (a plateau at the
                        # cap across two probes still signals the graph is populated).
                        probe_res = self.client.graph.search(graph_id=graph_id, query="a the of", scope='episodes', limit=50)
                        n_epis = len(probe_res.episodes) if probe_res and probe_res.episodes else 0
                        if n_epis > 0 and n_epis == last_n_episodes:
                            print(f"  Zep graph ready after {elapsed}s (episodes stable at {n_epis})")
                            break
                        if probe == max_probes - 1:
                            print(f"  WARNING: Zep episodes={n_epis} after {elapsed}s (not stabilized after {max_probes} probes); proceeding")
                            break
                        print(f"  Probe {probe+1}/{max_probes} at {elapsed}s: {n_epis} episodes (was {last_n_episodes}); sleeping 300s more...")
                        last_n_episodes = n_epis
                        time.sleep(300)
                        elapsed += 300
                    except Exception as e:
                        print(f"  Probe {probe+1}/{max_probes} exception ({e.__class__.__name__}): {e}; sleeping 300s more...")
                        time.sleep(300)
                        elapsed += 300
                self._zep_waited_for_context = context_id

            memory_construction_time = time.time() - self.agent_start_time

            # graph search with retry logic (Zep graph might still be processing)
            retrieval_query = get_retrieval_query(message)
            print(f"\n\n\nretrieval_query: {retrieval_query}\n\n\n")

            # Ablation env:
            #   ZEP_TOP_K       overrides self.retrieve_num (default 10 per Zep official)
            #                   → test if long-context (262k) crash is top-K cap driven
            #   ZEP_EDGES_ONLY  skip nodes + episodes fetching + compose
            #                   → isolate fact-level edges as the only memory granularity
            # Both env-gated, byte-identical to before when unset. See ZEP 262k ablation §.
            _zep_top_k = int(os.environ.get("ZEP_TOP_K", str(self.retrieve_num)))
            _zep_edges_only = os.environ.get("ZEP_EDGES_ONLY") == "1"

            def _search_with_retry(scope, retries=3, wait=30):
                for attempt in range(retries):
                    try:
                        return self.client.graph.search(graph_id=graph_id, query=retrieval_query[:399], scope=scope, limit=_zep_top_k)
                    except Exception as e:
                        if attempt < retries - 1:
                            print(f"  Zep search {scope} failed ({e.__class__.__name__}), retry {attempt+1}/{retries} after {wait}s...")
                            time.sleep(wait)
                        else:
                            raise
                return None

            _t_es = time.time(); edges_full = _search_with_retry('edges'); _t_ne = time.time() - _t_es
            if _zep_edges_only:
                nodes_full = None; eps_full = None; _t_nn = 0.0; _t_np = 0.0
            else:
                _t_es = time.time(); nodes_full = _search_with_retry('nodes'); _t_nn = time.time() - _t_es
                _t_es = time.time(); eps_full = _search_with_retry('episodes'); _t_np = time.time() - _t_es
            edges_results = edges_full.edges if edges_full else None
            node_results = nodes_full.nodes if nodes_full else None
            episode_results = eps_full.episodes if eps_full else None

            _zep_log("search_results", {
                "query_id": query_id, "context_id": context_id,
                "retrieval_query": retrieval_query,
                "edges": {"wall_s": round(_t_ne, 3),
                          "results": [_zep_to_dict(x) for x in (edges_results or [])]},
                "nodes": {"wall_s": round(_t_nn, 3),
                          "results": [_zep_to_dict(x) for x in (node_results or [])]},
                "episodes": {"wall_s": round(_t_np, 3),
                             "results": [_zep_to_dict(x) for x in (episode_results or [])]},
            })

            # thread search / currently we do not use the thread info
            # Retry with backoff on rate-limit / transient (FREE plan caps
            # get_user_context at 5/min — must wait per response Retry-After).
            _t_uc = time.time()
            _uc_retries = 6
            for _uc_i in range(_uc_retries):
                try:
                    memory = self.client.thread.get_user_context(thread_id=thread_id)
                    break
                except Exception as _uce:
                    _emsg = str(_uce)
                    if "429" in _emsg or "rate limit" in _emsg.lower():
                        _wait = 45 if _uc_i < 3 else 90
                        print(f"  get_user_context 429 rate-limit, retry {_uc_i+1}/{_uc_retries} after {_wait}s")
                        time.sleep(_wait)
                        continue
                    if _uc_i == _uc_retries - 1:
                        raise
                    print(f"  get_user_context transient ({_uce.__class__.__name__}), retry {_uc_i+1}/{_uc_retries} after 30s")
                    time.sleep(30)
            context_block = memory.context
            _zep_log("user_context", {
                "query_id": query_id, "context_id": context_id,
                "wall_s": round(time.time() - _t_uc, 3),
                "context_block": context_block,
                "memory_response": _zep_to_dict(memory),
            })

            # Prompt an LLM with relevant context
            retrieved_context = compose_search_context(edges_results, node_results, context_block, episode_results)
            import asyncio
            response = asyncio.run(llm_response(self.oai_client, retrieved_context, message))
            query_time_len = time.time() - self.agent_start_time - memory_construction_time

            output = self._create_standard_response(
                response,
                len(self.tokenizer.encode(retrieved_context, disallowed_special=())),
                len(self.tokenizer.encode(response, disallowed_special=())),
                memory_construction_time,
                query_time_len
            )
            self.agent_start_time = time.time()  # Reset time
            
            # save the context + structured graph data
            save_dir = f"./outputs/rag_retrieved/{self.agent_name}/k_{self.retrieve_num}/{self.sub_dataset}/chunksize_{self.chunk_size}/query_{query_id}_context_{context_id}.json"
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)

            def _serialize_edges(edges):
                # Include expired_at so we can attribute (invalid_at, expired_at)
                # -> Graphiti LLM label: (X, None)=temporal-extraction (source-parsed
                # event-end, NOT a KU judgment); (X, X)=contradicted (LLM labeled
                # `contradicted_facts`, deterministic system invalidated); (None,
                # None)=no conflict judged. duplicate_facts label produces NO edge
                # at all (invisible via cloud API).
                if not edges:
                    return []
                return [{"fact": e.fact, "name": e.name,
                         "source_node": e.source_node_uuid, "target_node": e.target_node_uuid,
                         "valid_at": str(e.valid_at) if e.valid_at else None,
                         "invalid_at": str(e.invalid_at) if e.invalid_at else None,
                         "expired_at": str(e.expired_at) if getattr(e, 'expired_at', None) else None,
                         "uuid": str(e.uuid_) if hasattr(e, 'uuid_') else None}
                        for e in edges]

            def _serialize_nodes(nodes):
                if not nodes:
                    return []
                return [{"name": n.name, "summary": n.summary,
                         "uuid": str(n.uuid_) if hasattr(n, 'uuid_') else None}
                        for n in nodes]

            def _serialize_episodes(episodes):
                if not episodes:
                    return []
                return [{"content": ep.content, "name": getattr(ep, 'name', None),
                         "uuid": str(ep.uuid_) if hasattr(ep, 'uuid_') else None}
                        for ep in episodes]

            with open(save_dir, "w", encoding="utf-8") as f:
                paragraphs = [p for p in retrieved_context.replace("\r\n", "\n").split("\n") if p.strip()]
                json.dump({
                    "retrieved_context_paragraphs": paragraphs,
                    "edges": _serialize_edges(edges_results),
                    "nodes": _serialize_nodes(node_results),
                    "episodes": _serialize_episodes(episode_results),
                    "context_block": context_block,
                    "response": response,
                }, f, ensure_ascii=False, indent=2)

            return output
    
    def _handle_rag_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for RAG agents."""
        if memorizing:
            # Add message to chunks and context
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            self.chunks.append(formatted_message)
            self.raw_chunks.append(message)  # raw content for HippoRAG (stable hash_ids)
            self.context_len = self.context_len + self.chunk_size

            # Truncate context if it exceeds limits
            if self.context_len > self.input_length_limit:
                self.chunks = self.chunks[1:]
                self.raw_chunks = self.raw_chunks[1:]
                self.context_len = self.context_len - self.chunk_size
            return ''
        else:
            # Handle query processing for different RAG types
            return self._process_rag_query(message, query_id, context_id)

    def _process_rag_query(self, message, query_id, context_id):
        """Process query for RAG agents with different retrieval strategies."""
                
        # Truncate context if needed
        tokenizer = self.tokenizer
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.input_length_limit:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.input_length_limit:])
        if self.context_len > self.input_length_limit:
            self.chunks = self.chunks[1:]
            self.context_len = self.context_len - self.chunk_size
        
        # Route to specific RAG implementation and get result
        rag_handlers = {
            "graph_rag": lambda: self._handle_graph_rag(message, context_id, tokenizer),
            "hippo_rag_v2_nv": lambda: self._handle_hippo_rag(message, context_id, tokenizer),
            "hippo_rag_v2_openai": lambda: self._handle_hippo_rag(message, context_id, tokenizer),
            "rag_bm25": lambda: self._handle_bm25_rag(message, context_id, tokenizer),
            "rag_contriever": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_text_embedding_3_large": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_text_embedding_3_small": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_qwen3_embedding_4b": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_raptor": lambda: self._handle_raptor_rag(message, context_id, tokenizer),
            "self_rag": lambda: self._handle_self_rag(message, context_id, tokenizer),
            "memo_rag": lambda: self._handle_memorag(message, context_id, tokenizer),
        }
        
        # Find matching handler
        handler = next((handler for agent_type, handler in rag_handlers.items() if self._is_agent_type(agent_type)), None)
        if not handler:
            raise NotImplementedError(f"RAG agent type not supported: {self.agent_name}")
        
        output = handler()

        # Save the retrieved context as JSON (if the method provides it)
        if output.get("retrieval_context"):
            save_dir = f"./outputs/rag_retrieved/{self.agent_name}/k_{self.retrieve_num}/{self.sub_dataset}/chunksize_{self.chunk_size}/query_{query_id}_context_{context_id}.json"
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            with open(save_dir, "w") as f:
                json.dump(output["retrieval_context"], f)
            
            # drop the retrieval_context       
            output.pop("retrieval_context")
        
        return output

    def _handle_graph_rag(self, message, context_id, tokenizer):
        """Handle Graph RAG processing."""
        start_time = time.time()

        # Build vectorstore if context changed
        if self.context_id != context_id:
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            try:
                from methods.graph_rag import GraphRAG
                self.graph_rag = GraphRAG(temperature=self.temperature, model_name=self.model, retrieve_num=self.retrieve_num, max_tokens=self.max_tokens)
                self.graph_rag.process_documents(docs)
                memory_construction_time = time.time() - start_time
            except Exception as e:
                print(f"\n\n\n\nError: {e}\n\n\n\n")
            print(f"\n\nGraph RAG build vectorstore finished...\n\n")
        else:
            memory_construction_time = 0
            print(f"\n\nContext {context_id} already processed, skipping Graph RAG build vectorstore...\n\n")

        # Process query
        try:
            response, retrieval_context = self.graph_rag.query(query=message)
        except Exception as e:
            response = f"{e}"
            retrieval_context = "ERROR"
            print(f"\n\n\n\nError: {e}\n\n\n\n")
        
        self.context_id = context_id
        
        print(f"\n\n\n\nResponse: {response}\n\n\n\n")
        if isinstance(response, str):
            response = response
        else:
            response = response.content
        query_time_len = time.time() - start_time - memory_construction_time
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    def _handle_hippo_rag(self, message, context_id, tokenizer):
        """Handle HippoRAG processing."""
        start_time = time.time()
        
        if self.context_id != context_id:
            docs = self.raw_chunks  # raw content (no template wrapper), ensures stable hash_ids across runs
            from methods.hipporag import HippoRAG
            if any(agent_name in self.agent_name for agent_name in ["hippo_rag_v2_nv"]):
                save_dir = os.path.join(f"./outputs/rag_retrieved/NV-Embed-v2", self.sub_dataset, f'chunksize_{self.chunk_size}', f'context_id_{context_id}')
                embedding_model_name = 'nvidia/NV-Embed-v2'
            elif any(agent_name in self.agent_name for agent_name in ["hippo_rag_v2_openai"]):
                save_dir = os.path.join(f"./outputs/rag_retrieved/OpenAIEmbedding", self.sub_dataset, f'chunksize_{self.chunk_size}', f'context_id_{context_id}') 
                embedding_model_name = 'text-embedding-ada-002'
            
            self.hipporag = HippoRAG(save_dir=save_dir,
                                llm_model_name=self.model,
                                embedding_model_name=embedding_model_name) 
            self.hipporag.index(docs=docs)
            memory_construction_time = time.time() - start_time
            print(f"\n\nHippoRAG build vectorstore finished...\n\n")
        else:
            memory_construction_time = 0
            print(f"\n\nContext {context_id} already processed, skipping HippoRAG build vectorstore...\n\n")
            
        # Retrieve and answer
        queries = [message]
        retrieval_results, top_k_docs = self.hipporag.retrieve(queries=queries, num_to_retrieve=self.retrieve_num)
        
        qa_results = self.hipporag.rag_qa(retrieval_results)
        response = qa_results[0][0].answer

        retrieval_context = "\n\n".join([f"Passage {i+1}:\n{text}" for i, text in enumerate(top_k_docs)])
        doc_scores = retrieval_results[0].doc_scores
        retrieval_scores = doc_scores.tolist() if doc_scores is not None else []
        query_time_len = time.time() - start_time - memory_construction_time

        self.context_id = context_id

        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
            "retrieval_scores": retrieval_scores,
        }

    # RAG implementation methods
    def _handle_bm25_rag(self, message, context_id, tokenizer):
        """Handle BM25 RAG processing."""
        start_time = time.time()
        
        # Extract retrieval query from message
        retrieval_query = self._extract_retrieval_query(message)
        print(f"\n\n\n\nretrieval_query: {retrieval_query}\n\n\n\n")
        
        # Build vectorstore if context changed
        if self.context_id != context_id:
            from langchain_community.retrievers import BM25Retriever
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            print(f"\n\nBM25 build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping BM25 build vectorstore...\n\n")
        
        # Retrieve documents
        self.bm25_retriever.k = self.retrieve_num
        bm25_documents = self.bm25_retriever.get_relevant_documents(retrieval_query)   
        retrieval_context = [f"{doc.page_content}\n" for doc in bm25_documents] 
        memory_construction_time = time.time() - start_time
        
        # Answer the query
        retrieval_memory_string = "\n".join([f"Memory {i+1}:\n{text}" for i, text in enumerate(retrieval_context)])
        
        # Format the message
        ask_llm_message = retrieval_memory_string + "\n" + message
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        format_message = format_chat(message=ask_llm_message, system_message=system_message)
        
        # Generate response
        response = self._create_oai_client().chat.completions.create(
            model=self.model,
            messages=format_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens if "gpt-4" in self.model else None
        )
        
        query_time_len = time.time() - start_time - memory_construction_time
        self.context_id = context_id
        
        return {
            "output": response.choices[0].message.content,
            "input_len": response.usage.prompt_tokens,
            "output_len": response.usage.completion_tokens,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }
    
    def _extract_retrieval_query(self, message):
        """Extract retrieval query from message using regex patterns."""
        patterns = [
            r"Now Answer the Question:\s*(.*)",
            r"Here is the conversation:\s*(.*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.DOTALL)
            if match:
                return ''.join(match.groups())
        
        return message
        
    def _handle_embedding_rag(self, message, context_id, tokenizer):
        """Handle embedding-based RAG processing (Contriever, Text-embedding models)."""
        from methods.embedding_retriever import TextRetriever, RAGSystem
        
        # Determine embedding model
        if any(agent_name in self.agent_name for agent_name in ["rag_contriever"]):
            embedding_model_name = "facebook/contriever"
        elif any(agent_name in self.agent_name for agent_name in ["rag_text_embedding_3_large"]):
            embedding_model_name = "text-embedding-3-large"
        elif any(agent_name in self.agent_name for agent_name in ["rag_text_embedding_3_small"]):
            embedding_model_name = "text-embedding-3-small"
        elif any(agent_name in self.agent_name for agent_name in ["rag_qwen3_embedding_4b"]):
            embedding_model_name = "Qwen/Qwen3-Embedding-4B"
        else:
            raise NotImplementedError
        
        # Build vectorstore if context changed
        if self.context_id != context_id:
            self.retriever = TextRetriever(embedding_model_name=embedding_model_name)
            self.retriever.build_vectorstore(self.chunks)
            print(f"\n\n{embedding_model_name} build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping {embedding_model_name} build vectorstore...\n\n")
                            
        # Retrieve relevant passages and answer the query
        rag_system = RAGSystem(self.retriever, self.model, self.temperature, self.max_tokens, use_azure=True, azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"), azure_api_key=os.environ.get("AZURE_OPENAI_API_KEY"), azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"))
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        result = rag_system.answer_query(
            query=message, 
            top_k=self.retrieve_num, 
            system_message=system_message
        )
        retrieval_context = result['context_used']
        
        self.context_id = context_id
        
        return {
            "output": result["answer"],
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(result["answer"], disallowed_special=())),
            "memory_construction_time": result.get("memory_construction_time", result.get("memory_construction_time", 0)),
            "query_time_len": result["query_time_len"],
            "retrieval_context": retrieval_context,
        }
        
    def _handle_raptor_rag(self, message, context_id, tokenizer):
        """Handle RAPTOR RAG processing."""
        # Build vectorstore if context changed
        if self.context_id != context_id:
            texts = self.chunks
            from methods.raptor import RAPTORMethod
            self.raptor_method = RAPTORMethod(texts, max_levels=3)
            print(f"\n\nRaptor build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping Raptor build vectorstore...\n\n")
        
        # Retrieve relevant passages and answer the query
        result = self.raptor_method.run(query=message, k=self.retrieve_num)
        response = result['answer']
        retrieval_context = result['context_used']
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": result.get("memory_construction_time", result.get("memory_construction_time", 0)),
            "query_time_len": result["query_time_len"],
            "retrieval_context": retrieval_context,
        }
        
    def _handle_self_rag(self, message, context_id, tokenizer):
        """Handle Self-RAG processing."""
        from methods.self_rag import SelfRAG
        start_time = time.time()
        
        # Build vectorstore if context changed
        if self.context_id != context_id:
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            self.self_rag = SelfRAG(documents=docs, temperature=self.temperature, top_k=self.retrieve_num)
            print(f"\n\nSelf-RAG build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping Self-RAG build vectorstore...\n\n")
        
        # Process query
        try:
            response, retrieval_context_list, memory_construction_time, query_time_len = self.self_rag.run(query=message)
        except Exception as e:
            response = f"{e}"
            retrieval_context_list = ["ERROR"]
            memory_construction_time = 0
            query_time_len = 0
            print(f"\n\n\n\nError: {e}\n\n\n\n")
        
        # Prepare the context
        retrieval_context = "\n\n".join([f"Passage {i+1}:\n{text}" 
                                        for i, text in enumerate(retrieval_context_list)])
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    # memorag
    def _handle_memorag(self, message, context_id, tokenizer):
        """Handle MemoRAG processing."""
        from methods.memorag import Agent, MemoRAG
        start_time = time.time()
        memory_construction_time = 0
        cache_context_save_dir=f"./outputs/rag_retrieved/MemoRAG/{self.sub_dataset}/chunksize_{self.chunk_size}/context_id_{context_id}"
        
        # build rag agent
        if self.context_id != context_id:
            # API configuration
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
            api_key=os.environ.get("AZURE_OPENAI_API_KEY")
            gen_model = Agent(model=self.model, source="azure", temperature=self.temperature, api_dict={"endpoint":endpoint, "api_version":api_version, "api_key":api_key})
            self.MemoRAG = MemoRAG(
                mem_model_name_or_path="TommyChien/memorag-qwen2-7b-inst",
                ret_model_name_or_path="BAAI/bge-m3",   
                customized_gen_model=gen_model,
                ret_hit=self.retrieve_num, 
                retrieval_chunk_size=self.chunk_size
            )
            # Use the loaded context / memorize the context for question answering
            context = " ".join(self.chunks)
            ## load the context from the cache
            if os.path.exists(f'{cache_context_save_dir}/memory.bin'):
                self.MemoRAG.load(cache_context_save_dir, print_stats=True)
            else:
                self.MemoRAG.memorize(context, save_dir=None, print_stats=True)
            memory_construction_time = time.time() - start_time
            print(f"Finish memorizing, time cost {memory_construction_time}")
        else:
            print(f"\n\nContext {context_id} already processed, skipping MemoRAG build vectorstore...\n\n")
            
        # Retrieve and answer
        if self.sub_dataset == "infbench_sum_eng_shots2":
            response, retrieval_context = self.MemoRAG(query=message, task_type="summarize", max_new_tokens=self.max_tokens)
        else:
            response, retrieval_context = self.MemoRAG(query=message, task_type="memorag", max_new_tokens=self.max_tokens)
        
        query_time_len = time.time() - start_time - memory_construction_time
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(str(retrieval_context) + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }
        
    def save_agent(self):
        """Save agent state to disk for persistence."""
        # Currently only implemented for Letta agents
        if not self._is_agent_type("letta") and not self._is_agent_type("zep"):
            print("\n\n Agent not saved (not implemented for this agent type) \n\n")
            return
        
        if self._is_agent_type("letta") and "api" not in self.agent_name:
            agent_save_folder = self.agent_save_to_folder
            os.makedirs(agent_save_folder, exist_ok=True)
            
            import shutil
            # Copy the SQLite database file to the target folder
            source_db_path = os.path.expanduser("~/.letta/sqlite.db")
            target_db_path = f"{agent_save_folder}/sqlite.db"
            shutil.copyfile(source_db_path, target_db_path)
            
            # Save the agent ID for future loading
            with open(f"{agent_save_folder}/agent_id.txt", "w") as f:
                f.write(self.agent_state.id)
        elif self._is_agent_type("zep"):
            # save the message that agent has processed
            messages = "agent finished memorization"
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(f"{self.agent_save_to_folder}/messages.txt", "w") as f:
                f.write(messages)
                
        print("\n\n Agent saved...\n\n")

    def load_agent(self):
        """Load agent state from disk."""
        agent_save_folder = self.agent_save_to_folder
        assert os.path.exists(agent_save_folder), f"Folder {agent_save_folder} does not exist."

        if not self._is_agent_type("letta") and not self._is_agent_type("zep"):
            print("\n\nAgent loading not implemented for this agent type\n\n")
            return None

        if self._is_agent_type("letta") and "api" not in self.agent_name:
            import shutil
            # Copy the database file back to the Letta directory
            source_db_path = f"{agent_save_folder}/sqlite.db"
            target_db_path = os.path.expanduser("~/.letta/sqlite.db")
            shutil.copyfile(source_db_path, target_db_path)

            # Load agent ID and find the corresponding agent state
            with open(f"{agent_save_folder}/agent_id.txt", "r") as f:
                agent_id = f.read()

            # Find the agent state with the matching ID
            for agent_state in self.client.list_agents():
                if agent_state.id == agent_id:
                    self.agent_state = agent_state
                    break
        elif self._is_agent_type("zep"):
            # load the message that agent has processed
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(f"{self.agent_save_to_folder}/messages.txt", "r") as f:
                messages = f.read()
        
        print("\n\n Agent loaded successfully...\n\n")
        