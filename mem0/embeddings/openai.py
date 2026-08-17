import os
import warnings
from typing import Literal, Optional

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class OpenAIEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-3-small"
        self.config.embedding_dims = self.config.embedding_dims or 1536

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = (
            self.config.openai_base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        if os.environ.get("OPENAI_API_BASE"):
            warnings.warn(
                "The environment variable 'OPENAI_API_BASE' is deprecated and will be removed in the 0.1.80. "
                "Please use 'OPENAI_BASE_URL' instead.",
                DeprecationWarning,
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        import time as _time
        _t0 = _time.time()
        resp = self.client.embeddings.create(input=[text], model=self.config.model, dimensions=self.config.embedding_dims)
        try:
            from methods.cost_logger import log as _costlog
            _costlog(f"embed_{memory_action or 'na'}", self.config.model,
                     getattr(getattr(resp, "usage", None), "prompt_tokens", 0), 0, _time.time() - _t0)
        except Exception:
            pass
        return resp.data[0].embedding

    def embed_batch(self, texts, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """Embed many texts in as few API calls as possible.

        The OpenAI embeddings endpoint accepts a LIST input (up to 2048 items)
        and returns one vector per item, so this collapses N sequential HTTP
        round-trips into ceil(N/2048) calls. The vectors are byte-identical to
        calling embed() once per text (same model, same per-text input after the
        identical newline normalization) -- this is purely a latency
        optimization and does not change any downstream result. Embeddings are
        returned in the SAME order as `texts`.
        """
        texts = [t.replace("\n", " ") for t in texts]
        # Robustness for weak backbones (GX10 weak-model regime): a weak extractor
        # (e.g. gemma) occasionally emits an empty/whitespace fact, which the OpenAI
        # embeddings endpoint rejects ("input cannot be an empty string"), aborting the
        # whole chunk's batch and the run. Substitute a single space so the list length
        # and index alignment with the caller (new_retrieved_facts / triples / _fact_embs)
        # are preserved; the vector is for a blank fact (semantically inert, never matches
        # a real query). Non-empty facts are untouched -> strong-backbone results unchanged.
        texts = [t if t.strip() else " " for t in texts]
        if not texts:
            return []
        import time as _time
        out = []
        for i in range(0, len(texts), 2048):
            batch = texts[i:i + 2048]
            _t0 = _time.time()
            resp = self.client.embeddings.create(
                input=batch, model=self.config.model, dimensions=self.config.embedding_dims
            )
            try:
                from methods.cost_logger import log as _costlog
                _costlog(f"embed_{memory_action or 'na'}", self.config.model,
                         getattr(getattr(resp, "usage", None), "prompt_tokens", 0), 0, _time.time() - _t0)
            except Exception:
                pass
            # API returns items carrying their input index; sort to guarantee order.
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return out
