import json
import os
import warnings
from typing import Dict, List, Optional

from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase


class OpenAILLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gpt-4o-mini"

        if os.environ.get("OPENROUTER_API_KEY"):  # Use OpenRouter
            self.client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.config.openrouter_base_url
                or os.getenv("OPENROUTER_API_BASE")
                or "https://openrouter.ai/api/v1",
            )
        else:
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

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using OpenAI.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        # GPT-5 family (and o1/o3/o4 reasoning models) rejected `max_tokens`
        # since March 2026 (HTTP 400 "unsupported_parameter"); require the
        # renamed `max_completion_tokens`. Detect by model-name prefix and
        # route accordingly so older-model callers are byte-identical.
        _model_name = str(self.config.model or "").lower()
        _needs_max_completion = any(
            _model_name.startswith(p) for p in ("gpt-5", "o1", "o3", "o4")
        )
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if self.config.max_tokens is not None:
            params["max_completion_tokens" if _needs_max_completion else "max_tokens"] = self.config.max_tokens

        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_params = {}
            if self.config.models:
                openrouter_params["models"] = self.config.models
                openrouter_params["route"] = self.config.route
                params.pop("model")

            if self.config.site_url and self.config.app_name:
                extra_headers = {
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.app_name,
                }
                openrouter_params["extra_headers"] = extra_headers

            params.update(**openrouter_params)

        if response_format:
            params["response_format"] = response_format
        if tools:  # TODO: Remove tools if no issues found with new memory addition logic
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        import time as _time
        _t0 = _time.time()
        response = self.client.chat.completions.create(**params)
        try:  # env-gated cost log; no-op when MEM0_COST_LOG unset
            from methods.cost_logger import log as _costlog
            _u = getattr(response, "usage", None)
            _costlog(os.environ.get("MEM0_COST_STAGE", "mem0_llm"),
                     params.get("model", self.config.model),
                     getattr(_u, "prompt_tokens", 0), getattr(_u, "completion_tokens", 0),
                     _time.time() - _t0)
        except Exception:
            pass
        return self._parse_response(response, tools)
