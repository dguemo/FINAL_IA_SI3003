"""Cliente LLM compatible con la API de OpenAI (Groq o NVIDIA NIM).

Devuelve mensajes normalizados como dict para que el loop no dependa del SDK
(y se pueda probar con un cliente simulado):

    {"content": str | None, "tool_calls": [{"id", "name", "arguments"}]}
"""
from __future__ import annotations

import logging
import os
import time

import openai

import config

log = logging.getLogger("llm")


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 temperature: float = config.LLM_TEMPERATURE, max_retries: int = 3):
        provider = (provider or config.LLM_PROVIDER).lower()
        if provider not in config.PROVIDERS:
            raise LLMError(f"Proveedor '{provider}' no soportado. Use: {list(config.PROVIDERS)}")
        spec = config.PROVIDERS[provider]
        api_key = os.getenv(spec["api_key_env"])
        if not api_key:
            raise LLMError(f"Falta la variable {spec['api_key_env']} (ver .env.example)")
        self.provider = provider
        self.model = model or config.LLM_MODEL or spec["default_model"]
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = openai.OpenAI(base_url=spec["base_url"], api_key=api_key, timeout=60)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str = "auto") -> dict:
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature,
                      max_tokens=1024)
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice)
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                calls = [{"id": c.id, "name": c.function.name,
                          "arguments": c.function.arguments or "{}"}
                         for c in (msg.tool_calls or [])]
                return {"content": msg.content, "tool_calls": calls}
            except (openai.RateLimitError, openai.APIConnectionError,
                    openai.APITimeoutError, openai.InternalServerError) as err:
                last = err
                wait = 2 ** (attempt + 1)
                log.warning("Error transitorio del LLM (%s). Reintento en %ss", type(err).__name__, wait)
                time.sleep(wait)
            except openai.BadRequestError as err:
                # Groq responde 400 'tool_use_failed' cuando el modelo emite una
                # llamada mal formada: se reintenta una vez con temperatura 0.
                last = err
                if "tool_use_failed" in str(err) and kwargs["temperature"] != 0:
                    kwargs["temperature"] = 0
                    continue
                break
            except openai.APIStatusError as err:
                last = err
                break
        raise LLMError(f"El servicio LLM falló: {last}")
