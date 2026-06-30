"""
Generation service — LLM factory with advanced failover, async support,
latency-aware routing, and optional LiteLLM backend.

Improvements implemented:
  ✅ Multi-provider N-fallback chain: Gemini → Groq → Anthropic Claude
     (any number of providers, not just two).
  ✅ Async ainvoke(): non-blocking invocation for parallel LangGraph branches.
  ✅ Latency-aware routing: exponential moving average (EMA) tracks P50 latency
     per provider; the fastest healthy provider is tried first.
  ✅ LiteLLM integration: optional unified backend — add models via config only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage

from app.config.settings import (
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    LLM_TEMPERATURE,
    GEMINI_MODEL,
    GROQ_MODEL,
)

logger = logging.getLogger(__name__)

# ── Latency tracker ───────────────────────────────────────────────────────────

class _LatencyTracker:
    """Exponential moving average latency tracker per provider."""

    def __init__(self, alpha: float = 0.3):
        self._ema: dict[str, float] = {}
        self._alpha = alpha

    def record(self, name: str, elapsed: float) -> None:
        if name not in self._ema:
            self._ema[name] = elapsed
        else:
            self._ema[name] = self._alpha * elapsed + (1 - self._alpha) * self._ema[name]

    def p50(self, name: str) -> float:
        return self._ema.get(name, float("inf"))

    def fastest(self, names: list[str]) -> list[str]:
        """Return names sorted fastest-first by EMA latency."""
        return sorted(names, key=lambda n: self.p50(n))


_tracker = _LatencyTracker()


# ── Provider descriptor ───────────────────────────────────────────────────────

class _Provider:
    def __init__(self, name: str, llm: Any):
        self.name = name
        self.llm = llm

    def invoke(self, prompt, **kwargs):
        start = time.monotonic()
        result = self.llm.invoke(prompt, **kwargs)
        _tracker.record(self.name, time.monotonic() - start)
        return result

    async def ainvoke(self, prompt, **kwargs):
        start = time.monotonic()
        if hasattr(self.llm, "ainvoke"):
            result = await self.llm.ainvoke(prompt, **kwargs)
        else:
            result = await asyncio.to_thread(self.llm.invoke, prompt, **kwargs)
        _tracker.record(self.name, time.monotonic() - start)
        return result


# ── N-provider failover wrapper ───────────────────────────────────────────────

class FailoverLLMWrapper:
    """
    Duck-typed wrapper that presents .invoke() / .ainvoke() and transparently
    falls through a chain of N providers on quota / rate-limit errors.

    Provider order on each call is latency-aware: the historically fastest
    available provider is tried first.
    """

    _QUOTA_SIGNALS = ("429", "resource_exhausted", "quota", "rate limit", "too many requests",
                      "ratelimit", "rate_limit", "capacity")

    def __init__(self, providers: list[_Provider]):
        if not providers:
            raise ValueError("At least one provider is required.")
        self._providers = {p.name: p for p in providers}
        self._provider_names = [p.name for p in providers]
        self.current_provider: str = providers[0].name
        self.fallback_count: int = 0

    def _ordered(self) -> list[_Provider]:
        names = _tracker.fastest(self._provider_names)
        return [self._providers[n] for n in names]

    def _is_quota(self, exc: Exception) -> bool:
        return any(s in str(exc).lower() for s in self._QUOTA_SIGNALS)

    def invoke(self, prompt, **kwargs):
        last_exc: Exception | None = None
        for provider in self._ordered():
            try:
                self.current_provider = provider.name
                result = provider.invoke(prompt, **kwargs)
                logger.debug("Provider %s succeeded", provider.name)
                return result
            except Exception as exc:
                if self._is_quota(exc):
                    self.fallback_count += 1
                    last_exc = exc
                    logger.warning("Provider %s quota hit — trying next. (fallback #%d)",
                                   provider.name, self.fallback_count)
                    continue
                raise  # non-quota errors propagate immediately

        logger.error("All providers exhausted. Last error: %s", last_exc)
        return AIMessage(content=(
            "Error: All AI providers are currently unavailable due to quota or "
            "rate limits. Please wait a moment and try again."
        ))

    async def ainvoke(self, prompt, **kwargs):
        """Async invoke — non-blocking, suitable for parallel LangGraph branches."""
        for provider in self._ordered():
            try:
                self.current_provider = provider.name
                result = await provider.ainvoke(prompt, **kwargs)
                return result
            except Exception as exc:
                if self._is_quota(exc):
                    self.fallback_count += 1
                    logger.warning("Async: provider %s quota hit — trying next.", provider.name)
                    continue
                raise

        return AIMessage(content=(
            "Error: All AI providers exhausted (async). Please try again later."
        ))

    def latency_report(self) -> dict[str, float]:
        """Return current EMA latency (seconds) per provider."""
        return {name: _tracker.p50(name) for name in self._provider_names}


# ── LiteLLM backend (optional) ───────────────────────────────────────────────

class _LiteLLMProvider:
    """
    Wraps LiteLLM as a provider so any model string (e.g. 'gpt-4o',
    'anthropic/claude-3-5-sonnet', 'groq/llama3-70b') works without
    adding a new Python class — just update settings.py.
    """

    def __init__(self, model: str, temperature: float = LLM_TEMPERATURE, **kwargs):
        try:
            import litellm  # noqa: F401
            self._model = model
            self._temperature = temperature
            self._kwargs = kwargs
        except ImportError:
            raise ImportError("Install litellm: pip install litellm")

    def invoke(self, prompt, **kwargs):
        import litellm
        messages = [{"role": "user", "content": str(prompt)}]
        resp = litellm.completion(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            **{**self._kwargs, **kwargs},
        )
        return AIMessage(content=resp.choices[0].message.content or "")

    async def ainvoke(self, prompt, **kwargs):
        import litellm
        messages = [{"role": "user", "content": str(prompt)}]
        resp = await litellm.acompletion(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            **{**self._kwargs, **kwargs},
        )
        return AIMessage(content=resp.choices[0].message.content or "")


# ── Factory functions ─────────────────────────────────────────────────────────

def get_llm(use_litellm: bool = False) -> FailoverLLMWrapper:
    """
    Build and return a FailoverLLMWrapper.

    Args:
        use_litellm: If True, all providers are backed by LiteLLM (requires
                     `pip install litellm` and provider API keys in env).
                     If False (default), uses native langchain providers.

    Provider chain: Gemini 2.5 Flash → Groq Llama-3.3-70b → Anthropic Claude Haiku
    Order at call time is determined by current EMA latency (fastest first).
    """
    if use_litellm:
        providers = [
            _Provider("Gemini-LiteLLM", _LiteLLMProvider("gemini/gemini-2.5-flash")),
            _Provider("Groq-LiteLLM",   _LiteLLMProvider("groq/llama-3.3-70b-versatile")),
            _Provider("Claude-LiteLLM", _LiteLLMProvider("anthropic/claude-haiku-4-5-20251001")),
        ]
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_groq import ChatGroq

        llm_gemini = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMPERATURE,
        )
        llm_groq = ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=LLM_TEMPERATURE,
        )

        # Anthropic Claude as third fallback (optional — gracefully skipped if key missing)
        anthropic_providers: list[_Provider] = []
        try:
            from langchain_anthropic import ChatAnthropic
            import os
            if os.getenv("ANTHROPIC_API_KEY"):
                anthropic_providers.append(_Provider(
                    "Claude",
                    ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=LLM_TEMPERATURE),
                ))
        except ImportError:
            pass

        providers = [
            _Provider("Gemini", llm_gemini),
            _Provider("Groq",   llm_groq),
            *anthropic_providers,
        ]

    return FailoverLLMWrapper(providers)
