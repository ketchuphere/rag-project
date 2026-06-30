"""
Unit tests – FailoverLLMWrapper N-provider chain behaviour.
Updated for the new list[_Provider] constructor.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import logging
from langchain_core.messages import AIMessage
from app.services.generation.llm_factory import FailoverLLMWrapper, _Provider

logging.basicConfig(level=logging.WARNING)


def _make(name, fail=False, msg=""):
    class M:
        def invoke(self, p, **kw):
            if fail:
                raise Exception(msg)
            return AIMessage(content=f"Hello from {name}!")
    return _Provider(name, M())


def test_primary_success():
    print("=== Test 1: Standard primary execution ===")
    llm = FailoverLLMWrapper([_make("Gemini"), _make("Groq")])
    resp = llm.invoke("Hi")
    assert "Gemini" in resp.content or "Groq" in resp.content
    assert llm.fallback_count == 0
    print(f"  ✓ Response: {resp.content}")
    print(f"  ✓ Fallbacks: {llm.fallback_count}")


def test_gemini_429_failover():
    print("\n=== Test 2: Gemini 429 → Groq Failover ===")
    llm = FailoverLLMWrapper([_make("Gemini", fail=True, msg="429 Resource Exhausted"), _make("Groq")])
    resp = llm.invoke("Hi")
    assert "Groq" in resp.content
    assert llm.fallback_count == 1
    print(f"  ✓ Response: {resp.content}")
    print(f"  ✓ Fallbacks: {llm.fallback_count}")


def test_both_fail():
    print("\n=== Test 3: Both Gemini AND Groq Fail ===")
    llm = FailoverLLMWrapper([
        _make("Gemini", fail=True, msg="429 Resource Exhausted"),
        _make("Groq",   fail=True, msg="Groq Rate Limit Reached"),
    ])
    resp = llm.invoke("Hi")
    assert "Error" in resp.content
    assert llm.fallback_count == 2
    print(f"  ✓ Response: {resp.content[:60]}")
    print(f"  ✓ Fallbacks: {llm.fallback_count}")


def test_three_provider_chain():
    print("\n=== Test 4: Three-provider chain (Gemini→Groq→Claude) ===")
    llm = FailoverLLMWrapper([
        _make("Gemini", fail=True, msg="429"),
        _make("Groq",   fail=True, msg="rate limit"),
        _make("Claude"),
    ])
    resp = llm.invoke("Hi")
    assert "Claude" in resp.content
    assert llm.fallback_count == 2
    print(f"  ✓ Response: {resp.content}")
    print(f"  ✓ Fallbacks: {llm.fallback_count}")


if __name__ == "__main__":
    test_primary_success()
    test_gemini_429_failover()
    test_both_fail()
    test_three_provider_chain()
    print("\n✅ All failover tests passed.")
