"""
Tests for src/image_extraction.py and src/message_facts.py, mocking
the AI calls (call_ollama / call_text_with_fallback) so they run with
no network access, no API key, and no local Ollama instance needed.

These exist specifically to catch the failure mode that slipped
through unnoticed during the Anthropic -> Groq migration: an API-level
error (e.g. a 404 for a model the account doesn't have access to) was
caught by the generic `except Exception` fallback and silently
recorded as a normal "couldn't extract" result (amount=None /
fact_type="no_op"), indistinguishable from a genuine per-item failure.
Any change to that fallback behavior should break one of these tests.

They also cover the Groq-primary / Ollama-fallback routing added
afterwards: message_facts.py must try Groq first and only reach for
Ollama when Groq itself fails, while image_extraction.py has no Groq
path at all (this account's Groq key has no vision model) and always
calls Ollama directly.

Run with: python -m pytest tests/test_ai_extraction.py -v
"""
import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import image_extraction, message_facts


def _fake_response(content: str):
    """Builds a minimal object matching response.choices[0].message.content."""
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


# ---------------------------------------------------------------------------
# image_extraction.py -- always Ollama, no Groq path
# ---------------------------------------------------------------------------

def test_image_extraction_happy_path_calls_ollama_directly(monkeypatch, tmp_path):
    monkeypatch.setattr(image_extraction, "_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(image_extraction, "OLLAMA_VISION_MODEL", "llama3.2-vision")

    calls = []

    def _fake_call_ollama(**kwargs):
        calls.append(kwargs)
        return _fake_response('{"amount": 42.5, "currency": "USD"}')

    monkeypatch.setattr(image_extraction, "call_ollama", _fake_call_ollama)

    img_path = tmp_path / "receipt.png"
    img_path.write_bytes(b"not a real png, just needs to exist")

    amount = image_extraction.extract_amount_from_image(str(img_path))
    assert amount == 42.5
    assert len(calls) == 1
    assert calls[0]["model"] == "llama3.2-vision"


def test_image_extraction_ollama_call_error_falls_back_to_none_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(image_extraction, "_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(image_extraction, "OLLAMA_VISION_MODEL", "llama3.2-vision")

    # A genuine per-call failure (e.g. malformed response, the model
    # couldn't parse the image) distinct from Ollama being unreachable
    # entirely -- that distinction is what test_..._unreachable_error_
    # propagates below covers.
    class SimulatedModelError(Exception):
        pass

    monkeypatch.setattr(
        image_extraction,
        "call_ollama",
        lambda **kwargs: (_ for _ in ()).throw(SimulatedModelError("boom")),
    )

    img_path = tmp_path / "receipt.png"
    img_path.write_bytes(b"not a real png, just needs to exist")

    amount = image_extraction.extract_amount_from_image(str(img_path))
    assert amount is None


def test_image_extraction_ollama_unreachable_error_propagates(monkeypatch, tmp_path):
    monkeypatch.setattr(image_extraction, "_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(image_extraction, "OLLAMA_VISION_MODEL", "llama3.2-vision")

    def _raise_unreachable(**kwargs):
        raise RuntimeError("Could not reach local Ollama at http://localhost:11434/v1")

    monkeypatch.setattr(image_extraction, "call_ollama", _raise_unreachable)

    img_path = tmp_path / "receipt.png"
    img_path.write_bytes(b"not a real png, just needs to exist")

    try:
        image_extraction.extract_amount_from_image(str(img_path))
        assert False, "expected RuntimeError to propagate"
    except RuntimeError:
        pass


def test_image_extraction_cache_key_includes_model(monkeypatch, tmp_path):
    # Regression test: an image cached under one OLLAMA_VISION_MODEL
    # must not be silently served as a cached result once
    # OLLAMA_VISION_MODEL changes to a different local model.
    cache_path = str(tmp_path / "cache.json")
    monkeypatch.setattr(image_extraction, "_CACHE_PATH", cache_path)

    img_path = tmp_path / "receipt.png"
    img_path.write_bytes(b"not a real png, just needs to exist")

    monkeypatch.setattr(image_extraction, "OLLAMA_VISION_MODEL", "model-a")
    monkeypatch.setattr(
        image_extraction,
        "call_ollama",
        lambda **kwargs: _fake_response('{"amount": 1.0, "currency": "USD"}'),
    )
    assert image_extraction.extract_amount_from_image(str(img_path)) == 1.0

    monkeypatch.setattr(image_extraction, "OLLAMA_VISION_MODEL", "model-b")
    monkeypatch.setattr(
        image_extraction,
        "call_ollama",
        lambda **kwargs: _fake_response('{"amount": 99.0, "currency": "USD"}'),
    )
    assert image_extraction.extract_amount_from_image(str(img_path)) == 99.0


# ---------------------------------------------------------------------------
# message_facts.py -- Groq primary, Ollama fallback on Groq failure
# ---------------------------------------------------------------------------

def test_message_facts_happy_path_uses_groq_primary(monkeypatch, tmp_path):
    monkeypatch.setattr(message_facts, "_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(
        message_facts,
        "call_text_with_fallback",
        lambda **kwargs: _fake_response(
            json.dumps(
                {
                    "fact_type": "salary_amount_change",
                    "amount": 5000000,
                    "currency": "IDR",
                    "effective_date": "2026-01-01",
                    "percent": None,
                    "category_hint": None,
                }
            )
        ),
    )

    fact = message_facts.extract_fact("msg_01", "user_01", "Your salary is now IDR 5,000,000.")
    assert fact.fact_type == "salary_amount_change"
    assert fact.amount == 5000000


def test_message_facts_api_error_falls_back_to_no_op_not_a_crash(monkeypatch, tmp_path):
    # This exercises message_facts.py's own except-block, not the
    # Groq->Ollama routing inside call_text_with_fallback (that's
    # tested directly against ai_client below) -- here, even after
    # call_text_with_fallback has done everything it can (including
    # trying Ollama), a genuine failure still must not crash the run.
    monkeypatch.setattr(message_facts, "_CACHE_PATH", str(tmp_path / "cache.json"))

    class SimulatedApiError(Exception):
        pass

    monkeypatch.setattr(
        message_facts,
        "call_text_with_fallback",
        lambda **kwargs: (_ for _ in ()).throw(SimulatedApiError("boom")),
    )

    fact = message_facts.extract_fact("msg_02", "user_01", "Some message text.")
    assert fact.fact_type == "no_op"


def test_message_facts_missing_api_key_error_propagates(monkeypatch, tmp_path):
    monkeypatch.setattr(message_facts, "_CACHE_PATH", str(tmp_path / "cache.json"))

    def _raise_missing_key(**kwargs):
        raise RuntimeError("GROQ_API_KEY is not set.")

    monkeypatch.setattr(message_facts, "call_text_with_fallback", _raise_missing_key)

    try:
        message_facts.extract_fact("msg_03", "user_01", "Some message text.")
        assert False, "expected RuntimeError to propagate"
    except RuntimeError:
        pass


def test_message_facts_cache_key_includes_model(monkeypatch, tmp_path):
    # Regression test: switching DEFAULT_MODEL (e.g. after discovering
    # the previous model 404s) must not silently reuse a fact cached
    # under a different model.
    monkeypatch.setattr(message_facts, "_CACHE_PATH", str(tmp_path / "cache.json"))

    monkeypatch.setattr(message_facts, "DEFAULT_MODEL", "model-a")
    monkeypatch.setattr(
        message_facts,
        "call_text_with_fallback",
        lambda **kwargs: _fake_response(
            json.dumps(
                {
                    "fact_type": "no_op",
                    "amount": None,
                    "currency": None,
                    "effective_date": None,
                    "percent": None,
                    "category_hint": None,
                }
            )
        ),
    )
    fact_a = message_facts.extract_fact("msg_04", "user_01", "Some message text.")
    assert fact_a.fact_type == "no_op"

    monkeypatch.setattr(message_facts, "DEFAULT_MODEL", "model-b")
    monkeypatch.setattr(
        message_facts,
        "call_text_with_fallback",
        lambda **kwargs: _fake_response(
            json.dumps(
                {
                    "fact_type": "internal_transfer",
                    "amount": 100,
                    "currency": "USD",
                    "effective_date": None,
                    "percent": None,
                    "category_hint": None,
                }
            )
        ),
    )
    fact_b = message_facts.extract_fact("msg_04", "user_01", "Some message text.")
    assert fact_b.fact_type == "internal_transfer"


# ---------------------------------------------------------------------------
# ai_client.py -- the Groq -> Ollama fallback routing itself
# ---------------------------------------------------------------------------

def test_call_text_with_fallback_uses_groq_when_it_succeeds(monkeypatch):
    from src import ai_client

    calls = {"groq": 0, "ollama": 0}

    def _fake_call_and_track(purpose, **kwargs):
        calls["groq"] += 1
        return _fake_response('{"ok": true}')

    def _fake_call_ollama(purpose, model, messages, max_tokens=300):
        calls["ollama"] += 1
        raise AssertionError("Ollama must not be called when Groq succeeds")

    monkeypatch.setattr(ai_client, "call_and_track", _fake_call_and_track)
    monkeypatch.setattr(ai_client, "call_ollama", _fake_call_ollama)

    response = ai_client.call_text_with_fallback(
        purpose="test", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    )
    assert response.choices[0].message.content == '{"ok": true}'
    assert calls == {"groq": 1, "ollama": 0}


def test_call_text_with_fallback_falls_back_to_ollama_when_groq_fails(monkeypatch):
    from src import ai_client

    calls = {"groq": 0, "ollama": 0}

    def _fake_call_and_track(purpose, **kwargs):
        calls["groq"] += 1
        raise Exception("simulated Groq failure (e.g. rate limit exhausted)")

    def _fake_call_ollama(purpose, model, messages, max_tokens=300):
        calls["ollama"] += 1
        assert model == ai_client.OLLAMA_TEXT_MODEL
        return _fake_response('{"ok": "from ollama"}')

    monkeypatch.setattr(ai_client, "call_and_track", _fake_call_and_track)
    monkeypatch.setattr(ai_client, "call_ollama", _fake_call_ollama)

    response = ai_client.call_text_with_fallback(
        purpose="test", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    )
    assert response.choices[0].message.content == '{"ok": "from ollama"}'
    assert calls == {"groq": 1, "ollama": 1}


def test_call_text_with_fallback_does_not_fall_back_on_missing_api_key(monkeypatch):
    from src import ai_client

    calls = {"groq": 0, "ollama": 0}

    def _fake_call_and_track(purpose, **kwargs):
        calls["groq"] += 1
        raise RuntimeError("GROQ_API_KEY is not set.")

    def _fake_call_ollama(purpose, model, messages, max_tokens=300):
        calls["ollama"] += 1
        raise AssertionError("Ollama must not be called for a missing-API-key config error")

    monkeypatch.setattr(ai_client, "call_and_track", _fake_call_and_track)
    monkeypatch.setattr(ai_client, "call_ollama", _fake_call_ollama)

    try:
        ai_client.call_text_with_fallback(
            purpose="test", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
        )
        assert False, "expected RuntimeError to propagate"
    except RuntimeError:
        pass
    assert calls == {"groq": 1, "ollama": 0}
