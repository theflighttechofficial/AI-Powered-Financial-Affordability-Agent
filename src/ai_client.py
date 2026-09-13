"""
Wrapper around the Groq API (primary, text) and a local Ollama
instance (vision, and text fallback) used by image_extraction.py and
message_facts.py. Centralizing the client here means:
  - one place to read the API key / configure the model
  - one place that accumulates token usage across every call, so
    evaluation/usage_report.md can be generated from real numbers
    after a run instead of hand-written
  - one place holding the Groq -> Ollama fallback / routing logic, so
    both call sites use it identically

Text extraction (message_facts.py): Groq is primary. If a Groq call
raises anything other than the missing-API-key config error, it falls
back once to a local Ollama model rather than letting a transient
Groq failure silently degrade into a "no_op" fact with no visibility
into which backend actually answered (see call_text_with_fallback).

Image extraction (image_extraction.py): Ollama is the default/only
path. This account's Groq key has no access to any vision-capable
model (meta-llama/llama-4-scout-17b-16e-instruct 404s for this key,
and client.models.list() returns only text/audio models), so there is
no working Groq vision call to try first.

Requires the GROQ_API_KEY environment variable to be set for the text
path. No key is ever hard-coded or written to disk. The Ollama path
requires a local Ollama instance running with the configured models
pulled (see OLLAMA_BASE_URL / OLLAMA_TEXT_MODEL / OLLAMA_VISION_MODEL
below) -- no API key needed for it.
"""
from __future__ import annotations

import json
import os
import re
import threading
import types
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from groq import Groq

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_json_fences(text: str) -> str:
    """
    Strips a leading/trailing markdown code fence (``` or ```json)
    around a JSON response, if present. Smaller/local models (observed
    with Ollama's llava-phi3) are much more prone to wrapping strict-
    JSON responses in a code fence than Groq's cloud models are, even
    when explicitly told to return only JSON -- without this, every
    such response fails json.loads and gets treated as a genuine
    per-call extraction failure.
    """
    return _JSON_FENCE_RE.sub("", text.strip()).strip()


# Matches an unquoted "amount" value written with thousands-separator
# commas, e.g. "amount": 431,896,027.35 -- valid to a human, but
# invalid JSON (a bare number can't contain commas). Observed from
# llava-phi3; not something a prompt tweak reliably prevents from a
# small local model, so it's sanitized here instead.
_UNQUOTED_AMOUNT_COMMA_RE = re.compile(
    r'"amount"\s*:\s*(-?\d{1,3}(?:,\d{1,3})+(?:\.\d+)?)(?=\s*[,}\]])'
)


def sanitize_unquoted_amount_commas(text: str) -> str:
    """
    Rewrites an unquoted, comma-grouped "amount" number (invalid JSON)
    into a valid bare number by removing the internal commas, so
    json.loads can parse the rest of the response normally. A quoted
    amount string (e.g. "amount": "$1,065,872") is left alone here --
    that's valid JSON already, and its commas/currency symbols are
    stripped later by coerce_amount instead.
    """
    return _UNQUOTED_AMOUNT_COMMA_RE.sub(
        lambda m: '"amount": ' + m.group(1).replace(",", ""), text
    )


_NON_NUMERIC_RE = re.compile(r"[^\d.\-]")


def coerce_amount(value) -> float | None:
    """
    Coerces a model's "amount" field into a float, tolerating the
    formats small/local models are prone to emitting despite being
    told to return a bare JSON number: a currency-symbol-prefixed
    string, thousands-separator commas inside a string, or a genuine
    number/None. Returns None for anything that isn't recoverably a
    number, rather than raising -- callers already treat None as "no
    amount found", which is the right fallback here too.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = _NON_NUMERIC_RE.sub("", value)
        if cleaned in ("", "-", "."):
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _load_dotenv():
    """
    Minimal .env loader (no external dependency): sets any KEY=VALUE
    pair found in a .env file at the repo root into os.environ, without
    overriding variables already set in the real environment.
    """
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()

# Structured extraction from short template-like messages doesn't need
# a large model. openai/gpt-oss-120b is confirmed available on this
# account's Groq key (checked via client.models.list()) and supports
# reliable structured JSON output. Override via env var if a different
# model is preferred.
DEFAULT_MODEL = os.environ.get("BUY_OR_WAIT_MODEL", "openai/gpt-oss-120b")

# Local Ollama instance: vision extraction's only path (Groq has none
# available to this key), and text extraction's fallback when Groq
# fails. Ollama's /v1/chat/completions is OpenAI-compatible, so the
# same message/response shape used for Groq is reused as-is.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "llama3.1:8b")
# moondream (~829MB, the smallest vision model in Ollama's library) was
# tried first but is a confirmed dead end: it returns an empty
# response for every image-containing prompt on this Ollama version,
# both through the OpenAI-compatible endpoint and Ollama's native
# /api/chat and /api/generate -- a known, longstanding issue (see
# ollama/ollama#4063, #6365; vikhyat/moondream#151), not specific to
# this dataset or setup. llava-phi3 (~2.9GB) is the next-smallest
# option and uses the mainline LLaVA architecture, which has been
# stable in Ollama for a long time. Override OLLAMA_VISION_MODEL for a
# larger/more accurate model (e.g. llama3.2-vision, ~8GB) if needed.
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llava-phi3")


@dataclass
class UsageTracker:
    """Accumulates token usage/cost across all calls in a single run."""
    calls: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, purpose: str, model: str, input_tokens: int, output_tokens: int):
        with self._lock:
            self.calls.append(
                {
                    "purpose": purpose,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )

    def summary(self) -> dict:
        with self._lock:
            calls = list(self.calls)
        by_model: dict[str, dict] = {}
        for c in calls:
            m = by_model.setdefault(
                c["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
            m["calls"] += 1
            m["input_tokens"] += c["input_tokens"]
            m["output_tokens"] += c["output_tokens"]
        total_calls = len(calls)
        total_input = sum(c["input_tokens"] for c in calls)
        total_output = sum(c["output_tokens"] for c in calls)
        return {
            "total_calls": total_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "by_model": by_model,
        }

    def write_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)


# Module-level singleton so both image_extraction.py and
# message_facts.py accumulate into the same tracker across one
# pipeline run without threading it through every function signature.
USAGE = UsageTracker()

_client = None
_client_lock = threading.Lock()


# The groq SDK already retries 429/408/409/5xx responses with
# exponential backoff internally (see Groq._should_retry) -- its
# default of 2 retries is thin for a ~200-call batch run against a
# free/low tier that's likely to hit rate limits partway through.
# Override via env var if a different budget is needed.
MAX_RETRIES = int(os.environ.get("BUY_OR_WAIT_MAX_RETRIES", "5"))


def get_client() -> Groq:
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. This pipeline calls the Groq "
                    "API for image and message understanding and requires "
                    "a valid key in the environment."
                )
            _client = Groq(api_key=api_key, max_retries=MAX_RETRIES)
    return _client


def call_and_track(purpose: str, **kwargs):
    """
    Wraps client.chat.completions.create, recording usage against
    `purpose` (e.g. "image_extraction", "message_facts") for the cost
    report.
    """
    client = get_client()
    model = kwargs.get("model", DEFAULT_MODEL)
    kwargs.setdefault("model", model)
    response = client.chat.completions.create(**kwargs)
    USAGE.record(
        purpose=purpose,
        model=model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )
    return response


def _response_from_openai_json(data: dict):
    """
    Wraps a raw OpenAI-shaped chat.completions JSON dict (what
    Ollama's /v1/chat/completions returns) in the same attribute-access
    shape the groq SDK's response object has, so callers parse a Groq
    response and an Ollama response identically.
    """
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=choice["message"]["content"])
            )
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        ),
    )


class OllamaUnavailableError(RuntimeError):
    """
    Raised when the local Ollama instance can't be reached at all
    (not installed, not running, or the target model not pulled) --
    distinct from a generic RuntimeError so callers can tell "this
    environment doesn't have Ollama set up" (an anticipated, gracefully
    handled gap -- e.g. a CI/grading sandbox with no local Ollama)
    apart from a genuine, unexpected config error that should still
    halt the run loudly.
    """


def call_ollama(purpose: str, model: str, messages: list, max_tokens: int = 300, temperature: float = 0):
    """
    Calls a local Ollama instance directly via its OpenAI-compatible
    endpoint, using only the standard library (urllib) so this doesn't
    pull in a new dependency just to talk to a local process. Raises
    OllamaUnavailableError if Ollama isn't reachable at all (not
    installed, not running, or the model not pulled).

    temperature defaults to 0: these are structured-JSON extraction
    calls, not creative generation, and smaller local vision models
    (observed with llava-phi3) are prone to giving wildly different
    numeric answers for the same image across repeated calls at the
    default temperature -- 0 substantially reduces (though doesn't
    fully eliminate) that variance.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise OllamaUnavailableError(
            f"Could not reach local Ollama at {OLLAMA_BASE_URL} for model "
            f"'{model}' ({e}). Install/start Ollama and `ollama pull {model}`."
        ) from e

    response = _response_from_openai_json(data)
    USAGE.record(
        purpose=purpose,
        model=f"ollama:{model}",
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )
    return response


def call_text_with_fallback(purpose: str, **kwargs):
    """
    Text-classification call used by message_facts.py: Groq
    (DEFAULT_MODEL) is primary. If Groq raises anything other than the
    missing-API-key RuntimeError -- rate limit exhausted after
    retries, a model access issue, a network error -- this falls back
    once to a local Ollama model (OLLAMA_TEXT_MODEL) instead of letting
    the failure propagate up into message_facts.py's generic
    except-block, which would otherwise silently record a "no_op" fact
    indistinguishable from a genuine per-message ambiguity.
    """
    try:
        return call_and_track(purpose, **kwargs)
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001 -- deliberately broad: any Groq call failure triggers the fallback
        print(
            f"WARNING: Groq call failed for purpose={purpose!r} ({e}); "
            f"falling back to local Ollama ({OLLAMA_TEXT_MODEL})."
        )
        messages = kwargs["messages"]
        max_tokens = kwargs.get("max_tokens", 300)
        return call_ollama(purpose, OLLAMA_TEXT_MODEL, messages, max_tokens=max_tokens)
