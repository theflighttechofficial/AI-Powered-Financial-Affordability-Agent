"""
Thin wrapper around the Groq API used by image_extraction.py and
message_facts.py. Centralizing the client here means:
  - one place to read the API key / configure the model
  - one place that accumulates token usage across every call, so
    evaluation/usage_report.md can be generated from real numbers
    after a run instead of hand-written
  - one place to add retry/backoff if needed

Requires the GROQ_API_KEY environment variable to be set. No key is
ever hard-coded or written to disk.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field

from groq import Groq


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

# Vision model for receipt-image extraction. As of this writing, this
# account's Groq key has NO access to any vision-capable model --
# meta-llama/llama-4-scout-17b-16e-instruct (Groq's documented vision
# model) returns 404 model_not_found for this key, and
# client.models.list() returns only text/audio models. Leave unset
# (None) until a vision-capable model is confirmed available, rather
# than pointing at a model that will 404 on every call. Set
# BUY_OR_WAIT_VISION_MODEL once one is available.
VISION_MODEL = os.environ.get("BUY_OR_WAIT_VISION_MODEL") or None


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
            _client = Groq(api_key=api_key)
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
