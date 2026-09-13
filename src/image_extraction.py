"""
Extracts the true amount for financial_events.csv rows with a blank
`amount`, by sending the linked receipt/invoice image to a
vision-capable model and asking for a single structured field back.

Every blank-amount event has exactly one linked image via
images.csv's related_event_id -> dataset/media/images/<image_id>.png.
This module resolves that join and makes one API call per image
(cached on disk so a second run of the pipeline against the same
dataset doesn't re-spend tokens).

The model is asked to return the single final/total amount actually
paid or payable -- matching how every other event in
financial_events.csv records one total, not a line-item breakdown --
and nothing else, as strict JSON.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Optional

import pandas as pd

from .ai_client import (
    OLLAMA_VISION_MODEL,
    OllamaUnavailableError,
    call_ollama,
    coerce_amount,
    sanitize_unquoted_amount_commas,
    strip_json_fences,
)

EXTRACTION_PROMPT = """You are extracting a single financial figure from a photographed or \
screenshotted receipt, invoice, bill, or payslip for a personal finance forecasting system.

Find the ONE final total amount that was actually paid or is payable -- the same kind of \
single total that a bank statement or transaction log would record for this event. Concretely:
- For a purchase receipt or invoice: the grand total / net amount / total paid (after tax, \
after delivery, after any discount already applied) -- not a subtotal or a single line item.
- For a bill (utility, rent, maintenance, water, telecom): the amount due / balance due / \
total amount received for that billing period.
- For a payslip: the NET pay actually credited to the employee (take-home pay), not gross \
earnings and not any single allowance or deduction line.
- For a hospital or medical bill: the balance / total bill amount payable.

Respond with ONLY a JSON object, no other text, no markdown fences:
{"amount": <number>, "currency": "<3-letter code if visible, else null>"}

If you genuinely cannot find a total amount in the image, respond with:
{"amount": null, "currency": null}
"""

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", ".image_extraction_cache.json")


def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict):
    try:
        with open(_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass


def _media_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def _cache_key(image_path: str) -> str:
    # Keyed on model too, not just image path: switching
    # OLLAMA_VISION_MODEL (e.g. to a bigger/different local model)
    # must not silently reuse a result cached under a different model.
    return f"{image_path}::{OLLAMA_VISION_MODEL}"


def extract_amount_from_image(image_path: str) -> Optional[float]:
    """
    Sends one receipt image to a local Ollama vision model and returns
    the extracted total amount, or None if the model couldn't find one
    / the call fails / Ollama isn't available in this environment.
    Ollama is the only path here -- this account's Groq key has no
    vision-capable model available (see src/ai_client.py's module
    docstring), so there's no cloud call to try first. Results are
    cached on disk keyed by (image path, model) so repeat pipeline
    runs against the same dataset/model don't re-call the API -- except
    an Ollama-unavailable result, which is deliberately left uncached
    so a later run with Ollama actually running picks it up.
    """
    cache = _load_cache()
    key = _cache_key(image_path)
    if key in cache:
        return cache[key]["amount"]

    if not os.path.exists(image_path):
        return None

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    media_type = _media_type_for(image_path)

    try:
        response = call_ollama(
            purpose="image_extraction",
            model=OLLAMA_VISION_MODEL,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                            },
                        },
                        {"type": "text", "text": EXTRACTION_PROMPT},
                    ],
                }
            ],
        )
        text = response.choices[0].message.content
        cleaned = sanitize_unquoted_amount_commas(strip_json_fences(text))
        parsed = json.loads(cleaned)
        amount = coerce_amount(parsed.get("amount"))
    except OllamaUnavailableError as e:
        # Ollama isn't installed/running in this environment (e.g. a
        # fresh clone or a CI/grading sandbox with no local Ollama and
        # no GPU) -- an anticipated, documented gap, not a bug. Warn
        # loudly and leave this event's amount unresolved rather than
        # halting the whole pipeline run over it; still don't cache
        # this as a normal "no result" (see _cache_key) so a later run
        # with Ollama actually available doesn't skip it.
        print(f"WARNING: {e} -- {image_path} amount left unresolved (None).")
        return None
    except Exception:  # noqa: BLE001 -- genuine per-call API/parsing failure -> unresolved
        amount = None

    cache[key] = {"amount": amount}
    _save_cache(cache)
    return amount


def extract_amount_for_event(
    event_id: str,
    images_df: pd.DataFrame,
    media_dir: str,
) -> Optional[float]:
    """
    Resolves event_id -> its linked image (via images.csv's
    related_event_id) -> media_dir/<image_id>.png -> extracted amount.
    Returns None if there's no linked image or extraction fails.
    """
    matches = images_df[images_df["related_event_id"] == event_id]
    if matches.empty:
        return None
    image_id = matches.iloc[0]["image_id"]
    image_path = os.path.join(media_dir, f"{image_id}.png")
    return extract_amount_from_image(image_path)
