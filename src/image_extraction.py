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

from .ai_client import VISION_MODEL, call_and_track

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


def extract_amount_from_image(image_path: str) -> Optional[float]:
    """
    Sends one receipt image to the model and returns the extracted
    total amount, or None if the model couldn't find one / the call
    fails. Results are cached on disk keyed by image path so repeat
    pipeline runs against the same dataset don't re-call the API.
    """
    cache = _load_cache()
    if image_path in cache:
        return cache[image_path]["amount"]

    if not os.path.exists(image_path):
        return None

    if VISION_MODEL is None:
        # No vision-capable model is configured/available on this
        # account's Groq key -- calling anyway would just 404 on every
        # image and get masked by the except-block below as an
        # ordinary per-call failure, silently corrupting these 16
        # events' amounts with no visible signal. Surface it loudly
        # once per image instead, but don't halt the whole pipeline --
        # the caller (forecast.py) treats a None amount as unresolved
        # and the run is expected to complete with this known gap.
        print(
            f"WARNING: no vision-capable model configured -- "
            f"{image_path} amount left unresolved (None). Set "
            f"BUY_OR_WAIT_VISION_MODEL once one is available."
        )
        cache[image_path] = {"amount": None}
        _save_cache(cache)
        return None

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    media_type = _media_type_for(image_path)

    try:
        response = call_and_track(
            purpose="image_extraction",
            model=VISION_MODEL,
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
        parsed = json.loads(text.strip())
        amount = parsed.get("amount")
        amount = float(amount) if amount is not None else None
    except RuntimeError:
        # Missing/invalid API key -- this is a configuration error, not
        # a per-image extraction failure, so it must not be silently
        # swallowed into a cached "None" result. Let it propagate.
        raise
    except Exception:  # noqa: BLE001 -- genuine per-call API/parsing failure -> unresolved
        amount = None

    cache[image_path] = {"amount": amount}
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
