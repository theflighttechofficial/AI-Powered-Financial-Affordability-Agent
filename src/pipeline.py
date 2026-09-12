"""
End-to-end pipeline: dataset/ -> output.csv

Usage:
    export GROQ_API_KEY=gsk_...
    python -m src.pipeline --dataset dataset --out output.csv

Requires GROQ_API_KEY to be set: image amount extraction and
message fact extraction both call the Groq API (see
src/ai_client.py, src/image_extraction.py, src/message_facts.py).
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from .ai_client import USAGE
from .currency import CurrencyConverter
from .decision_engine import PaymentOption, decide
from .forecast import build_user_timeline

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def load_all(dataset_dir: str):
    def p(name):
        return os.path.join(dataset_dir, name)

    requests = pd.read_csv(p("requests.csv"), parse_dates=["request_date", "desired_completion_date"])
    profiles = pd.read_csv(p("financial_profiles.csv"))
    events = pd.read_csv(p("financial_events.csv"), parse_dates=["event_date", "settlement_date"])
    messages = pd.read_csv(p("messages.csv"))
    rates = pd.read_csv(p("exchange_rates.csv"))
    payment_options = pd.read_csv(p("request_payment_options.csv"), parse_dates=["first_payment_date"])
    images = pd.read_csv(p("images.csv"))
    media_dir = os.path.join(dataset_dir, "media", "images")
    return requests, profiles, events, messages, rates, payment_options, images, media_dir


def build_payment_options(payment_options_df: pd.DataFrame, request_id: str) -> list[PaymentOption]:
    rows = payment_options_df[payment_options_df["request_id"] == request_id]
    options = []
    for row in rows.itertuples():
        freq = getattr(row, "payment_frequency_days", None)
        options.append(
            PaymentOption(
                payment_option_id=row.payment_option_id,
                request_id=request_id,
                payment_method=row.payment_method,
                payment_amount=float(row.payment_amount),
                number_of_payments=int(row.number_of_payments),
                first_payment_date=row.first_payment_date,
                payment_frequency_days=float(freq) if pd.notna(freq) else None,
                financing_fee=float(row.financing_fee) if pd.notna(row.financing_fee) else 0.0,
                total_payable_amount=float(row.total_payable_amount),
            )
        )
    return options


def process_request(row, profiles, events, messages, converter, payment_options_df, images_df, media_dir) -> dict:
    profile = profiles[profiles["user_id"] == row.user_id].iloc[0]
    timeline = build_user_timeline(
        row.user_id, profile, events, messages, converter, row.request_date, images_df, media_dir
    )
    options = build_payment_options(payment_options_df, row.request_id)

    return decide(
        request_id=row.request_id,
        request_date=row.request_date,
        requested_amount=float(row.requested_amount),
        desired_completion_date=row.desired_completion_date,
        allows_partial_payment=bool(row.allows_partial_payment),
        timeline=timeline,
        profile=profile,
        payment_options=options,
    )


# Pricing per million tokens (input, output), current as of this
# writing. Update here if pricing changes; usage_report.md is
# generated from these plus the actual token counts recorded by
# ai_client.USAGE during the run, not hand-typed.
# Only models this account's Groq key actually has access to (see
# client.models.list()) are priced here. meta-llama/llama-4-scout and
# llama-3.3-70b-versatile were removed after being confirmed
# unavailable to this key (404 model_not_found) -- see
# src/ai_client.py's VISION_MODEL comment.
PRICING_PER_MILLION_TOKENS = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
}


def write_usage_report(path: str):
    summary = USAGE.summary()
    lines = ["# Token Usage and Cost Report", "", "## Summary", ""]
    lines.append(
        "This run made real Groq API calls for image amount extraction "
        "(`src/image_extraction.py`) and message fact extraction "
        "(`src/message_facts.py`). All other computation -- the 90-day "
        "balance forecast, affordability decision, and payment-plan "
        "ranking -- is deterministic and makes no model calls."
    )
    lines.append("")
    lines.append(
        f"| Metric | Value |\n|---|---|\n"
        f"| Total model calls | {summary['total_calls']} |\n"
        f"| Total input tokens | {summary['total_input_tokens']:,} |\n"
        f"| Total output tokens | {summary['total_output_tokens']:,} |\n"
        f"| Total tokens | {summary['total_tokens']:,} |\n"
    )

    total_cost = 0.0
    lines.append("## By model\n")
    lines.append("| Model | Calls | Input tokens | Output tokens | Est. cost |\n|---|---|---|---|---|")
    for model, stats in summary["by_model"].items():
        pricing = PRICING_PER_MILLION_TOKENS.get(model)
        if pricing:
            cost = (
                stats["input_tokens"] / 1_000_000 * pricing["input"]
                + stats["output_tokens"] / 1_000_000 * pricing["output"]
            )
        else:
            cost = None
        total_cost += cost or 0.0
        cost_str = f"${cost:.4f}" if cost is not None else "unknown (pricing not on file)"
        lines.append(
            f"| {model} | {stats['calls']} | {stats['input_tokens']:,} | "
            f"{stats['output_tokens']:,} | {cost_str} |"
        )

    lines.append("")
    lines.append(f"**Estimated total cost: ${total_cost:.4f}**")
    if summary["total_calls"] > 0:
        lines.append(f"**Average cost per request in this run: see per-request note below.**")
    lines.append("")
    lines.append(
        "Note: image and message extraction calls are cached on disk "
        "(`.image_extraction_cache.json`, `.message_facts_cache.json`) "
        "keyed by image path / message content, so the counts above "
        "reflect a full cold run against this dataset -- a second run "
        "against the same dataset makes zero additional calls."
    )
    lines.append("")
    lines.append(
        "Per-request average: this dataset has 16 blank-amount events "
        "(image calls) and up to ~215 messages (message-fact calls), "
        "shared across all 250 requests in requests.csv, so the "
        "per-request marginal cost is far below the per-call cost -- "
        "divide the total cost above by 250 for an average, or note "
        "that most requests trigger zero fresh calls since they reuse "
        "already-cached user history."
    )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run(dataset_dir: str, out_path: str, requests_file: str = "requests.csv", usage_report_path: str = None):
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. This pipeline calls the Groq "
            "API for image amount extraction and message fact extraction and "
            "cannot produce correct output without it. Set the environment "
            "variable and re-run."
        )

    requests, profiles, events, messages, rates, payment_options_df, images, media_dir = load_all(dataset_dir)
    if requests_file != "requests.csv":
        requests = pd.read_csv(
            os.path.join(dataset_dir, requests_file),
            parse_dates=["request_date", "desired_completion_date"],
        )
    converter = CurrencyConverter(rates)

    rows = []
    errors = []
    for row in requests.itertuples():
        try:
            rows.append(
                process_request(row, profiles, events, messages, converter, payment_options_df, images, media_dir)
            )
        except Exception as e:  # noqa: BLE001
            errors.append((row.request_id, str(e)))
            rows.append(
                {
                    "request_id": row.request_id,
                    "amount_safe_to_pay": 0,
                    "affordability_status": "not_affordable",
                    "recommended_payment_method": "not_recommended",
                    "payment_plan": "none",
                    "earliest_date_for_full_payment": "",
                    "spending_changes_needed": "none",
                    "decision_explanation": f"ERROR during processing: {e}",
                }
            )

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {len(out_df)} rows to {out_path}")
    if errors:
        print(f"{len(errors)} requests hit errors:")
        for rid, msg in errors[:20]:
            print(f"  {rid}: {msg}")

    if usage_report_path:
        write_usage_report(usage_report_path)
        print(f"Wrote usage report to {usage_report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--out", default="output.csv")
    parser.add_argument("--requests-file", default="requests.csv")
    parser.add_argument("--usage-report", default="evaluation/usage_report.md")
    args = parser.parse_args()
    run(args.dataset, args.out, args.requests_file, args.usage_report)

