"""
Builds the full 90-day cashflow forecast for one user and answers the
two core safety questions:

  1. max_safe_payment_today: largest amount payable on request_date
     without breaking the 90-day minimum-balance floor.
  2. earliest_safe_full_payment_date: first date the FULL requested
     amount is safe to pay in one shot.

This module composes the pieces built earlier:
  - event_resolution.classify_and_clean / detect_recurring_series /
    expand_series_to_cashflow  (recurring pattern detection + projection)
  - message_facts.extract_all_facts                (structured facts)
  - image_extraction.extract_amount_for_event       (blank-amount fill)
  - currency.CurrencyConverter                      (home-currency normalization)

Everything downstream of `build_user_timeline` is pure arithmetic on a
list of (date, signed_amount) cashflow points -- fully deterministic
and independent of any LLM call, per the design principle that
narrative/interpretive work (reading receipts, parsing messages) is a
one-time or rule-based step, while the actual affordability math never
depends on a model call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import pandas as pd

from .currency import CurrencyConverter
from .event_resolution import (
    RecurringSeries,
    classify_and_clean,
    detect_recurring_series,
    expand_series_to_cashflow,
)
from .image_extraction import extract_amount_for_event
from .message_facts import MessageFact, extract_fact

FORECAST_HORIZON_DAYS = 90


@dataclass
class CashflowPoint:
    date: pd.Timestamp
    amount: float  # signed: positive inflow, negative outflow
    source_event_id: str
    category: str
    flexibility: str = "fixed"
    minimum_allowed_amount: Optional[float] = None
    is_recurring_projection: bool = False


@dataclass
class UserTimeline:
    user_id: str
    home_currency: str
    start_balance: float
    minimum_balance_to_keep: float
    points: list[CashflowPoint]
    recurring_series: list[RecurringSeries]
    notes: list[str] = field(default_factory=list)


def _fill_blank_amounts(events: pd.DataFrame, images_df: pd.DataFrame, media_dir: str) -> pd.DataFrame:
    df = events.copy()
    blank_mask = df["amount"].isna()
    for idx in df[blank_mask].index:
        event_id = df.at[idx, "event_id"]
        amount = extract_amount_for_event(event_id, images_df, media_dir)
        if amount is not None:
            df.at[idx, "amount"] = amount
    return df


def _apply_message_facts_to_series(
    series_list: list[RecurringSeries],
    leftover: pd.DataFrame,
    facts: list[MessageFact],
    request_date: pd.Timestamp,
) -> tuple[list[RecurringSeries], pd.DataFrame, list[tuple[pd.Timestamp, float, str, str]], list[str]]:
    """
    Applies message-derived facts to the detected recurring series and
    one-off leftover events. Returns:
      - the (possibly amount-overridden / skipped) series list
      - the (possibly filtered) leftover one-off events
      - a list of extra one-off cashflow points to inject
        (date, signed_amount, source_id, category)
      - human-readable notes for traceability
    """
    notes: list[str] = []
    extra_points: list[tuple[pd.Timestamp, float, str, str]] = []
    series_overrides: dict[int, float] = {}
    series_skip: set[int] = set()
    series_date_shift: dict[int, pd.Timestamp] = {}
    rent_multiplier: dict[int, float] = {}

    salary_indices = [i for i, s in enumerate(series_list) if s.category == "salary" or s.event_type == "income"]
    rent_indices = [i for i, s in enumerate(series_list) if s.category == "rent"]

    for fact in facts:
        if fact.fact_type == "no_op":
            continue

        if fact.fact_type == "salary_ended":
            for i in salary_indices:
                series_skip.add(i)
            notes.append(f"{fact.message_id}: salary series stopped (employment/contract ended)")

        elif fact.fact_type == "salary_amount_change" and fact.amount is not None:
            for i in salary_indices:
                series_overrides[i] = fact.amount
            notes.append(f"{fact.message_id}: salary amount overridden to {fact.amount} {fact.currency}")

        elif fact.fact_type == "salary_date_shift" and fact.effective_date is not None:
            for i in salary_indices:
                series_date_shift[i] = fact.effective_date
            notes.append(f"{fact.message_id}: salary date shifted to {fact.effective_date.date()}")

        elif fact.fact_type == "salary_new_source" and fact.amount is not None and fact.effective_date is not None:
            # A brand new recurring salary starting at effective_date.
            # We don't have a prior interval for it, so treat it as a
            # monthly series (~30 days), which matches every observed
            # salary cadence in this dataset.
            extra_points.append((fact.effective_date, fact.amount, fact.message_id, "salary"))
            # Also register it as an ongoing monthly series so it
            # continues to recur through the rest of the 90-day window.
            series_list.append(
                RecurringSeries(
                    user_id=fact.user_id,
                    category="salary",
                    event_type="income",
                    direction="credit",
                    currency=fact.currency or "",
                    interval_days=30.0,
                    representative_amount=fact.amount,
                    last_date=fact.effective_date,
                    last_event_id=fact.message_id,
                    flexibility="fixed",
                    minimum_allowed_amount=None,
                    occurrences=1,
                )
            )
            notes.append(f"{fact.message_id}: new recurring salary of {fact.amount} {fact.currency} from {fact.effective_date.date()}")

        elif fact.fact_type == "confirmed_one_off_income" and fact.amount is not None:
            date = fact.effective_date if fact.effective_date is not None else request_date
            extra_points.append((date, fact.amount, fact.message_id, fact.category_hint or "other_income"))
            notes.append(f"{fact.message_id}: confirmed one-off income {fact.amount} {fact.currency} on {date.date()}")

        elif fact.fact_type == "rent_increase_pct" and fact.percent is not None:
            for i in rent_indices:
                rent_multiplier[i] = 1.0 + fact.percent / 100.0
            notes.append(f"{fact.message_id}: rent increased by {fact.percent}%")

        elif fact.fact_type == "income_unconfirmed":
            # Nothing to add -- explicitly do NOT add unconfirmed
            # bonus/commission/prize/refund income to the forecast.
            # For gig-style income (category_hint == "gig_income"),
            # the message additionally warns that even the ONGOING
            # payout series is unreliable ("can change until closed,
            # not withdrawable until completed") -- so we stop
            # projecting that recurring income series forward rather
            # than just skipping a single occurrence, since there is
            # no confirmed future payout to count on.
            if fact.category_hint == "gig_income":
                for i, s in enumerate(series_list):
                    if s.event_type == "income":
                        series_skip.add(i)
                notes.append(f"{fact.message_id}: gig income series stopped (payout unconfirmed/pending)")
            else:
                notes.append(f"{fact.message_id}: unconfirmed income excluded ({fact.category_hint})")

        elif fact.fact_type == "income_scam":
            notes.append(f"{fact.message_id}: advance-fee scam pattern recognized and ignored")

        elif fact.fact_type == "internal_transfer":
            notes.append(f"{fact.message_id}: internal transfer between own accounts noted (net-zero, no forecast impact)")

    # Apply overrides / skips / shifts / rent multiplier to the series list.
    new_series_list = []
    for i, s in enumerate(series_list):
        if i in series_skip:
            continue
        amount = series_overrides.get(i, s.representative_amount)
        if i in rent_multiplier:
            amount = s.representative_amount * rent_multiplier[i]
        last_date = series_date_shift.get(i, s.last_date)
        new_series_list.append(
            RecurringSeries(
                user_id=s.user_id,
                category=s.category,
                event_type=s.event_type,
                direction=s.direction,
                currency=s.currency,
                interval_days=s.interval_days,
                representative_amount=amount,
                last_date=last_date,
                last_event_id=s.last_event_id,
                flexibility=s.flexibility,
                minimum_allowed_amount=s.minimum_allowed_amount,
                occurrences=s.occurrences,
            )
        )

    return new_series_list, leftover, extra_points, notes


def build_user_timeline(
    user_id: str,
    profile: pd.Series,
    all_events: pd.DataFrame,
    all_messages: pd.DataFrame,
    converter: CurrencyConverter,
    request_date: pd.Timestamp,
    images_df: pd.DataFrame,
    media_dir: str,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> UserTimeline:
    home_currency = str(profile["home_currency"])
    start_balance = float(profile["current_available_balance"])
    min_balance = float(profile["minimum_balance_to_keep"])
    # Build points out to 2x horizon_days: a candidate payment date can
    # itself be up to horizon_days after request_date (the last day
    # scanned by earliest_safe_full_payment_date), and that candidate
    # then needs a further full horizon_days of points to check against
    # (see run_forecast's rolling-window docstring).
    horizon_end = request_date + timedelta(days=horizon_days * 2)

    user_events = all_events[all_events["user_id"] == user_id].copy()
    user_events = _fill_blank_amounts(user_events, images_df, media_dir)
    clean = classify_and_clean(user_events)

    series_list, leftover = detect_recurring_series(clean, request_date)

    user_messages = all_messages[all_messages["user_id"] == user_id]
    facts = [
        extract_fact(row.message_id, row.user_id, row.message_text)
        for row in user_messages.itertuples()
        if isinstance(getattr(row, "message_text", None), str)
    ]

    series_list, leftover, extra_points, notes = _apply_message_facts_to_series(
        series_list, leftover, facts, request_date
    )

    points: list[CashflowPoint] = []

    # Recurring series -> projected points within the forecast window.
    for s in series_list:
        for date, signed_amount, source_id in expand_series_to_cashflow(s, request_date, horizon_end):
            converted = converter.convert(abs(signed_amount), s.currency, home_currency, date)
            signed_converted = converted if signed_amount >= 0 else -converted
            points.append(
                CashflowPoint(
                    date=date,
                    amount=signed_converted,
                    source_event_id=source_id,
                    category=s.category,
                    flexibility=s.flexibility,
                    minimum_allowed_amount=s.minimum_allowed_amount,
                    is_recurring_projection=True,
                )
            )

    # One-off leftover events that fall within the forecast window
    # (both already-past-but-unabsorbed history is irrelevant here --
    # only events at/after request_date affect the forward forecast).
    in_window = leftover[
        (leftover["event_date"] >= request_date) & (leftover["event_date"] <= horizon_end)
    ]
    for row in in_window.itertuples():
        direction = getattr(row, "direction")
        amount = getattr(row, "amount")
        if pd.isna(amount):
            continue
        sign = 1.0 if direction == "credit" else -1.0
        currency = getattr(row, "currency")
        converted = converter.convert(abs(amount), currency, home_currency, row.event_date)
        flexibility = str(getattr(row, "flexibility", "fixed"))
        min_allowed = getattr(row, "minimum_allowed_amount", None)
        min_allowed = float(min_allowed) if pd.notna(min_allowed) else None
        points.append(
            CashflowPoint(
                date=row.event_date,
                amount=sign * converted,
                source_event_id=str(row.event_id),
                category=str(row.category),
                flexibility=flexibility,
                minimum_allowed_amount=min_allowed,
                is_recurring_projection=False,
            )
        )

    # Message-derived extra one-off points (confirmed invoice income,
    # arrears, new-job first salary already also added as a series).
    for date, amount, source_id, category in extra_points:
        if request_date <= date <= horizon_end:
            converted = converter.convert(abs(amount), home_currency, home_currency, date)
            points.append(
                CashflowPoint(
                    date=date,
                    amount=converted if amount >= 0 else -converted,
                    source_event_id=source_id,
                    category=category,
                    flexibility="fixed",
                    is_recurring_projection=False,
                )
            )

    points.sort(key=lambda p: p.date)

    return UserTimeline(
        user_id=user_id,
        home_currency=home_currency,
        start_balance=start_balance,
        minimum_balance_to_keep=min_balance,
        points=points,
        recurring_series=series_list,
        notes=notes,
    )


def run_forecast(
    timeline: UserTimeline,
    request_date: pd.Timestamp,
    extra_payments: Optional[list[tuple[pd.Timestamp, float]]] = None,
    excluded_source_ids: Optional[set[str]] = None,
    reduced_sources: Optional[dict[str, float]] = None,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> tuple[bool, float, pd.Timestamp]:
    """
    Walk the timeline plus any candidate extra_payments (negative =
    outflow) chronologically, applying optional spending changes:
      - excluded_source_ids: source_event_ids to fully drop (stop:)
      - reduced_sources: source_event_id -> new_amount to clamp a
        recurring projection down to (reduce_to:), applied to the
        magnitude of matching points.

    The safety window is fixed at 90 days from request_date (not from
    the candidate payment date). Confirmed against sample_requests.csv:
    a payment recommended near the end of the 90-day span is judged
    safe based on the remaining days within that same fixed window,
    not a fresh 90-day lookout restarting from the payment date --
    otherwise a normal monthly dip occurring just after the window
    would incorrectly veto an otherwise-safe date.

    Returns (is_safe, min_balance, min_balance_date).
    """
    excluded_source_ids = excluded_source_ids or set()
    reduced_sources = reduced_sources or {}
    horizon_end = request_date + timedelta(days=horizon_days)

    balance = timeline.start_balance
    min_balance = balance
    min_balance_date = request_date

    events: list[tuple[pd.Timestamp, float]] = []
    for p in timeline.points:
        if p.date < request_date or p.date > horizon_end:
            continue
        if p.source_event_id in excluded_source_ids:
            continue
        amount = p.amount
        if p.source_event_id in reduced_sources and amount < 0:
            new_cap = reduced_sources[p.source_event_id]
            amount = -min(abs(amount), new_cap)
        events.append((p.date, amount))

    for date, amount in extra_payments or []:
        events.append((date, amount))

    events.sort(key=lambda e: e[0])

    for date, amount in events:
        balance += amount
        if balance < min_balance:
            min_balance = balance
            min_balance_date = date

    is_safe = min_balance >= timeline.minimum_balance_to_keep
    return is_safe, min_balance, min_balance_date


def max_safe_payment_today(
    timeline: UserTimeline,
    request_date: pd.Timestamp,
    requested_amount: float,
    excluded_source_ids: Optional[set[str]] = None,
    reduced_sources: Optional[dict[str, float]] = None,
    tolerance: float = 0.01,
) -> float:
    def is_safe(amount: float) -> bool:
        safe, _, _ = run_forecast(
            timeline, request_date,
            extra_payments=[(request_date, -amount)],
            excluded_source_ids=excluded_source_ids,
            reduced_sources=reduced_sources,
        )
        return safe

    if requested_amount <= 0:
        return 0.0
    if is_safe(requested_amount):
        return round(requested_amount, 2)
    if not is_safe(0.0):
        return 0.0

    lo, hi = 0.0, requested_amount
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        if is_safe(mid):
            lo = mid
        else:
            hi = mid
    return round(lo, 2)


def earliest_safe_full_payment_date(
    timeline: UserTimeline,
    request_date: pd.Timestamp,
    requested_amount: float,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> Optional[pd.Timestamp]:
    """
    Scan forward day by day for the first date on which paying the full
    requested_amount as a single payment is safe, WITHOUT any optional
    spending changes (per spec, this is measured independently of the
    chosen recommendation).
    """
    for offset in range(0, horizon_days + 1):
        candidate_date = request_date + timedelta(days=offset)
        safe, _, _ = run_forecast(
            timeline, request_date,
            extra_payments=[(candidate_date, -requested_amount)],
        )
        if safe:
            return candidate_date
    return None
