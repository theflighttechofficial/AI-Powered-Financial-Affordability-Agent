"""
Turns raw financial_events.csv rows for one user into a clean set of
CashflowEvent objects ready for 90-day forecasting.

Key facts learned from the real dataset that drive this design:

  * There is NO explicit "is_recurring" flag or frequency column.
    Recurrence must be detected from history: group a user's events by
    `category`, and if a category has multiple occurrences with a
    fairly consistent interval, treat it as recurring and project it
    forward using the median interval and a representative amount.

  * `status` values: settled, scheduled, pending, cancelled, failed,
    unrealized.
      - cancelled / failed -> always excluded.
      - unrealized -> always excluded (non-cash investment marks).
      - pending credit (direction=credit) -> excluded (not confirmed
        received yet, per spec: "ignore pending credits").
      - pending debit (direction=debit) -> INCLUDED as a known future
        obligation (a bill that's due but not yet paid still has to be
        paid; excluding it would be the financially UNSAFE reading).
      - scheduled -> included as-is (confirmed future event, e.g. next
        salary, or a bill already on the calendar).
      - settled -> included as-is (it already happened / is history
        used to build the recurring pattern).

  * `linked_event_id` chains an amendment/settlement to an earlier
    event (e.g. a cancelled charge that was corrected, or an
    investment purchase that matures into a later sale). We keep both
    ends of the chain but the excluded-status rule above naturally
    drops the superseded (cancelled/failed) one.

  * `direction`: debit (outflow), credit (inflow), non_cash (investment
    valuation marks -> always excluded).

  * Blank `amount` occurs only on rows that have a matching image in
    images.csv (`related_event_id`); the true amount must be extracted
    from that receipt image.

  * `flexibility`: fixed | stoppable | reducible | reducible_or_stoppable.
    Combined with the user's financial_profiles.csv category lists
    (`expense_categories_user_is_willing_to_stop` /
    `..._to_reduce`) to determine which recurring expenses are valid
    spending-change candidates. When reducing, the floor is the
    event's own `minimum_allowed_amount`.

  * Messages can amend the picture: confirm/replace a salary amount
    and effective date, mark income as unconfirmed (exclude it),
    flag a payout/refund as still-pending (exclude), note an internal
    transfer between the user's own accounts (net to zero), or note a
    category-wide change like a rent increase. See
    `message_facts.py` for extraction; this module only applies
    already-extracted structured facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import pandas as pd

EXCLUDED_STATUSES_ALWAYS = {"cancelled", "failed"}
NON_CASH_DIRECTION = "non_cash"

# Categories that represent salary/regular income (used to decide which
# recurring series a "salary changed" message fact should override).
INCOME_EVENT_TYPE = "income"


@dataclass
class RecurringSeries:
    user_id: str
    category: str
    event_type: str
    direction: str  # debit / credit
    currency: str
    interval_days: float
    representative_amount: float
    last_date: pd.Timestamp
    last_event_id: str
    flexibility: str
    minimum_allowed_amount: Optional[float]
    occurrences: int  # how many historical points this was built from


@dataclass
class ResolvedUserFinances:
    user_id: str
    one_off_events: pd.DataFrame  # single, non-recurring events still in-window
    recurring_series: list[RecurringSeries]
    notes: list[str] = field(default_factory=list)


def _is_recurring_category(event_type: str) -> bool:
    # One-off types are never treated as recurring series, even if a
    # category name repeats by coincidence.
    return event_type in {"expense", "subscription", "income", "debt_payment"}


def classify_and_clean(events: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the status/direction exclusion rules described above.
    Does NOT yet apply message-derived amendments — see
    apply_message_facts in message_facts.py, invoked by the resolver.
    """
    df = events.copy()

    # Always drop cancelled/failed.
    df = df[~df["status"].isin(EXCLUDED_STATUSES_ALWAYS)]

    # Drop unrealized / non-cash marks entirely.
    df = df[df["direction"] != NON_CASH_DIRECTION]

    # Drop pending credits (not yet confirmed received).
    pending_credit_mask = (df["status"] == "pending") & (df["direction"] == "credit")
    df = df[~pending_credit_mask]

    return df


def detect_recurring_series(
    history: pd.DataFrame,
    as_of_date: pd.Timestamp,
    min_occurrences: int = 2,
) -> tuple[list[RecurringSeries], pd.DataFrame]:
    """
    Group a user's events by (category, direction, event_type) and
    detect recurring series. Returns (series_list, leftover_one_off_events).

    A category is treated as recurring when it has >= min_occurrences
    points total (settled history AND already-scheduled future
    occurrences both count -- e.g. one settled salary plus one already
    scheduled next salary is enough to infer a monthly cadence). The
    representative amount is the median of the most recent occurrences
    (robust to one-off spikes); the interval is the median gap between
    consecutive dates. Projection into the forecast window always
    starts strictly after the latest known occurrence, so an
    already-scheduled future row is used as history for cadence
    detection but is not itself duplicated by the projection.
    """
    series_list: list[RecurringSeries] = []
    used_indices = set()

    history = history.sort_values("event_date")

    group_cols = ["user_id", "category", "direction", "event_type"]
    for key, group in history.groupby(group_cols):
        user_id, category, direction, event_type = key
        if not _is_recurring_category(event_type):
            continue
        if len(group) < min_occurrences:
            continue

        dates = pd.to_datetime(group["event_date"])
        diffs = dates.diff().dropna().dt.days
        if diffs.empty:
            continue
        interval = float(diffs.median())
        if interval <= 0:
            continue

        # Use the median of the last 3 occurrences as the representative
        # recurring amount. Pure "last value" is fooled by a one-off
        # outlier landing as the most recent point (e.g. a bulk/unusual
        # purchase in an otherwise steady weekly category); a longer
        # median (6+) is instead too slow to reflect a genuine level
        # shift (a pay cut or rent increase already visible in the last
        # 2-3 occurrences). A 3-point median balances both: a single
        # outlier is outvoted by the other two, while a shift sustained
        # across 2+ of the last 3 points comes through correctly.
        ordered = group.sort_values("event_date")
        last_row = ordered.iloc[-1]
        recent_amounts = ordered["amount"].tail(3)
        representative_amount = float(recent_amounts.median())

        flexibility = str(last_row.get("flexibility", "fixed"))
        min_allowed = last_row.get("minimum_allowed_amount", None)
        min_allowed = float(min_allowed) if pd.notna(min_allowed) else None

        series_list.append(
            RecurringSeries(
                user_id=user_id,
                category=category,
                event_type=event_type,
                direction=direction,
                currency=str(last_row["currency"]),
                interval_days=interval,
                representative_amount=representative_amount,
                last_date=dates.max(),
                last_event_id=str(last_row["event_id"]),
                flexibility=flexibility,
                minimum_allowed_amount=min_allowed,
                occurrences=len(group),
            )
        )
        used_indices.update(group.index)

    # Anything not absorbed into a recurring series (fewer than
    # min_occurrences points in its category/direction group) is
    # treated as a one-off event and included directly rather than
    # projected forward.
    leftover = history[~history.index.isin(used_indices)]
    return series_list, leftover


def expand_series_to_cashflow(
    series: RecurringSeries,
    start_date: pd.Timestamp,
    horizon_end: pd.Timestamp,
    override_amount: Optional[float] = None,
    skip: bool = False,
) -> list[tuple[pd.Timestamp, float, str]]:
    """
    Project a recurring series forward into [start_date, horizon_end].
    Returns a list of (date, signed_amount, source_event_id) tuples.

    `override_amount` lets a spending-change candidate (reduce_to) or a
    message-derived fact (e.g. new salary amount) replace the projected
    amount from the point it takes effect.
    """
    if skip:
        return []

    sign = 1.0 if series.direction == "credit" else -1.0
    amount = override_amount if override_amount is not None else series.representative_amount

    out = []
    cursor = series.last_date
    step = timedelta(days=round(series.interval_days))
    if step.days <= 0:
        return out

    # Walk forward from the last known occurrence until we're inside
    # the forecast window, then keep stepping through the window.
    while cursor < start_date:
        cursor = cursor + step
    while cursor <= horizon_end:
        out.append((cursor, sign * amount, series.last_event_id))
        cursor = cursor + step

    return out
