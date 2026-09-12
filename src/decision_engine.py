"""
Turns a user's forecast timeline + one request into the final decision
fields required by output.csv. Pure rule-based logic -- no LLM calls --
so results are fully reproducible given the same inputs.

Key mechanics confirmed against sample_requests.csv:

  * affordable_now requires the full amount to be safe TODAY WITHOUT
    any spending changes, and the user must accept full_payment.
  * If spending changes are needed to make even a same-day full
    payment safe, the method can still be full_payment, but the
    status becomes affordable_with_plan (not affordable_now).
  * earliest_date_for_full_payment is ALWAYS computed without
    optional spending changes, independent of the chosen method.
  * spending_changes_needed candidates: stop:<event_id> removes a
    stoppable recurring expense entirely; reduce_to:<event_id>:<amt>
    clamps a reducible expense down to exactly its own
    minimum_allowed_amount (confirmed against real examples -- the
    reduced amount is never an arbitrary computed value). A category
    is eligible only when the event's own `flexibility` field allows
    it AND the user's profile lists that category under
    willing_to_stop / willing_to_reduce.
  * installments must exactly match a supplied payment_option row
    (payment_option_id, schedule, total_payable_amount) -- never a
    freely computed schedule.
  * partial_payment is exactly two payments: amount_safe_to_pay on
    request_date, then the remainder on earliest_date_for_full_payment,
    only when allows_partial_payment is True, the user accepts
    partial_payment, 0 < amount_safe_to_pay < requested_amount, and
    earliest_date_for_full_payment <= desired_completion_date.
  * wait is eligible when full payment becomes safe on some date after
    request_date and the user accepts full_payment.
  * Ranking of otherwise-safe/eligible candidates (in order): complete
    by desired_completion_date > requires no spending changes > lowest
    total amount paid > earliest start date > fewest payments > lowest
    payment_option_id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import pandas as pd

from .forecast import (
    UserTimeline,
    earliest_safe_full_payment_date,
    max_safe_payment_today,
    run_forecast,
)

def _format_date(d) -> str:
    # strftime("%-d %B %Y") is a glibc-only extension (no leading-zero
    # day); it raises ValueError on Windows' MSVCRT. d.day is the
    # portable equivalent.
    return f"{d.day} {d.strftime('%B %Y')}"


AFFORDABLE_NOW = "affordable_now"
AFFORDABLE_WITH_PLAN = "affordable_with_plan"
AFFORDABLE_LATER = "affordable_later"
NOT_AFFORDABLE = "not_affordable"

FULL_PAYMENT = "full_payment"
PARTIAL_PAYMENT = "partial_payment"
INSTALLMENTS = "installments"
WAIT = "wait"
NOT_RECOMMENDED = "not_recommended"


@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: float
    number_of_payments: int
    first_payment_date: pd.Timestamp
    payment_frequency_days: Optional[float]
    financing_fee: float
    total_payable_amount: float

    def to_schedule(self) -> list[tuple[pd.Timestamp, float]]:
        if self.number_of_payments <= 1 or not self.payment_frequency_days:
            return [(self.first_payment_date, self.payment_amount)]
        freq = timedelta(days=int(self.payment_frequency_days))
        return [
            (self.first_payment_date + freq * i, self.payment_amount)
            for i in range(self.number_of_payments)
        ]


@dataclass
class SpendingChangeCandidate:
    kind: str
    event_id: str
    category: str
    new_amount: Optional[float] = None


@dataclass
class CandidatePlan:
    method: str
    payments: list[tuple[pd.Timestamp, float]]
    payment_option_id: Optional[str] = None
    spending_changes: list[SpendingChangeCandidate] = field(default_factory=list)
    completes_by_deadline: bool = False
    total_paid: float = 0.0
    first_payment_date: Optional[pd.Timestamp] = None
    num_payments: int = 0

    def sort_key(self):
        return (
            0 if self.completes_by_deadline else 1,
            1 if self.spending_changes else 0,
            round(self.total_paid, 2),
            self.first_payment_date if self.first_payment_date is not None else pd.Timestamp.max,
            self.num_payments,
            self.payment_option_id or "",
        )


def _format_amount(amount: float) -> str:
    """
    Format a number for machine-readable output fields (payment_plan,
    spending_changes_needed): plain decimal, no scientific notation,
    no unnecessary trailing zeros/decimal point, but full precision
    preserved (large IDR amounts can be in the tens of millions, where
    Python's default :g switches to scientific notation).
    """
    rounded = round(amount, 2)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def format_payment_plan(payments: list[tuple[pd.Timestamp, float]]) -> str:
    if not payments:
        return "none"
    return "|".join(f"{d.strftime('%Y-%m-%d')}:{_format_amount(amt)}" for d, amt in payments)


def format_spending_changes(changes: list[SpendingChangeCandidate]) -> str:
    if not changes:
        return "none"
    parts = []
    for c in changes[:3]:
        if c.kind == "stop":
            parts.append(f"stop:{c.event_id}")
        else:
            parts.append(f"reduce_to:{c.event_id}:{_format_amount(c.new_amount)}")
    return "|".join(parts)


def get_spending_change_candidates(
    timeline: UserTimeline,
    profile: pd.Series,
    request_date: pd.Timestamp,
    horizon_days: int = 90,
) -> list[SpendingChangeCandidate]:
    willing_to_stop = set(str(profile.get("expense_categories_user_is_willing_to_stop", "") or "").split("|"))
    willing_to_reduce = set(str(profile.get("expense_categories_user_is_willing_to_reduce", "") or "").split("|"))

    horizon_end = request_date + timedelta(days=horizon_days)
    candidates = []
    seen_categories = set()

    for s in timeline.recurring_series:
        if s.category in seen_categories:
            continue
        has_occurrence_in_window = any(
            p.date >= request_date and p.date <= horizon_end and p.source_event_id == s.last_event_id
            for p in timeline.points
        )
        if not has_occurrence_in_window:
            continue

        can_stop = s.flexibility in ("stoppable", "reducible_or_stoppable") and s.category in willing_to_stop
        can_reduce = (
            s.flexibility in ("reducible", "reducible_or_stoppable")
            and s.category in willing_to_reduce
            and s.minimum_allowed_amount is not None
        )

        if can_stop:
            candidates.append(SpendingChangeCandidate("stop", s.last_event_id, s.category))
            seen_categories.add(s.category)
        elif can_reduce:
            candidates.append(
                SpendingChangeCandidate("reduce_to", s.last_event_id, s.category, s.minimum_allowed_amount)
            )
            seen_categories.add(s.category)

    return candidates


def _spending_change_sets(candidates: list[SpendingChangeCandidate]) -> list[list[SpendingChangeCandidate]]:
    sets: list[list[SpendingChangeCandidate]] = [[]]
    for c in candidates:
        sets.append([c])
    if len(candidates) > 1:
        sets.append(candidates[:3])
    return sets


def _apply_changes_forecast_ok(
    timeline: UserTimeline,
    request_date: pd.Timestamp,
    changes: list[SpendingChangeCandidate],
    extra_payments: list[tuple[pd.Timestamp, float]],
) -> bool:
    excluded = {c.event_id for c in changes if c.kind == "stop"}
    reduced = {c.event_id: c.new_amount for c in changes if c.kind == "reduce_to"}
    safe, _, _ = run_forecast(
        timeline, request_date,
        extra_payments=extra_payments,
        excluded_source_ids=excluded,
        reduced_sources=reduced,
    )
    return safe


def decide(
    request_id: str,
    request_date: pd.Timestamp,
    requested_amount: float,
    desired_completion_date: pd.Timestamp,
    allows_partial_payment: bool,
    timeline: UserTimeline,
    profile: pd.Series,
    payment_options: list[PaymentOption],
) -> dict:
    payment_methods = set(
        str(profile.get("payment_methods_user_will_consider", "") or "").split("|")
    )

    amount_safe = max_safe_payment_today(timeline, request_date, requested_amount)
    earliest_full_date = earliest_safe_full_payment_date(timeline, request_date, requested_amount)

    spending_candidates = get_spending_change_candidates(timeline, profile, request_date)
    change_sets = _spending_change_sets(spending_candidates)

    candidates: list[CandidatePlan] = []

    if FULL_PAYMENT in payment_methods:
        for changes in change_sets:
            if _apply_changes_forecast_ok(
                timeline, request_date, changes, [(request_date, -requested_amount)]
            ):
                candidates.append(
                    CandidatePlan(
                        method=FULL_PAYMENT,
                        payments=[(request_date, requested_amount)],
                        spending_changes=changes,
                        completes_by_deadline=request_date <= desired_completion_date,
                        total_paid=requested_amount,
                        first_payment_date=request_date,
                        num_payments=1,
                    )
                )
                break

    if FULL_PAYMENT in payment_methods and earliest_full_date is not None and earliest_full_date > request_date:
        candidates.append(
            CandidatePlan(
                method=WAIT,
                payments=[(earliest_full_date, requested_amount)],
                completes_by_deadline=earliest_full_date <= desired_completion_date,
                total_paid=requested_amount,
                first_payment_date=earliest_full_date,
                num_payments=1,
            )
        )

    if (
        PARTIAL_PAYMENT in payment_methods
        and allows_partial_payment
        and 0 < amount_safe < requested_amount
        and earliest_full_date is not None
        and earliest_full_date <= desired_completion_date
    ):
        remaining = round(requested_amount - amount_safe, 2)
        candidates.append(
            CandidatePlan(
                method=PARTIAL_PAYMENT,
                payments=[(request_date, amount_safe), (earliest_full_date, remaining)],
                completes_by_deadline=True,
                total_paid=requested_amount,
                first_payment_date=request_date,
                num_payments=2,
            )
        )

    if INSTALLMENTS in payment_methods:
        max_months = profile.get("max_installment_months", None)
        max_months = float(max_months) if pd.notna(max_months) else None

        for opt in payment_options:
            if opt.payment_method != "installments":
                continue
            if max_months is not None and opt.payment_frequency_days:
                span_months = (opt.number_of_payments * opt.payment_frequency_days) / 30.44
                if span_months > max_months + 0.5:
                    continue

            schedule = opt.to_schedule()
            extra_payments = [(d, -amt) for d, amt in schedule]
            if _apply_changes_forecast_ok(timeline, request_date, [], extra_payments):
                last_payment_date = max(d for d, _ in schedule)
                candidates.append(
                    CandidatePlan(
                        method=INSTALLMENTS,
                        payments=schedule,
                        payment_option_id=opt.payment_option_id,
                        completes_by_deadline=last_payment_date <= desired_completion_date,
                        total_paid=opt.total_payable_amount,
                        first_payment_date=schedule[0][0],
                        num_payments=len(schedule),
                    )
                )

    usable = [c for c in candidates if c.completes_by_deadline]

    if usable:
        usable.sort(key=lambda c: c.sort_key())
        best = usable[0]

        if best.method == FULL_PAYMENT and not best.spending_changes:
            status = AFFORDABLE_NOW
        elif best.method == WAIT:
            status = AFFORDABLE_LATER
        else:
            status = AFFORDABLE_WITH_PLAN

        explanation = _build_explanation(
            best, requested_amount, timeline.home_currency, timeline.minimum_balance_to_keep
        )

        return {
            "request_id": request_id,
            "amount_safe_to_pay": amount_safe,
            "affordability_status": status,
            "recommended_payment_method": best.method,
            "payment_plan": format_payment_plan(best.payments),
            "earliest_date_for_full_payment": (
                request_date.strftime("%Y-%m-%d")
                if status == AFFORDABLE_NOW
                else (earliest_full_date.strftime("%Y-%m-%d") if earliest_full_date is not None else "")
            ),
            "spending_changes_needed": format_spending_changes(best.spending_changes),
            "decision_explanation": explanation,
        }

    return {
        "request_id": request_id,
        "amount_safe_to_pay": amount_safe,
        "affordability_status": NOT_AFFORDABLE,
        "recommended_payment_method": NOT_RECOMMENDED,
        "payment_plan": "none",
        "earliest_date_for_full_payment": (
            earliest_full_date.strftime("%Y-%m-%d") if earliest_full_date is not None else ""
        ),
        "spending_changes_needed": "none",
        "decision_explanation": (
            f"Do not make this payment by {_format_date(desired_completion_date)}. "
            f"None of the available options keeps the {timeline.home_currency} "
            f"{timeline.minimum_balance_to_keep:,.0f} minimum protected."
        ),
    }


def _build_explanation(
    plan: CandidatePlan,
    requested_amount: float,
    currency: str,
    min_balance: float,
) -> str:
    if plan.method == FULL_PAYMENT:
        base = f"Pay {currency} {requested_amount:,.2f} today."
    elif plan.method == WAIT:
        base = f"Wait until {_format_date(plan.payments[0][0])}, then pay {currency} {requested_amount:,.2f} in full."
    elif plan.method == PARTIAL_PAYMENT:
        base = (
            f"Pay {currency} {plan.payments[0][1]:,.2f} today and the remaining "
            f"{currency} {plan.payments[1][1]:,.2f} on {_format_date(plan.payments[1][0])}."
        )
    elif plan.method == INSTALLMENTS:
        base = (
            f"Use {plan.num_payments} installments of {currency} {plan.payments[0][1]:,.2f}, "
            f"starting {_format_date(plan.payments[0][0])}."
        )
    else:
        base = ""

    if plan.spending_changes:
        change_desc = " and ".join(
            f"stop the {c.category} expense" if c.kind == "stop" else f"reduce {c.category} to {_format_amount(c.new_amount)}"
            for c in plan.spending_changes
        )
        change_desc = change_desc[0].upper() + change_desc[1:]
        base = f"{change_desc}, then {base[0].lower()}{base[1:]}"

    return f"{base} This leaves at least {currency} {min_balance:,.0f} available over the next 90 days."
