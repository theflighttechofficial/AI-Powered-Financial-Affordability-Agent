"""
Synthetic smoke tests for src/forecast.py's core safety-check math --
no real dataset needed. These construct a UserTimeline by hand so the
tests are independent of recurring-series detection, message parsing,
or currency conversion, and only exercise run_forecast /
max_safe_payment_today / earliest_safe_full_payment_date.

Run with: python -m pytest tests/test_forecast.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.forecast import (
    CashflowPoint,
    UserTimeline,
    run_forecast,
    max_safe_payment_today,
    earliest_safe_full_payment_date,
)


def d(s):
    return pd.Timestamp(s)


def make_timeline(start_balance, min_balance, points):
    return UserTimeline(
        user_id="test_user",
        home_currency="USD",
        start_balance=start_balance,
        minimum_balance_to_keep=min_balance,
        points=points,
        recurring_series=[],
    )


def pt(date, amount, source="ev", category="other"):
    return CashflowPoint(date=d(date), amount=amount, source_event_id=source, category=category)


def test_simple_safe_forecast():
    timeline = make_timeline(
        1000, 0,
        [pt("2026-01-15", -200, "rent"), pt("2026-01-30", 3000, "salary")],
    )
    safe, min_bal, _ = run_forecast(timeline, d("2026-01-01"))
    assert safe
    assert min_bal == 800  # 1000 - 200, before salary arrives


def test_unsafe_forecast_breaches_minimum():
    timeline = make_timeline(1000, 0, [pt("2026-01-05", -1500, "big_bill")])
    safe, min_bal, _ = run_forecast(timeline, d("2026-01-01"))
    assert not safe
    assert min_bal == -500


def test_minimum_balance_threshold_respected():
    timeline = make_timeline(1000, 700, [pt("2026-01-05", -400, "bill")])
    safe, _, _ = run_forecast(timeline, d("2026-01-01"))
    assert not safe  # 1000-400=600 < 700

    timeline2 = make_timeline(1000, 500, [pt("2026-01-05", -400, "bill")])
    safe2, _, _ = run_forecast(timeline2, d("2026-01-01"))
    assert safe2


def test_max_safe_payment_today_binary_search():
    timeline = make_timeline(1000, 200, [])
    amount = max_safe_payment_today(timeline, d("2026-01-01"), requested_amount=1000)
    assert abs(amount - 800) < 0.02


def test_max_safe_payment_capped_at_requested_amount():
    timeline = make_timeline(1000, 200, [])
    amount = max_safe_payment_today(timeline, d("2026-01-01"), requested_amount=300)
    assert amount == 300


def test_earliest_safe_full_payment_date_future():
    timeline = make_timeline(500, 0, [pt("2026-01-10", 2000, "salary")])
    earliest = earliest_safe_full_payment_date(timeline, d("2026-01-01"), requested_amount=2000)
    assert earliest == d("2026-01-10")


def test_earliest_safe_full_payment_none_within_horizon():
    timeline = make_timeline(100, 0, [])
    earliest = earliest_safe_full_payment_date(timeline, d("2026-01-01"), requested_amount=100000)
    assert earliest is None


def test_excluded_source_id_removes_expense():
    timeline = make_timeline(1000, 900, [pt("2026-01-05", -150, "streaming_sub")])
    safe_without_exclusion, _, _ = run_forecast(timeline, d("2026-01-01"))
    assert not safe_without_exclusion  # 1000-150=850 < 900

    safe_with_exclusion, _, _ = run_forecast(
        timeline, d("2026-01-01"), excluded_source_ids={"streaming_sub"}
    )
    assert safe_with_exclusion  # expense dropped entirely


def test_reduced_source_clamps_expense():
    timeline = make_timeline(1000, 900, [pt("2026-01-05", -150, "dining_out")])
    safe, min_bal, _ = run_forecast(
        timeline, d("2026-01-01"), reduced_sources={"dining_out": 50}
    )
    assert safe
    assert min_bal == 950  # 1000 - min(150, 50)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
