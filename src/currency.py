"""
Currency conversion using fixed, dated rates from exchange_rates.csv.

Confirmed schema: rate_date, from_currency, to_currency, rate.
Rates are only provided monthly (typically the 15th). Cross-currency
events in financial_events.csv (e.g. USD salary for an IDR/INR user)
are dated to align with an available rate row for that same date.

For any date without an exact rate row, we fall back to the nearest
available rate for that currency pair, since these are described as
"fixed, dated conversion rates" rather than a continuous market feed.
"""
from __future__ import annotations

import pandas as pd


class RateNotFoundError(Exception):
    pass


class CurrencyConverter:
    def __init__(self, exchange_rates: pd.DataFrame):
        self.rates = exchange_rates.copy()
        if "rate_date" in self.rates.columns:
            self.rates["rate_date"] = pd.to_datetime(self.rates["rate_date"])

    def convert(self, amount: float, from_ccy: str, to_ccy: str, on_date) -> float:
        if from_ccy == to_ccy:
            return amount
        if isinstance(on_date, str):
            on_date = pd.to_datetime(on_date)

        rate = self._find_rate(from_ccy, to_ccy, on_date)
        return amount * rate

    def _find_rate(self, from_ccy: str, to_ccy: str, on_date: pd.Timestamp) -> float:
        exact = self.rates[
            (self.rates["from_currency"] == from_ccy)
            & (self.rates["to_currency"] == to_ccy)
            & (self.rates["rate_date"] == on_date)
        ]
        if not exact.empty:
            return float(exact.iloc[0]["rate"])

        exact_inv = self.rates[
            (self.rates["from_currency"] == to_ccy)
            & (self.rates["to_currency"] == from_ccy)
            & (self.rates["rate_date"] == on_date)
        ]
        if not exact_inv.empty:
            return 1.0 / float(exact_inv.iloc[0]["rate"])

        direct = self.rates[
            (self.rates["from_currency"] == from_ccy) & (self.rates["to_currency"] == to_ccy)
        ].copy()
        if not direct.empty:
            direct["diff"] = (direct["rate_date"] - on_date).abs()
            row = direct.sort_values("diff").iloc[0]
            return float(row["rate"])

        inverse = self.rates[
            (self.rates["from_currency"] == to_ccy) & (self.rates["to_currency"] == from_ccy)
        ].copy()
        if not inverse.empty:
            inverse["diff"] = (inverse["rate_date"] - on_date).abs()
            row = inverse.sort_values("diff").iloc[0]
            return 1.0 / float(row["rate"])

        # Bridge via USD if no direct/inverse pair exists at any date
        if from_ccy != "USD" and to_ccy != "USD":
            to_usd = None
            try:
                to_usd = self._find_rate(from_ccy, "USD", on_date)
            except RateNotFoundError:
                pass
            if to_usd is not None:
                try:
                    usd_to_target = self._find_rate("USD", to_ccy, on_date)
                    return to_usd * usd_to_target
                except RateNotFoundError:
                    pass

        raise RateNotFoundError(f"No exchange rate found for {from_ccy}->{to_ccy} near {on_date}")
