"""Small, dependency-free helpers shared across modules.

Time conventions used throughout the project
--------------------------------------------
* Internal hourly frames are indexed by tz-aware timestamps in MARKET_TZ
  (Europe/Brussels), each stamp marking the *start* of the delivery hour.
* A "market day" / "delivery day" is a local calendar day. It has 23 hours
  on the spring DST switch and 25 hours in autumn, so never assume 24 rows.
* "Same hour on the previous day" is matched on (local date, local hour),
  not on a 24-row shift, so DST days do not misalign the comparison.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config import MARKET_TZ


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------
def to_market_tz(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert any tz-aware index to market time; naive input is assumed UTC."""
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(MARKET_TZ)


def local_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive midnight timestamps of the local delivery day for each row."""
    return to_market_tz(index).tz_localize(None).normalize()


def market_day_bounds(start_date, end_date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) tz-aware bounds covering whole local delivery days."""
    start = pd.Timestamp(start_date).normalize().tz_localize(MARKET_TZ)
    end = (pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)).tz_localize(MARKET_TZ)
    return start, end


def slice_market_days(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Rows whose local delivery day lies in [start_date, end_date] (inclusive)."""
    start, end = market_day_bounds(start_date, end_date)
    return df.loc[(df.index >= start) & (df.index < end)]


def same_hour_lag(s: pd.Series, days: int = 1) -> pd.Series:
    """Value at the same local delivery hour `days` market days earlier.

    Matching on (local date, local hour) keeps DST days aligned. If the
    reference hour does not exist (spring-forward) the result is NaN; on the
    autumn 25-hour day the first occurrence of the repeated hour is used.
    """
    dates = local_dates(s.index)
    hours = to_market_tz(s.index).hour
    lookup = pd.Series(s.to_numpy(), index=pd.MultiIndex.from_arrays([dates, hours]))
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    keys = pd.MultiIndex.from_arrays([dates - pd.Timedelta(days=days), hours])
    return pd.Series(lookup.reindex(keys).to_numpy(), index=s.index, name=s.name)


def daily_to_hourly(daily: pd.Series, hourly_index: pd.DatetimeIndex) -> pd.Series:
    """Broadcast a series indexed by naive local dates onto an hourly index."""
    mapped = daily.reindex(local_dates(hourly_index))
    return pd.Series(mapped.to_numpy(), index=hourly_index, name=daily.name)


def full_daily_index(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Continuous daily range spanning the given (naive) dates."""
    return pd.date_range(dates.min(), dates.max(), freq="D")


# ---------------------------------------------------------------------------
# Market-language helpers
# ---------------------------------------------------------------------------
def period_of_day(hour: int) -> str:
    """Trader shorthand for a delivery hour."""
    if hour < 6:
        return "overnight"
    if hour < 10:
        return "morning ramp"
    if hour < 16:
        return "midday"
    if hour < 18:
        return "afternoon"
    if hour < 21:
        return "evening peak"
    return "late evening"


def describe_hours(hours: Iterable[int]) -> str:
    """Summarise a set of hours as the dominant part(s) of the day."""
    hours = list(hours)
    if not hours:
        return "no particular part of the day"
    counts = pd.Series([period_of_day(h) for h in hours]).value_counts()
    top = counts.index[:2].tolist()
    if len(top) == 1 or counts.iloc[1] < 0.5 * counts.iloc[0]:
        return f"the {top[0]}"
    return f"the {top[0]} and {top[1]}"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
MINUS = "−"  # typographic minus, reads better than a hyphen in prose


def fmt_eur(x: float, decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    sign = MINUS if x < 0 and round(abs(x), decimals) > 0 else ""
    return f"{sign}€{abs(x):,.{decimals}f}/MWh"


def fmt_gw(mw: float, decimals: int = 1, signed: bool = False) -> str:
    if mw is None or pd.isna(mw):
        return "n/a"
    sign = MINUS if mw < 0 else ("+" if signed and mw > 0 else "")
    return f"{sign}{abs(mw) / 1000:,.{decimals}f} GW"


def fmt_pct(x: float, decimals: int = 0, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    sign = MINUS if x < 0 else ("+" if signed and x > 0 else "")
    return f"{sign}{abs(x) * 100:.{decimals}f}%"


def fmt_signed(x: float, decimals: int = 1, unit: str = "") -> str:
    if x is None or pd.isna(x):
        return "n/a"
    sign = MINUS if x < 0 else "+"
    return f"{sign}{abs(x):,.{decimals}f}{unit}"


def safe_div(a: float, b: float) -> float:
    if b is None or a is None or pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
