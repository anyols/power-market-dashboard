"""Cleaning and validation of hourly market frames.

ENTSO-E data is messy in predictable ways: mixed resolutions (15-min load in
DE, 15-min day-ahead prices since the SDAC 15-minute MTU go-live on
1 Oct 2025), occasional gaps, duplicated revisions and DST quirks. This module
normalises everything onto one hourly grid in market time and produces a
data-quality report that the dashboard surfaces to the user instead of
silently papering over problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import MARKET_TZ, PRICE_CAP, PRICE_FLOOR
from src.utils import market_day_bounds, to_market_tz

CORE_COLUMNS = [
    "price",
    "load_actual",
    "load_forecast",
    "wind_actual",
    "wind_forecast",
    "solar_actual",
    "solar_forecast",
]

MAX_INTERPOLATION_GAP_H = 3


@dataclass
class DataQualityReport:
    rows: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    missing_before: dict[str, int] = field(default_factory=dict)
    interpolated: dict[str, int] = field(default_factory=dict)
    missing_after: dict[str, int] = field(default_factory=dict)
    longest_gap_h: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.warnings

    def coverage(self, column: str) -> float:
        if not self.rows:
            return 0.0
        return 1 - self.missing_after.get(column, 0) / self.rows

    def to_frame(self) -> pd.DataFrame:
        cols = sorted(set(self.missing_before) | set(self.missing_after))
        return pd.DataFrame(
            {
                "missing (raw)": [self.missing_before.get(c, 0) for c in cols],
                "interpolated": [self.interpolated.get(c, 0) for c in cols],
                "still missing": [self.missing_after.get(c, 0) for c in cols],
                "longest gap (h)": [self.longest_gap_h.get(c, 0) for c in cols],
                "coverage": [f"{self.coverage(c):.1%}" for c in cols],
            },
            index=pd.Index(cols, name="column"),
        )


def to_hourly(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Resample (sub-)hourly MW or EUR/MWh data to hourly means.

    ENTSO-E quantities are average power over each interval (MW) and prices
    are per MWh, so the hourly value is the arithmetic mean of the intervals.
    This is also how the hourly index is derived from 15-minute day-ahead
    prices after the SDAC 15-minute MTU transition.
    """
    if obj.empty:
        return obj
    obj = obj.copy()
    obj.index = to_market_tz(pd.DatetimeIndex(obj.index))
    obj = obj[~obj.index.duplicated(keep="last")].sort_index()
    # Resample in UTC so that DST transitions never create ambiguous bins.
    utc = obj.tz_convert("UTC")
    hourly = utc.resample("h").mean()
    return hourly.tz_convert(MARKET_TZ)


def align_to_hourly_grid(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Reindex onto every delivery hour between two local dates (inclusive)."""
    start, end = market_day_bounds(start_date, end_date)
    grid = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")
    out = df.copy()
    out.index = to_market_tz(pd.DatetimeIndex(out.index)).tz_convert("UTC")
    out = out[~out.index.duplicated(keep="last")].reindex(grid)
    out.index = out.index.tz_convert(MARKET_TZ)
    out.index.name = "timestamp"
    return out


def _nan_runs(s: pd.Series) -> pd.Series:
    """Length of the NaN run each element belongs to (0 for valid values)."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    return run_len.where(isna, 0).astype(int)


def fill_short_gaps(df: pd.DataFrame, max_gap_h: int = MAX_INTERPOLATION_GAP_H) -> tuple[pd.DataFrame, dict[str, int]]:
    """Linearly interpolate interior gaps of at most `max_gap_h` hours.

    Longer gaps are left as NaN on purpose: inventing several hours of wind
    or load would contaminate forecast-error statistics and backtests.
    """
    out = df.copy()
    filled: dict[str, int] = {}
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        s = out[col]
        runs = _nan_runs(s)
        fillable = (runs > 0) & (runs <= max_gap_h)
        if not fillable.any():
            continue
        interp = s.interpolate(method="time", limit_area="inside")
        fillable &= interp.notna()
        out[col] = s.where(~fillable, interp)
        filled[col] = int(fillable.sum())
    return out, filled


def validate_market_frame(df: pd.DataFrame, report: DataQualityReport | None = None) -> DataQualityReport:
    """Run sanity checks and collect human-readable warnings."""
    report = report or DataQualityReport()
    report.rows = len(df)
    if len(df):
        report.start, report.end = df.index.min(), df.index.max()

    for col in CORE_COLUMNS:
        if col not in df.columns:
            report.warnings.append(f"Column '{col}' is unavailable for this zone/period.")
            continue
        n_missing = int(df[col].isna().sum())
        report.missing_after[col] = n_missing
        longest = int(_nan_runs(df[col]).max()) if n_missing else 0
        report.longest_gap_h[col] = longest
        if n_missing == len(df):
            report.warnings.append(f"'{col}' is entirely missing.")
        elif longest > MAX_INTERPOLATION_GAP_H:
            report.warnings.append(
                f"'{col}' has {n_missing} missing hours (longest gap {longest} h); "
                "affected hours are excluded from statistics rather than imputed."
            )

    if "price" in df:
        bad = df["price"].dropna()
        out_of_bounds = int(((bad < PRICE_FLOOR) | (bad > PRICE_CAP)).sum())
        if out_of_bounds:
            report.warnings.append(
                f"{out_of_bounds} price values fall outside the SDAC limits "
                f"[{PRICE_FLOOR:.0f}, {PRICE_CAP:.0f}] EUR/MWh."
            )
    for col in ("load_actual", "load_forecast", "wind_actual", "solar_actual", "wind_forecast", "solar_forecast"):
        if col in df and (df[col] < 0).any():
            report.warnings.append(f"'{col}' contains negative values; check the source data.")

    if len(df) > 1:
        step = df.index.to_series().diff().dropna()
        if (step != pd.Timedelta(hours=1)).any():
            report.warnings.append("Index is not a continuous hourly grid.")
    return report


def clean_market_frame(raw: pd.DataFrame, start_date=None, end_date=None) -> tuple[pd.DataFrame, DataQualityReport]:
    """Hourly-align, gap-fill (short gaps only) and validate a market frame."""
    report = DataQualityReport()
    df = to_hourly(raw)
    if start_date is not None and end_date is not None:
        df = align_to_hourly_grid(df, start_date, end_date)
    report.missing_before = {c: int(df[c].isna().sum()) for c in df.columns}
    df, report.interpolated = fill_short_gaps(df)
    # Renewable output and load cannot be negative; tiny negatives from
    # interpolation or TSO netting are clipped.
    for col in ("wind_actual", "wind_forecast", "solar_actual", "solar_forecast"):
        if col in df:
            df[col] = df[col].clip(lower=0)
    validate_market_frame(df, report)
    return df, report


def summarise_flows_quality(flows: pd.DataFrame) -> list[str]:
    """Warnings for the long-format flow table (timestamp, from_zone, to_zone, flow_mw)."""
    if flows is None or flows.empty:
        return ["No cross-border flow data available."]
    warnings = []
    n_missing = int(flows["flow_mw"].isna().sum())
    if n_missing:
        warnings.append(f"{n_missing} missing flow values.")
    if (flows["flow_mw"].dropna() < 0).any():
        warnings.append("Negative directional flows found; ENTSO-E reports each direction as >= 0.")
    return warnings
