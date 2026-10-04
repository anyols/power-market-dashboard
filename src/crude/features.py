"""Crude-oil fundamentals: seasonal inventory comparisons, curve, positioning.

Oil-market intuition encoded here
---------------------------------
* Inventories are the oil market's shock absorber. Stocks below their
  seasonal norm mean less buffer against disruptions, so prompt barrels
  command a premium: prices and time spreads (backwardation) rise.
* Seasonality is large (spring builds, summer draws), so levels and weekly
  changes are always judged against the *same week* in previous years.
* Cushing, Oklahoma is the delivery point of the NYMEX WTI contract. Low
  Cushing stocks tighten the front of the WTI curve directly; near tank
  bottoms (~20 mb) squeezes become possible.
* The futures curve is the market's own read of tightness: backwardation
  (front above deferred) signals scarcity and pays a roll yield to holders
  of the front contract; contango signals surplus and pays for storage.
* Positioning: when speculators are already very long, the marginal buyer
  is scarce, and the asymmetry favours a pullback (and vice versa).

All seasonal statistics use only *previous* years, so a value is never
compared with a benchmark that contains itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.crude.contracts import roll_adjusted_changes

STOCK_COLUMNS = ["crude_stocks", "cushing_stocks", "gasoline_stocks", "distillate_stocks"]
SEASONAL_COLUMNS = STOCK_COLUMNS + ["refinery_utilization", "product_supplied", "total_commercial_stocks"]
WEEKLY_RELEASE_LAG = pd.Timedelta(days=6)  # week ending Friday -> known by Thursday (Wed release, Thu in holiday weeks)
COT_RELEASE_LAG = pd.Timedelta(days=3)  # Tuesday positions -> Friday release


def iso_year_week(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    iso = index.isocalendar()
    return iso.year.to_numpy().astype(int), np.minimum(iso.week.to_numpy().astype(int), 52)


def seasonal_stats(s: pd.Series, years: int = 5, min_years: int = 3) -> pd.DataFrame:
    """Average / min / max of the same ISO week over the previous `years` years.

    Week 53 is folded into week 52. Fewer than `min_years` prior observations
    for that week gives NaN. The current year is never part of its own benchmark.
    """
    year, week = iso_year_week(s.index)
    frame = pd.DataFrame({"v": s.to_numpy(), "year": year, "week": week})
    all_years = np.unique(year)
    table = frame.pivot_table(index="year", columns="week", values="v", aggfunc="mean").reindex(
        index=all_years, columns=range(1, 53))
    parts = []
    for y in all_years:
        prior = table.loc[(table.index >= y - years) & (table.index < y)]
        enough = prior.notna().sum() >= min_years
        parts.append(pd.DataFrame({"year": y, "week": range(1, 53), "avg": prior.mean().where(enough).to_numpy(),
                                   "min": prior.min().where(enough).to_numpy(),
                                   "max": prior.max().where(enough).to_numpy()}))
    lookup = pd.concat(parts).set_index(["year", "week"])
    out = lookup.reindex(pd.MultiIndex.from_arrays([year, week]))
    out.index = s.index
    return out


def add_weekly_features(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly.copy()
    out["total_commercial_stocks"] = out["crude_stocks"] + out["gasoline_stocks"] + out["distillate_stocks"]
    for col in SEASONAL_COLUMNS:
        st = seasonal_stats(out[col])
        out[f"{col}_5y_avg"], out[f"{col}_5y_min"], out[f"{col}_5y_max"] = st["avg"], st["min"], st["max"]
        out[f"{col}_vs_5y"] = out[col] - st["avg"]
        out[f"{col}_vs_5y_pct"] = out[f"{col}_vs_5y"] / st["avg"]
    for col in STOCK_COLUMNS + ["total_commercial_stocks"]:
        chg = out[col].diff()
        out[f"{col}_chg"] = chg
        out[f"{col}_chg_5y_avg"] = seasonal_stats(chg)["avg"]
        # "Surprise" vs the seasonal norm. A true surprise would be vs analyst
        # consensus, which is not freely available - stated in the UI.
        out[f"{col}_surprise"] = chg - out[f"{col}_chg_5y_avg"]
    out["days_of_cover"] = out["crude_stocks"] / out["refinery_crude_input"]
    out["product_supplied_4w"] = out["product_supplied"].rolling(4, min_periods=4).mean()
    out["available_date"] = out.index + WEEKLY_RELEASE_LAG
    return out


def add_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["brent_wti_spot"] = out["brent_spot"] - out["wti_spot"]
    out["m1_m2"] = out["cl1"] - out["cl2"]  # > 0: backwardation
    out["m1_m4"] = out["cl1"] - out["cl4"]
    out["cl1_roll_adj_change"] = roll_adjusted_changes(out["cl1"], out["cl2"])
    prev = out["wti_spot"].shift(1)
    # WTI went negative on 20 Apr 2020; returns are undefined off a non-positive base.
    ret = (out["wti_spot"] - prev) / prev.where(prev > 0)
    out["realized_vol_20d"] = ret.rolling(20, min_periods=15).std() * np.sqrt(252)
    return out


def trailing_percentile(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Share of the previous `window` observations below the current value (excludes itself)."""
    mp = min_periods or window // 2

    def pct(x: np.ndarray) -> float:
        hist, cur = x[:-1], x[-1]
        hist = hist[~np.isnan(hist)]
        return np.nan if len(hist) < mp or np.isnan(cur) else float((hist < cur).mean())

    return s.rolling(window + 1, min_periods=mp + 1).apply(pct, raw=True)


def add_cot_features(cot: pd.DataFrame, window_weeks: int = 156) -> pd.DataFrame:
    out = cot.copy()
    out["mm_net"] = out["mm_long"] - out["mm_short"]
    out["mm_net_pct_oi"] = out["mm_net"] / out["open_interest"]
    out["mm_net_change"] = out["mm_net"].diff()
    out["mm_net_pctile"] = trailing_percentile(out["mm_net_pct_oi"], window_weeks)
    out["available_date"] = out.index + COT_RELEASE_LAG
    return out


def seasonal_profile(s: pd.Series, year: int | None = None, years: int = 5) -> pd.DataFrame:
    """Week-of-year view for charts: previous `years` range/average vs the current and previous year."""
    yr, wk = iso_year_week(s.index)
    frame = pd.DataFrame({"v": s.to_numpy(), "year": yr, "week": wk}).dropna()
    year = int(year or frame["year"].max())
    table = frame.pivot_table(index="week", columns="year", values="v", aggfunc="mean").reindex(range(1, 53))
    prior = table[[c for c in table.columns if year - years <= c < year]]
    return pd.DataFrame({
        "min": prior.min(axis=1), "max": prior.max(axis=1), "avg": prior.mean(axis=1),
        "current": table.get(year), "last_year": table.get(year - 1),
    }, index=pd.Index(range(1, 53), name="week"))
