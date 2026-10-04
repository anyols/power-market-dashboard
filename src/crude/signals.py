"""Weekly, rule-based crude signal with an explicit, release-calendar-aware information set.

Decision: every Thursday at the NYMEX settlement (14:30 ET). Position held
until the following Thursday's settlement. CL offers Trade-at-Settlement
(TAS) orders, so settlement-to-settlement PnL is close to executable - unlike
the power proxy, this is a realistic entry/exit convention.

Votes (each -1 / 0 / +1):
* Inventory surprise  - last reported weekly crude-stock change vs the 5-year
  average change for that week. Bigger draw than normal = +1.
* Cushing tightness   - Cushing stocks vs their 5-year seasonal average.
  Unusually low = +1 (prompt WTI tightness).
* Curve (carry)       - front minus second-month spread. Backwardation = +1:
  the holder of the front contract earns roll yield.
* Positioning         - managed-money net length percentile over 3 years.
  Crowded long = -1, crowded short = +1 (contrarian).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.crude.config import BARRELS_PER_CONTRACT

CRUDE_INFORMATION_SET = [
    {"data": "EIA weekly balances, week ending Friday F", "known from": "Thursday F+6",
     "why": "Released Wednesday 10:30 ET (Thursday 11:00 ET in holiday weeks) - always before Thursday's settlement."},
    {"data": "CFTC COT positions as of Tuesday T", "known from": "Friday T+3",
     "why": "Published Friday 15:30 ET, so Thursday decisions use the previous week's report."},
    {"data": "Futures settlement prices / curve on day D", "known from": "D (settlement)",
     "why": "Entry is at settlement via TAS; the curve vote uses the same settlement."},
    {"data": "Anything after D", "known from": "Never (target only)",
     "why": "The target is the next week's settlement-to-settlement move."},
]

COMPONENTS = {"comp_inventory": "Inventory surprise", "comp_cushing": "Cushing tightness",
              "comp_curve": "Curve (carry)", "comp_positioning": "Positioning (contrarian)"}


class CrudeSignalParams(BaseModel):
    target: Literal["futures", "spot"] = Field(
        "futures", description="futures = roll-adjusted NYMEX front month (EIA, to Apr 2024); spot = WTI Cushing spot.")
    use_curve: bool = True
    use_positioning: bool = True
    surprise_z: float = Field(1.0, gt=0)
    cushing_z: float = Field(1.0, gt=0)
    curve_threshold: float = Field(0.10, ge=0, description="$/bbl M1-M2 dead-band")
    crowding_pctile: float = Field(0.85, gt=0.5, lt=1)
    entry_threshold: int = Field(2, ge=1, le=4)
    zscore_window_weeks: int = Field(104, ge=26, le=520)


def thursdays(start, end) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="W-THU")


def _asof(dates: pd.DatetimeIndex, table: pd.DataFrame, on: str | None, cols: list[str],
          tolerance: pd.Timedelta | None = None) -> pd.DataFrame:
    """Latest row of `table` whose availability date (column `on`, or the index) is <= each date."""
    left = pd.DataFrame({"decision": pd.DatetimeIndex(dates)})
    right = table.copy()
    right["_key"] = right[on] if on else right.index
    right["_row_date"] = right.index
    right = right.dropna(subset=["_key"]).sort_values("_key")
    merged = pd.merge_asof(left, right[["_key", "_row_date", *cols]], left_on="decision", right_on="_key",
                           direction="backward", tolerance=tolerance)
    return merged.set_index("decision").drop(columns="_key")


def _asof_series(dates: pd.DatetimeIndex, s: pd.Series, tolerance: pd.Timedelta | None = None) -> pd.Series:
    """Latest non-missing value of `s` on or before each date."""
    s = s.dropna()
    left = pd.DataFrame({"d": pd.DatetimeIndex(dates)})
    right = pd.DataFrame({"d": s.index, "v": s.to_numpy()})
    return pd.merge_asof(left, right, on="d", direction="backward", tolerance=tolerance).set_index("d")["v"]


def _vote(x: pd.Series, upper: float, lower: float, sign: int = 1) -> pd.Series:
    out = pd.Series(np.select([x > upper, x < lower], [sign, -sign], 0), index=x.index, dtype=float)
    return out.where(x.notna())


def prepare_weekly_inputs(weekly_feat: pd.DataFrame, params: CrudeSignalParams) -> pd.DataFrame:
    """Standardise the weekly votes' raw inputs using trailing windows (row-wise known data only)."""
    w = params.zscore_window_weeks
    mp = max(26, w // 3)
    out = weekly_feat.copy()
    surprise = out["crude_stocks_surprise"]
    out["inventory_z"] = surprise / surprise.rolling(w, min_periods=mp).std()
    dev = out["cushing_stocks_vs_5y_pct"]
    out["cushing_z"] = (dev - dev.rolling(w, min_periods=mp).mean()) / dev.rolling(w, min_periods=mp).std()
    return out


def build_crude_signals(daily_feat: pd.DataFrame, weekly_feat: pd.DataFrame, cot_feat: pd.DataFrame | None,
                        params: CrudeSignalParams | None = None, dates: pd.DatetimeIndex | None = None,
                        curve_override: pd.Series | None = None) -> pd.DataFrame:
    """Signal inputs, votes, score, position and realised target for each decision date.

    `curve_override` (date-indexed M1-M2 values) fills the curve where EIA
    futures are missing - used only for the live reading with Yahoo's curve.
    """
    params = params or CrudeSignalParams()
    if dates is None:
        last = daily_feat["wti_spot"].dropna().index.max()
        dates = thursdays(weekly_feat.index.min() + pd.Timedelta(days=7), last)

    wk = _asof(dates, prepare_weekly_inputs(weekly_feat, params), "available_date",
               ["inventory_z", "cushing_z", "crude_stocks_surprise", "cushing_stocks_vs_5y_pct"])
    out = pd.DataFrame(index=pd.DatetimeIndex(dates, name="decision_date"))
    out["date"] = out.index
    out["week_used"] = wk["_row_date"]
    out["inventory_z"] = wk["inventory_z"]
    out["cushing_z"] = wk["cushing_z"]

    curve = daily_feat["m1_m2"]
    if curve_override is not None and len(curve_override):
        curve = curve.combine_first(curve_override)
    stale = pd.Timedelta(days=5)  # never carry a price more than a few days
    out["m1_m2"] = _asof_series(dates, curve, stale).to_numpy()

    if cot_feat is not None and not cot_feat.empty:
        pos = _asof(dates, cot_feat, "available_date", ["mm_net_pctile", "mm_net_pct_oi"])
        out["cot_used"] = pos["_row_date"]
        out["mm_pctile"] = pos["mm_net_pctile"]
    else:
        out["cot_used"] = pd.NaT
        out["mm_pctile"] = np.nan

    out["comp_inventory"] = _vote(out["inventory_z"], params.surprise_z, -params.surprise_z, sign=-1)
    out["comp_cushing"] = _vote(out["cushing_z"], params.cushing_z, -params.cushing_z, sign=-1)
    out["comp_curve"] = _vote(out["m1_m2"], params.curve_threshold, -params.curve_threshold) if params.use_curve else np.nan
    crowd = params.crowding_pctile
    out["comp_positioning"] = _vote(out["mm_pctile"], crowd, 1 - crowd, sign=-1) if params.use_positioning else np.nan
    active = ["comp_inventory", "comp_cushing"] + (["comp_curve"] if params.use_curve else []) + \
             (["comp_positioning"] if params.use_positioning else [])
    out["score"] = out[active].sum(axis=1, min_count=len(active))
    out["position"] = np.sign(out["score"]).where(out["score"].abs() >= params.entry_threshold, 0).fillna(0)

    # ---- realised target: next decision date's settlement vs this one --------------------
    nxt = out.index.to_series().shift(-1)
    if params.target == "futures":
        ref = _asof_series(dates, daily_feat["cl1"], stale)
        # Back-adjusted continuous front month: cumulative roll-adjusted changes.
        level = daily_feat["cl1_roll_adj_change"].fillna(0).cumsum().where(daily_feat["cl1"].notna())
        lv = _asof_series(dates, level, stale)
        target = lv.shift(-1) - lv
    else:
        ref = _asof_series(dates, daily_feat["wti_spot"], stale)
        target = ref.shift(-1) - ref
    target[nxt.isna().to_numpy()] = np.nan
    ref, target = ref.to_numpy(), target.to_numpy()
    out["reference_price"] = ref
    out["target"] = target
    out["delivered_price"] = out["reference_price"] + out["target"]
    out["hours"] = float(BARRELS_PER_CONTRACT)  # units per position, reused by the generic backtester
    return out


def component_diagnostics(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = signals["target"].notna() & (signals["target"] != 0)
    direction = np.sign(signals["target"])
    for col, label in {**COMPONENTS, "score": "Combined score"}.items():
        if col not in signals or signals[col].isna().all():
            continue
        vote = np.sign(signals[col])
        active = valid & vote.notna() & (vote != 0)
        n = int(active.sum())
        rows.append({
            "component": label,
            "active weeks": n,
            "hit rate": float((vote[active] == direction[active]).mean()) if n else np.nan,
            "avg. signed move ($/bbl)": float((vote[active] * signals.loc[active, "target"]).mean()) if n else np.nan,
        })
    return pd.DataFrame(rows).set_index("component")
