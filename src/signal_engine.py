"""Transparent, rule-based fundamental signals with an explicit information set.

This is a research tool, not a trading strategy. The point is to show *how*
fundamentals map to directional price pressure, and to test that mapping
honestly. No parameters are fitted to the data.

What exactly are we predicting?
-------------------------------
All 24 day-ahead prices of delivery day D are set at once in a single
auction on D-1. Predicting "next hour's day-ahead price change" from
information at hour t is therefore meaningless: by then that price has been
public for a day. Instead we predict, *before the D-1 auction*:

* ``hourly`` horizon: price(D, h) - price(D-1, h)   (same delivery hour)
* ``daily``  horizon: baseload(D) - baseload(D-1)

The reference price (D-1) is public at decision time, so the target is a
genuine forecast of an unknown quantity: "will tomorrow clear above or below
today?". The PnL proxy treats the reference price as the entry level - a
simplification, since there is no product that trades at exactly that price.

Information set at decision time (D-1, ~11:00 CET, before the 12:00 gate)
------------------------------------------------------------------------
See INFORMATION_SET below. The tests in tests/test_no_lookahead.py perturb
everything outside this set and assert the signal does not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SignalParams
from src.utils import daily_to_hourly, full_daily_index, same_hour_lag

INFORMATION_SET = [
    {
        "data": "Day-ahead prices up to and including D-1",
        "known": "Yes",
        "why": "D-1 prices were published after the D-2 auction (~12:45 CET).",
        "used for": "Reference price, momentum",
    },
    {
        "data": "Day-ahead prices for D",
        "known": "No",
        "why": "Set by the auction we are positioning for.",
        "used for": "Target / PnL only",
    },
    {
        "data": "Actual load, wind, solar up to end of D-2",
        "known": "Yes",
        "why": "Conservative cut-off; D-1 morning actuals are ignored to allow for publication lags.",
        "used for": "Load and renewable surprise components",
    },
    {
        "data": "Actual load, wind, solar on D-1 and D",
        "known": "No",
        "why": "Not yet (fully) published at decision time.",
        "used for": "Nothing",
    },
    {
        "data": "Day-ahead load / wind / solar forecasts for D",
        "known": "Assumed yes",
        "why": "Assumption: forecasts of this quality are available before gate closure "
        "(TSO publication times vary; traders typically use vendor forecasts).",
        "used for": "Residual-demand component",
    },
]

COMPONENT_COLUMNS = ["comp_rd", "comp_load", "comp_res"]
COMPONENT_LABELS = {
    "comp_rd": "Residual demand",
    "comp_load": "Load surprise",
    "comp_res": "Renewable surprise",
}


def _min_periods(window: int) -> int:
    return max(5, window // 3)


def _daily_mean(s: pd.Series, dates: pd.Series) -> pd.Series:
    daily = s.groupby(dates.to_numpy()).mean()
    daily.index = pd.DatetimeIndex(daily.index)
    return daily.reindex(full_daily_index(daily.index))


def known_daily_surprise(error: pd.Series, dates: pd.Series, window_days: int, lag_days: int = 2) -> pd.Series:
    """Standardised daily forecast surprise that is *known* on decision day.

    The value for day D is the mean forecast error on day D-`lag_days`,
    divided by the trailing std of daily mean errors up to that same day.
    With the default lag of 2 the newest surprise used is from D-2, matching
    the conservative information set.
    """
    daily = _daily_mean(error, dates)
    sigma = daily.rolling(window_days, min_periods=_min_periods(window_days)).std()
    z = daily / sigma.replace(0, np.nan)
    return z.shift(lag_days)


def residual_demand_z_hourly(feat: pd.DataFrame, params: SignalParams) -> pd.Series:
    """Standardised forecast residual demand for each delivery hour of D.

    ``change`` mode: forecast residual demand for (D, h) minus the forecast
    for (D-1, h) - i.e. how much tighter tomorrow looks than the day the
    reference price was set - scaled by the trailing std of such changes.
    ``level`` mode: forecast residual demand vs its trailing mean / std.
    Trailing statistics are taken at the last hour of D-1 and applied to all
    hours of D, so nothing from D's actual outcome leaks in.
    """
    rd_fc = feat["residual_demand_forecast"]
    dates = feat["date"]
    window_h = params.zscore_window_days * 24
    mp = _min_periods(window_h)

    def end_of_previous_day(stat: pd.Series) -> pd.Series:
        daily = stat.groupby(dates.to_numpy()).last()
        daily.index = pd.DatetimeIndex(daily.index)
        daily = daily.reindex(full_daily_index(daily.index)).shift(1)
        return daily_to_hourly(daily, feat.index)

    if params.rd_mode == "change":
        delta = rd_fc - same_hour_lag(rd_fc, 1)
        sigma = end_of_previous_day(delta.rolling(window_h, min_periods=mp).std())
        return delta / sigma.replace(0, np.nan)
    mean = end_of_previous_day(rd_fc.rolling(window_h, min_periods=mp).mean())
    sigma = end_of_previous_day(rd_fc.rolling(window_h, min_periods=mp).std())
    return (rd_fc - mean) / sigma.replace(0, np.nan)


def residual_demand_z_daily(feat: pd.DataFrame, params: SignalParams) -> pd.Series:
    rd = _daily_mean(feat["residual_demand_forecast"], feat["date"])
    w, mp = params.zscore_window_days, _min_periods(params.zscore_window_days)
    if params.rd_mode == "change":
        delta = rd - rd.shift(1)
        sigma = delta.rolling(w, min_periods=mp).std().shift(1)
        return delta / sigma.replace(0, np.nan)
    mean = rd.rolling(w, min_periods=mp).mean().shift(1)
    sigma = rd.rolling(w, min_periods=mp).std().shift(1)
    return (rd - mean) / sigma.replace(0, np.nan)


def known_momentum(feat: pd.DataFrame) -> pd.Series:
    """Baseload momentum known at decision time: MA3 - MA7 of daily baseload up to D-1."""
    base = _daily_mean(feat["price"], feat["date"])
    return (base.rolling(3, min_periods=2).mean() - base.rolling(7, min_periods=4).mean()).shift(1)


def _ternary(x: pd.Series, upper: float, lower: float, up_value: int = 1) -> pd.Series:
    out = pd.Series(np.select([x > upper, x < lower], [up_value, -up_value], 0), index=x.index, dtype=float)
    return out.where(x.notna())


def score_components(rd_z: pd.Series, load_z: pd.Series, res_z: pd.Series, params: SignalParams) -> pd.DataFrame:
    """Map standardised inputs to {-1, 0, +1} votes and a -3..+3 score.

    +1 residual demand z  >  threshold   (system tighter than reference)
    +1 load surprise      >  threshold   (demand recently above forecast)
    +1 renewable surprise < -threshold   (wind/solar recently under-delivered)
    and the mirror images for -1. The score is NaN (no trade) while any
    input is still in its warm-up period.
    """
    t, s = params.rd_z_threshold, params.surprise_z_threshold
    comps = pd.DataFrame(
        {
            "comp_rd": _ternary(rd_z, t, -t),
            "comp_load": _ternary(load_z, s, -s),
            "comp_res": _ternary(res_z, s, -s, up_value=-1),
        }
    )
    comps["score"] = comps[COMPONENT_COLUMNS].sum(axis=1, min_count=3)
    return comps


def positions_from_score(score: pd.Series, momentum: pd.Series, params: SignalParams) -> pd.Series:
    pos = np.sign(score).where(score.abs() >= params.entry_threshold, 0).fillna(0)
    if params.use_momentum_filter:
        agrees = np.sign(momentum) == pos
        pos = pos.where(agrees, 0)
    return pos.astype(float)


def build_signal_frame(feat: pd.DataFrame, params: SignalParams | None = None) -> pd.DataFrame:
    """Signals, positions and realised targets for every delivery hour/day.

    `feat` must contain the columns produced by feature_engineering.build_features.
    Rows are indexed by delivery hour (hourly horizon) or delivery date (daily).
    """
    params = params or SignalParams()
    dates = feat["date"]
    w = params.zscore_window_days
    load_z_d = known_daily_surprise(feat["load_error"], dates, w)
    res_z_d = known_daily_surprise(feat["renewable_error"], dates, w)
    mom_d = known_momentum(feat)

    if params.horizon == "hourly":
        out = pd.DataFrame(index=feat.index)
        out["date"] = dates
        out["hour"] = feat["hour"]
        out["rd_z"] = residual_demand_z_hourly(feat, params)
        out["load_surprise_z"] = daily_to_hourly(load_z_d, feat.index)
        out["res_surprise_z"] = daily_to_hourly(res_z_d, feat.index)
        out["momentum"] = daily_to_hourly(mom_d, feat.index)
        out["reference_price"] = same_hour_lag(feat["price"], 1)
        out["delivered_price"] = feat["price"]
        out["hours"] = 1.0
    else:
        base = _daily_mean(feat["price"], dates)
        n_hours = feat.groupby(dates.to_numpy())["price"].size()
        n_hours.index = pd.DatetimeIndex(n_hours.index)
        out = pd.DataFrame(index=base.index)
        out.index.name = "delivery_day"
        out["date"] = out.index
        out["rd_z"] = residual_demand_z_daily(feat, params)
        out["load_surprise_z"] = load_z_d
        out["res_surprise_z"] = res_z_d
        out["momentum"] = mom_d
        out["reference_price"] = base.shift(1)
        out["delivered_price"] = base
        out["hours"] = n_hours.reindex(out.index).astype(float)

    comps = score_components(out["rd_z"], out["load_surprise_z"], out["res_surprise_z"], params)
    out = out.join(comps)
    out["position"] = positions_from_score(out["score"], out["momentum"], params)
    out["target"] = out["delivered_price"] - out["reference_price"]
    return out


def component_diagnostics(signals: pd.DataFrame) -> pd.DataFrame:
    """How often each vote pointed the right way, taken on its own."""
    rows = []
    valid = signals["target"].notna() & (signals["target"] != 0)
    direction = np.sign(signals["target"])
    for col in COMPONENT_COLUMNS + ["score"]:
        vote = np.sign(signals[col])
        active = valid & vote.notna() & (vote != 0)
        n = int(active.sum())
        hit = float((vote[active] == direction[active]).mean()) if n else np.nan
        avg_move = float((vote[active] * signals.loc[active, "target"]).mean()) if n else np.nan
        rows.append(
            {
                "component": COMPONENT_LABELS.get(col, "Combined score"),
                "active observations": n,
                "hit rate": hit,
                "avg. signed move (EUR/MWh)": avg_move,
            }
        )
    return pd.DataFrame(rows).set_index("component")
