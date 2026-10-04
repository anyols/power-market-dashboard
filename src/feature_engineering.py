"""Feature engineering: turning raw fundamentals into trader-relevant variables.

Two kinds of features live here:

* *Descriptive* features (spike flags over a chosen period, correlations,
  full-period percentiles) are used to explain what happened. They may use
  any data inside the analysed window.
* *Trailing* features (rolling means, z-scores, volatility regimes) only ever
  look backwards: the value at time t uses data up to t-1. These are safe
  to feed into signals, although the signal engine adds its own, stricter
  information-set lags on top (see signal_engine.py).

Sign conventions
----------------
Forecast errors are always ``actual - forecast``:

* ``load_error > 0``      demand came in above forecast  -> system shorter
* ``wind_error < 0``      wind under-delivered            -> system shorter
* ``residual_demand_error = load_error - renewable_error``
  is the net surprise the rest of the system had to absorb. Positive means
  the residual (mostly thermal) fleet had to deliver more than planned,
  which is bullish for intraday/imbalance prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PEAK_END_HOUR, PEAK_START_HOUR, SPIKE_PERCENTILE
from src.utils import local_dates, same_hour_lag, to_market_tz

HOURS_PER_DAY = 24


# ---------------------------------------------------------------------------
# Generic rolling statistics (trailing only)
# ---------------------------------------------------------------------------
def rolling_mean(s: pd.Series, window: int, min_periods: int | None = None, exclude_current: bool = False) -> pd.Series:
    """Trailing mean over `window` observations.

    With ``exclude_current=True`` the value at t uses observations t-window..t-1,
    i.e. what was known *before* t.
    """
    m = s.rolling(window, min_periods=min_periods or max(2, window // 4)).mean()
    return m.shift(1) if exclude_current else m


def rolling_volatility(s: pd.Series, window: int, min_periods: int | None = None, on_changes: bool = True) -> pd.Series:
    """Trailing standard deviation of hour-on-hour price changes (EUR/MWh).

    Power prices can be zero or negative, so percentage/log returns are
    meaningless; volatility is measured in absolute EUR/MWh changes.
    """
    x = s.diff() if on_changes else s
    return x.rolling(window, min_periods=min_periods or max(2, window // 4)).std()


def zscore(s: pd.Series, window: int, min_periods: int | None = None, exclude_current: bool = True) -> pd.Series:
    """Rolling z-score of `s` against its own trailing window.

    By default the mean/std are computed on t-window..t-1 so the current
    observation never contaminates its own benchmark (no look-ahead).
    """
    mp = min_periods or max(2, window // 4)
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std()
    if exclude_current:
        mean, std = mean.shift(1), std.shift(1)
    return (s - mean) / std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------
def add_residual_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Residual demand = load - wind - solar.

    Market intuition: wind and solar have ~zero marginal cost and are
    dispatched first, so the *residual* must be met by dispatchable plants
    (gas, coal, lignite, hydro, nuclear, imports). Where residual demand lands
    on that merit order largely sets the price: high residual demand pushes
    expensive gas peakers onto the margin (bullish); low or negative residual
    demand leaves inflexible plants and subsidised renewables competing to
    stay on, which is when negative prices appear (bearish).
    """
    out = df.copy()
    out["renewable_actual"] = out["wind_actual"] + out["solar_actual"]
    out["renewable_forecast"] = out["wind_forecast"] + out["solar_forecast"]
    out["residual_demand"] = out["load_actual"] - out["renewable_actual"]
    out["residual_demand_forecast"] = out["load_forecast"] - out["renewable_forecast"]
    out["renewable_share"] = out["renewable_actual"] / out["load_actual"]
    return out


def add_forecast_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Day-ahead forecast errors (actual - forecast), all in MW.

    The day-ahead auction clears on forecasts. Whatever the forecast missed
    has to be re-balanced in intraday markets or, ultimately, through
    balancing energy - so these errors are the main driver of intraday and
    imbalance price moves around the day-ahead level.
    """
    out = df.copy()
    out["load_error"] = out["load_actual"] - out["load_forecast"]
    out["wind_error"] = out["wind_actual"] - out["wind_forecast"]
    out["solar_error"] = out["solar_actual"] - out["solar_forecast"]
    out["renewable_error"] = out["wind_error"] + out["solar_error"]
    out["residual_demand_error"] = out["load_error"] - out["renewable_error"]
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features in local market time.

    Peak = Mon-Fri 08:00-20:00 (EEX-style); everything else is off-peak.
    Weekends and the overnight trough carry lower demand and, with solar,
    the midday weekend hours are where negative prices cluster.
    """
    out = df.copy()
    idx = to_market_tz(out.index)
    out["date"] = local_dates(idx)
    out["hour"] = idx.hour
    out["weekday"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = out["weekday"] >= 5
    out["is_peak"] = (~out["is_weekend"]) & (out["hour"] >= PEAK_START_HOUR) & (out["hour"] < PEAK_END_HOUR)
    return out


# ---------------------------------------------------------------------------
# Price features
# ---------------------------------------------------------------------------
def add_price_features(df: pd.DataFrame, momentum_fast_h: int = 24, momentum_slow_h: int = 168) -> pd.DataFrame:
    """Price changes, momentum and volatility (all trailing)."""
    out = df.copy()
    p = out["price"]
    out["price_change_1h"] = p.diff()
    # Change vs the same delivery hour yesterday: removes the daily shape,
    # which otherwise dominates hour-on-hour changes.
    out["price_lag_24h"] = same_hour_lag(p, 1)
    out["price_change_24h"] = p - out["price_lag_24h"]
    out["price_momentum"] = rolling_mean(p, momentum_fast_h) - rolling_mean(p, momentum_slow_h)
    out["price_vol_24h"] = rolling_volatility(p, 24)
    out["price_vol_7d"] = rolling_volatility(p, 168)
    out["is_negative"] = p < 0
    return out


def flag_spikes(price: pd.Series, percentile: float = SPIKE_PERCENTILE, threshold: float | None = None) -> tuple[pd.Series, float]:
    """Spike flag for the analysed window.

    Descriptive definition: price above the given percentile *of the window
    being analysed*, or above an absolute threshold if one is supplied.
    This uses the whole window, so never feed it into a signal.
    """
    level = float(threshold) if threshold is not None else float(price.quantile(percentile))
    return (price > level).rename("is_spike"), level


def price_spread(price_a: pd.Series, price_b: pd.Series) -> pd.Series:
    """Spread b - a (EUR/MWh). Positive: market b is more expensive."""
    return (price_b - price_a).rename("spread")


def volatility_regime(vol: pd.Series, min_history: int = 24 * 14) -> pd.Series:
    """Classify volatility as low / normal / high against its *own past*.

    Thresholds are the expanding 33rd/67th percentiles of the volatility
    series up to t-1, so the regime at t never uses future information.
    """
    lo = vol.expanding(min_periods=min_history).quantile(1 / 3).shift(1)
    hi = vol.expanding(min_periods=min_history).quantile(2 / 3).shift(1)
    regime = pd.Series("n/a", index=vol.index, dtype=object)
    regime[vol <= lo] = "low"
    regime[(vol > lo) & (vol <= hi)] = "normal"
    regime[vol > hi] = "high"
    return regime.rename("vol_regime")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def add_trailing_zscores(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """z-scores of residual demand and forecast errors vs the trailing window."""
    out = df.copy()
    w = window_days * HOURS_PER_DAY
    for col in ["residual_demand", "load_error", "wind_error", "solar_error", "renewable_error", "residual_demand_error"]:
        out[f"{col}_z"] = zscore(out[col], w)
    out["residual_demand_vs_30d"] = out["residual_demand"] - rolling_mean(out["residual_demand"], w, exclude_current=True)
    return out


def build_features(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Full feature set used by the dashboard pages."""
    out = add_time_features(df)
    out = add_residual_demand(out)
    out = add_forecast_errors(out)
    out = add_price_features(out)
    out = add_trailing_zscores(out, window_days)
    out["vol_regime"] = volatility_regime(out["price_vol_7d"])
    return out


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-delivery-day statistics (baseload, peak, off-peak, extremes)."""
    g = df.groupby("date")
    out = pd.DataFrame(
        {
            "baseload": g["price"].mean(),
            "peak": df[df["is_peak"]].groupby("date")["price"].mean(),
            "offpeak": df[~df["is_peak"]].groupby("date")["price"].mean(),
            "min": g["price"].min(),
            "max": g["price"].max(),
            "intraday_range": g["price"].max() - g["price"].min(),
            "negative_hours": g["is_negative"].sum(),
            "residual_demand_gw": g["residual_demand"].mean() / 1000,
            "wind_gw": g["wind_actual"].mean() / 1000,
            "solar_gw": g["solar_actual"].mean() / 1000,
            "load_gw": g["load_actual"].mean() / 1000,
            "hours": g["price"].size(),
        }
    )
    out.index.name = "date"
    return out


def correlation_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "price": "Price",
        "residual_demand": "Residual demand",
        "load_actual": "Load",
        "wind_actual": "Wind",
        "solar_actual": "Solar",
        "load_error": "Load error",
        "renewable_error": "RES error",
        "residual_demand_error": "Resid. demand error",
    }
    return df[list(cols)].rename(columns=cols).corr()


# ---------------------------------------------------------------------------
# Cross-border flows
# ---------------------------------------------------------------------------
def net_flows_by_border(flows: pd.DataFrame, zone: str) -> pd.DataFrame:
    """Hourly net physical flow *into* `zone` from each neighbour (MW).

    ENTSO-E publishes each direction separately as a non-negative number;
    net = imports - exports. Positive values mean the neighbour is supplying
    the zone, which relieves local tightness; negative values mean the zone
    exports, adding to its own generation requirement.
    """
    if flows is None or flows.empty:
        return pd.DataFrame()
    imp = flows[flows["to_zone"] == zone].pivot_table(index="timestamp", columns="from_zone", values="flow_mw", aggfunc="sum")
    exp = flows[flows["from_zone"] == zone].pivot_table(index="timestamp", columns="to_zone", values="flow_mw", aggfunc="sum")
    net = imp.sub(exp, fill_value=0).sort_index()
    net.columns.name = None
    return net


def border_summary(net: pd.DataFrame, zone_prices: pd.DataFrame, zone: str, capacities: dict[str, float | None]) -> pd.DataFrame:
    """Per-border statistics linking flows and price spreads.

    * convergence: share of hours where the two prices are within 1 EUR/MWh
      (market coupling cleared without a binding constraint).
    * flow with spread: share of hours where power flowed from the cheaper
      to the more expensive zone - the economically intuitive direction.
      Physical flows can run against the spread because of loop flows and
      flow-based coupling.
    * near capacity: share of hours with |flow| above 90% of the indicative
      capacity in that direction - a proxy for congestion.
    """
    rows = []
    for nb in net.columns:
        if nb not in zone_prices or zone not in zone_prices:
            continue
        spread = (zone_prices[nb] - zone_prices[zone]).reindex(net.index)  # > 0: neighbour pricier
        inflow = net[nb]
        valid = spread.notna() & inflow.notna()
        s, f = spread[valid], inflow[valid]
        cap_in = capacities.get(f"{nb}->{zone}")
        cap_out = capacities.get(f"{zone}->{nb}")
        near = pd.Series(False, index=f.index)
        if cap_in:
            near |= f > 0.9 * cap_in
        if cap_out:
            near |= -f > 0.9 * cap_out
        moving = s.abs() > 1
        rows.append(
            {
                "neighbour": nb,
                "avg net import (MW)": f.mean(),
                "avg spread nb - zone (EUR/MWh)": s.mean(),
                "price convergence": (s.abs() <= 1).mean(),
                "flow with spread": (np.sign(f[moving]) == -np.sign(s[moving])).mean() if moving.any() else np.nan,
                "near capacity": near.mean() if (cap_in or cap_out) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("neighbour") if rows else pd.DataFrame()
