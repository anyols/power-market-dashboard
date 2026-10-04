"""Generate the SYNTHETIC crude-oil sample dataset in data/sample/crude/.

    python scripts/generate_sample_crude.py

Same schema as the real store (data/store/crude/), so the dashboard and tests
run without API keys. NOT real market data - see data/sample/README.md.

DGP in brief
------------
* Weekly balances: trend + seasonal pattern + autocorrelated deviations
  (crude builds in spring and draws in summer, gasoline draws into the driving
  season, distillates draw in winter, refinery maintenance dips in spring and
  autumn). A synthetic demand shock is included to stress the analytics.
* Spot price: log mean-reverting around a slowly moving anchor, with a
  contemporaneous jump on each report day proportional to the crude-stock
  surprise vs the seasonal norm. The surprise is priced *immediately*, so it
  should carry no information about the following week - by design.
* Futures curve: F_k = S * exp(-y * tau_k), where the convenience yield y
  rises when Cushing and US stocks are low (theory of storage) and tau_k uses
  the real CL expiry calendar. Holding the front contract therefore earns a
  roll yield in backwardation - the one genuine "edge" built into the data.
* Positioning: managed-money net length follows past returns (trend
  followers) plus noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SAMPLE_DIR  # noqa: E402
from src.crude.contracts import BDAY, front_contract, last_trade_schedule  # noqa: E402

SEED = 20100101
START, END = "2010-01-01", "2025-12-31"
OUT = SAMPLE_DIR / "crude"
rng = np.random.default_rng(SEED)


def ar1(n: int, phi: float, sigma: float) -> np.ndarray:
    eps = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def path(index: pd.DatetimeIndex, anchors: dict[float, float]) -> np.ndarray:
    """Piecewise-linear path through {decimal_year: value} anchors."""
    t = index.year + (index.dayofyear - 1) / 365.25
    xs, ys = zip(*sorted(anchors.items()))
    return np.interp(t, xs, ys)


# ---------------------------------------------------------------------------
# Weekly balances (thousand barrels; flows in kb/d)
# ---------------------------------------------------------------------------
weeks = pd.date_range(START, END, freq="W-FRI")
W = len(weeks)
woy = np.minimum(weeks.isocalendar().week.to_numpy(), 52)


def seasonal(amplitude: float, peak_week: int) -> np.ndarray:
    return amplitude * np.cos(2 * np.pi * (woy - peak_week) / 52)


shock = np.zeros(W)  # synthetic demand shock: sharp drop, slow recovery
i0 = np.searchsorted(weeks, pd.Timestamp("2020-03-20"))
shock[i0:] = np.exp(-np.arange(W - i0) / 20.0)

crude = (path(weeks, {2010: 350e3, 2014: 370e3, 2016.3: 525e3, 2018.5: 420e3, 2019.5: 450e3, 2022: 420e3, 2025: 430e3})
         + seasonal(16e3, 19) + ar1(W, 0.95, 9e3) + 90e3 * shock)
cushing = np.clip(path(weeks, {2010: 35e3, 2014: 25e3, 2016.3: 65e3, 2018.5: 28e3, 2019.5: 45e3, 2022: 27e3, 2025: 24e3})
                  + seasonal(2e3, 18) + ar1(W, 0.92, 3.5e3) + 25e3 * shock, 18e3, 78e3)
spr = path(weeks, {2010: 727e3, 2017: 688e3, 2019: 645e3, 2021.9: 600e3, 2022.5: 470e3, 2023.5: 347e3, 2025: 400e3})
gasoline = 228e3 + seasonal(12e3, 6) + ar1(W, 0.9, 4e3) + 15e3 * shock
distillate = 135e3 + seasonal(12e3, 42) + ar1(W, 0.9, 4e3)
util = np.clip(90 + seasonal(3.0, 29) - 3.0 * ((woy >= 8) & (woy <= 14)) - 2.0 * ((woy >= 39) & (woy <= 44))
               + ar1(W, 0.7, 1.2) - 18 * shock, 65, 97)
capacity = path(weeks, {2010: 17.6e3, 2025: 18.4e3})
production = path(weeks, {2010: 5.5e3, 2012: 6.5e3, 2014: 8.7e3, 2015.5: 9.6e3, 2016.7: 8.5e3, 2018: 10.0e3,
                          2019.9: 12.9e3, 2020.4: 10.6e3, 2021: 11.0e3, 2023: 12.9e3, 2025: 13.5e3}) + rng.normal(0, 80, W)
net_imports = path(weeks, {2010: 9.0e3, 2014: 7.0e3, 2018: 5.5e3, 2020: 2.8e3, 2025: 2.3e3}) + rng.normal(0, 700, W)
product_supplied = 19.6e3 + seasonal(700, 30) + rng.normal(0, 600, W) - 4.5e3 * shock

weekly = pd.DataFrame(
    {
        "crude_stocks": crude, "cushing_stocks": cushing, "spr_stocks": spr, "gasoline_stocks": gasoline,
        "distillate_stocks": distillate, "refinery_utilization": util, "crude_production": production,
        "crude_net_imports": net_imports, "refinery_crude_input": util / 100 * capacity,
        "product_supplied": product_supplied,
    },
    index=pd.DatetimeIndex(weeks, name="date"),
)

# Crude-stock surprise vs the seasonal norm of the weekly change (used for report-day price jumps)
chg = weekly["crude_stocks"].diff()
norm = chg.groupby(woy).transform("mean")
surprise_z = ((chg - norm) / (chg - norm).std()).fillna(0)

# ---------------------------------------------------------------------------
# Daily prices
# ---------------------------------------------------------------------------
days = pd.date_range(START, END, freq=BDAY)
D = len(days)
anchor = path(days, {2010: 80, 2011.3: 105, 2014.5: 102, 2015.2: 50, 2016.1: 32, 2018.8: 74, 2019.2: 56,
                     2020.3: 22, 2021: 55, 2022.4: 112, 2023: 78, 2024.3: 82, 2025: 66, 2026: 64})
# Deviation of log price from the anchor: slow mean reversion. Report-day
# surprises enter as ordinary innovations (a draw bigger than normal lifts the
# price that same day), so they are no more predictable afterwards than any other shock.
release = pd.Series(surprise_z.to_numpy(), index=weeks + pd.Timedelta(days=5))  # Wednesday after week end
jumps = -0.008 * release.reindex(days).fillna(0).to_numpy()
eps = rng.normal(0, 0.021, D)
x = np.zeros(D)
for t in range(1, D):
    x[t] = 0.995 * x[t - 1] + eps[t] + jumps[t]
spot = np.exp(np.log(anchor) + x)

daily_w = weekly.reindex(days, method="ffill").bfill()
cush_dev = (daily_w["cushing_stocks"] - 35e3) / 35e3
crude_dev = (daily_w["crude_stocks"] - 440e3) / 440e3
conv_yield = -0.01 - 0.45 * cush_dev.to_numpy() - 0.25 * crude_dev.to_numpy() + ar1(D, 0.995, 0.03)

schedule = last_trade_schedule(days.min() - pd.Timedelta(days=40), days.max() + pd.Timedelta(days=200))
front = front_contract(days, schedule)
futures = {}
for k in range(4):
    expiry = pd.DatetimeIndex(schedule.reindex(front + k).to_numpy())
    tau = (expiry - days).days.to_numpy() / 365.0 + 1 / 365.0
    futures[f"cl{k + 1}"] = spot * np.exp(-conv_yield * tau)

brent_spread = path(days, {2010: 1.0, 2011.5: 14.0, 2013: 7.0, 2015: 3.0, 2020: 3.0, 2025: 4.0}) + ar1(D, 0.99, 1.0)
daily = pd.DataFrame({"wti_spot": spot, "brent_spot": spot + brent_spread, **futures}, index=pd.DatetimeIndex(days, name="date"))

# ---------------------------------------------------------------------------
# Positioning (CFTC disaggregated, managed money), report date = Tuesday
# ---------------------------------------------------------------------------
report_dates = weeks - pd.Timedelta(days=3)
weekly_ret = np.log(daily["cl1"]).reindex(report_dates, method="ffill").diff(6).fillna(0).to_numpy()
oi = path(report_dates, {2010: 1.4e6, 2018: 2.6e6, 2020: 2.2e6, 2025: 1.9e6}) * (1 + ar1(W, 0.9, 0.03))
net_pct = np.clip(0.08 + ar1(W, 0.92, 0.04) + 0.35 * weekly_ret, -0.05, 0.25)
mm_long = oi * (0.08 + np.maximum(net_pct, 0) + 0.02 * np.abs(rng.normal(size=W)))
mm_short = np.maximum(mm_long - net_pct * oi, 0.02 * oi)
cot = pd.DataFrame(
    {
        "open_interest": oi, "mm_long": mm_long, "mm_short": mm_short, "mm_spread": 0.15 * oi,
        "producer_long": 0.30 * oi, "producer_short": 0.18 * oi, "swap_long": 0.06 * oi, "swap_short": 0.30 * oi,
    },
    index=pd.DatetimeIndex(report_dates, name="report_date"),
).round(0)

yahoo = pd.DataFrame({"wti_front": daily["cl1"], "brent_front": daily["brent_spot"] - 0.8}, index=daily.index)

OUT.mkdir(parents=True, exist_ok=True)
daily.round(2).to_csv(OUT / "eia_daily.csv")
weekly.round(1).to_csv(OUT / "eia_weekly.csv")
cot.astype("int64").to_csv(OUT / "cot_wti.csv")
yahoo.round(2).to_csv(OUT / "yahoo_daily.csv")
(OUT / "metadata.json").write_text(json.dumps({
    "notice": "SYNTHETIC SAMPLE DATA - generated by scripts/generate_sample_crude.py. NOT real market data.",
    "seed": SEED, "period": [START, END],
    "files": ["eia_daily.csv", "eia_weekly.csv", "cot_wti.csv", "yahoo_daily.csv"],
    "built_in_relationships": [
        "convenience yield (backwardation) rises when Cushing/US stocks are low -> roll yield for front-month longs",
        "stock surprises move price on the report day only (no next-week predictability)",
        "managed-money positioning follows past returns",
    ],
}, indent=2))
print(f"Synthetic crude sample written to {OUT}: daily {daily.shape}, weekly {weekly.shape}, cot {cot.shape}")
print(daily[["wti_spot", "cl1", "cl2"]].resample("YE").mean().round(1).T.to_string())
print("M1-M2 spread percentiles:", np.percentile(daily["cl1"] - daily["cl2"], [5, 50, 95]).round(2))
