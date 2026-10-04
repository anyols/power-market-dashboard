"""Generate the SYNTHETIC sample dataset in data/sample/.

    python scripts/generate_sample_data.py

The output mimics the *structure* of ENTSO-E Transparency Platform data and
the broad *behaviour* of European power markets, so the dashboard can run
without an API token. It is NOT real market data and must never be presented
as such. Every file is listed in data/sample/metadata.json with that notice.

Data-generating process (DGP), in one paragraph
------------------------------------------------
Weather drives everything: a temperature process drives load (heating /
cooling), a latent wind process drives wind output through a power-curve-like
logistic transform, and a cloud process scales a clear-sky solar profile.
TSO-style day-ahead forecasts are produced first; actuals = forecast + an
autocorrelated forecast error (with a persistent daily bias component and an
hourly component). Day-ahead prices are cleared on *forecast* fundamentals
(the auction happens the day before delivery) using a per-zone merit-order
curve built from fuel/carbon costs, plus a simplified market-coupling loop in
which interconnector flows move power from cheap to expensive zones until
prices converge or the line congests. To mimic the fact that market
participants' own forecasts are better than the published TSO forecast, the
price-setting residual demand includes ~50% of each day's systematic
forecast bias. That assumption is the *only* reason lagged forecast errors
carry any information about prices in this dataset - it is documented so
nobody mistakes a property of the DGP for a market discovery.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    BORDERS,
    MARKET_TZ,
    PRICE_CAP,
    PRICE_FLOOR,
    SAMPLE_DIR,
    ZONES,
)

SEED = 20240101
START = "2024-01-01 00:00"
END = "2024-07-01 00:00"  # exclusive
GENERATOR_VERSION = "1.0"
KAPPA_BIAS = 0.5  # share of daily forecast bias anticipated by the market

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
idx = pd.date_range(START, END, freq="h", tz=MARKET_TZ, inclusive="left")
T = len(idx)
hour = idx.hour.to_numpy()
weekday = idx.dayofweek.to_numpy()
doy = idx.dayofyear.to_numpy()
local_dates = idx.tz_localize(None).normalize()
day_codes, day_index = pd.factorize(local_dates)
D = len(day_index)
day_doy = day_index.dayofyear.to_numpy()
utc_hour = (idx.tz_convert("UTC").hour.to_numpy() + 0.5)  # mid-interval

# Seasonal weight: 0 in mid-January, 1 in mid-July
season_w = (1 - np.cos(2 * np.pi * (doy - 15) / 365)) / 2
season_w_day = (1 - np.cos(2 * np.pi * (day_doy - 15) / 365)) / 2

HOLIDAYS = {
    "DE_LU": ["2024-01-01", "2024-03-29", "2024-04-01", "2024-05-01", "2024-05-09", "2024-05-20"],
    "FR": ["2024-01-01", "2024-04-01", "2024-05-01", "2024-05-08", "2024-05-09", "2024-05-20"],
    "NL": ["2024-01-01", "2024-04-01", "2024-05-09", "2024-05-20"],
    "BE": ["2024-01-01", "2024-04-01", "2024-05-01", "2024-05-09", "2024-05-20"],
    "ES": ["2024-01-01", "2024-01-06", "2024-03-29", "2024-05-01"],
    "IT_NORD": ["2024-01-01", "2024-01-06", "2024-04-01", "2024-04-25", "2024-05-01"],
}


def ar1(n: int, phi: float, sigma: float) -> np.ndarray:
    """Stationary AR(1) path with marginal standard deviation `sigma`."""
    eps = rng.normal(0.0, sigma * np.sqrt(1 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def daily_to_hourly(x_day: np.ndarray) -> np.ndarray:
    return x_day[day_codes]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Zone parameters (rough 2024 orders of magnitude, MW and degC)
# ---------------------------------------------------------------------------
ZP = {
    "DE_LU": dict(lat=51.0, lon=10.0, temp=(0.5, 19.0), temp_rho=0.9,
                  load_base=50_500, heat=450, cool=0,
                  wind_on=61_000, wind_off=8_900, wind_mu=(0.95, 1.7), wind_rho=0.92,
                  solar=85_000, cloud=(0.72, 0.50), cloud_rho=0.75,
                  exo_import=(2500, 1500, 800)),
    "FR": dict(lat=46.5, lon=2.5, temp=(4.5, 21.0), temp_rho=0.85,
               load_base=39_500, heat=1_800, cool=150,
               wind_on=22_000, wind_off=1_500, wind_mu=(1.0, 1.7), wind_rho=0.75,
               solar=19_000, cloud=(0.65, 0.40), cloud_rho=0.65,
               exo_import=(-1500, 600, 1200)),
    "NL": dict(lat=52.2, lon=5.3, temp=(3.5, 18.0), temp_rho=0.9,
               load_base=12_500, heat=120, cool=0,
               wind_on=6_500, wind_off=4_700, wind_mu=(0.7, 1.4), wind_rho=0.92,
               solar=22_000, cloud=(0.72, 0.50), cloud_rho=0.75,
               exo_import=(500, 300, 700)),
    "BE": dict(lat=50.6, lon=4.5, temp=(3.5, 18.5), temp_rho=0.9,
               load_base=8_800, heat=120, cool=0,
               wind_on=3_200, wind_off=2_300, wind_mu=(0.9, 1.6), wind_rho=0.88,
               solar=9_000, cloud=(0.72, 0.50), cloud_rho=0.75,
               exo_import=(0, 200, 500)),
    "ES": dict(lat=40.0, lon=-3.7, temp=(8.0, 26.0), temp_rho=0.4,
               load_base=27_000, heat=350, cool=900,
               wind_on=31_000, wind_off=0, wind_mu=(0.9, 1.3), wind_rho=0.35,
               solar=30_000, cloud=(0.30, 0.12), cloud_rho=0.30,
               exo_import=(-600, 200, 400)),
    "IT_NORD": dict(lat=45.3, lon=9.5, temp=(2.5, 24.0), temp_rho=0.6,
                    load_base=17_500, heat=150, cool=800,
                    wind_on=300, wind_off=0, wind_mu=(1.6, 2.0), wind_rho=0.30,
                    solar=11_000, cloud=(0.55, 0.30), cloud_rho=0.40,
                    exo_import=(5000, 600, 900)),
}
ZONE_LIST = list(ZP)
assert all(ZONES[z].in_sample for z in ZONE_LIST)

# Hourly load shapes (normalised below). Winter: lighting-driven evening peak;
# summer: flatter, with a midday plateau.
SHAPE_WINTER = np.array([0.80, 0.77, 0.75, 0.75, 0.77, 0.84, 0.96, 1.06, 1.10, 1.11, 1.11, 1.10,
                         1.08, 1.07, 1.06, 1.06, 1.09, 1.15, 1.16, 1.12, 1.06, 1.00, 0.93, 0.86])
SHAPE_SUMMER = np.array([0.82, 0.79, 0.77, 0.76, 0.77, 0.82, 0.91, 1.00, 1.06, 1.10, 1.12, 1.13,
                         1.13, 1.12, 1.11, 1.10, 1.09, 1.08, 1.07, 1.05, 1.03, 1.00, 0.95, 0.88])
SHAPE_WINTER /= SHAPE_WINTER.mean()
SHAPE_SUMMER /= SHAPE_SUMMER.mean()

# ---------------------------------------------------------------------------
# Common weather factors (shared synoptic systems across Europe)
# ---------------------------------------------------------------------------
temp_common = ar1(D, 0.8, 3.0)
wind_common = ar1(T, 0.985, 1.0)
cloud_common = ar1(D, 0.6, 1.0)

out: dict[str, pd.DataFrame] = {}
for z, p in ZP.items():
    # ---------------- temperature -> load -----------------------------------
    t_jan, t_jul = p["temp"]
    t_clim = t_jan + (t_jul - t_jan) * season_w_day
    t_anom = p["temp_rho"] * temp_common + np.sqrt(1 - p["temp_rho"] ** 2) * ar1(D, 0.7, 2.0)
    temp_day = t_clim + t_anom
    hdd = np.maximum(0, 15 - temp_day)
    cdd = np.maximum(0, temp_day - 21)

    shape = (1 - season_w) * SHAPE_WINTER[hour] + season_w * SHAPE_SUMMER[hour]
    hol = np.isin(local_dates, pd.to_datetime(HOLIDAYS[z]))
    sunday_like = (weekday == 6) | hol
    saturday = (weekday == 5) & ~hol
    day_level = np.where(sunday_like, 0.80, np.where(saturday, 0.88, 1.0))
    shape = np.where(sunday_like | saturday, 1 + 0.6 * (shape - 1), shape)  # flatter weekends

    level = p["load_base"] + p["heat"] * daily_to_hourly(hdd) + p["cool"] * daily_to_hourly(cdd)
    load_true = level * day_level * shape * (1 + ar1(T, 0.8, 0.005))
    mean_load = load_true.mean()
    # Forecast error = persistent daily bias + autocorrelated hourly noise
    load_err = daily_to_hourly(ar1(D, 0.7, 0.010 * mean_load)) + ar1(T, 0.8, 0.008 * mean_load)
    load_fc = load_true - load_err

    # ---------------- wind --------------------------------------------------
    mu = p["wind_mu"][0] + (p["wind_mu"][1] - p["wind_mu"][0]) * season_w
    x_fc = p["wind_rho"] * wind_common + np.sqrt(1 - p["wind_rho"] ** 2) * ar1(T, 0.97, 1.0)
    x_fc = x_fc + 0.10 * np.sin(2 * np.pi * (hour - 9) / 24)  # slight afternoon peak onshore
    wind_err_latent = daily_to_hourly(ar1(D, 0.6, 0.10)) + ar1(T, 0.9, 0.15)
    x_act = x_fc + wind_err_latent

    def cf_on(x):
        return 0.88 * sigmoid((x - mu) / 0.7)

    def cf_off(x):
        return 0.92 * sigmoid((x - (mu - 0.6)) / 0.7)

    wind_on_fc, wind_on_act = p["wind_on"] * cf_on(x_fc), p["wind_on"] * cf_on(x_act)
    wind_off_fc, wind_off_act = p["wind_off"] * cf_off(x_fc), p["wind_off"] * cf_off(x_act)

    # ---------------- solar -------------------------------------------------
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    lat = np.deg2rad(p["lat"])
    hour_angle = np.deg2rad(15 * (utc_hour + p["lon"] / 15 - 12))
    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    cf_clear = 0.80 * np.clip(sin_elev, 0, None) ** 1.2

    c_w, c_s = p["cloud"]
    c0 = np.log(c_w / (1 - c_w)) + (np.log(c_s / (1 - c_s)) - np.log(c_w / (1 - c_w))) * season_w_day
    cloud_latent = p["cloud_rho"] * cloud_common + np.sqrt(1 - p["cloud_rho"] ** 2) * ar1(D, 0.6, 1.0)
    cloud_fc = np.clip(sigmoid(daily_to_hourly(c0 + 1.2 * cloud_latent)) + ar1(T, 0.9, 0.06), 0, 1)
    cloud_err = daily_to_hourly(ar1(D, 0.4, 0.05)) + ar1(T, 0.85, 0.06)
    cloud_act = np.clip(cloud_fc + cloud_err, 0, 1)
    solar_fc = p["solar"] * cf_clear * (1 - 0.75 * cloud_fc)
    solar_act = p["solar"] * cf_clear * (1 - 0.75 * cloud_act)

    out[z] = pd.DataFrame(
        {
            "load_actual": load_true, "load_forecast": load_fc,
            "wind_onshore": wind_on_act, "wind_offshore": wind_off_act,
            "solar": solar_act,
            "wind_onshore_forecast": wind_on_fc, "wind_offshore_forecast": wind_off_fc,
            "solar_forecast": solar_fc,
        },
        index=idx,
    )

# ---------------------------------------------------------------------------
# Fuel & carbon paths (daily; anchored to rough H1-2024 monthly levels)
# ---------------------------------------------------------------------------
months = day_index.month.to_numpy()
ttf_anchor = {1: 30, 2: 25, 3: 27, 4: 29, 5: 32, 6: 34}
eua_anchor = {1: 70, 2: 56, 3: 58, 4: 65, 5: 70, 6: 68}
gas = pd.Series([ttf_anchor[m] for m in months], dtype=float).rolling(15, center=True, min_periods=1).mean().to_numpy()
eua = pd.Series([eua_anchor[m] for m in months], dtype=float).rolling(15, center=True, min_periods=1).mean().to_numpy()
gas = gas + ar1(D, 0.95, 1.5)
eua = eua + ar1(D, 0.95, 2.5)
gas_h, eua_h = daily_to_hourly(gas), daily_to_hourly(eua)

lignite = 6 + 1.10 * eua_h
coal = 27.5 + 0.85 * eua_h
ccgt = gas_h / 0.55 + 0.37 * eua_h + 3
ocgt = gas_h / 0.38 + 0.53 * eua_h + 5
ccgt_it = (gas_h + 3.0) / 0.52 + 0.37 * eua_h + 4  # Italian PSV premium over TTF, older fleet
nuclear_cost = np.full(T, 12.0)

# Seasonal hydro water values (wet spring in 2024 Iberia/France)
hydro_es = 0.85 - 0.50 * np.exp(-(((doy - 95) / 45) ** 2))
hydro_fr = 0.90 - 0.25 * np.exp(-(((doy - 120) / 40) ** 2))


def outage_process(rate_per_day=1 / 25) -> np.ndarray:
    """Unplanned thermal outages: multi-day events removing 3-8% of capacity."""
    o = np.zeros(D)
    for d in range(D):
        if rng.random() < rate_per_day:
            dur = rng.integers(1, 6)
            o[d : d + dur] += rng.uniform(0.03, 0.08)
    return 1 - np.clip(daily_to_hourly(o), 0, 0.15)


fr_nuclear_avail = np.clip(0.90 - 0.08 * season_w + daily_to_hourly(ar1(D, 0.95, 0.03)), 0.70, 0.98)

# Merit-order segments: (name, capacity MW, cost_lo, cost_hi, scaled_by_availability)
ONE = np.ones(T)
STACKS = {
    "DE_LU": [("must_run", 9000, -0.5 * ONE, 5 * ONE, False), ("bio_hydro", 6000, 5 * ONE, 45 * ONE, True),
              ("lignite", 12000, lignite * 0.95, lignite * 1.12, True), ("hard_coal", 5000, coal, coal * 1.20, True),
              ("ccgt", 12000, ccgt, ccgt * 1.35, True), ("ocgt", 6000, ocgt, ocgt * 1.45, True),
              ("oil", 3000, 200 * ONE, 300 * ONE, True), ("scarcity", 2000, 300 * ONE, 1500 * ONE, False)],
    "FR": [("must_run", 18000, -12 * ONE, -1 * ONE, False),  # inflexible nuclear + run-of-river
           ("nuclear_flex", 14000, -1 * ONE, 15 * ONE, "fr_nuc"), ("nuclear", 22000, 15 * ONE, 40 * ONE, "fr_nuc"),
           ("hydro", 9000, hydro_fr * ccgt * 0.85, hydro_fr * ccgt * 1.05, True),
           ("ccgt", 6500, ccgt, ccgt * 1.30, True), ("coal", 1500, coal, coal * 1.15, True),
           ("ocgt_oil", 4000, ocgt, 250 * ONE, True), ("scarcity", 1500, 300 * ONE, 1500 * ONE, False)],
    "NL": [("must_run", 3500, -0.5 * ONE, 5 * ONE, False), ("coal", 3500, coal, coal * 1.08, True),
           ("ccgt", 12000, ccgt, ccgt * 1.35, True), ("ocgt", 3000, ocgt, ocgt * 1.45, True),
           ("scarcity", 1000, 300 * ONE, 1500 * ONE, False)],
    "BE": [("must_run", 1500, -5 * ONE, 0 * ONE, False), ("nuclear", 3900, 0 * ONE, 18 * ONE, True),
           ("ccgt", 6000, ccgt, ccgt * 1.35, True), ("ocgt", 1500, ocgt, ocgt * 1.45, True),
           ("scarcity", 800, 300 * ONE, 1500 * ONE, False)],
    "ES": [("must_run", 10000, -0.5 * ONE, 1 * ONE, False),
           ("hydro", 12000, hydro_es * ccgt * 0.65, hydro_es * ccgt * 1.05, True),
           ("ccgt", 24000, ccgt, ccgt * 1.30, True), ("coal", 1500, coal, coal * 1.15, True),
           ("ocgt_oil", 2000, ocgt, 250 * ONE, True), ("scarcity", 1000, 300 * ONE, 1500 * ONE, False)],
    "IT_NORD": [("must_run", 3500, 0 * ONE, 5 * ONE, False), ("hydro", 6000, 0.95 * ccgt_it, 1.02 * ccgt_it, True),
                ("ccgt", 22000, ccgt_it, ccgt_it * 1.35, True), ("ocgt_oil", 2500, ocgt + 5, 260 * ONE, True),
                ("scarcity", 1000, 300 * ONE, 1500 * ONE, False)],
}
# Price behaviour below the must-run level: renewables curtail at negative bids.
NEG_KNOTS = {
    "DE_LU": [(-30000, -200), (-8000, -40), (-1000, -3), (0, -0.5)],
    "FR": [(-25000, -150), (-6000, -40), (-1000, -14), (0, -12)],
    "NL": [(-12000, -200), (-3000, -40), (-400, -3), (0, -0.5)],
    "BE": [(-8000, -150), (-2000, -30), (-300, -7), (0, -5)],
    "ES": [(-15000, -5), (-3000, -1), (0, -0.2)],  # Iberian bids cluster at ~0
    "IT_NORD": [(-10000, 0), (0, 0)],  # no negative prices in Italy in 2024
}
GAP_MW = 200.0  # transition width between merit-order blocks


def build_knots(z: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-hour merit-order knots (T, M), sorted by cost each hour."""
    avail = outage_process()
    segs = STACKS[z]
    K = len(segs)
    caps = np.zeros((T, K))
    clo = np.zeros((T, K))
    chi = np.zeros((T, K))
    for k, (_, cap, lo, hi, scaled) in enumerate(segs):
        if scaled == "fr_nuc":
            caps[:, k] = cap * fr_nuclear_avail * avail
        elif scaled:
            caps[:, k] = cap * avail
        else:
            caps[:, k] = cap
        clo[:, k], chi[:, k] = lo, hi
    order = np.argsort(clo, axis=1, kind="stable")
    caps = np.take_along_axis(caps, order, 1)
    clo = np.take_along_axis(clo, order, 1)
    chi = np.take_along_axis(chi, order, 1)
    starts = np.concatenate([np.zeros((T, 1)), np.cumsum(caps + GAP_MW, axis=1)[:, :-1]], axis=1)
    xs = np.stack([starts, starts + caps], axis=2).reshape(T, 2 * K)
    fs = np.stack([clo, chi], axis=2).reshape(T, 2 * K)
    end = xs[:, -1:]
    neg = NEG_KNOTS[z]
    x_neg = np.tile([x for x, _ in neg[:-1]], (T, 1))
    f_neg = np.tile([f for _, f in neg[:-1]], (T, 1))
    xk = np.concatenate([x_neg, xs, end + 3000, end + 6000], axis=1)
    fk = np.concatenate([f_neg, fs, np.full((T, 1), 3000.0), np.full((T, 1), PRICE_CAP)], axis=1)
    fk = np.maximum.accumulate(fk, axis=1)  # merit order is monotone
    return xk, fk


def interp_rows(n: np.ndarray, xk: np.ndarray, fk: np.ndarray) -> np.ndarray:
    m = xk.shape[1]
    j = np.clip((xk <= n[:, None]).sum(axis=1) - 1, 0, m - 2)
    r = np.arange(len(n))
    x0, x1, f0, f1 = xk[r, j], xk[r, j + 1], fk[r, j], fk[r, j + 1]
    return f0 + (n - x0) / (x1 - x0) * (f1 - f0)


knots = {z: build_knots(z) for z in ZONE_LIST}



def zone_price(z: str, n: np.ndarray) -> np.ndarray:
    base = interp_rows(n, *knots[z])
    p = base * (1 + markups[z]) + adders[z]
    if z == "IT_NORD":
        p = np.maximum(p, 0.0)
    return np.clip(p, PRICE_FLOOR, PRICE_CAP)


# ---------------------------------------------------------------------------
# Price-setting residual demand (what the auction "sees")
# ---------------------------------------------------------------------------
rd_market, exo = {}, {}
for z in ZONE_LIST:
    df = out[z]
    rd_fc = df.load_forecast - df.wind_onshore_forecast - df.wind_offshore_forecast - df.solar_forecast
    rd_act = df.load_actual - df.wind_onshore - df.wind_offshore - df.solar
    daily_bias = (rd_act - rd_fc).groupby(day_codes).transform("mean").to_numpy()
    rd_market[z] = rd_fc.to_numpy() + KAPPA_BIAS * daily_bias
    # Imports from neighbours outside the modelled set respond to local tightness
    base, k_tight, sd = ZP[z]["exo_import"]
    z_rd = (rd_market[z] - rd_market[z].mean()) / rd_market[z].std()
    exo[z] = base + k_tight * z_rd + ar1(T, 0.95, sd)

# Bidding noise: shared regional component + idiosyncratic component, plus a
# flexibility premium when residual demand ramps up quickly (sunset in summer,
# morning pick-up in winter): ramping plants and storage price in scarcity of
# flexibility, which is why evening peaks are sharper than the merit order alone implies.
# Daily shocks stand in for everything the fundamentals here do not capture
# (gas/carbon news, outages, hydro strategy, risk appetite): they move whole
# days up or down independently of residual demand.
regional_markup = ar1(T, 0.8, 0.035) + daily_to_hourly(ar1(D, 0.3, 0.07))
markups = {z: regional_markup + ar1(T, 0.6, 0.02) + daily_to_hourly(ar1(D, 0.3, 0.04)) for z in ZONE_LIST}
adders = {}
for z in ZONE_LIST:
    ramp = np.diff(rd_market[z], prepend=rd_market[z][0]) / out[z]["load_forecast"].mean()
    adders[z] = ar1(T, 0.7, 2.0) + 80.0 * np.clip(ramp, 0, None)

# ---------------------------------------------------------------------------
# Simplified market coupling: Newton-style flow updates until spreads close
# or interconnectors congest. Positive flow = zone_a -> zone_b.
# ---------------------------------------------------------------------------
sample_borders = [b for b in BORDERS if b.zone_a in ZP and b.zone_b in ZP]
zi = {z: i for i, z in enumerate(ZONE_LIST)}
flows = np.zeros((T, len(sample_borders)))
neff: dict[str, np.ndarray] = {}
for it in range(120):
    net_exp = np.zeros((T, len(ZONE_LIST)))
    for k, b in enumerate(sample_borders):
        net_exp[:, zi[b.zone_a]] += flows[:, k]
        net_exp[:, zi[b.zone_b]] -= flows[:, k]
    prices, slopes = {}, {}
    for z in ZONE_LIST:
        n = rd_market[z] - exo[z] + net_exp[:, zi[z]]
        neff[z] = n
        prices[z] = zone_price(z, n)
        slopes[z] = (zone_price(z, n + 100) - zone_price(z, n - 100)) / 200 + 1e-4
    damping = 0.5 if it < 60 else 0.25
    for k, b in enumerate(sample_borders):
        spread = prices[b.zone_b] - prices[b.zone_a]
        step = spread / (slopes[b.zone_a] + slopes[b.zone_b])
        flows[:, k] = np.clip(flows[:, k] + damping * step, -b.capacity_ba_mw, b.capacity_ab_mw)

for z in ZONE_LIST:
    out[z]["price"] = np.round(prices[z], 2)

# ---------------------------------------------------------------------------
# Inject realistic data-quality issues (ENTSO-E data has gaps)
# ---------------------------------------------------------------------------
def blank(z: str, col: str, start: str, end: str) -> None:
    mask = (idx >= pd.Timestamp(start, tz=MARKET_TZ)) & (idx <= pd.Timestamp(end, tz=MARKET_TZ))
    out[z].loc[mask, col] = np.nan


blank("BE", "load_actual", "2024-02-14 10:00", "2024-02-14 11:00")  # short gap -> interpolated
blank("BE", "load_actual", "2024-03-05 06:00", "2024-03-05 12:00")  # long gap -> flagged
blank("FR", "solar", "2024-05-02 12:00", "2024-05-02 14:00")
blank("IT_NORD", "wind_onshore_forecast", "2024-04-10 00:00", "2024-04-10 23:00")

# ---------------------------------------------------------------------------
# Write ENTSO-E-shaped CSVs (UTC timestamps, MW / EUR/MWh)
# ---------------------------------------------------------------------------
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
ts_fmt = "%Y-%m-%dT%H:%MZ"


def long_frame(cols: dict[str, str], decimals: int = 0) -> pd.DataFrame:
    frames = []
    for z in ZONE_LIST:
        f = out[z][list(cols)].rename(columns=cols).round(decimals)
        if decimals == 0:
            f = f.astype("Int64")  # MW values as integers, gaps stay empty
        f.insert(0, "zone", z)
        f.index = f.index.tz_convert("UTC")
        f.index.name = "timestamp_utc"
        frames.append(f)
    return pd.concat(frames).reset_index()


files = {
    "hourly_prices.csv": long_frame({"price": "price_eur_mwh"}, 2),
    "load_actual.csv": long_frame({"load_actual": "load_mw"}),
    "load_forecast.csv": long_frame({"load_forecast": "load_forecast_mw"}),
    "renewable_actual.csv": long_frame(
        {"wind_onshore": "wind_onshore_mw", "wind_offshore": "wind_offshore_mw", "solar": "solar_mw"}
    ),
    "renewable_forecast.csv": long_frame(
        {"wind_onshore_forecast": "wind_onshore_forecast_mw",
         "wind_offshore_forecast": "wind_offshore_forecast_mw",
         "solar_forecast": "solar_forecast_mw"}
    ),
}
# Physical flows are reported per direction as non-negative values (as on ENTSO-E).
flow_rows = []
utc_idx = idx.tz_convert("UTC")
for k, b in enumerate(sample_borders):
    loop = ar1(T, 0.9, 0.02 * max(b.capacity_ab_mw, b.capacity_ba_mw))  # loop-flow noise
    f = flows[:, k] + loop
    flow_rows.append(pd.DataFrame({"timestamp_utc": utc_idx, "from_zone": b.zone_a, "to_zone": b.zone_b,
                                   "flow_mw": np.round(np.clip(f, 0, None)).astype(int)}))
    flow_rows.append(pd.DataFrame({"timestamp_utc": utc_idx, "from_zone": b.zone_b, "to_zone": b.zone_a,
                                   "flow_mw": np.round(np.clip(-f, 0, None)).astype(int)}))
files["flows.csv"] = pd.concat(flow_rows, ignore_index=True)

for name, frame in files.items():
    frame.to_csv(SAMPLE_DIR / name, index=False, date_format=ts_fmt)

metadata = {
    "notice": "SYNTHETIC SAMPLE DATA - generated by scripts/generate_sample_data.py. "
              "NOT real market data. Do not use for trading or present as real.",
    "generator_version": GENERATOR_VERSION,
    "seed": SEED,
    "period_local": [START, END],
    "market_timezone": MARKET_TZ,
    "timestamps": "UTC, start of delivery hour",
    "zones": ZONE_LIST,
    "borders": [[b.zone_a, b.zone_b] for b in sample_borders],
    "files": sorted(files),
    "dgp_assumptions": {
        "price_formation": "merit-order curve per zone from synthetic gas/carbon paths, cleared on forecast fundamentals",
        "market_coupling": "iterative flow updates from low- to high-price zones, capped by indicative capacities",
        "forecast_bias_anticipated_by_market": KAPPA_BIAS,
        "injected_gaps": ["BE load_actual 2024-02-14 10-11h", "BE load_actual 2024-03-05 06-12h",
                          "FR solar 2024-05-02 12-14h", "IT_NORD wind_onshore_forecast 2024-04-10"],
    },
}
(SAMPLE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

# ---------------------------------------------------------------------------
# Calibration summary (sanity check against rough real-world magnitudes)
# ---------------------------------------------------------------------------
print(f"Generated {T} hours x {len(ZONE_LIST)} zones -> {SAMPLE_DIR}")
rows = []
for z in ZONE_LIST:
    d = out[z]
    wind = d.wind_onshore + d.wind_offshore
    rd = d.load_actual - wind - d.solar
    rows.append({
        "zone": z,
        "avg": d.price.mean(), "min": d.price.min(), "max": d.price.max(),
        "p90": d.price.quantile(0.9), "neg_h": int((d.price < 0).sum()),
        "load_GW": d.load_actual.mean() / 1e3, "wind_GW": wind.mean() / 1e3, "solar_GW": d.solar.mean() / 1e3,
        "corr_rd_px": rd.corr(d.price),
    })
summary = pd.DataFrame(rows).set_index("zone").round(2)
print(summary.to_string())
monthly = pd.DataFrame({z: out[z].price for z in ZONE_LIST}).resample("MS").mean().round(1)
print("\nMonthly average prices (EUR/MWh):")
print(monthly.to_string())
pairs = [("DE_LU", "NL"), ("DE_LU", "FR"), ("FR", "ES"), ("FR", "IT_NORD"), ("NL", "BE")]
print("\nShare of hours with |spread| < 1 EUR/MWh:")
for a, b in pairs:
    print(f"  {a}-{b}: {(np.abs(out[a].price - out[b].price) < 1).mean():.0%}")

