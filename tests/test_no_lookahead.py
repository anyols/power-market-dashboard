"""Look-ahead tests: perturb everything a trader could NOT have known and check nothing moves.

For a decision about delivery day D (taken on D-1 before the day-ahead gate):
* prices from D onwards are unknown,
* actual load / wind / solar from D-1 onwards are treated as unknown,
* day-ahead forecasts from D+1 onwards are unknown.
If any signal input for day D changes when those values are scrambled, the
signal is peeking into the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import SignalParams
from src.data_cleaning import clean_market_frame
from src.data_loader import load_sample_frame
from src.feature_engineering import build_features
from src.market_commentary import generate_daily_brief
from src.signal_engine import build_signal_frame
from src.utils import local_dates

ACTUALS = ["load_actual", "wind_actual", "solar_actual"]
FORECASTS = ["load_forecast", "wind_forecast", "solar_forecast"]
SIGNAL_INPUTS = ["rd_z", "load_surprise_z", "res_surprise_z", "momentum", "comp_rd", "comp_load", "comp_res",
                 "score", "position", "reference_price"]


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    frame, _ = clean_market_frame(load_sample_frame("DE_LU"), "2024-01-01", "2024-06-30")
    return frame


def scramble(raw: pd.DataFrame, price_from, actuals_from, forecasts_from, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = raw.copy()
    dates = local_dates(out.index)
    m = dates >= pd.Timestamp(price_from)
    out.loc[m, "price"] = rng.uniform(-500, 4000, m.sum())
    m = dates >= pd.Timestamp(actuals_from)
    for c in ACTUALS:
        out.loc[m, c] = rng.uniform(0, 90_000, m.sum())
    m = dates >= pd.Timestamp(forecasts_from)
    for c in FORECASTS:
        out.loc[m, c] = rng.uniform(0, 90_000, m.sum())
    return out


def _rows_for_day(sig: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    return sig[sig["date"] == day]


@pytest.mark.parametrize("horizon", ["hourly", "daily"])
@pytest.mark.parametrize("rd_mode", ["change", "level"])
@pytest.mark.parametrize("decision_day", ["2024-03-14", "2024-03-31", "2024-05-20"])
def test_signal_ignores_information_outside_the_set(raw, horizon, rd_mode, decision_day):
    params = SignalParams(horizon=horizon, rd_mode=rd_mode, use_momentum_filter=True)
    D = pd.Timestamp(decision_day)
    base = _rows_for_day(build_signal_frame(build_features(raw), params), D)

    future = scramble(raw, price_from=D, actuals_from=D - pd.Timedelta(days=1), forecasts_from=D + pd.Timedelta(days=1))
    pert = _rows_for_day(build_signal_frame(build_features(future), params), D)

    assert len(base) > 0
    pd.testing.assert_frame_equal(base[SIGNAL_INPUTS], pert[SIGNAL_INPUTS])
    # Sanity check that the perturbation really hit the realised outcome.
    assert not np.allclose(base["target"].to_numpy(), pert["target"].to_numpy(), equal_nan=True)


def test_signal_does_use_forecasts_for_the_delivery_day(raw):
    """Counter-test: the residual-demand input must react to day-D forecasts (they ARE in the information set)."""
    params = SignalParams(horizon="hourly")
    D = pd.Timestamp("2024-03-14")
    base = _rows_for_day(build_signal_frame(build_features(raw), params), D)
    future = scramble(raw, price_from=D + pd.Timedelta(days=5), actuals_from=D + pd.Timedelta(days=5), forecasts_from=D)
    pert = _rows_for_day(build_signal_frame(build_features(future), params), D)
    assert not np.allclose(base["rd_z"].to_numpy(), pert["rd_z"].to_numpy())


def test_actuals_from_previous_day_are_not_used(raw):
    """Conservative cut-off: D-1 actuals must not influence the signal for D."""
    params = SignalParams(horizon="hourly")
    D = pd.Timestamp("2024-04-10")
    base = _rows_for_day(build_signal_frame(build_features(raw), params), D)
    future = raw.copy()
    m = local_dates(future.index) == D - pd.Timedelta(days=1)
    future.loc[m, ACTUALS] = future.loc[m, ACTUALS] * 3
    pert = _rows_for_day(build_signal_frame(build_features(future), params), D)
    pd.testing.assert_frame_equal(base[SIGNAL_INPUTS], pert[SIGNAL_INPUTS])


def test_trailing_features_do_not_peek(raw):
    feat = build_features(raw)
    cutoff = pd.Timestamp("2024-04-01")
    future = scramble(raw, cutoff, cutoff, cutoff)
    feat2 = build_features(future)
    before = local_dates(feat.index) < cutoff
    cols = [c for c in feat.columns if c.endswith("_z")] + ["residual_demand_vs_30d", "price_vol_7d", "price_momentum",
                                                             "vol_regime"]
    pd.testing.assert_frame_equal(feat.loc[before, cols], feat2.loc[before, cols])


def test_daily_brief_uses_no_future_data(raw):
    D = pd.Timestamp("2024-05-15")
    feat = build_features(raw)
    brief = generate_daily_brief(feat, "DE_LU", D)
    # Scramble prices/actuals after D and forecasts after D+1 (D+1 forecasts are published on D).
    future = scramble(raw, price_from=D + pd.Timedelta(days=1), actuals_from=D + pd.Timedelta(days=1),
                      forecasts_from=D + pd.Timedelta(days=2))
    brief2 = generate_daily_brief(build_features(future), "DE_LU", D)
    assert brief.to_markdown() == brief2.to_markdown()


def test_daily_brief_is_deterministic(raw):
    feat = build_features(raw)
    a = generate_daily_brief(feat, "DE_LU", "2024-02-20").to_markdown()
    b = generate_daily_brief(feat, "DE_LU", "2024-02-20").to_markdown()
    assert a == b
    assert a.startswith("# Daily Power Market Brief — Germany/Luxembourg — 2024-02-20")
    for section in ["Price action", "Demand and residual load", "Renewables", "Forecast errors",
                    "Volatility / spike risk", "What to watch next"]:
        assert section in a
