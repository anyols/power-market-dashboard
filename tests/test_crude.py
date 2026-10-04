"""Crude module: contract calendar, seasonal statistics, release-aware signals, brief."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import run_backtest
from src.config import BacktestParams
from src.crude.commentary import generate_crude_brief
from src.crude.contracts import cl_last_trade, front_contract, roll_adjusted_changes
from src.crude.data import load_crude_data
from src.crude.features import (
    add_cot_features,
    add_daily_features,
    add_weekly_features,
    seasonal_profile,
    seasonal_stats,
    trailing_percentile,
)
from src.crude.signals import CrudeSignalParams, build_crude_signals, component_diagnostics

# Published CME last-trade dates for CL contracts
KNOWN_EXPIRIES = {
    (2024, 1): "2023-12-19", (2024, 2): "2024-01-22", (2024, 3): "2024-02-20", (2024, 4): "2024-03-20",
    (2024, 5): "2024-04-22", (2024, 6): "2024-05-21", (2024, 7): "2024-06-20", (2024, 8): "2024-07-22",
    (2024, 9): "2024-08-20", (2024, 10): "2024-09-20", (2024, 11): "2024-10-22", (2024, 12): "2024-11-20",
    (2025, 1): "2024-12-19",
}


@pytest.fixture(scope="module")
def sample():
    from src.config import SAMPLE_DIR

    data = load_crude_data(store_root=SAMPLE_DIR / "__no_store__")  # force the deterministic synthetic sample
    assert data.is_sample
    return data


@pytest.fixture(scope="module")
def feats(sample):
    return add_daily_features(sample.daily), add_weekly_features(sample.weekly), add_cot_features(sample.cot)


# ---------------------------------------------------------------------------
# Contract calendar
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("delivery,expected", KNOWN_EXPIRIES.items())
def test_cl_last_trade_matches_cme(delivery, expected):
    assert cl_last_trade(*delivery) == pd.Timestamp(expected)


def test_front_contract_rolls_the_day_after_expiry():
    dates = pd.DatetimeIndex(["2024-02-20", "2024-02-21"])  # CLH24 last trade 20 Feb
    front = front_contract(dates)
    assert str(front[0]) == "2024-03" and str(front[1]) == "2024-04"


def test_roll_adjusted_change_uses_previous_second_contract():
    idx = pd.DatetimeIndex(["2024-02-16", "2024-02-20", "2024-02-21"])
    cl1 = pd.Series([78.0, 79.0, 77.0], index=idx)  # 21 Feb: new front (April), which traded at 78.5 on the 20th
    cl2 = pd.Series([77.5, 78.5, 76.0], index=idx)
    chg = roll_adjusted_changes(cl1, cl2)
    assert chg.iloc[1] == pytest.approx(1.0)
    assert chg.iloc[2] == pytest.approx(77.0 - 78.5)  # not 77 - 79 (that would book the spread as PnL)


# ---------------------------------------------------------------------------
# Seasonal statistics
# ---------------------------------------------------------------------------
def _weekly_series(levels: dict[int, float]) -> pd.Series:
    idx = pd.date_range("2010-01-08", "2020-12-25", freq="W-FRI")
    return pd.Series([levels.get(d.isocalendar().year, np.nan) for d in idx], index=idx, dtype=float)


def test_seasonal_stats_use_previous_five_years_only():
    s = _weekly_series({y: float(y) for y in range(2009, 2021)})
    st = seasonal_stats(s)
    row = st.loc[s.index[s.index.year == 2016][10]]
    assert row["avg"] == pytest.approx(np.mean([2011, 2012, 2013, 2014, 2015]))
    assert row["min"] == 2011 and row["max"] == 2015


def test_seasonal_stats_ignore_the_future():
    s = _weekly_series({y: float(y) for y in range(2009, 2021)})
    s2 = s.copy()
    s2[s2.index.year >= 2018] = 1e9
    before = s.index.year < 2018
    pd.testing.assert_frame_equal(seasonal_stats(s)[before], seasonal_stats(s2)[before])


def test_weekly_surprise_definition(feats):
    _, w, _ = feats
    row = w.dropna(subset=["crude_stocks_surprise"]).iloc[-1]
    assert row["crude_stocks_surprise"] == pytest.approx(row["crude_stocks_chg"] - row["crude_stocks_chg_5y_avg"])
    assert (w["available_date"] - w.index == pd.Timedelta(days=6)).all()


def test_trailing_percentile_excludes_current():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0])
    assert trailing_percentile(s, window=4, min_periods=4).iloc[-1] == 1.0
    assert trailing_percentile(pd.Series([5.0, 4, 3, 2, 1]), window=4, min_periods=4).iloc[-1] == 0.0


def test_seasonal_profile_shape(feats):
    _, w, _ = feats
    prof = seasonal_profile(w["crude_stocks"], 2024)
    assert list(prof.columns) == ["min", "max", "avg", "current", "last_year"]
    assert len(prof) == 52 and (prof["min"] <= prof["max"]).all()


# ---------------------------------------------------------------------------
# Signals: release-calendar-aware look-ahead tests
# ---------------------------------------------------------------------------
INPUT_COLS = ["inventory_z", "cushing_z", "m1_m2", "mm_pctile", "comp_inventory", "comp_cushing", "comp_curve",
              "comp_positioning", "score", "position", "reference_price", "week_used", "cot_used"]


def _scramble(sample, decision: pd.Timestamp, seed: int = 3):
    rng = np.random.default_rng(seed)
    daily, weekly, cot = sample.daily.copy(), sample.weekly.copy(), sample.cot.copy()
    m = daily.index > decision
    daily.loc[m] = rng.uniform(1, 200, (m.sum(), daily.shape[1]))
    m = weekly.index + pd.Timedelta(days=6) > decision  # not yet released at decision time
    weekly.loc[m] = rng.uniform(1e3, 9e5, (m.sum(), weekly.shape[1]))
    m = cot.index + pd.Timedelta(days=3) > decision
    cot.loc[m] = rng.uniform(1e4, 3e6, (m.sum(), cot.shape[1]))
    return add_daily_features(daily), add_weekly_features(weekly), add_cot_features(cot)


@pytest.mark.parametrize("target", ["futures", "spot"])
@pytest.mark.parametrize("decision", ["2016-11-24", "2019-03-14", "2022-06-16", "2023-12-28"])
def test_crude_signal_ignores_unreleased_information(sample, feats, target, decision):
    D = pd.Timestamp(decision)
    params = CrudeSignalParams(target=target)
    dates = pd.date_range(D - pd.Timedelta(weeks=3), D + pd.Timedelta(weeks=2), freq="W-THU")
    base = build_crude_signals(*feats, params, dates=dates)
    pert = build_crude_signals(*_scramble(sample, D), params, dates=dates)
    pd.testing.assert_series_equal(base.loc[D, INPUT_COLS], pert.loc[D, INPUT_COLS], check_names=False)
    assert base.loc[D, "target"] != pytest.approx(pert.loc[D, "target"])  # the outcome did change


def test_latest_released_report_is_used(sample, feats):
    """Counter-test: the report released on decision day (week ending D-6) must feed the signal."""
    D = pd.Timestamp("2019-03-14")
    sig = build_crude_signals(*feats, dates=pd.DatetimeIndex([D]))
    assert sig.loc[D, "week_used"] == D - pd.Timedelta(days=6)
    assert sig.loc[D, "cot_used"] == D - pd.Timedelta(days=9)  # previous Tuesday's report, published last Friday


def test_spot_target_without_curve_runs_to_latest_data(feats):
    sig = build_crude_signals(*feats, CrudeSignalParams(target="spot", use_curve=False))
    assert sig["target"].dropna().index.max() > pd.Timestamp("2025-06-01")


def test_weekly_backtest_and_diagnostics(feats):
    sig = build_crude_signals(*feats)
    res = run_backtest(sig, BacktestParams(cost_per_mwh=0.03), period_freq="W-THU")
    assert res.metrics["n_trades"] > 50
    assert (res.daily_pnl.index.dayofweek == 3).all()  # weekly Thursday periods
    # PnL is per 1,000-barrel contract
    t = res.trades.iloc[0]
    assert t["pnl"] == pytest.approx(t["position"] * t["target"] * 1000 - 0.03 * 1000)
    diag = component_diagnostics(sig)
    assert {"Inventory surprise", "Curve (carry)", "Combined score"} <= set(diag.index)


# ---------------------------------------------------------------------------
# Weekly brief
# ---------------------------------------------------------------------------
def test_crude_brief_structure_and_determinism(feats):
    d, w, c = feats
    a = generate_crude_brief(w, d, c, "2023-06-09").to_markdown()
    b = generate_crude_brief(w, d, c, "2023-06-09").to_markdown()
    assert a == b
    assert a.startswith("# Weekly Crude Brief — week ending 2023-06-09 (EIA release 2023-06-14)")
    for section in ["Price action", "Inventories", "Refining & supply", "Curve", "Positioning", "Signal & what to watch"]:
        assert section in a


def test_crude_brief_uses_no_future_data(sample, feats):
    F = pd.Timestamp("2023-06-09")
    R = F + pd.Timedelta(days=5)
    d, w, c = feats
    brief = generate_crude_brief(w, d, c, F).to_markdown()
    rng = np.random.default_rng(1)
    daily, weekly, cot = sample.daily.copy(), sample.weekly.copy(), sample.cot.copy()
    daily.loc[daily.index > R] = rng.uniform(1, 200, ((daily.index > R).sum(), daily.shape[1]))
    weekly.loc[weekly.index > F] = rng.uniform(1e3, 9e5, ((weekly.index > F).sum(), weekly.shape[1]))
    cot.loc[cot.index + pd.Timedelta(days=3) > R] = 1e6
    brief2 = generate_crude_brief(add_weekly_features(weekly), add_daily_features(daily), add_cot_features(cot), F)
    assert brief == brief2.to_markdown()
