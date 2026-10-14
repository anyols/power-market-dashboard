"""Cross-market analytics: board stats, ratios, pair statistics, seasonality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.markets import (
    ASSETS,
    PAIRS,
    RATIOS,
    correlation_matrix,
    decoupling,
    lead_lag,
    load_prices,
    log_returns,
    month_stats,
    monthly_returns,
    performance_table,
    ratio_stats,
    rolling_beta_corr,
    seasonal_paths,
    synthetic_prices,
    weekly_returns,
)


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices()


def test_universe_is_consistent():
    assert len(ASSETS) == len({a.ticker for a in ASSETS.values()})
    for pair in PAIRS:
        assert pair.commodity in ASSETS and pair.stock in ASSETS
    for r in RATIOS.values():
        assert all(k in ASSETS for k in r.needs)


def test_synthetic_source_is_offline(monkeypatch):
    monkeypatch.setenv("MARKETS_SOURCE", "synthetic")
    mp = load_prices()
    assert mp.is_synthetic and set(ASSETS) <= set(mp.prices.columns)


def test_performance_table_returns(prices):
    perf = performance_table(prices, ["gold", "wti"])
    s = prices["gold"].dropna()
    assert perf.loc["gold", "1D"] == pytest.approx(s.iloc[-1] / s.iloc[-2] - 1)
    assert perf.loc["gold", "1W"] == pytest.approx(s.iloc[-1] / s.iloc[-6] - 1)
    assert 0 <= perf.loc["gold", "range_pos"] <= 1


def test_log_returns_skip_non_positive_prices():
    p = pd.DataFrame({"wti": [18.0, -37.6, 10.0, 12.0]})
    r = log_returns(p)["wti"]
    assert r.iloc[1:3].isna().all() and r.iloc[3] == pytest.approx(np.log(12 / 10))


def test_ratio_formulas():
    p = pd.DataFrame({"gold": [2000.0], "silver": [25.0], "wti": [80.0], "natgas": [2.5], "gasoline": [2.5],
                      "heating_oil": [3.0], "brent": [84.0], "copper": [4.0], "soybeans": [1200.0], "corn": [500.0],
                      "ttf": [34.12], "eurusd": [1.10], "platinum": [1000.0]})
    assert RATIOS["gold_silver"].compute(p).iloc[0] == 80
    assert RATIOS["gold_oil"].compute(p).iloc[0] == 25
    assert RATIOS["oil_gas"].compute(p).iloc[0] == 32
    assert RATIOS["crack_321"].compute(p).iloc[0] == pytest.approx((2 * 2.5 * 42 + 3.0 * 42 - 3 * 80) / 3)
    assert RATIOS["ttf_hh"].compute(p).iloc[0] == pytest.approx(34.12 * 1.10 / 3.412 - 2.5)
    assert RATIOS["soy_corn"].compute(p).iloc[0] == 2.4


def test_gold_oil_undefined_when_oil_negative():
    p = pd.DataFrame({"gold": [1700.0, 1700.0], "wti": [-37.63, 20.0]})
    r = RATIOS["gold_oil"].compute(p)
    assert np.isnan(r.iloc[0]) and r.iloc[1] == 85


def test_ratio_stats_percentile():
    s = pd.Series(np.arange(100, dtype=float))
    st = ratio_stats(s)
    assert st["last"] == 99 and st["pctile"] == pytest.approx(0.99)


def test_rolling_beta_recovers_known_beta():
    rng = np.random.default_rng(0)
    rc = pd.Series(rng.normal(0, 0.03, 400))
    rs = 1.5 * rc + pd.Series(rng.normal(0, 0.005, 400))
    rb = rolling_beta_corr(rs, rc, 52)
    assert rb["beta"].iloc[-1] == pytest.approx(1.5, abs=0.1)
    assert rb["corr"].iloc[-1] > 0.95


def test_lead_lag_finds_the_leader():
    rng = np.random.default_rng(1)
    rc = pd.Series(rng.normal(0, 1, 500))
    rs = rc.shift(2).fillna(0) + pd.Series(rng.normal(0, 0.3, 500))  # stock follows the commodity by 2 periods
    ll = lead_lag(rs, rc, max_lag=4)
    assert ll.idxmax() == 2


def test_decoupling_uses_beta_from_before_the_window():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2015-01-02", periods=300, freq="W-FRI")
    rc = pd.Series(rng.normal(0, 0.03, 300), index=idx)
    rs = rc.copy()
    rs.iloc[-13:] += 0.02  # stock drifts up 2%/week for 13 weeks, unrelated to the commodity
    dec = decoupling(rs, rc)
    assert dec["residual"].iloc[-1] == pytest.approx(0.26, abs=0.03)
    assert dec["z"].iloc[-1] > 3


def test_weekly_returns_and_correlation_matrix(prices):
    w = weekly_returns(prices[["gold", "silver"]])
    assert (w.index.dayofweek == 4).all()
    c = correlation_matrix(prices, ["gold", "silver", "hsy", "cocoa"], 104)
    assert c.loc["Gold", "Gold"] == pytest.approx(1.0)
    assert c.loc["Hershey", "Cocoa"] < 0  # synthetic data has a built-in negative link


def test_monthly_returns_and_seasonal_paths():
    idx = pd.bdate_range("2020-01-01", "2023-12-31")
    s = pd.Series(np.where(idx.month == 3, 1.002, 1.0).cumprod() * 100, index=idx)  # rises only in March
    table = monthly_returns(s)
    stats = month_stats(table)
    assert stats["mean"].idxmax() == 3 and stats.loc[3, "share_up"] == 1.0
    paths = seasonal_paths(s, years=3)
    assert set(paths.columns) == {2020, 2021, 2022, 2023}
    assert paths[2021].iloc[0] == pytest.approx(100)
