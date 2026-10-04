"""Backtester arithmetic, metrics and edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    exposure_by_hour,
    max_drawdown,
    run_backtest,
    sharpe_like,
    stress_spike_hours,
)
from src.config import BacktestParams


def _signals(positions, targets, hours=None, start="2024-01-01") -> pd.DataFrame:
    n = len(positions)
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "hour": 12,
            "position": np.asarray(positions, dtype=float),
            "target": np.asarray(targets, dtype=float),
            "reference_price": 50.0,
            "delivered_price": 50.0 + np.asarray(targets, dtype=float),
            "score": np.asarray(positions, dtype=float) * 2,
            "hours": 1.0 if hours is None else np.asarray(hours, dtype=float),
        },
        index=dates,
    )


def test_pnl_includes_costs():
    sig = _signals([1, -1, 0, 1], [10, -5, 7, -4])
    res = run_backtest(sig, BacktestParams(cost_per_mwh=1.0))
    # gross: +10, +5, 0, -4 ; costs: 1 per traded MWh on 3 trades
    assert res.metrics["gross_pnl"] == pytest.approx(11)
    assert res.metrics["total_cost"] == pytest.approx(3)
    assert res.metrics["total_pnl"] == pytest.approx(8)
    assert res.metrics["n_trades"] == 3
    assert res.metrics["hit_rate"] == pytest.approx(2 / 3)


def test_no_positions_means_no_pnl():
    res = run_backtest(_signals([0, 0, 0], [5, -5, 1]))
    assert res.metrics["n_trades"] == 0
    assert res.metrics["total_pnl"] == 0
    assert np.isnan(res.metrics["hit_rate"])
    assert (res.daily_pnl == 0).all()


def test_perfect_foresight_hits_everything():
    targets = np.array([3.0, -2.0, 8.0, -1.0, 4.0])
    res = run_backtest(_signals(np.sign(targets), targets), BacktestParams(cost_per_mwh=0.5))
    assert res.metrics["hit_rate"] == 1.0
    assert res.metrics["total_pnl"] == pytest.approx(np.abs(targets).sum() - 0.5 * len(targets))


def test_daily_horizon_scales_by_hours_in_day():
    # A 23-hour DST day: baseload position earns the move on 23 MWh.
    res = run_backtest(_signals([1], [2.0], hours=[23]), BacktestParams(cost_per_mwh=0.0))
    assert res.metrics["total_pnl"] == pytest.approx(46)


def test_max_drawdown_known_example():
    cum = pd.Series([0.0, 10.0, 5.0, 15.0, 3.0])
    assert max_drawdown(cum) == pytest.approx(-12.0)
    assert max_drawdown(pd.Series([-5.0, -3.0])) == pytest.approx(-5.0)  # losing from the start counts
    assert max_drawdown(pd.Series(dtype=float)) == 0.0


def test_sharpe_like():
    daily = pd.Series([1.0, -1.0, 2.0, 0.0])
    assert sharpe_like(daily) == pytest.approx(daily.mean() / daily.std() * np.sqrt(365))
    assert np.isnan(sharpe_like(pd.Series([1.0, 1.0])))


def test_idle_days_are_kept_in_daily_pnl():
    sig = _signals([1, 0, 0, 1], [1, 1, 1, 1])
    sig = sig.drop(sig.index[1])  # a missing day in the ledger
    res = run_backtest(sig, BacktestParams(cost_per_mwh=0))
    assert len(res.daily_pnl) == 4
    assert res.daily_pnl.iloc[1] == 0


def test_rows_without_target_are_ignored():
    sig = _signals([1, 1], [np.nan, 5.0])
    res = run_backtest(sig, BacktestParams(cost_per_mwh=0))
    assert res.metrics["eligible_rows"] == 1
    assert res.metrics["total_pnl"] == 5


def test_exclusion_mask_and_stress_table():
    sig = _signals([1, 1, 1, 1], [100.0, 1.0, 1.0, -1.0])
    spike = pd.Series([True, False, False, False], index=sig.index)
    excl = run_backtest(sig, BacktestParams(cost_per_mwh=0), exclude=spike)
    assert excl.metrics["total_pnl"] == pytest.approx(1.0)
    table = stress_spike_hours(sig, spike, BacktestParams(cost_per_mwh=0))
    assert table.loc["All observations", "total_pnl"] == pytest.approx(101.0)
    assert table.loc["Spike rows only", "total_pnl"] == pytest.approx(100.0)


def test_confusion_matrix_counts():
    res = run_backtest(_signals([1, 1, -1, 0], [2, -2, -2, 3]))
    conf = res.confusion
    assert conf.loc["Long", "Price up"] == 1
    assert conf.loc["Long", "Price down"] == 1
    assert conf.loc["Short", "Price down"] == 1
    assert conf.loc["No trade", "Price up"] == 1


def test_exposure_by_hour():
    sig = _signals([1, -1, 1], [1, 1, 1])
    sig["hour"] = [7, 7, 18]
    expo = exposure_by_hour(run_backtest(sig).trades)
    assert expo.loc[7, "Long"] == 1 and expo.loc[7, "Short"] == 1 and expo.loc[18, "Long"] == 1
    assert len(expo) == 24
