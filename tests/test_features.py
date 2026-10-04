"""Feature engineering: definitions, sign conventions, DST handling and trailing windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import MARKET_TZ
from src.feature_engineering import (
    add_forecast_errors,
    add_residual_demand,
    add_time_features,
    build_features,
    daily_summary,
    flag_spikes,
    net_flows_by_border,
    price_spread,
    rolling_mean,
    rolling_volatility,
    volatility_regime,
    zscore,
)
from src.utils import same_hour_lag


def _frame(n_hours: int = 24 * 60, start: str = "2024-01-01", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_hours, freq="h", tz=MARKET_TZ)
    load = 50_000 + 5_000 * np.sin(np.arange(n_hours) / 24 * 2 * np.pi) + rng.normal(0, 500, n_hours)
    wind = np.clip(15_000 + rng.normal(0, 4_000, n_hours), 0, None)
    solar = np.clip(10_000 * np.sin((idx.hour - 6) / 12 * np.pi), 0, None)
    return pd.DataFrame(
        {
            "price": 60 + (load - wind - solar - 25_000) / 500 + rng.normal(0, 5, n_hours),
            "load_actual": load,
            "load_forecast": load - rng.normal(0, 600, n_hours),
            "wind_actual": wind,
            "wind_forecast": np.clip(wind - rng.normal(0, 1_500, n_hours), 0, None),
            "solar_actual": solar,
            "solar_forecast": solar * 0.95,
        },
        index=idx,
    )


def test_residual_demand_is_load_minus_wind_minus_solar():
    df = add_residual_demand(_frame(48))
    expected = df["load_actual"] - df["wind_actual"] - df["solar_actual"]
    pd.testing.assert_series_equal(df["residual_demand"], expected, check_names=False)
    expected_fc = df["load_forecast"] - df["wind_forecast"] - df["solar_forecast"]
    pd.testing.assert_series_equal(df["residual_demand_forecast"], expected_fc, check_names=False)


def test_forecast_error_sign_conventions():
    df = pd.DataFrame(
        {
            "load_actual": [100.0], "load_forecast": [90.0],      # demand above forecast
            "wind_actual": [20.0], "wind_forecast": [30.0],       # wind under-delivered
            "solar_actual": [5.0], "solar_forecast": [5.0],
        }
    )
    out = add_forecast_errors(df).iloc[0]
    assert out["load_error"] == 10
    assert out["wind_error"] == -10
    assert out["renewable_error"] == -10
    # Both surprises make the system shorter: residual-demand error must be positive.
    assert out["residual_demand_error"] == 20


def test_zscore_uses_only_past_observations():
    s = pd.Series(np.arange(20, dtype=float) ** 1.5)
    z = zscore(s, window=5, min_periods=5)
    t = 10
    past = s.iloc[t - 5 : t]
    expected = (s.iloc[t] - past.mean()) / past.std()
    assert z.iloc[t] == pytest.approx(expected)
    # Changing the future must not change the past.
    s2 = s.copy()
    s2.iloc[t + 1 :] = 1e6
    pd.testing.assert_series_equal(zscore(s2, 5, 5).iloc[: t + 1], z.iloc[: t + 1])


def test_rolling_mean_exclude_current():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rolling_mean(s, 2, min_periods=2, exclude_current=True).iloc[4] == pytest.approx(3.5)
    assert rolling_mean(s, 2, min_periods=2).iloc[4] == pytest.approx(4.5)


def test_rolling_volatility_is_in_absolute_price_changes():
    s = pd.Series([10.0, -10.0, 10.0, -10.0, 10.0])  # swings through zero: log returns undefined
    vol = rolling_volatility(s, 4, min_periods=4)
    assert vol.iloc[-1] == pytest.approx(pd.Series([-20.0, 20.0, -20.0, 20.0]).std())


def test_same_hour_lag_handles_spring_dst():
    idx = pd.date_range("2024-03-30 00:00", "2024-04-01 23:00", freq="h", tz=MARKET_TZ)
    s = pd.Series(idx.day * 100 + idx.hour, index=idx, dtype=float)
    lag = same_hour_lag(s)
    # 31 March has no 02:00, so 1 April 02:00 has no reference hour.
    assert np.isnan(lag[pd.Timestamp("2024-04-01 02:00", tz=MARKET_TZ)])
    # 31 March 03:00 (first hour after the switch) compares with 30 March 03:00, not 02:00.
    assert lag[pd.Timestamp("2024-03-31 03:00", tz=MARKET_TZ)] == 3003
    assert lag[pd.Timestamp("2024-04-01 03:00", tz=MARKET_TZ)] == 3103


def test_same_hour_lag_handles_autumn_dst():
    idx = pd.date_range("2024-10-26 22:00", "2024-10-28 05:00", freq="h", tz=MARKET_TZ)
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    lag = same_hour_lag(s)
    target = pd.Timestamp("2024-10-28 02:00", tz=MARKET_TZ)
    first_two_am = s[(s.index.day == 27) & (s.index.hour == 2)].iloc[0]
    assert lag[target] == first_two_am


def test_peak_and_weekend_flags():
    idx = pd.DatetimeIndex(
        ["2024-01-08 08:00", "2024-01-08 19:00", "2024-01-08 20:00", "2024-01-13 12:00"]
    ).tz_localize(MARKET_TZ)
    df = add_time_features(pd.DataFrame({"price": [1.0, 2, 3, 4]}, index=idx))
    assert df["is_peak"].tolist() == [True, True, False, False]
    assert df["is_weekend"].tolist() == [False, False, False, True]


def test_spike_flags_percentile_and_threshold():
    price = pd.Series(np.arange(1, 101, dtype=float))
    mask, level = flag_spikes(price, percentile=0.9)
    assert mask.sum() == 10 and level == pytest.approx(price.quantile(0.9))
    mask, level = flag_spikes(price, threshold=95)
    assert mask.sum() == 5 and level == 95


def test_negative_price_flag_and_daily_summary_dst():
    df = build_features(_frame(24 * 100, start="2024-02-01"))
    df.loc[df.index[5], "price"] = -12.0
    df = build_features(df[["price", "load_actual", "load_forecast", "wind_actual", "wind_forecast",
                            "solar_actual", "solar_forecast"]])
    assert df["is_negative"].iloc[5]
    daily = daily_summary(df)
    assert daily.loc[pd.Timestamp("2024-03-31"), "hours"] == 23
    assert daily.loc[pd.Timestamp("2024-03-30"), "hours"] == 24


def test_volatility_regime_has_no_lookahead():
    vol = pd.Series(np.random.default_rng(1).gamma(2, 2, 24 * 40))
    regime = volatility_regime(vol, min_history=24 * 14)
    vol2 = vol.copy()
    vol2.iloc[24 * 30 :] *= 50
    regime2 = volatility_regime(vol2, min_history=24 * 14)
    pd.testing.assert_series_equal(regime.iloc[: 24 * 30], regime2.iloc[: 24 * 30])
    assert set(regime.unique()) <= {"low", "normal", "high", "n/a"}


def test_price_spread_and_net_flows():
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz=MARKET_TZ)
    assert price_spread(pd.Series([50.0, 60.0], index=idx), pd.Series([55.0, 40.0], index=idx)).tolist() == [5.0, -20.0]
    flows = pd.DataFrame(
        {
            "timestamp": list(idx) * 2,
            "from_zone": ["FR", "FR", "DE_LU", "DE_LU"],
            "to_zone": ["DE_LU", "DE_LU", "FR", "FR"],
            "flow_mw": [1000.0, 0.0, 0.0, 2500.0],
        }
    )
    net = net_flows_by_border(flows, "DE_LU")
    assert net["FR"].tolist() == [1000.0, -2500.0]  # + import, - export


def test_build_features_has_expected_columns():
    df = build_features(_frame())
    for col in ["residual_demand", "residual_demand_error", "price_change_24h", "price_momentum",
                "residual_demand_z", "vol_regime", "is_peak", "date"]:
        assert col in df
