"""Cleaning, validation and the sample-data loader."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings
from src.data_cleaning import (
    CORE_COLUMNS,
    align_to_hourly_grid,
    clean_market_frame,
    fill_short_gaps,
    to_hourly,
)
from src.data_loader import (
    SAMPLE_NOTICE,
    load_market_data,
    sample_metadata,
    sample_period,
)


def test_short_gaps_filled_long_gaps_kept():
    idx = pd.date_range("2024-01-01", periods=12, freq="h", tz="Europe/Brussels")
    s = pd.Series(np.arange(12, dtype=float), index=idx)
    s.iloc[2:4] = np.nan  # 2h gap -> fill
    s.iloc[6:11] = np.nan  # 5h gap -> keep entirely (no partial filling)
    out, filled = fill_short_gaps(s.to_frame("x"), max_gap_h=3)
    assert out["x"].iloc[2:4].tolist() == [2.0, 3.0]
    assert out["x"].iloc[6:11].isna().all()
    assert filled == {"x": 2}


def test_to_hourly_averages_quarter_hours_across_dst():
    idx = pd.date_range("2024-03-31 00:00", "2024-03-31 04:45", freq="15min", tz="Europe/Brussels")
    s = pd.Series(1.0, index=idx)
    hourly = to_hourly(s)
    assert len(hourly) == 4  # 02:00 does not exist on the spring DST day
    assert (hourly == 1.0).all()


def test_align_grid_has_23_hours_on_spring_dst():
    df = pd.DataFrame({"price": [1.0]}, index=pd.DatetimeIndex(["2024-03-31 00:00"]).tz_localize("Europe/Brussels"))
    grid = align_to_hourly_grid(df, "2024-03-31", "2024-03-31")
    assert len(grid) == 23


def test_validation_flags_out_of_bounds_and_gaps():
    idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="Europe/Brussels")
    raw = pd.DataFrame({c: 100.0 for c in CORE_COLUMNS}, index=idx)
    raw.loc[idx[3], "price"] = 9999
    raw.loc[idx[5:12], "load_actual"] = np.nan
    _, report = clean_market_frame(raw)
    text = " ".join(report.warnings)
    assert "SDAC limits" in text
    assert "load_actual" in text
    assert report.longest_gap_h["load_actual"] == 7


def test_sample_loader_shape_and_labels(monkeypatch):
    monkeypatch.setenv("DATA_SOURCE", "sample")
    first, last = sample_period()
    md = load_market_data("DE_LU", first, last, warmup_days=0, settings=Settings(data_source="sample"))
    assert md.is_sample and md.notices[0] == SAMPLE_NOTICE
    assert list(md.frame.columns) == CORE_COLUMNS
    assert str(md.frame.index.tz) == "Europe/Brussels"
    assert md.frame.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert md.has_flows
    assert "SYNTHETIC" in sample_metadata()["notice"]


def test_auto_source_without_token_uses_sample():
    md = load_market_data("FR", "2024-02-01", "2024-02-10", settings=Settings(data_source="auto", entsoe_api_token=None))
    assert md.source == "sample"


def test_injected_long_gap_is_reported():
    md = load_market_data("BE", "2024-03-01", "2024-03-10", warmup_days=0, settings=Settings(data_source="sample"))
    assert any("load_actual" in w for w in md.quality.warnings)
    assert md.quality.interpolated.get("load_actual", 0) == 0  # the Feb gap is outside this window
