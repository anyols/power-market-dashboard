"""Sanity checks on the committed data store.

Run by the daily GitHub Action *before* committing refreshed data, so a broken
API response can never silently corrupt the published dashboard.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.crude.config import T_COT, T_DAILY, T_WEEKLY
from src.data_store import STORE_DIR, read_status, read_table


def _table(name: str, utc: bool = False) -> pd.DataFrame:
    df = read_table(name, STORE_DIR, utc=utc)
    if df is None:
        pytest.skip(f"{name} not in store")
    return df


def test_status_file_is_valid_json():
    assert isinstance(read_status(), dict)


@pytest.mark.parametrize("name", [T_DAILY, T_WEEKLY, T_COT])
def test_crude_tables_are_clean(name):
    df = _table(name)
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert df.index.max() <= pd.Timestamp.now() + pd.Timedelta(days=7)


def test_crude_prices_in_plausible_range():
    df = _table(T_DAILY)
    for col in [c for c in ["wti_spot", "brent_spot", "cl1", "cl2", "cl3", "cl4"] if c in df]:
        s = df[col].dropna()
        assert s.between(-60, 400).all(), col  # WTI briefly settled at -$37.63 in April 2020


def test_weekly_balances_plausible():
    df = _table(T_WEEKLY)
    if "crude_stocks" in df:
        assert df["crude_stocks"].dropna().between(150_000, 800_000).all()
    if "refinery_utilization" in df:
        assert df["refinery_utilization"].dropna().between(40, 105).all()


def test_power_tables_clean():
    if not (STORE_DIR / "power").exists():
        pytest.skip("no power data in store")
    for path in (STORE_DIR / "power").glob("*.csv"):
        df = read_table(f"power/{path.stem}", STORE_DIR, utc=True)
        assert not df.index.duplicated().any(), path.name
        if "price" in df:
            assert df["price"].dropna().between(-500, 4000).all(), path.name
