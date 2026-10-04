"""EIA / CFTC / Yahoo clients, the data store and the daily update job - all offline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import src.data_store as store
from src.cftc_client import parse_cot_rows
from src.eia_client import EiaClient, EiaError
from src.market_prices import MarketPriceError, tidy_yahoo_frame


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


class FakeEiaSession:
    """Serves a daily series newest-first in pages, like the real seriesid endpoint."""

    def __init__(self, n=12000, status=200):
        self.dates = pd.bdate_range(end="2024-04-05", periods=n)[::-1]
        self.calls, self.status = [], status

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        if self.status != 200:
            return FakeResponse({"error": "rate limited"}, self.status)
        off, length = params["offset"], params["length"]
        rows = [{"period": d.strftime("%Y-%m-%d"), "value": 50 + i % 7} for i, d in enumerate(self.dates[off:off + length])]
        return FakeResponse({"response": {"total": len(self.dates), "data": rows}})


def test_eia_stops_paging_once_past_start():
    session = FakeEiaSession()
    s = EiaClient("key", session=session).get_series("PET.RCLC1.D", start="2015-01-01")
    assert s.index.min() >= pd.Timestamp("2015-01-01")
    assert s.index.is_monotonic_increasing
    assert len(session.calls) == 1  # ~2,400 business days since 2015 fit in one 5,000-row page
    assert session.calls[0]["sort[0][direction]"] == "desc"
    deep = FakeEiaSession()
    s = EiaClient("key", session=deep).get_series("PET.RCLC1.D", start="2000-01-01")
    assert len(deep.calls) == 2 and s.index.min() >= pd.Timestamp("2000-01-01")  # pages until it passes start


def test_eia_last_n_costs_one_request():
    session = FakeEiaSession()
    s = EiaClient("key", session=session).get_series("PET.RWTC.D", last_n=90)
    assert len(s) == 90 and len(session.calls) == 1


def test_eia_error_payload_raises():
    class ErrSession:
        def get(self, *a, **k):
            return FakeResponse({"error": {"code": "API_KEY_INVALID", "message": "bad key"}}, 403)

    with pytest.raises(EiaError, match="bad key"):
        EiaClient("bad", session=ErrSession()).get_series("PET.RWTC.D")


def test_eia_demo_key_rate_limit_fails_fast():
    with pytest.raises(EiaError, match="rate limit"):
        EiaClient("DEMO_KEY", session=FakeEiaSession(status=429)).get_series("PET.RWTC.D")


def test_eia_requires_key():
    with pytest.raises(EiaError):
        EiaClient(None)


def test_parse_cot_rows():
    rows = [{"report_date_as_yyyy_mm_dd": "2026-09-29T00:00:00.000", "open_interest_all": "1878576",
             "m_money_positions_long_all": "209028", "m_money_positions_short_all": "129436",
             "m_money_positions_spread": "302987", "prod_merc_positions_long": "615887",
             "prod_merc_positions_short": "296348", "swap_positions_long_all": "113558",
             "swap__positions_short_all": "575715"}]
    df = parse_cot_rows(rows)
    assert df.index[0] == pd.Timestamp("2026-09-29")
    assert df.loc["2026-09-29", "mm_long"] - df.loc["2026-09-29", "mm_short"] == 209028 - 129436
    assert parse_cot_rows([]).empty


def test_tidy_yahoo_frame():
    idx = pd.DatetimeIndex(["2026-10-01", "2026-10-02"])
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["BZ=F", "CL=F"]])
    raw = pd.DataFrame(np.arange(8, dtype=float).reshape(2, 4), index=idx, columns=cols)
    out = tidy_yahoo_frame(raw, {"wti_front": "CL=F", "brent_front": "BZ=F"})
    assert list(out.columns) == ["wti_front", "brent_front"]
    assert out.loc["2026-10-02", "wti_front"] == 5.0
    with pytest.raises(MarketPriceError):
        tidy_yahoo_frame(pd.DataFrame(), {"wti_front": "CL=F"})


# ---------------------------------------------------------------------------
# Store + daily job
# ---------------------------------------------------------------------------
def test_merge_update_prefers_fresh_values():
    old = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3))
    new = pd.DataFrame({"a": [20.0, 40.0]}, index=pd.date_range("2024-01-02", periods=2))
    merged = store.merge_update(old, new)
    assert merged["a"].tolist() == [1.0, 20.0, 40.0]


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(store, "STATUS_PATH", tmp_path / "status.json")
    return tmp_path


def test_status_bookkeeping(tmp_store):
    store.update_status("x", ok=True, source="S", latest="2024-01-01", rows=3, path=tmp_store / "status.json")
    store.update_status("x", ok=False, source="S", message="boom", path=tmp_store / "status.json")
    st = store.read_status(tmp_store / "status.json")["x"]
    assert st["ok"] is False and st["latest_observation"] == "2024-01-01" and st["message"] == "boom"
    assert "last_success_utc" in st


def test_daily_update_isolates_failures(tmp_store, monkeypatch):
    import scripts.daily_update as job

    monkeypatch.setattr(job, "update_status", lambda *a, **k: store.update_status(*a, **k, path=tmp_store / "status.json"))

    class FakeEia:
        def __init__(self, key):
            pass

        def get_series(self, sid, start=None, last_n=None):
            if sid == "PET.RCLC4.D":
                raise EiaError("boom")
            idx = pd.date_range("2024-01-01", periods=30, freq="W-FRI" if sid.endswith(".W") else "B")
            return pd.Series(np.arange(30, dtype=float), index=idx, name=sid)

    monkeypatch.setattr(job, "EiaClient", FakeEia)
    monkeypatch.setenv("EIA_API_KEY", "x")
    assert job.update_eia(full=True) is False  # one series failed ...
    daily = store.read_table("crude/eia_daily", tmp_store)
    assert {"wti_spot", "cl1"} <= set(daily.columns)  # ... but the rest was stored
    status = store.read_status(tmp_store / "status.json")
    assert status["crude/eia_daily"]["ok"] is False and "cl4" in status["crude/eia_daily"]["message"]
    assert status["crude/eia_weekly"]["ok"] is True

    monkeypatch.setattr(job, "get_disaggregated", lambda code, start: parse_cot_rows([
        {"report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000", "open_interest_all": "100",
         "m_money_positions_long_all": "60", "m_money_positions_short_all": "20"}]))
    assert job.update_cot(full=True) is True
    assert store.read_table("crude/cot_wti", tmp_store).shape[0] == 1


def test_daily_update_skips_eia_without_key(tmp_store, monkeypatch):
    import scripts.daily_update as job

    monkeypatch.setattr(job, "update_status", lambda *a, **k: store.update_status(*a, **k, path=tmp_store / "status.json"))
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    assert job.update_eia(full=False) is False
    assert "not configured" in store.read_status(tmp_store / "status.json")["crude/eia_daily"]["message"]


def test_power_store_round_trip(tmp_store):
    """Data written in the store layout is served by load_market_data(source='store')."""
    from src.config import Settings
    from src.data_cleaning import clean_market_frame
    from src.data_loader import load_market_data, load_sample_frame, store_power_zones

    for zone in ["DE_LU", "FR"]:
        frame, _ = clean_market_frame(load_sample_frame(zone), "2024-03-01", "2024-03-31")
        frame.index = frame.index.tz_convert("UTC")
        store.write_table(f"power/{zone}", frame)
    flows = pd.DataFrame({"DE_LU->FR": 1000.0, "FR->DE_LU": 0.0}, index=frame.index)
    store.write_table("power/flows", flows)

    assert store_power_zones() == ["DE_LU", "FR"]
    md = load_market_data("DE_LU", "2024-03-10", "2024-03-20", settings=Settings(data_source="store"))
    assert md.source == "store" and not md.is_sample
    assert md.frame.index.min().date().isoformat() == "2024-03-01"  # warm-up clipped to store coverage
    assert md.has_flows and set(md.flows["to_zone"]) <= {"FR", "DE_LU"}
    assert "FR" in md.zone_prices
