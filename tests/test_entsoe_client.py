"""ENTSO-E client: XML parsing, resolution handling, caching and error paths (no network)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_cleaning import to_hourly
from src.entsoe_client import (
    EntsoeAuthError,
    EntsoeClient,
    EntsoeError,
    ResponseCache,
    collapse_to_series,
    month_chunks,
    parse_resolution,
    parse_timeseries_xml,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fx(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeResponse:
    def __init__(self, text: str, status: int = 200):
        self.text, self.status_code = text, status

    @property
    def ok(self) -> bool:
        return self.status_code < 400


class FakeSession:
    def __init__(self, body: str, status: int = 200):
        self.body, self.status, self.calls = body, status, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return FakeResponse(self.body, self.status)


def test_parse_resolution():
    assert parse_resolution("PT15M") == pd.Timedelta(minutes=15)
    assert parse_resolution("PT60M") == pd.Timedelta(hours=1)
    assert parse_resolution("PT1H") == pd.Timedelta(hours=1)
    assert parse_resolution("P1D") == pd.Timedelta(days=1)
    with pytest.raises(EntsoeError):
        parse_resolution("P1M2")


def test_a03_curve_compression_is_forward_filled():
    long = parse_timeseries_xml(fx("prices_a03.xml"), "price.amount")
    s = collapse_to_series(long, "price")
    assert s.tolist() == [50.0, 50.0, 45.5, 45.5]
    assert s.index[0] == pd.Timestamp("2024-01-01T23:00Z")
    assert str(s.index.tz) == "UTC"


def test_15min_prices_preferred_and_averaged_to_hourly():
    s = collapse_to_series(parse_timeseries_xml(fx("prices_15min.xml"), "price.amount"), "price")
    # The duplicate hourly series (999) must be dropped in favour of the 15-minute one.
    assert 999 not in s.tolist()
    hourly = to_hourly(s)
    assert hourly.tolist() == [86.0, -4.0]
    assert str(hourly.index.tz) == "Europe/Brussels"
    assert hourly.index[0] == pd.Timestamp("2025-10-02 00:00", tz="Europe/Brussels")


def test_load_quantity_parsing():
    s = collapse_to_series(parse_timeseries_xml(fx("load_actual.xml"), "quantity"), "load")
    assert len(s) == 8
    assert to_hourly(s).tolist() == [41500.0, 44000.0]


def test_no_data_acknowledgement_returns_empty():
    assert parse_timeseries_xml(fx("ack_no_data.xml"), "price.amount").empty


def test_other_acknowledgements_raise():
    with pytest.raises(EntsoeError, match="exceeds allowed limit"):
        parse_timeseries_xml(fx("ack_error.xml"), "price.amount")


def test_month_chunks_cover_range():
    start, end = pd.Timestamp("2024-01-15", tz="UTC"), pd.Timestamp("2024-03-02", tz="UTC")
    chunks = month_chunks(start, end)
    assert chunks[0][0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert chunks[-1][1] == pd.Timestamp("2024-04-01", tz="UTC")
    assert len(chunks) == 3


def test_client_requires_token():
    with pytest.raises(EntsoeAuthError):
        EntsoeClient(None)


def test_client_request_parameters_and_cache(tmp_path):
    session = FakeSession(fx("prices_a03.xml"))
    client = EntsoeClient("dummy-token", cache=ResponseCache(tmp_path / "c.sqlite"), session=session)
    s = client.get_day_ahead_prices("DE_LU", "2024-01-01T23:00Z", "2024-01-02T03:00Z")
    assert s.tolist() == [50.0, 50.0, 45.5, 45.5]
    params = session.calls[0]
    assert params["documentType"] == "A44"
    assert params["in_Domain"] == params["out_Domain"] == "10Y1001A1001A82H"
    assert params["securityToken"] == "dummy-token"
    # Second identical call is served from the cache.
    client.get_day_ahead_prices("DE_LU", "2024-01-01T23:00Z", "2024-01-02T03:00Z")
    assert len(session.calls) == 1


def test_cache_key_excludes_token(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    a = EntsoeClient("token-a", cache=cache, session=FakeSession(fx("prices_a03.xml")))
    a.get_day_ahead_prices("DE_LU", "2024-01-01T23:00Z", "2024-01-02T03:00Z")
    b_session = FakeSession(fx("prices_a03.xml"))
    EntsoeClient("token-b", cache=cache, session=b_session).get_day_ahead_prices(
        "DE_LU", "2024-01-01T23:00Z", "2024-01-02T03:00Z"
    )
    assert b_session.calls == []


def test_generation_keeps_only_generation_series(tmp_path):
    client = EntsoeClient("t", cache=None, session=FakeSession(fx("generation_wind.xml")))
    gen = client.get_wind_solar_generation("DE_LU", "2024-01-01T23:00Z", "2024-01-02T01:00Z")
    assert gen["wind_onshore"].tolist() == [12000.0, 13000.0]


def test_http_401_raises_auth_error():
    client = EntsoeClient("bad", session=FakeSession("Unauthorized", status=401))
    with pytest.raises(EntsoeAuthError):
        client.get_actual_load("FR", "2024-01-01", "2024-01-02")


def test_unknown_zone_raises():
    client = EntsoeClient("t", session=FakeSession(fx("ack_no_data.xml")))
    with pytest.raises(EntsoeError, match="Unknown bidding zone"):
        client.get_day_ahead_prices("XX", "2024-01-01", "2024-01-02")
