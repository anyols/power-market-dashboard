"""Minimal, dependency-light client for the ENTSO-E Transparency Platform REST API.

Only the handful of endpoints this project needs are implemented, but each
one is implemented properly: chunked requests, a local SQLite response cache,
retries on throttling, namespace-agnostic XML parsing, curve-type A03
compression, and mixed 15/60-minute resolutions.

API reference: "Transparency Platform RESTful API - user guide" (ENTSO-E).
Getting a token: register on transparency.entsoe.eu, then request API access
(see README). The token is read from the ENTSOE_API_TOKEN environment variable.

Endpoint map
------------
=============================  ============  ===========  ==========================
Dataset                        documentType  processType  Domain parameters
=============================  ============  ===========  ==========================
Day-ahead prices               A44           -            in_Domain = out_Domain
Actual total load              A65           A16          outBiddingZone_Domain
Day-ahead load forecast        A65           A01          outBiddingZone_Domain
Actual generation per type     A75           A16          in_Domain (+ psrType)
Wind & solar DA forecast       A69           A01          in_Domain (+ psrType)
Physical cross-border flows    A11           -            in_Domain (to), out_Domain (from)
=============================  ============  ===========  ==========================
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import CACHE_DB_PATH, ZONES

logger = logging.getLogger(__name__)

BASE_URL = "https://web-api.tp.entsoe.eu/api"
PSR_SOLAR = "B16"
PSR_WIND_OFFSHORE = "B18"
PSR_WIND_ONSHORE = "B19"
PSR_COLUMNS = {PSR_SOLAR: "solar", PSR_WIND_OFFSHORE: "wind_offshore", PSR_WIND_ONSHORE: "wind_onshore"}

# Data for recent periods is still being published/revised, so it is only
# cached briefly. Older chunks are cached indefinitely.
RECENT_DATA_TTL_S = 3600
RECENT_WINDOW = pd.Timedelta(days=3)


class EntsoeError(RuntimeError):
    """Any error returned by, or encountered while talking to, the API."""


class EntsoeAuthError(EntsoeError):
    """Missing or rejected security token."""


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
class ResponseCache:
    """Tiny SQLite key-value store for raw XML responses.

    Raw XML (not parsed frames) is cached so parser fixes never require
    re-downloading data.
    """

    def __init__(self, path: Path = CACHE_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "key TEXT PRIMARY KEY, body TEXT NOT NULL, fetched_at REAL NOT NULL)"
            )

    def get(self, key: str, max_age_s: float | None = None) -> str | None:
        with sqlite3.connect(self.path) as con:
            row = con.execute("SELECT body, fetched_at FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        body, fetched_at = row
        if max_age_s is not None and time.time() - fetched_at > max_age_s:
            return None
        return body

    def set(self, key: str, body: str) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute(
                "INSERT OR REPLACE INTO responses (key, body, fetched_at) VALUES (?, ?, ?)",
                (key, body, time.time()),
            )


def cache_key(params: dict) -> str:
    public = {k: v for k, v in params.items() if k != "securityToken"}
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------
_RES_RE = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?$")


def parse_resolution(text: str) -> pd.Timedelta:
    """ISO-8601 duration such as PT15M / PT60M / PT1H / P1D -> Timedelta."""
    m = _RES_RE.match(text.strip()) if text else None
    if not m or not any(m.groups()):
        raise EntsoeError(f"Unsupported resolution '{text}'")
    days, hours, minutes = (int(g) if g else 0 for g in m.groups())
    return pd.Timedelta(days=days, hours=hours, minutes=minutes)


def _strip_namespaces(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _child_text(el: ET.Element, *names: str) -> str | None:
    for name in names:
        found = el.find(name)
        if found is not None and found.text:
            return found.text.strip()
    return None


def parse_acknowledgement(root: ET.Element) -> tuple[str | None, str]:
    code = _child_text(root, "Reason/code")
    text = _child_text(root, "Reason/text") or "Unknown error"
    return code, text


def parse_timeseries_xml(xml_text: str, value_tag: str) -> pd.DataFrame:
    """Parse any ENTSO-E time-series document into a long DataFrame.

    Returns columns: timestamp (UTC), value, resolution_min, psr_type,
    in_domain, out_domain, series. An "Acknowledgement" document saying
    "No matching data found" yields an empty frame; any other
    acknowledgement (e.g. "requested data exceeds allowed limit") raises
    EntsoeError.
    """
    root = _strip_namespaces(ET.fromstring(xml_text))
    if root.tag == "Acknowledgement_MarketDocument":
        code, text = parse_acknowledgement(root)
        # Code 999 is used for every kind of refusal, so the text decides:
        # only "no matching data" means an empty (but valid) answer.
        if "no matching data" in text.lower():
            return _empty_long_frame()
        raise EntsoeError(f"ENTSO-E returned an error (code {code}): {text}")

    frames = []
    for series_no, ts in enumerate(root.iter("TimeSeries")):
        psr_type = _child_text(ts, "MktPSRType/psrType")
        in_domain = _child_text(ts, "inBiddingZone_Domain.mRID", "in_Domain.mRID")
        out_domain = _child_text(ts, "outBiddingZone_Domain.mRID", "out_Domain.mRID")
        curve_type = _child_text(ts, "curveType") or "A01"
        for period in ts.iter("Period"):
            start = pd.Timestamp(_child_text(period, "timeInterval/start"))
            end = pd.Timestamp(_child_text(period, "timeInterval/end"))
            res = parse_resolution(_child_text(period, "resolution") or "")
            n = int((end - start) / res)
            values = np.full(n, np.nan)
            for point in period.iter("Point"):
                fields = {child.tag: child.text for child in point}
                pos = int(fields.get("position", 0))
                raw = fields.get(value_tag)
                if raw is not None and 1 <= pos <= n:
                    values[pos - 1] = float(raw)
            if curve_type == "A03":
                # A03 = variable-sized blocks: a missing position repeats the
                # previous value (ENTSO-E compresses runs of equal prices).
                values = pd.Series(values).ffill().to_numpy()
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.date_range(start, periods=n, freq=res),
                        "value": values,
                        "resolution_min": res / pd.Timedelta(minutes=1),
                        "psr_type": psr_type,
                        "in_domain": in_domain,
                        "out_domain": out_domain,
                        "series": series_no,
                    }
                )
            )
    if not frames:
        return _empty_long_frame()
    return pd.concat(frames, ignore_index=True)


def _empty_long_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex([], tz="UTC"),
            "value": pd.Series(dtype=float),
            "resolution_min": pd.Series(dtype=float),
            "psr_type": pd.Series(dtype=object),
            "in_domain": pd.Series(dtype=object),
            "out_domain": pd.Series(dtype=object),
            "series": pd.Series(dtype=int),
        }
    )


def collapse_to_series(long: pd.DataFrame, name: str) -> pd.Series:
    """One value per timestamp, preferring the finest published resolution.

    Around the 15-minute MTU transition some zones publish both hourly and
    quarter-hourly series; keeping the finest one avoids double counting
    when the result is later averaged to hourly.
    """
    if long.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name=name)
    ordered = long.sort_values(["resolution_min", "series"]).drop_duplicates("timestamp", keep="first")
    s = ordered.set_index("timestamp")["value"].sort_index()
    s.name = name
    return s


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
def _to_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def month_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) into whole UTC calendar months (cache-friendly)."""
    first = start.tz_convert("UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    edges = pd.date_range(first, end + pd.offsets.MonthBegin(1), freq="MS")
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > start and a < end]


class EntsoeClient:
    """Thin wrapper around the ENTSO-E REST API returning pandas objects (UTC)."""

    def __init__(
        self,
        token: str | None,
        timeout: int = 60,
        cache: ResponseCache | None = None,
        session: requests.Session | None = None,
        max_retries: int = 3,
        base_url: str = BASE_URL,
    ):
        if not token:
            raise EntsoeAuthError(
                "No ENTSO-E API token configured. Add ENTSOE_API_TOKEN to your .env file "
                "(see README) or run the dashboard on the bundled sample data."
            )
        self.token = token
        self.timeout = timeout
        self.cache = cache
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.base_url = base_url

    # -- low level ---------------------------------------------------------
    def _request(self, params: dict) -> str:
        query = {**params, "securityToken": self.token}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(self.base_url, params=query, timeout=self.timeout)
            except requests.RequestException as exc:  # network trouble: retry
                last_error = exc
                time.sleep(2**attempt)
                continue
            if resp.status_code == 401:
                raise EntsoeAuthError("ENTSO-E rejected the API token (HTTP 401). Check ENTSOE_API_TOKEN.")
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = EntsoeError(f"HTTP {resp.status_code} from ENTSO-E")
                time.sleep(2**attempt * (5 if resp.status_code == 429 else 1))
                continue
            if resp.status_code == 400 and "Acknowledgement_MarketDocument" in resp.text:
                return resp.text  # parsed (and possibly raised) by the XML parser
            if not resp.ok:
                raise EntsoeError(f"HTTP {resp.status_code} from ENTSO-E: {resp.text[:300]}")
            return resp.text
        raise EntsoeError(f"ENTSO-E request failed after {self.max_retries + 1} attempts: {last_error}")

    def _cached_request(self, params: dict, chunk_end: pd.Timestamp) -> str:
        key = cache_key(params)
        recent = chunk_end > pd.Timestamp.now(tz="UTC") - RECENT_WINDOW
        if self.cache is not None:
            body = self.cache.get(key, max_age_s=RECENT_DATA_TTL_S if recent else None)
            if body is not None:
                return body
        body = self._request(params)
        if self.cache is not None:
            self.cache.set(key, body)
        return body

    def _query(self, params: dict, start, end, value_tag: str) -> pd.DataFrame:
        start, end = _to_utc(start), _to_utc(end)
        frames = []
        for a, b in month_chunks(start, end):
            chunk_params = {**params, "periodStart": a.strftime("%Y%m%d%H%M"), "periodEnd": b.strftime("%Y%m%d%H%M")}
            frames.append(parse_timeseries_xml(self._cached_request(chunk_params, b), value_tag))
        frames = [f for f in frames if not f.empty]
        if not frames:
            return _empty_long_frame()
        long = pd.concat(frames, ignore_index=True)
        return long[(long["timestamp"] >= start) & (long["timestamp"] < end)]

    @staticmethod
    def _eic(zone: str) -> str:
        if zone not in ZONES:
            raise EntsoeError(f"Unknown bidding zone '{zone}'. Configure it in src/config.py.")
        return ZONES[zone].eic

    # -- public endpoints -----------------------------------------------------
    def get_day_ahead_prices(self, zone: str, start, end) -> pd.Series:
        """Day-ahead (SDAC) clearing prices in EUR/MWh."""
        eic = self._eic(zone)
        params = {"documentType": "A44", "in_Domain": eic, "out_Domain": eic, "contract_MarketAgreement.type": "A01"}
        return collapse_to_series(self._query(params, start, end, "price.amount"), "price")

    def get_actual_load(self, zone: str, start, end) -> pd.Series:
        """Actual total load (MW)."""
        params = {"documentType": "A65", "processType": "A16", "outBiddingZone_Domain": self._eic(zone)}
        return collapse_to_series(self._query(params, start, end, "quantity"), "load_actual")

    def get_load_forecast(self, zone: str, start, end) -> pd.Series:
        """Day-ahead total load forecast (MW)."""
        params = {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": self._eic(zone)}
        return collapse_to_series(self._query(params, start, end, "quantity"), "load_forecast")

    def _per_psr(self, base: dict, zone: str, start, end, suffix: str) -> pd.DataFrame:
        out = {}
        for psr, col in PSR_COLUMNS.items():
            long = self._query({**base, "psrType": psr}, start, end, "quantity")
            # A75 documents also contain consumption series (out domain); keep generation only.
            if not long.empty and long["in_domain"].notna().any():
                long = long[long["in_domain"].notna()]
            out[col + suffix] = collapse_to_series(long, col + suffix)
        return pd.DataFrame(out)

    def get_wind_solar_generation(self, zone: str, start, end) -> pd.DataFrame:
        """Actual wind onshore/offshore and solar generation (MW)."""
        base = {"documentType": "A75", "processType": "A16", "in_Domain": self._eic(zone)}
        return self._per_psr(base, zone, start, end, "")

    def get_wind_solar_forecast(self, zone: str, start, end) -> pd.DataFrame:
        """Day-ahead wind onshore/offshore and solar generation forecast (MW)."""
        base = {"documentType": "A69", "processType": "A01", "in_Domain": self._eic(zone)}
        return self._per_psr(base, zone, start, end, "_forecast")

    def get_cross_border_flows(self, zone_from: str, zone_to: str, start, end) -> pd.Series:
        """Physical flow from `zone_from` into `zone_to` (MW, >= 0)."""
        params = {"documentType": "A11", "in_Domain": self._eic(zone_to), "out_Domain": self._eic(zone_from)}
        return collapse_to_series(self._query(params, start, end, "quantity"), f"{zone_from}->{zone_to}")
