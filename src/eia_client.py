"""U.S. Energy Information Administration (EIA) API v2 client.

Uses the v2 "seriesid" endpoint, which accepts the familiar v1 series IDs
(e.g. PET.RWTC.D) and returns JSON. Register for a free key at
https://www.eia.gov/opendata/register.php and set EIA_API_KEY.

Publication rhythm (matters for look-ahead-free research):
* Weekly Petroleum Status Report (stocks, refinery runs, production): data
  for the week ending Friday is released the following Wednesday 10:30 ET
  (Thursday in weeks with a Monday holiday).
* Daily spot and futures prices are published with a lag of several days,
  typically refreshed weekly - fine for research, not a live price feed.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

BASE_URL = "https://api.eia.gov/v2/seriesid/"
PAGE_SIZE = 5000  # API maximum for JSON


class EiaError(RuntimeError):
    pass


class EiaClient:
    def __init__(self, api_key: str | None, session: requests.Session | None = None, timeout: int = 60,
                 max_retries: int = 3, base_url: str = BASE_URL):
        if not api_key:
            raise EiaError("No EIA API key. Register at https://www.eia.gov/opendata/register.php and set EIA_API_KEY.")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url

    def _get(self, series_id: str, params: dict) -> dict:
        url = f"{self.base_url}{series_id}"
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params={**params, "api_key": self.api_key}, timeout=self.timeout)
            except requests.RequestException as exc:
                last = exc
                time.sleep(2**attempt)
                continue
            if resp.status_code == 429 and self.api_key == "DEMO_KEY":
                raise EiaError("EIA DEMO_KEY rate limit reached (HTTP 429). Use your own free EIA_API_KEY.")
            if resp.status_code in (429, 500, 502, 503, 504):
                last = EiaError(f"HTTP {resp.status_code}")
                time.sleep(min(2**attempt * 2, 10))
                continue
            try:
                payload = resp.json()
            except ValueError as exc:
                raise EiaError(f"Non-JSON response from EIA (HTTP {resp.status_code}): {resp.text[:200]}") from exc
            if "error" in payload:
                err = payload["error"]
                msg = err.get("message", err) if isinstance(err, dict) else err
                raise EiaError(f"EIA error for {series_id}: {msg}")
            return payload
        raise EiaError(f"EIA request for {series_id} failed after retries: {last}")

    def get_series(self, series_id: str, start: str | pd.Timestamp | None = None, last_n: int | None = None) -> pd.Series:
        """One series as a float Series indexed by date.

        Pages newest-first and stops as soon as it passes `start` (or has
        `last_n` rows), so an incremental update costs a single request. The
        seriesid endpoint does not filter by date server-side.
        """
        length = min(PAGE_SIZE, last_n) if last_n else PAGE_SIZE
        params = {"sort[0][column]": "period", "sort[0][direction]": "desc", "length": length, "offset": 0}
        start_ts = pd.Timestamp(start) if start is not None else None
        rows: list[dict] = []
        while True:
            payload = self._get(series_id, params)
            response = payload.get("response", {})
            data = response.get("data", [])
            rows.extend(data)
            total = int(response.get("total", len(rows)) or 0)
            if not data or len(rows) >= total or (last_n and len(rows) >= last_n):
                break
            if start_ts is not None and pd.Timestamp(data[-1]["period"]) < start_ts:
                break
            params["offset"] += length
        s = parse_series_rows(rows, series_id)
        return s[s.index >= start_ts] if start_ts is not None else s


def parse_series_rows(rows: list[dict], name: str) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float, name=name, index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(rows)
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(), index=pd.to_datetime(df["period"]), name=name)
    s.index.name = "date"
    return s[~s.index.duplicated(keep="last")].sort_index()
