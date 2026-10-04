"""CFTC Commitments of Traders (COT) client - public Socrata API, no key needed.

Dataset: Disaggregated Futures-Only report (publicreporting.cftc.gov, id 72hh-3qpy).
Positions are as of Tuesday and published the following Friday at 15:30 ET,
so a Tuesday snapshot is only *known* three days later.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

DISAGGREGATED_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
WTI_CODE = "067651"  # WTI-PHYSICAL, NYMEX (light sweet crude)
PAGE_SIZE = 5000
RELEASE_LAG = pd.Timedelta(days=3)  # Tuesday positions -> Friday publication

FIELDS = {
    "report_date_as_yyyy_mm_dd": "report_date",
    "open_interest_all": "open_interest",
    "m_money_positions_long_all": "mm_long",
    "m_money_positions_short_all": "mm_short",
    "m_money_positions_spread": "mm_spread",
    "prod_merc_positions_long": "producer_long",
    "prod_merc_positions_short": "producer_short",
    "swap_positions_long_all": "swap_long",
    "swap__positions_short_all": "swap_short",
}


class CftcError(RuntimeError):
    pass


def get_disaggregated(contract_code: str = WTI_CODE, start: str | pd.Timestamp | None = None,
                      session: requests.Session | None = None, timeout: int = 60) -> pd.DataFrame:
    session = session or requests.Session()
    where = f"cftc_contract_market_code='{contract_code}'"
    if start is not None:
        where += f" AND report_date_as_yyyy_mm_dd >= '{pd.Timestamp(start):%Y-%m-%dT00:00:00}'"
    rows: list[dict] = []
    offset = 0
    while True:
        params = {"$where": where, "$order": "report_date_as_yyyy_mm_dd ASC", "$limit": PAGE_SIZE, "$offset": offset,
                  "$select": ",".join(FIELDS)}
        for attempt in range(4):
            try:
                resp = session.get(DISAGGREGATED_URL, params=params, timeout=timeout)
                if resp.status_code < 500:
                    break
            except requests.RequestException:
                pass
            time.sleep(2**attempt)
        else:
            raise CftcError("CFTC API unreachable after retries.")
        if not resp.ok:
            raise CftcError(f"CFTC API error HTTP {resp.status_code}: {resp.text[:200]}")
        batch = resp.json()
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return parse_cot_rows(rows)


def parse_cot_rows(rows: list[dict]) -> pd.DataFrame:
    cols = list(FIELDS.values())[1:]
    if not rows:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="report_date"))
    df = pd.DataFrame(rows).rename(columns=FIELDS)
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.normalize()
    for c in cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    out = df.set_index("report_date")[cols].sort_index()
    return out[~out.index.duplicated(keep="last")]
