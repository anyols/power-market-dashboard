"""Live (indicative) crude prices and forward curve from Yahoo Finance.

Fetched by the app at view time and cached briefly; never stored (Yahoo's
terms restrict redistribution). In sample mode a synthetic equivalent is
built from the sample tables so the pages behave the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.crude.contracts import front_contract
from src.crude.data import CrudeData
from src.market_prices import MarketPriceError, fetch_yahoo_closes

MONTH_CODES = "FGHJKMNQUVXZ"
YAHOO_LABEL = "Yahoo Finance (unofficial, indicative)"


@dataclass
class LiveCrude:
    fronts: pd.DataFrame  # columns wti_front, brent_front (daily closes)
    curve: pd.Series  # WTI futures by contract month label, latest close
    curve_date: pd.Timestamp | None
    source: str
    ok: bool
    message: str = ""

    @property
    def m1_m2(self) -> float:
        return float(self.curve.iloc[0] - self.curve.iloc[1]) if len(self.curve) >= 2 else float("nan")

    @property
    def latest_date(self) -> pd.Timestamp | None:
        return None if self.fronts.empty else self.fronts["wti_front"].dropna().index.max()


def curve_tickers(asof: pd.Timestamp, n: int = 6) -> list[tuple[str, str]]:
    """(label, Yahoo ticker) for the next `n` WTI contracts trading on `asof`."""
    front = front_contract(pd.DatetimeIndex([pd.Timestamp(asof)]))[0]
    out = []
    for k in range(n):
        p = front + k
        out.append((p.strftime("%b-%y"), f"CL{MONTH_CODES[p.month - 1]}{p.year % 100:02d}.NYM"))
    return out


def fetch_live_crude(history_start: str = "2018-01-01", n_contracts: int = 6) -> LiveCrude:
    messages = []
    try:
        fronts = fetch_yahoo_closes(start=history_start)
    except MarketPriceError as exc:
        fronts = pd.DataFrame(columns=["wti_front", "brent_front"])
        messages.append(str(exc))
    curve, curve_date = pd.Series(dtype=float), None
    try:
        asof = fronts.index.max() if not fronts.empty else pd.Timestamp.now().normalize()
        pairs = curve_tickers(asof, n_contracts)
        closes = fetch_yahoo_closes({label: ticker for label, ticker in pairs},
                                    start=(asof - pd.Timedelta(days=10)).strftime("%Y-%m-%d"))
        last = closes.dropna(how="all")
        if not last.empty:
            curve_date = last.index.max()
            curve = last.loc[curve_date][[label for label, _ in pairs if label in last.columns]].dropna()
    except MarketPriceError as exc:
        messages.append(f"curve: {exc}")
    ok = not fronts.empty
    return LiveCrude(fronts, curve, curve_date, YAHOO_LABEL, ok, "; ".join(messages))


def sample_live(data: CrudeData) -> LiveCrude:
    """Synthetic stand-in for the live layer (sample mode only)."""
    from src.config import SAMPLE_DIR
    from src.data_store import read_table

    fronts = read_table("crude/yahoo_daily", SAMPLE_DIR)
    last = data.daily[["cl1", "cl2", "cl3", "cl4"]].dropna().iloc[-1]
    front = front_contract(pd.DatetimeIndex([last.name]))[0]
    labels = [(front + k).strftime("%b-%y") for k in range(4)]
    curve = pd.Series(last.to_numpy(), index=labels)
    return LiveCrude(fronts, curve, last.name, "synthetic sample", True, "")
