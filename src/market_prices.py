"""Indicative daily futures prices from Yahoo Finance (unofficial).

Yahoo's continuous front-month tickers (CL=F, BZ=F) give same-day closes,
which official free sources do not. Caveats, shown in the UI:
* unofficial and unsupported - it can break or be rate-limited at any time;
* the continuous series jumps at contract rolls on Yahoo's own schedule;
* Yahoo's terms restrict redistribution, so these prices are fetched live by
  the app and cached briefly - never written to the committed data store.
Research and backtests use the official EIA futures series instead.
"""

from __future__ import annotations

import pandas as pd

YAHOO_TICKERS = {"wti_front": "CL=F", "brent_front": "BZ=F"}


class MarketPriceError(RuntimeError):
    pass


def fetch_yahoo_closes(tickers: dict[str, str] | None = None, start: str = "2015-01-01") -> pd.DataFrame:
    """Daily closes, one column per name in `tickers`, indexed by trading date."""
    tickers = tickers or YAHOO_TICKERS
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise MarketPriceError("yfinance is not installed.") from exc
    try:
        raw = yf.download(list(tickers.values()), start=start, progress=False, auto_adjust=False, threads=False)
    except Exception as exc:  # yfinance raises a variety of network errors
        raise MarketPriceError(f"Yahoo Finance download failed: {exc}") from exc
    return tidy_yahoo_frame(raw, tickers)


def tidy_yahoo_frame(raw: pd.DataFrame, tickers: dict[str, str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise MarketPriceError("Yahoo Finance returned no data (possibly rate-limited).")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": next(iter(tickers.values()))})
    inverse = {v: k for k, v in tickers.items()}
    out = close.rename(columns=inverse)[[c for c in tickers if c in close.rename(columns=inverse)]]
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.index.name = "date"
    out.columns.name = None
    return out.dropna(how="all").astype(float)
