"""About: data sources, how it updates, caveats."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_store import read_status, status_frame


def render(_=None) -> None:
    st.title("About")
    st.markdown(
        """
A side project to keep the commodity markets I follow in one place, and to try a few ideas properly with real
data. Code: [github.com/anyols/power-market-dashboard](https://github.com/anyols/power-market-dashboard).

**Where the numbers come from**

| | Source | Updated |
|---|---|---|
| Commodity & stock prices | Yahoo Finance (front-month futures, delayed) | when you open the app, cached 1h |
| EU power prices, load, wind, solar, flows | ENTSO-E Transparency Platform | daily, by a GitHub Action |
| US crude, Cushing, products, refinery runs | EIA weekly report | daily check, new data on Wednesdays |
| Hedge fund positioning | CFTC Commitments of Traders | daily check, new data on Fridays |

Yahoo data isn't stored anywhere - it's fetched live. Everything else is committed to the repo by the morning job,
so the app itself doesn't need API keys.

**Caveats**

- Yahoo's continuous futures jump when contracts roll. The oil backtest builds a roll-adjusted WTI series from the
  CME expiry calendar instead.
- EIA stopped publishing NYMEX futures prices after April 2024, so the official curve history stops there; the live
  curve comes from Yahoo.
- Pages that don't have real data yet fall back to clearly-labelled synthetic sample data (power, until an
  ENTSO-E token is added).
- Nothing here is investment advice.
"""
    )
    st.markdown("**Last refresh of the stored data**")
    status = read_status()
    if status:
        df = status_frame(status)
        df.insert(0, "ok", df.pop("ok").map({True: "✓", False: "✗"}))
        st.dataframe(df, column_config={"rows": st.column_config.NumberColumn(format="%d"),
                                        "message": st.column_config.TextColumn(width="large")})
        st.caption(f"Last attempt {pd.to_datetime(df['last_attempt_utc']).max():%d %b %Y %H:%M} UTC.")
    else:
        st.caption("The refresh job hasn't run yet.")
