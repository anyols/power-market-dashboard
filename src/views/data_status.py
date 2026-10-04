"""Data pipeline status: what the morning job refreshed, and when."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_store import read_status, status_frame


def render(_ctx=None) -> None:
    st.title("Data Pipeline Status")
    st.caption("Every morning a scheduled GitHub Action runs `scripts/daily_update.py`, fetches new data from each API, "
               "merges it into `data/store/` and commits it. Streamlit Cloud redeploys from the repository, so the app "
               "itself needs no API keys.")
    status = read_status()
    if not status:
        st.warning("No refresh has run yet: the store is empty and the dashboard is using synthetic sample data. "
                   "Run `python scripts/daily_update.py` locally or trigger the GitHub Action.")
    else:
        df = status_frame(status)
        df.insert(0, "state", df["ok"].map({True: "✅", False: "⚠️"}))
        st.dataframe(df.drop(columns=["ok"]), column_config={
            "rows": st.column_config.NumberColumn(format="%d"),
            "message": st.column_config.TextColumn(width="large"),
        })
        last = pd.to_datetime(df["last_attempt_utc"]).max()
        st.caption(f"Most recent refresh attempt: {last:%Y-%m-%d %H:%M} UTC.")

    st.subheader("Sources and release rhythm")
    st.markdown(
        """
| Source | Data | Key | Published |
|---|---|---|---|
| ENTSO-E Transparency | Day-ahead prices, load, wind, solar, flows | `ENTSOE_API_TOKEN` | Day-ahead prices ~13:00 CET for the next day; actuals within hours |
| EIA API v2 | WTI/Brent spot, weekly crude & product balances | `EIA_API_KEY` | Weekly report Wed 10:30 ET; daily prices with a lag of several days |
| EIA API v2 | NYMEX futures (contracts 1–4) | `EIA_API_KEY` | **Discontinued after 5 Apr 2024** (history only) |
| CFTC COT | Managed-money positioning (WTI) | none | Fri 15:30 ET, positions as of Tuesday |
| Yahoo Finance | Front-month and individual WTI contracts | none | Live, unofficial; fetched by the app, never stored |
"""
    )
    st.caption("A job run is marked failed if any source fails, but whatever succeeded is still committed, so one "
               "broken API never blocks the others.")
