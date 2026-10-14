"""Home: the whole commodity complex on one screen."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.config import PROJECT_ROOT
from src.markets import ASSETS, COMMODITY_GROUPS, performance_table
from src.views.common import cached_crude, cached_markets, mb, pct

NOTES_FILE = PROJECT_ROOT / "notes" / "watchlist.md"
TILES_PER_ROW = 6


def _tiles(prices: pd.DataFrame, perf: pd.DataFrame, group: str) -> None:
    keys = [k for k in perf.index if perf.loc[k, "group"] == group]
    for i in range(0, len(keys), TILES_PER_ROW):
        cols = st.columns(TILES_PER_ROW)
        for col, k in zip(cols, keys[i:i + TILES_PER_ROW]):
            a, row = ASSETS[k], perf.loc[k]
            spark = prices[k].dropna().iloc[-63:].round(4).tolist()
            col.metric(
                a.name, f"{row['last']:,.{a.decimals}f}",
                delta=pct(row["1D"], 1, signed=True) if pd.notna(row["1D"]) else None,
                chart_data=spark, chart_type="line", border=True,
                help=f"{a.ticker} · {a.unit} · last close {row['date']:%d %b}",
            )


def _movers(perf: pd.DataFrame) -> str:
    wk = perf["1W"].dropna().sort_values()
    up = " · ".join(f"{perf.loc[k, 'name']} {pct(v, 1, signed=True)}" for k, v in wk.iloc[::-1][:3].items())
    down = " · ".join(f"{perf.loc[k, 'name']} {pct(v, 1, signed=True)}" for k, v in wk[:3].items())
    return f"**Up most this week:** {up}  \n**Down most:** {down}"


def render(_=None) -> None:
    mp = cached_markets()
    st.title("Spot & Spread")
    if mp.prices.empty:
        st.error("Couldn't get prices from Yahoo right now. " + " ".join(mp.errors))
        return
    st.caption(f"Energy, metals and ags in one place · last close {mp.asof:%a %d %b %Y} · "
               f"{'synthetic data (offline mode)' if mp.is_synthetic else 'front-month futures via Yahoo, delayed'}")

    commodities = [a.key for a in ASSETS.values() if a.group in COMMODITY_GROUPS]
    perf = performance_table(mp.prices, commodities)
    for group in COMMODITY_GROUPS:
        st.markdown(f"#### {group}")
        _tiles(mp.prices, perf, group)

    st.markdown(_movers(perf))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Returns")
        st.plotly_chart(charts.returns_heatmap(perf, ["1W", "1M", "YTD", "1Y"]), key="board_heat")
    with right:
        st.markdown("#### Where in the 52-week range")
        st.plotly_chart(charts.range_strip(perf), key="board_range")
    st.caption("Continuous front-month contracts jump at rolls (nat gas especially), so treat 1-month+ numbers "
               "on those as approximate.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        _oil_card()
    with c2:
        _notes_card()


def _oil_card() -> None:
    data, _, weekly, _ = cached_crude()
    w = weekly.dropna(subset=["crude_stocks"])
    if w.empty:
        return
    last = w.iloc[-1]
    with st.container(border=True):
        st.markdown(f"**Latest EIA report** · week to {last.name:%d %b}"
                    + (" · *synthetic*" if data.is_sample else ""))
        a, b, c = st.columns(3)
        a.metric("US crude stocks", mb(last["crude_stocks"]), delta=mb(last["crude_stocks_chg"], True),
                 delta_color="inverse")
        b.metric("vs 5y average", pct(last["crude_stocks_vs_5y_pct"], 1, signed=True))
        c.metric("Cushing", mb(last["cushing_stocks"]), delta=mb(last["cushing_stocks_chg"], True),
                 delta_color="inverse")
        st.caption("More on the Oil page.")


def _notes_card() -> None:
    with st.container(border=True):
        st.markdown("**What I'm watching**")
        if NOTES_FILE.exists():
            st.markdown(NOTES_FILE.read_text(encoding="utf-8"))
        else:
            st.caption("Add notes in notes/watchlist.md")
