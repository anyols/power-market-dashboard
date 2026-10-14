"""Oil deep dive: prices, the curve, US inventories and positioning."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.crude.config import LABELS
from src.crude.features import iso_year_week, seasonal_profile
from src.views.common import CrudeCtx, build_crude_context, mb, pct, synthetic_note, usd

STOCKS = [("crude_stocks", "US commercial crude stocks"), ("cushing_stocks", "Cushing, Oklahoma"),
          ("gasoline_stocks", "US gasoline stocks"), ("distillate_stocks", "US distillate stocks")]


def render(_=None) -> None:
    st.title("Oil")
    st.caption("WTI and Brent, the futures curve, the weekly EIA inventory numbers and what hedge funds are doing.")
    years = st.select_slider("Chart window", [1, 2, 3, 5, 10, 16], value=3, format_func=lambda y: f"{y}y",
                             key="oil_years")
    ctx = build_crude_context(years)
    if ctx.data.is_sample:
        synthetic_note("Oil")
    tabs = st.tabs(["Prices & curve", "Inventories", "Positioning"])
    with tabs[0]:
        _prices(ctx)
    with tabs[1]:
        _inventories(ctx)
    with tabs[2]:
        _positioning(ctx)


def _prices(ctx: CrudeCtx) -> None:
    d, live = ctx.daily, ctx.live
    if live.ok and not live.fronts.empty:
        f = live.fronts.dropna(subset=["wti_front"])
        wti, brent, asof = f["wti_front"].iloc[-1], f["brent_front"].dropna().iloc[-1], f.index[-1]
    else:
        wti, brent, asof = d["wti_spot"].dropna().iloc[-1], d["brent_spot"].dropna().iloc[-1], d["wti_spot"].dropna().index[-1]
    m12 = live.m1_m2 if len(live.curve) >= 2 else d["m1_m2"].dropna().iloc[-1]
    k = st.columns(4)
    k[0].metric("WTI", usd(wti), help=f"close {asof:%d %b}", border=True)
    k[1].metric("Brent", usd(brent), border=True)
    k[2].metric("Brent − WTI", usd(brent - wti), border=True)
    k[3].metric("1st − 2nd month", usd(m12, signed=True), border=True,
                help="Positive = backwardation: prompt oil is worth more than later oil, i.e. the market is tight.")

    a, b = st.columns([3, 2])
    if live.ok:
        fr = live.fronts[live.fronts.index >= ctx.start]
        a.plotly_chart(charts.price_lines(fr, {"wti_front": "WTI", "brent_front": "Brent"},
                                          "Front-month futures ($/bbl)"), key="oil_px")
    else:
        sp = d[d.index >= ctx.start]
        a.plotly_chart(charts.price_lines(sp, {"wti_spot": "WTI", "brent_spot": "Brent"}, "Spot ($/bbl, EIA)"),
                       key="oil_px")
    if len(live.curve) >= 2:
        b.plotly_chart(charts.forward_curve_chart(live.curve, f"WTI curve, {live.curve_date:%d %b}"), key="oil_curve")
    spread = d["brent_wti_spot"][d.index >= ctx.start]
    st.plotly_chart(charts.zero_line_chart(spread, "Brent − WTI (spot)", "Brent − WTI"), key="oil_bw")


def _inventories(ctx: CrudeCtx) -> None:
    w = ctx.weekly.dropna(subset=["crude_stocks"])
    latest = w.iloc[-1]
    years = sorted(set(iso_year_week(w.index)[0]), reverse=True)
    year = st.selectbox("Highlight year", years, key="oil_inv_year")
    st.markdown(f"Latest report: week to **{latest.name:%d %b %Y}**. Crude stocks "
                f"{'fell' if latest['crude_stocks_chg'] < 0 else 'rose'} {mb(abs(latest['crude_stocks_chg']))} "
                f"against a usual {mb(latest['crude_stocks_chg_5y_avg'], True)} for the time of year, leaving them "
                f"{pct(latest['crude_stocks_vs_5y_pct'], 1, signed=True)} vs the 5-year average.")
    cols = st.columns(2)
    for i, (col, title) in enumerate(STOCKS):
        prof = seasonal_profile(w[col], year)
        cols[i % 2].plotly_chart(charts.seasonal_band_chart(prof, title, "mb", year, scale=1000), key=f"oil_band_{col}")
    rows = []
    for col in ["crude_stocks", "cushing_stocks", "gasoline_stocks", "distillate_stocks"]:
        rows.append({"": LABELS[col], "level (mb)": latest[col] / 1000, "change (mb)": latest[f"{col}_chg"] / 1000,
                     "usual change (mb)": latest[f"{col}_chg_5y_avg"] / 1000,
                     "vs 5y avg": latest[f"{col}_vs_5y_pct"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "level (mb)": st.column_config.NumberColumn(format="%.1f"),
        "change (mb)": st.column_config.NumberColumn(format="%+.1f"),
        "usual change (mb)": st.column_config.NumberColumn(format="%+.1f"),
        "vs 5y avg": st.column_config.NumberColumn(format="percent"),
    })
    st.caption(f"Refinery utilisation {latest['refinery_utilization']:.1f}% · production "
               f"{latest['crude_production'] / 1000:.2f} mb/d · net imports {latest['crude_net_imports'] / 1000:.2f} mb/d. "
               "Source: EIA Weekly Petroleum Status Report.")


def _positioning(ctx: CrudeCtx) -> None:
    d, w, cot = ctx.daily, ctx.weekly, ctx.cot
    st.markdown("Theory of storage in one chart: when Cushing (the WTI delivery point) is short of oil, the front of "
                "the curve trades at a premium.")
    weekly_spread = d["m1_m2"].reindex(w.index, method="ffill", limit=3)
    st.plotly_chart(charts.storage_vs_spread(w["cushing_stocks_vs_5y_pct"], weekly_spread,
                                             pd.Series(w.index.year, index=w.index)), key="oil_storage")
    if ctx.data.has_cot:
        last = cot.dropna(subset=["mm_net"]).iloc[-1]
        st.markdown(f"Hedge funds (CFTC 'managed money') were net **{'long' if last['mm_net'] >= 0 else 'short'} "
                    f"{abs(last['mm_net']) / 1000:,.0f}k contracts** on {last.name:%d %b}, the "
                    f"{last['mm_net_pctile'] * 100:.0f}th percentile of the last three years.")
        st.plotly_chart(charts.positioning_chart(cot[cot.index >= ctx.start]), key="oil_cot")
