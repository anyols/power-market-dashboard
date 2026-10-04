"""Crude page 2 - inventories and balances vs seasonal norms."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.crude.config import LABELS
from src.crude.features import iso_year_week, seasonal_profile
from src.views.common import CrudeCtx, crude_header, mb, pct

STOCKS = [("crude_stocks", "US commercial crude stocks (excl. SPR)"), ("cushing_stocks", "Cushing, OK crude stocks"),
          ("gasoline_stocks", "US gasoline stocks"), ("distillate_stocks", "US distillate stocks")]


def render(ctx: CrudeCtx) -> None:
    crude_header(ctx, "Inventories & Balances", "Weekly Petroleum Status Report vs 5-year seasonal norms")
    w = ctx.weekly.dropna(subset=["crude_stocks"])
    latest = w.iloc[-1]
    years = sorted(set(iso_year_week(w.index)[0]), reverse=True)
    year = st.selectbox("Year to highlight", years, key="inv_year")
    st.caption(f"Latest report: week ending {latest.name:%a %d %b %Y}, released {latest.name + pd.Timedelta(days=5):%a %d %b}. "
               "Seasonal bands use the five calendar years *before* the highlighted year.")

    k = st.columns(6)
    k[0].metric("Crude stocks w/w", mb(latest["crude_stocks_chg"], True),
                delta=f"5y norm {mb(latest['crude_stocks_chg_5y_avg'], True)}", delta_color="off", border=True)
    k[1].metric("Surprise vs seasonal", mb(latest["crude_stocks_surprise"], True), border=True,
                help="Weekly change minus the 5-year average change for the same week. Negative = bigger draw than "
                     "normal (bullish). Analyst-consensus surprises need paid data.")
    k[2].metric("Cushing", mb(latest["cushing_stocks"]), delta=mb(latest["cushing_stocks_chg"], True), delta_color="inverse",
                border=True)
    k[3].metric("Gasoline w/w", mb(latest["gasoline_stocks_chg"], True), border=True)
    k[4].metric("Refinery utilisation", f"{latest['refinery_utilization']:.1f}%",
                delta=f"{latest['refinery_utilization'] - latest['refinery_utilization_5y_avg']:+.1f} pp vs 5y",
                delta_color="off", border=True)
    k[5].metric("Days of crude cover", f"{latest['days_of_cover']:.1f}", border=True,
                help="Commercial crude stocks divided by refinery crude runs.")

    cols = st.columns(2)
    for i, (col, title) in enumerate(STOCKS):
        prof = seasonal_profile(w[col], year)
        cols[i % 2].plotly_chart(charts.seasonal_band_chart(prof, title, "mb", year, scale=1000), key=f"band_{col}")

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.seasonal_band_chart(seasonal_profile(w["refinery_utilization"], year),
                                               "Refinery utilisation", "%", year), key="band_util")
    c2.plotly_chart(charts.seasonal_band_chart(seasonal_profile(w["product_supplied_4w"], year),
                                               "Implied product demand (4-week average)", "mb/d", year, scale=1000),
                    key="band_demand")

    win = w[w.index >= ctx.start]
    supply = pd.DataFrame({"Production": win["crude_production"], "Net imports": win["crude_net_imports"],
                           "Refinery crude runs": win["refinery_crude_input"]}) / 1000
    st.plotly_chart(charts.price_lines(supply, {c: c for c in supply.columns}, "US crude supply and refinery runs",
                                       unit="mb/d"), key="supply")

    st.subheader("Latest report vs seasonal norms")
    rows = []
    for col in ["crude_stocks", "cushing_stocks", "gasoline_stocks", "distillate_stocks", "total_commercial_stocks"]:
        rows.append({
            "series": LABELS.get(col, "Total commercial (crude + gasoline + distillate)"),
            "level (mb)": latest[col] / 1000,
            "w/w change (mb)": latest[f"{col}_chg"] / 1000,
            "5y avg change (mb)": latest[f"{col}_chg_5y_avg"] / 1000,
            "vs seasonal (mb)": latest[f"{col}_surprise"] / 1000,
            "level vs 5y avg": latest[f"{col}_vs_5y_pct"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "level (mb)": st.column_config.NumberColumn(format="%.1f"),
        "w/w change (mb)": st.column_config.NumberColumn(format="%+.1f"),
        "5y avg change (mb)": st.column_config.NumberColumn(format="%+.1f"),
        "vs seasonal (mb)": st.column_config.NumberColumn(format="%+.1f"),
        "level vs 5y avg": st.column_config.NumberColumn(format="percent"),
    })

    with st.expander("How to read inventories", expanded=False):
        st.markdown(
            """
- **Stocks are the shock absorber.** Below-normal inventories leave less buffer against outages, so the market pays up
  for prompt barrels: flat price and time spreads rise.
- **Always compare with the same week in prior years.** Crude typically builds in spring (refinery maintenance) and
  draws through summer (peak runs); gasoline draws into the driving season, distillates in winter.
- **Cushing is special for WTI**: it is the contract's delivery point, so Cushing tightness shows up directly in the
  front spreads, and levels near ~20 mb (tank bottoms) raise squeeze risk.
- **A draw is not automatically bullish**: a crude draw caused by record refinery runs is different from one caused
  by an export surge or lower imports. Read the balance together with runs, production and trade.
"""
        )
    st.caption(f"Units: thousand barrels converted to mb (million barrels); flows in mb/d. Latest values: crude "
               f"{pct(latest['crude_stocks_vs_5y_pct'], 1)} vs 5y, Cushing {pct(latest['cushing_stocks_vs_5y_pct'], 1)} vs 5y.")
