"""Ratios and spreads across commodities."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.markets import ASSETS, RATIOS, ratio_stats
from src.views.common import cached_markets, pct


def render(_=None) -> None:
    st.title("Ratios & spreads")
    st.caption("Relative value is where commodities get interesting. Each ratio is shown against its own history "
               "since 2010 (TTF since 2017).")
    mp = cached_markets()
    p = mp.prices
    if p.empty:
        st.error("No price data available right now.")
        return

    rows = []
    for r in RATIOS.values():
        if not all(k in p and p[k].notna().any() for k in r.needs):
            continue
        st_ = ratio_stats(r.compute(p))
        rows.append({"ratio": r.name, "unit": r.unit, "now": st_["last"], "percentile": st_["pctile"],
                     "1y z-score": st_["z_1y"], "1m change": st_["chg_1m"]})
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, column_config={
        "now": st.column_config.NumberColumn(format="%.2f"),
        "percentile": st.column_config.ProgressColumn("vs history", min_value=0, max_value=1, format="percent"),
        "1y z-score": st.column_config.NumberColumn(format="%+.1f"),
        "1m change": st.column_config.NumberColumn(format="%+.2f"),
    })

    st.divider()
    names = {r.name: k for k, r in RATIOS.items() if r.name in set(table["ratio"])}
    choice = st.pills("Ratio", list(names), default=list(names)[1] if len(names) > 1 else None, key="ratio_pick")
    if choice:
        r = RATIOS[names[choice]]
        s = r.compute(p)
        st.markdown(r.why)
        companion = p["us10y"] if r.key == "copper_gold" and "us10y" in p else None
        st.plotly_chart(charts.ratio_chart(s, r.name, r.unit, companion, "US 10-year yield (%)"), key="ratio_chart")
        stats = ratio_stats(s)
        a, b, c = st.columns(3)
        a.metric("Now", f"{stats['last']:,.2f}")
        b.metric("Percentile since 2010", pct(stats["pctile"]))
        c.metric("Median", f"{stats['median']:,.2f}")

    st.divider()
    st.markdown("#### Make your own")
    keys = [k for k in p.columns if p[k].notna().any()]
    c1, c2 = st.columns(2)
    num = c1.selectbox("Numerator", keys, index=keys.index("coffee") if "coffee" in keys else 0,
                       format_func=lambda k: ASSETS[k].name, key="r_num")
    den = c2.selectbox("Denominator", keys, index=keys.index("sbux") if "sbux" in keys else 1,
                       format_func=lambda k: ASSETS[k].name, key="r_den")
    if num != den:
        custom = (p[num] / p[den].where(p[den] > 0)).dropna()
        st.plotly_chart(charts.ratio_chart(custom, f"{ASSETS[num].name} / {ASSETS[den].name}",
                                           f"{ASSETS[num].unit} per {ASSETS[den].unit}"), key="ratio_custom")
