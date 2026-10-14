"""Stocks vs the commodity they depend on - plus a correlation matrix."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.markets import ASSETS, PAIRS, correlation_matrix, decoupling, lead_lag, rolling_beta_corr, weekly_returns
from src.views.common import cached_markets, pct

DEFAULT_MATRIX = ["wti", "natgas", "ttf", "gold", "copper", "wheat", "coffee", "cocoa",
                  "xom", "rwe", "fcx", "hsy", "spx", "dxy", "us10y"]


def render(_=None) -> None:
    st.title("Stocks vs the stuff they depend on")
    st.caption("Every commodity has a listed company that lives or dies by it. Does the stock actually follow? "
               "All on weekly returns, which avoids US/European closing-time noise.")
    mp = cached_markets()
    p = mp.prices
    if p.empty:
        st.error("No price data available right now.")
        return

    labels = {f"{ASSETS[x.commodity].name} → {ASSETS[x.stock].name}": x for x in PAIRS}
    pick = st.pills("Pair", list(labels), default=list(labels)[0], key="pair_pick")
    if not pick:
        return
    pair = labels[pick]
    cname, sname = ASSETS[pair.commodity].name, ASSETS[pair.stock].name
    st.markdown(pair.story)

    c1, c2 = st.columns([1, 1])
    years = c1.select_slider("History", [1, 2, 3, 5, 10, 15], value=5, format_func=lambda y: f"{y}y", key="pair_years")
    window = c2.select_slider("Rolling window", [13, 26, 52, 104], value=52, format_func=lambda w: f"{w} weeks",
                              key="pair_window")

    r = weekly_returns(p[[pair.commodity, pair.stock]])
    rc, rs = r[pair.commodity], r[pair.stock]
    start = p.index.max() - pd.DateOffset(years=years)
    shown = r[r.index >= start]

    beta_corr = rolling_beta_corr(rs, rc, window)
    full = pd.concat({"c": rc, "s": rs}, axis=1).dropna()
    recent = full[full.index >= start]
    corr_now = recent["s"].corr(recent["c"])
    beta_now = recent["s"].cov(recent["c"]) / recent["c"].var() if recent["c"].var() else float("nan")
    dec = decoupling(rs, rc)
    last_dec = dec.dropna().iloc[-1] if dec.dropna().shape[0] else None

    k = st.columns(3)
    k[0].metric(f"Correlation ({years}y)", f"{corr_now:+.2f}")
    k[1].metric(f"Beta ({years}y)", f"{beta_now:+.2f}",
                help=f"A 1% weekly move in {cname} has come with a {beta_now:+.2f}% move in {sname} on average.")
    if last_dec is not None:
        k[2].metric("vs what the commodity implies (13w)", pct(last_dec["residual"], 1, signed=True),
                    delta=f"z {last_dec['z']:+.1f}", delta_color="off",
                    help=f"{sname}'s last-13-week return minus beta × {cname}'s, with the beta measured before the "
                         "window. Big values = the stock has decoupled from its commodity.")

    a, b = st.columns(2)
    a.plotly_chart(charts.indexed_performance(p.loc[p.index >= start, [pair.commodity, pair.stock]],
                                              {pair.commodity: cname, pair.stock: sname}), key="pair_idx")
    b.plotly_chart(charts.weekly_scatter(shown[pair.commodity], shown[pair.stock], cname, sname), key="pair_scatter")
    c, d = st.columns(2)
    c.plotly_chart(charts.rolling_pair_chart(beta_corr[beta_corr.index >= start], f"{window}w"), key="pair_roll")
    with d:
        st.plotly_chart(charts.lead_lag_chart(lead_lag(shown[pair.stock], shown[pair.commodity]), cname, sname),
                        key="pair_ll")
        st.caption("Bars to the right of zero: the commodity's move came first and the stock followed. "
                   "Most of these are close to zero - markets price this stuff fast.")

    st.divider()
    st.markdown("#### Correlation matrix")
    keys = [k for k in p.columns if p[k].notna().any()]
    m1, m2 = st.columns([4, 1])
    chosen = m1.multiselect("Assets", keys, default=[k for k in DEFAULT_MATRIX if k in keys],
                            format_func=lambda k: ASSETS[k].name, key="corr_assets")
    weeks = m2.selectbox("Window", [13, 52, 156, 520], index=1, format_func=lambda w: f"{w} weeks", key="corr_weeks")
    if len(chosen) >= 2:
        st.plotly_chart(charts.correlation_heatmap(correlation_matrix(p, chosen, weeks)), key="corr_matrix")
        st.caption("Weekly log returns. Compare the 13-week view with the 520-week one: correlations are anything "
                   "but stable.")
