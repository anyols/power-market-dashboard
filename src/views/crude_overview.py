"""Crude page 1 - market overview: prices, spreads, curve and data freshness."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.views.common import CrudeCtx, crude_header, mb, pct, usd


def freshness_table(ctx: CrudeCtx) -> pd.DataFrame:
    rows = []
    for name, label in [("crude/eia_daily", "EIA daily prices"), ("crude/eia_weekly", "EIA weekly balances"),
                        ("crude/cot_wti", "CFTC positioning")]:
        st_ = ctx.data.status.get(name, {})
        rows.append({
            "dataset": label,
            "latest observation": st_.get("latest_observation", "—" if not ctx.data.is_sample else "synthetic"),
            "last refreshed (UTC)": st_.get("last_success_utc", "—"),
            "status": ("✅ ok" if st_.get("ok") else "⚠️ " + st_.get("message", "")[:60]) if st_ else "—",
        })
    live = ctx.live
    rows.append({
        "dataset": "Indicative futures (Yahoo)",
        "latest observation": str(live.latest_date.date()) if live.latest_date is not None else "—",
        "last refreshed (UTC)": "on page load (cached 1 h)" if not ctx.data.is_sample else "synthetic",
        "status": "✅ ok" if live.ok else f"⚠️ {live.message[:60]}",
    })
    return pd.DataFrame(rows)


def render(ctx: CrudeCtx) -> None:
    crude_header(ctx, "Crude Oil Overview", "Prices, spreads, curve shape and inventories at a glance")
    d, w, live = ctx.daily, ctx.weekly, ctx.live
    latest_w = w.dropna(subset=["crude_stocks"]).iloc[-1]

    if live.ok and not live.fronts.empty:
        f = live.fronts.dropna(subset=["wti_front"])
        wti, wti_date = f["wti_front"].iloc[-1], f.index[-1]
        prev = f["wti_front"][f.index <= wti_date - pd.Timedelta(days=7)]
        brent = f["brent_front"].dropna().iloc[-1] if f["brent_front"].notna().any() else float("nan")
        price_src = f"{live.source}, {wti_date:%d %b}"
    else:
        s = d["wti_spot"].dropna()
        wti, wti_date = s.iloc[-1], s.index[-1]
        prev = s[s.index <= wti_date - pd.Timedelta(days=7)]
        brent = d["brent_spot"].dropna().iloc[-1]
        price_src = f"EIA spot, {wti_date:%d %b}"
    wk_chg = wti - prev.iloc[-1] if len(prev) else float("nan")

    k = st.columns(6)
    k[0].metric("WTI", usd(wti), delta=f"{wk_chg:+.2f} w/w" if pd.notna(wk_chg) else None, border=True, help=price_src)
    k[1].metric("Brent", usd(brent), border=True, help=price_src)
    k[2].metric("Brent − WTI", usd(brent - wti), border=True,
                help="Wide spreads pull US crude into exports; narrow spreads discourage them.")
    m12 = live.m1_m2 if len(live.curve) >= 2 else d["m1_m2"].dropna().iloc[-1]
    k[3].metric("WTI M1 − M2", usd(m12, signed=True), border=True,
                help="Positive = backwardation (tight prompt market). From the live curve when available.")
    k[4].metric("US crude stocks vs 5y", pct(latest_w["crude_stocks_vs_5y_pct"], 1),
                delta=f"{mb(latest_w['crude_stocks_chg'], True)} w/w", delta_color="inverse", border=True,
                help=f"Commercial stocks excl. SPR, week ending {latest_w.name:%d %b %Y}.")
    k[5].metric("Cushing", mb(latest_w["cushing_stocks"]), delta=f"{pct(latest_w['cushing_stocks_vs_5y_pct'], 0)} vs 5y",
                delta_color="inverse", border=True, help="WTI delivery hub. Operational lows are around 20 mb.")

    tab_live, tab_eia = st.tabs(["Front-month futures (Yahoo, indicative)", "Official spot prices (EIA)"])
    with tab_live:
        if live.ok:
            fr = live.fronts[live.fronts.index >= ctx.start]
            st.plotly_chart(charts.price_lines(fr, {"wti_front": "WTI front month", "brent_front": "Brent front month"},
                                               "WTI and Brent front-month futures"), key="cr_live")
            st.caption("Continuous front-month series from Yahoo Finance: unofficial, indicative, and they jump at "
                       "contract rolls. Research and backtests use official EIA data.")
        else:
            st.warning(f"Live prices unavailable ({live.message}). Showing official data only.")
    with tab_eia:
        sp = d[d.index >= ctx.start]
        st.plotly_chart(charts.price_lines(sp, {"wti_spot": "WTI Cushing spot", "brent_spot": "Brent spot"},
                                           "WTI and Brent spot (EIA)"), key="cr_spot")
        st.caption("EIA publishes daily spot prices with a lag of several days.")

    c1, c2 = st.columns(2)
    with c1:
        if len(live.curve) >= 2:
            st.plotly_chart(charts.forward_curve_chart(live.curve, f"WTI forward curve, {live.curve_date:%d %b %Y}"),
                            key="cr_curve")
            st.caption(f"Source: {live.source}. A downward-sloping curve is backwardation.")
        else:
            st.info("Live forward curve unavailable.")
    with c2:
        spread = d["brent_wti_spot"][d.index >= ctx.start]
        st.plotly_chart(charts.zero_line_chart(spread, "Brent − WTI spot spread", "Brent − WTI"), key="cr_bw")

    st.subheader("Data freshness")
    st.dataframe(freshness_table(ctx), hide_index=True)
    st.caption("EIA and CFTC data are refreshed every morning by a scheduled GitHub Action and committed to the "
               "repository; Yahoo prices are fetched live and never stored.")
