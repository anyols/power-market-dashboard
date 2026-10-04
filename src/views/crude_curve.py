"""Crude page 3 - futures curve, theory of storage and positioning."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.views.common import CrudeCtx, crude_header, insight, pct


def render(ctx: CrudeCtx) -> None:
    crude_header(ctx, "Curve & Positioning", "What the futures curve and speculative positioning say about tightness")
    d, w, cot = ctx.daily, ctx.weekly, ctx.cot

    insight("Theory of storage: when inventories are scarce, holding physical oil has a *convenience yield*, so prompt "
            "barrels trade above deferred ones (backwardation). When stocks are ample, the curve must pay for storage "
            "(contango).")

    weekly_spread = d["m1_m2"].reindex(w.index, method="ffill", limit=3)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(charts.storage_vs_spread(w["cushing_stocks_vs_5y_pct"], weekly_spread,
                                             pd.Series(w.index.year, index=w.index)), key="storage")
    with c2:
        valid = pd.DataFrame({"x": w["cushing_stocks_vs_5y_pct"], "y": weekly_spread}).dropna()
        st.metric("Correlation: Cushing vs 5y and M1–M2", f"{valid['x'].corr(valid['y']):.2f}" if len(valid) > 20 else "n/a",
                  border=True, help="Expected negative: lower stocks, more backwardation.")
        st.metric("Weekly observations", f"{len(valid):,}", border=True)
        if d["cl1"].notna().any():
            st.caption(f"Official futures data runs to {d['cl1'].dropna().index.max():%d %b %Y}. EIA stopped publishing "
                       "NYMEX futures prices after 5 April 2024, so newer weeks lack an official curve.")

    sp = d[d.index >= ctx.start][["m1_m2", "m1_m4"]].dropna(how="all")
    if not sp.empty:
        st.plotly_chart(charts.price_lines(sp, {"m1_m2": "M1 − M2", "m1_m4": "M1 − M4"},
                                           "WTI time spreads (EIA settlements; + = backwardation)"), key="spreads")
    else:
        st.info("No official futures data in the selected window (EIA futures history ends April 2024). "
                "Widen the window in the sidebar to see it.")
    if len(ctx.live.curve) >= 2:
        st.plotly_chart(charts.forward_curve_chart(ctx.live.curve, f"Current WTI curve, {ctx.live.curve_date:%d %b %Y} "
                                                                   f"({ctx.live.source})"), key="live_curve")

    st.subheader("Speculative positioning (CFTC Commitments of Traders)")
    if not ctx.data.has_cot:
        st.warning("CFTC positioning data is not available yet.")
        return
    cw = cot[cot.index >= ctx.start]
    last = cot.dropna(subset=["mm_net"]).iloc[-1]
    k = st.columns(4)
    k[0].metric("Managed money net", f"{last['mm_net'] / 1000:+,.0f}k contracts".replace("-", "−"),
                delta=f"{last['mm_net_change'] / 1000:+,.0f}k w/w", border=True)
    k[1].metric("Net as % of open interest", pct(last["mm_net_pct_oi"], 1), border=True)
    k[2].metric("3-year percentile", pct(last["mm_net_pctile"]), border=True,
                help="Share of the previous 156 weekly readings below the current one.")
    k[3].metric("Report date", f"{last.name:%d %b %Y}", delta=f"published {last['available_date']:%a %d %b}",
                delta_color="off", border=True)
    st.plotly_chart(charts.positioning_chart(cw), key="cot")
    st.caption("Managed money = hedge funds and CTAs (CFTC disaggregated report, WTI-Physical on NYMEX). Positions are "
               "as of Tuesday and published on Friday. Extreme readings (above the 85th or below the 15th percentile) "
               "flag crowded trades.")
