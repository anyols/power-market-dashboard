"""Crude page 4 - weekly signal lab with release-calendar-aware backtest."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.backtester import run_backtest
from src.config import BacktestParams
from src.crude.signals import COMPONENTS, CRUDE_INFORMATION_SET, build_crude_signals, component_diagnostics
from src.views.common import CrudeCtx, crude_header, crude_signal_controls, pct, usd


def live_reading(ctx: CrudeCtx, params) -> pd.Series:
    """Votes as of today using everything published so far (live curve fills the missing official curve)."""
    today = pd.Timestamp.now().normalize()
    override = None
    if len(ctx.live.curve) >= 2 and ctx.live.curve_date is not None:
        override = pd.Series({pd.Timestamp(ctx.live.curve_date).normalize(): ctx.live.m1_m2})
    asof = max(today, ctx.daily.index.max()) if not ctx.data.is_sample else ctx.daily.index.max()
    sig = build_crude_signals(ctx.daily, ctx.weekly, ctx.cot, params, dates=pd.DatetimeIndex([asof]),
                              curve_override=override)
    return sig.iloc[0]


def render(ctx: CrudeCtx) -> None:
    crude_header(ctx, "Crude Signal Lab", "Weekly fundamental votes, tested settlement-to-settlement")
    st.warning("**Research model, not a trading recommendation.** Fixed rules, no fitting, costs on every barrel, and "
               "each input is only used after its official release time.", icon="⚠️")
    if ctx.data.is_sample:
        st.error("Synthetic data: its generator builds in a roll-yield premium in backwardation and makes inventory "
                 "surprises unpredictable by design. Use these results to learn the method, not as evidence.", icon="🧪")

    with st.container(border=True):
        st.markdown("**Signal settings**")
        params, bp = crude_signal_controls()

    reading = live_reading(ctx, params)
    with st.container(border=True):
        st.markdown(f"**Current reading** (all data published by {reading.name:%d %b %Y})")
        cols = st.columns(5)
        for col, (key, label) in zip(cols, COMPONENTS.items()):
            v = reading.get(key)
            col.metric(label, "—" if pd.isna(v) else f"{int(v):+d}")
        cols[4].metric("Score", "—" if pd.isna(reading["score"]) else f"{int(reading['score']):+d}",
                       delta={1.0: "long", -1.0: "short", 0.0: "flat"}.get(reading["position"], "flat"), delta_color="off")
        st.caption(f"Uses EIA week ending {pd.Timestamp(reading['week_used']):%d %b} and the CFTC report of "
                   f"{pd.Timestamp(reading['cot_used']):%d %b}" if pd.notna(reading.get("cot_used")) else
                   f"Uses EIA week ending {pd.Timestamp(reading['week_used']):%d %b}.")

    sig = build_crude_signals(ctx.daily, ctx.weekly, ctx.cot, params)
    sig = sig[sig["target"].notna() | sig["score"].notna()]
    res = run_backtest(sig, bp, period_freq="W-THU")
    m = res.metrics
    if m["eligible_rows"] == 0:
        st.warning("No weeks with both a complete signal and a realised target. With the futures target the official "
                   "data ends in April 2024; try the spot target with the curve vote switched off.")
        return
    ledger = res.ledger
    st.caption(f"Backtest window: {ledger.index.min():%d %b %Y} – {ledger.index.max():%d %b %Y} "
               f"({m['eligible_rows']} weeks). One NYMEX contract = 1,000 bbl.")

    k = st.columns(6)
    k[0].metric("Trades", f"{m['n_trades']:,}", delta=f"{pct(m['trade_frequency'])} of weeks", delta_color="off", border=True)
    k[1].metric("Hit rate", pct(m["hit_rate"], 1), delta=f"base rate up {pct(m['base_rate_up'])}", delta_color="off", border=True)
    k[2].metric("Total PnL (1 contract)", usd(m["total_pnl"], 0), delta=f"costs {usd(-m['total_cost'], 0)}",
                delta_color="off", border=True)
    k[3].metric("Avg PnL per bbl", usd(m["avg_pnl_per_mwh"], 3), border=True)
    k[4].metric("Sharpe-like", f"{m['sharpe_like']:.2f}" if pd.notna(m["sharpe_like"]) else "n/a", border=True,
                help="Annualised mean/std of weekly PnL, idle weeks included.")
    k[5].metric("Max drawdown", usd(m["max_drawdown"], 0), border=True)

    label = "Front-month futures, roll-adjusted reference ($/bbl)" if params.target == "futures" else "WTI Cushing spot ($/bbl)"
    st.plotly_chart(charts.weekly_signal_chart(sig, label), key="csig_chart")

    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(charts.target_by_score(sig, unit="$/bbl"), key="csig_box")
    c2.plotly_chart(charts.confusion_heatmap(res.confusion), key="csig_conf")

    st.subheader("Which vote carries information?")
    st.dataframe(component_diagnostics(sig), column_config={
        "active weeks": st.column_config.NumberColumn(format="%d"),
        "hit rate": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
        "avg. signed move ($/bbl)": st.column_config.NumberColumn(format="%+.3f"),
    })
    st.caption("Inventory surprises are priced within minutes of the 10:30 ET release, so a Thursday-settlement entry "
               "should find little left - a useful negative result. The curve vote earns roll yield by construction "
               "of how futures converge to spot.")

    st.plotly_chart(charts.equity_curve(res.daily_pnl, ccy="$", clip="1 contract, weekly"), key="csig_eq")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Cost sensitivity**")
        rows = []
        for cost in [0.0, 0.02, 0.03, 0.05, 0.10]:
            r = run_backtest(sig, BacktestParams(cost_per_mwh=cost), period_freq="W-THU").metrics
            rows.append({"cost $/bbl": cost, "total PnL $": r["total_pnl"], "Sharpe-like": r["sharpe_like"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
            "cost $/bbl": st.column_config.NumberColumn(format="%.2f"),
            "total PnL $": st.column_config.NumberColumn(format="%+,.0f"),
            "Sharpe-like": st.column_config.NumberColumn(format="%.2f")})
    with c4:
        st.markdown("**Execution realism**")
        st.caption("Entry and exit at Thursday settlement are achievable with NYMEX Trade-at-Settlement (TAS) orders, "
                   "so this PnL is much closer to executable than the power proxy. Still missing: margin, position "
                   "limits, slippage in fast markets, and the roll mechanics of a live position.")

    with st.expander("Information set: when is each input known?", expanded=False):
        st.dataframe(pd.DataFrame(CRUDE_INFORMATION_SET), hide_index=True)
        st.caption("Tests in tests/test_crude.py scramble every input published after the decision time and check "
                   "the signal does not move.")
