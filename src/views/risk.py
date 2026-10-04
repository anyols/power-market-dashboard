"""Page 8 - Risk Dashboard for the research signal."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.backtester import exposure_by_hour, run_backtest, stress_spike_hours
from src.views.common import (
    Ctx,
    current_params,
    eur,
    page_header,
    pct,
    require_rows,
    signal_controls,
)
from src.views.signal_lab import windowed_signals


def _regime_known_at_decision(ctx: Ctx, dates: pd.Series) -> pd.Series:
    """Volatility regime as of the last hour of D-1 (no peeking at day D)."""
    daily = ctx.feat.groupby("date")["vol_regime"].last().shift(1)
    return pd.Series(daily.reindex(pd.DatetimeIndex(dates)).to_numpy(), index=dates.index)


def render(ctx: Ctx) -> None:
    page_header(ctx, "Risk Dashboard", "PnL proxy, drawdowns, exposure and stress for the Signal Lab strategy")
    if not require_rows(ctx.view, 24 * 14):
        return
    st.caption("Educational/research backtest of the Signal Lab rules on a fixed 1 MW clip. "
               "Settings are shared with the Signal Lab page.")
    with st.expander("Strategy settings", expanded=False):
        signal_controls()
    sp, bp = current_params()

    sig = windowed_signals(ctx, sp)
    sig = sig.assign(vol_regime=_regime_known_at_decision(ctx, sig["date"]))
    res = run_backtest(sig, bp)
    m = res.metrics
    if m["eligible_rows"] == 0:
        st.warning("No eligible observations in this window.")
        return

    k = st.columns(6)
    k[0].metric("Total PnL proxy", eur(m["total_pnl"]), border=True)
    k[1].metric("Hit rate", pct(m["hit_rate"], 1), border=True)
    k[2].metric("Sharpe-like", f"{m['sharpe_like']:.2f}" if pd.notna(m["sharpe_like"]) else "n/a", border=True)
    k[3].metric("Max drawdown", eur(m["max_drawdown"]), border=True)
    k[4].metric("Worst single trade", eur(m["worst_trade"]), border=True)
    k[5].metric("Profit factor", f"{m['profit_factor']:.2f}" if pd.notna(m["profit_factor"]) else "n/a",
                help="Gross gains / gross losses on trades.", border=True)

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.daily_pnl_bars(res.daily_pnl), key="risk_daily")
    c2.plotly_chart(charts.equity_curve(res.daily_pnl), key="risk_eq")

    st.subheader("Worst 10 trades")
    worst = res.trades.nsmallest(10, "pnl")
    label = worst.index.strftime("%Y-%m-%d %H:%M") if sp.horizon == "hourly" else worst.index.strftime("%Y-%m-%d")
    st.dataframe(
        pd.DataFrame({
            "delivery": label,
            "side": worst["position"].map({1.0: "Long", -1.0: "Short"}),
            "score": worst["score"],
            "reference €/MWh": worst["reference_price"],
            "delivered €/MWh": worst["delivered_price"],
            "move €/MWh": worst["target"],
            "PnL €": worst["pnl"],
            "vol regime": worst["vol_regime"],
        }),
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn(format="%+d"),
            "reference €/MWh": st.column_config.NumberColumn(format="%.2f"),
            "delivered €/MWh": st.column_config.NumberColumn(format="%.2f"),
            "move €/MWh": st.column_config.NumberColumn(format="%+.2f"),
            "PnL €": st.column_config.NumberColumn(format="%+,.0f"),
        },
    )

    c3, c4 = st.columns(2)
    if sp.horizon == "hourly":
        c3.plotly_chart(charts.exposure_chart(exposure_by_hour(res.trades)), key="risk_expo")
    else:
        c3.info("Exposure by hour applies to the hourly horizon. In daily mode every trade is a baseload block.")
    c4.plotly_chart(charts.pnl_by_regime(res.ledger), key="risk_regime")

    st.subheader("Stress scenario: spike hours included vs excluded")
    if sp.horizon == "hourly":
        spike_mask = ctx.spike_mask.reindex(sig.index).fillna(False).astype(bool)
    else:
        spike_days = ctx.view.loc[ctx.spike_mask, "date"].unique()
        spike_mask = sig["date"].isin(spike_days)
    stress = stress_spike_hours(sig, spike_mask, bp)
    st.dataframe(
        stress,
        column_config={
            "n_trades": st.column_config.NumberColumn("trades", format="%d"),
            "hit_rate": st.column_config.NumberColumn("hit rate", format="percent"),
            "total_pnl": st.column_config.NumberColumn("total PnL €", format="%+,.0f"),
            "avg_pnl_per_trade": st.column_config.NumberColumn("avg PnL / trade €", format="%+,.1f"),
            "sharpe_like": st.column_config.NumberColumn("Sharpe-like", format="%.2f"),
            "max_drawdown": st.column_config.NumberColumn("max drawdown €", format="%+,.0f"),
        },
    )
    share = stress.loc["Spike rows only", "total_pnl"] / m["total_pnl"] if m["total_pnl"] else float("nan")
    st.caption(
        f"Spike rows ({ctx.spike_label}) contribute {pct(share)} of total PnL. A strategy whose result hinges on a few "
        "extreme hours is really selling or buying tail risk - and tails are where limits, liquidity and slippage hurt "
        "most. Excluding spikes is an ex-post diagnostic, not something a trader could do in real time."
        if pd.notna(share) else "No PnL to attribute."
    )
