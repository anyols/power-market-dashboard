"""Page 6 - Trading Signal Lab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.backtester import run_backtest
from src.config import BacktestParams, SignalParams
from src.signal_engine import INFORMATION_SET, build_signal_frame, component_diagnostics
from src.views.common import Ctx, eur, page_header, pct, require_rows, signal_controls

DISCLAIMER = (
    "**Research toy model - not a production strategy.** Rules are fixed in advance (no fitting), use only "
    "information available before the day-ahead gate closure, and pay a cost on every MWh. Even so, the PnL is a "
    "*proxy*: it measures calls against yesterday's price, not against a tradeable pre-auction market price."
)


def windowed_signals(ctx: Ctx, sp: SignalParams) -> pd.DataFrame:
    """Signals computed on the full history (for warm-up), evaluated on the selected window."""
    sig = build_signal_frame(ctx.feat, sp)
    start, end = pd.Timestamp(ctx.start), pd.Timestamp(ctx.end)
    return sig[(sig["date"] >= start) & (sig["date"] <= end)]


def render(ctx: Ctx) -> None:
    page_header(ctx, "Trading Signal Lab", "Transparent fundamental signals, tested honestly")
    if not require_rows(ctx.view, 24 * 14):
        return
    st.warning(DISCLAIMER, icon="⚠️")
    if ctx.data.is_sample:
        st.error("You are looking at **synthetic data**. Its generator clears prices on forecast residual demand, so a "
                 "residual-demand signal works partly *by construction*. Treat these numbers as a demonstration of the "
                 "method, not as evidence about real markets.", icon="🧪")

    with st.container(border=True):
        st.markdown("**Signal settings**")
        sp, bp = signal_controls()

    sig = windowed_signals(ctx, sp)
    result = run_backtest(sig, bp)
    m = result.metrics
    if m["eligible_rows"] == 0:
        st.warning("No eligible observations - widen the window or reduce the trailing window length.")
        return

    k = st.columns(6)
    k[0].metric("Trades", f"{m['n_trades']:,}", delta=f"{pct(m['trade_frequency'])} of {'hours' if sp.horizon == 'hourly' else 'days'}",
                delta_color="off", border=True)
    k[1].metric("Hit rate", pct(m["hit_rate"], 1), delta=f"base rate up-moves {pct(m['base_rate_up'])}", delta_color="off",
                border=True, help="Share of trades whose direction matched the realised move. Compare with the base rate.")
    k[2].metric("Total PnL proxy", eur(m["total_pnl"]), delta=f"costs {eur(-m['total_cost'])}", delta_color="off", border=True)
    k[3].metric("Avg PnL per MWh", eur(m["avg_pnl_per_mwh"], 2), border=True)
    k[4].metric("Sharpe-like", f"{m['sharpe_like']:.2f}" if pd.notna(m["sharpe_like"]) else "n/a",
                help="Annualised mean / std of daily PnL, idle days included. No capital base, so not a true Sharpe ratio.",
                border=True)
    k[5].metric("Max drawdown", eur(m["max_drawdown"]), border=True)

    st.plotly_chart(charts.signal_overview(sig, sp.horizon), key="sig_overview")

    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(charts.target_by_score(sig), key="sig_box")
    with c2:
        st.plotly_chart(charts.confusion_heatmap(result.confusion), key="sig_conf")
        st.caption(f"Long hit rate {pct(m['hit_rate_long'], 1)} ({m['n_long']} trades) · "
                   f"short hit rate {pct(m['hit_rate_short'], 1)} ({m['n_short']} trades).")

    st.subheader("Which component carries the information?")
    diag = component_diagnostics(sig)
    st.dataframe(
        diag,
        column_config={
            "active observations": st.column_config.NumberColumn(format="%d"),
            "hit rate": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "avg. signed move (EUR/MWh)": st.column_config.NumberColumn(format="%+.2f"),
        },
    )
    st.caption(
        "Each vote evaluated on its own whenever it is non-zero. The lagged surprise components use errors from D-2: "
        "forecast misses mostly matter for intraday and imbalance prices on the day, so expect little information "
        "about the *next* day-ahead auction. That is a useful negative result, not a bug."
    )

    rd_only = sig.copy()
    rd_only["position"] = rd_only["comp_rd"].fillna(0)
    rd_result = run_backtest(rd_only, bp)
    st.plotly_chart(charts.equity_curve(result.daily_pnl, {"Residual-demand vote only": rd_result.daily_pnl}), key="sig_eq")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Cost sensitivity**")
        rows = []
        for cost in [0.0, 0.5, 1.0, 2.0, 5.0]:
            r = run_backtest(sig, BacktestParams(cost_per_mwh=cost, volume_mw=bp.volume_mw)).metrics
            rows.append({"cost €/MWh": cost, "total PnL €": r["total_pnl"], "Sharpe-like": r["sharpe_like"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={"total PnL €": st.column_config.NumberColumn(format="%+,.0f"),
                                    "Sharpe-like": st.column_config.NumberColumn(format="%.2f")})
        traded = result.trades["mwh"].sum()
        if traded:
            st.caption(f"Break-even cost: {eur(m['gross_pnl'] / traded, 2)} per MWh traded.")
    with c4:
        st.markdown("**Why beating yesterday's price is not the same as beating the market**")
        st.caption(
            "Before the auction, day-ahead products already trade OTC and on exchange at prices that embed the same "
            "public forecasts this signal uses. A residual-demand-change signal therefore mostly rediscovers what the "
            "market already prices. A realistic test needs pre-auction forward prices as the entry level (not available "
            "from ENTSO-E), or should target the day-ahead vs intraday/imbalance spread, where forecast errors genuinely "
            "drive value."
        )

    with st.expander("Information set: what was known when the signal was formed?", expanded=False):
        st.markdown("Decision time: **D-1, ~11:00 CET**, before the 12:00 CET SDAC gate closure for delivery day **D**.")
        st.dataframe(pd.DataFrame(INFORMATION_SET), hide_index=True)
        st.markdown(
            """
**Score construction** (−3 … +3):
- +1 if the residual-demand z-score > threshold, −1 if < −threshold
- +1 if the latest known load surprise (D-2) is positive beyond the dead-band, −1 if negative
- +1 if the latest known renewable surprise (D-2) is negative beyond the dead-band, −1 if positive

Position = sign(score) when |score| ≥ entry threshold; 1 MW per delivery hour (hourly) or baseload for the day (daily).
PnL proxy = position × (delivered price − reference price) × MWh − cost × MWh. Tests in `tests/test_no_lookahead.py`
perturb all data outside the information set and verify the signal does not change.
"""
        )
