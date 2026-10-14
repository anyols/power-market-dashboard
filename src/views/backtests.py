"""Two rule-based ideas, backtested without peeking at the future."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.backtester import run_backtest
from src.crude.signals import COMPONENTS, build_crude_signals
from src.crude.signals import component_diagnostics as oil_diagnostics
from src.signal_engine import build_signal_frame
from src.signal_engine import component_diagnostics as power_diagnostics
from src.views.common import (
    CrudeCtx,
    build_crude_context,
    crude_signal_controls,
    eur,
    pct,
    power_picker,
    signal_controls,
    synthetic_note,
    usd,
)


def render(_=None) -> None:
    st.title("Backtests")
    st.caption("Simple, fixed rules - nothing optimised. Every input is only used once it would actually have been "
               "published, and every trade pays a cost. There are tests in the repo that check this.")
    oil, power = st.tabs(["Oil: weekly inventory/curve signal", "Power: residual-load signal"])
    with oil:
        _oil()
    with power:
        _power()


def _metrics(m: dict, money, per_unit: str) -> None:
    k = st.columns(5)
    k[0].metric("Trades", f"{m['n_trades']:,}")
    k[1].metric("Hit rate", pct(m["hit_rate"], 1), delta=f"coin flip = {pct(m['base_rate_up'])} up",
                delta_color="off")
    k[2].metric("Total PnL", money(m["total_pnl"]), help=per_unit)
    k[3].metric("Sharpe (ish)", f"{m['sharpe_like']:.2f}" if pd.notna(m["sharpe_like"]) else "n/a",
                help="Mean / std of period PnL, annualised. No capital base, so only roughly a Sharpe ratio.")
    k[4].metric("Max drawdown", money(m["max_drawdown"]))


# ---------------------------------------------------------------------------
# Oil
# ---------------------------------------------------------------------------
def oil_reading(ctx: CrudeCtx, params) -> pd.Series:
    """Votes as of now, with the live curve filling in for the missing official one."""
    override = None
    if len(ctx.live.curve) >= 2 and ctx.live.curve_date is not None:
        override = pd.Series({pd.Timestamp(ctx.live.curve_date).normalize(): ctx.live.m1_m2})
    asof = ctx.daily.index.max() if ctx.data.is_sample else max(pd.Timestamp.now().normalize(), ctx.daily.index.max())
    return build_crude_signals(ctx.daily, ctx.weekly, ctx.cot, params, dates=pd.DatetimeIndex([asof]),
                               curve_override=override).iloc[0]


def _oil() -> None:
    ctx = build_crude_context(None)
    if ctx.data.is_sample:
        synthetic_note("Oil")
    st.markdown(
        "Every Thursday, after the EIA report, score four things: was the weekly crude draw bigger than usual for the "
        "time of year, is Cushing tight, is the curve backwardated, and are hedge funds crowded on one side. "
        "Go long or short WTI until the next Thursday when two or more agree."
    )
    with st.expander("Settings"):
        params, bp = crude_signal_controls()

    now = oil_reading(ctx, params)
    votes = "  ·  ".join(f"{label} **{int(now[c]):+d}**" for c, label in COMPONENTS.items() if pd.notna(now.get(c)))
    st.markdown(f"This week: {votes}" + (f"  →  score **{int(now['score']):+d}**" if pd.notna(now["score"]) else ""))

    sig = build_crude_signals(ctx.daily, ctx.weekly, ctx.cot, params)
    res = run_backtest(sig, bp, period_freq="W-THU")
    m = res.metrics
    if not m["eligible_rows"]:
        st.warning("Nothing to test with these settings (official futures data stops in April 2024 - try spot "
                   "without the curve vote).")
        return
    st.caption(f"{res.ledger.index.min():%b %Y} – {res.ledger.index.max():%b %Y}, one 1,000-barrel contract, "
               f"${bp.cost_per_mwh:.2f}/bbl costs.")
    _metrics(m, lambda x: usd(x, 0), "per 1,000-barrel contract")
    st.plotly_chart(charts.equity_curve(res.daily_pnl, ccy="$", clip="1 contract"), key="bt_oil_eq")
    diag = oil_diagnostics(sig)
    st.dataframe(diag, column_config={
        "active weeks": st.column_config.NumberColumn(format="%d"),
        "hit rate": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
        "avg. signed move ($/bbl)": st.column_config.NumberColumn(format="%+.2f"),
    })
    edge = m["hit_rate"] - m["base_rate_up"]
    verdict = "barely better than" if 0 < edge < 0.03 else ("better than" if edge >= 0.03 else "no better than")
    st.markdown(
        f"Over {m['eligible_rows']} weeks the combined score was right {pct(m['hit_rate'], 1)} of the time, "
        f"{verdict} the {pct(m['base_rate_up'], 1)} of weeks that went up anyway. The weekly inventory number is "
        "public at 10:30 ET on Wednesday and priced within minutes, so by Thursday's close there's little left in it."
    )


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------
def _power() -> None:
    ctx = power_picker()
    if ctx is None:
        return
    st.markdown(
        "Before the day-ahead auction (D-1, before noon) compare tomorrow's forecast residual load with today's. "
        "Add two weaker votes from the most recent load and wind/solar forecast misses I'd actually have seen."
    )
    with st.expander("Settings"):
        sp, bp = signal_controls()
    sig = build_signal_frame(ctx.feat, sp)
    sig = sig[(sig["date"] >= pd.Timestamp(ctx.start)) & (sig["date"] <= pd.Timestamp(ctx.end))]
    res = run_backtest(sig, bp)
    m = res.metrics
    if not m["eligible_rows"]:
        st.warning("Window too short for this test.")
        return
    _metrics(m, lambda x: eur(x), "per 1 MW traded each hour")
    st.plotly_chart(charts.equity_curve(res.daily_pnl), key="bt_pw_eq")
    st.dataframe(power_diagnostics(sig), column_config={
        "active observations": st.column_config.NumberColumn(format="%d"),
        "hit rate": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
        "avg. signed move (EUR/MWh)": st.column_config.NumberColumn(format="%+.2f"),
    })
    st.markdown("The catch: the 'entry price' here is yesterday's auction price, which you can't actually trade "
                "at. The market already knows tomorrow's wind forecast, so beating yesterday's price isn't the same "
                "as beating the market.")
