"""Page 1 - Market Overview."""

from __future__ import annotations

import streamlit as st

from src import charts
from src.feature_engineering import daily_summary
from src.market_commentary import market_regime_notes
from src.views.common import Ctx, eur, gw, page_header, regime_cards, require_rows


def render(ctx: Ctx) -> None:
    page_header(ctx, "Market Overview", "Day-ahead prices and the fundamentals behind them")
    v = ctx.view
    if not require_rows(v):
        return

    k = st.columns(6)
    k[0].metric("Average price", eur(v["price"].mean(), 1), help="Baseload: simple average of hourly day-ahead prices (€/MWh).", border=True)
    k[1].metric("Min / max €/MWh", f"{v['price'].min():,.0f} / {v['price'].max():,.0f}".replace("-", "−"), border=True)
    k[2].metric("Volatility", eur(v["price_change_1h"].std(), 1),
                help="Standard deviation of hour-on-hour price changes (€/MWh). Absolute, not %, because prices can be negative.",
                border=True)
    k[3].metric("Spike hours", f"{int(ctx.spike_mask.sum()):,}", help=f"Spike = {ctx.spike_label}.", border=True)
    k[4].metric("Negative hours", f"{int(v['is_negative'].sum()):,}", help="Hours with a day-ahead price below 0 €/MWh.", border=True)
    k[5].metric("Residual demand", gw(v["residual_demand"].mean()),
                help="Load minus wind minus solar: the demand dispatchable plants and imports must cover.", border=True)

    st.subheader("Trader read-out")
    st.caption("Rule-based interpretation of the last 7 days of the selected window, using trailing statistics only.")
    regime_cards(market_regime_notes(v))

    st.plotly_chart(charts.fundamentals_stack(v, ctx.spike_mask), key="stack")

    daily = daily_summary(v)
    st.plotly_chart(charts.daily_range_chart(daily), key="daily_range")

    with st.expander("How to read this page", expanded=False):
        st.markdown(
            """
- **High residual demand = tighter system = bullish pressure.** More of the demand must be met by
  dispatchable plants, so pricier units (gas, then peakers) set the price.
- **Low residual demand = looser system = bearish pressure.** When wind and solar cover most of the
  load, cheap or inflexible units are at the margin and prices can fall below zero.
- **Large renewable forecast errors** are a potential intraday/imbalance driver: the day-ahead auction
  cleared on the forecast, so the gap has to be re-traded closer to delivery.
- **High volatility** means more opportunity but also more risk: wider ranges, larger drawdowns and
  worse execution.
"""
        )

    with st.expander("Daily statistics table"):
        table = daily.copy()
        table.index = table.index.date
        st.dataframe(
            table,
            column_config={
                "baseload": st.column_config.NumberColumn("Base €/MWh", format="%.1f"),
                "peak": st.column_config.NumberColumn("Peak €/MWh", format="%.1f"),
                "offpeak": st.column_config.NumberColumn("Off-peak €/MWh", format="%.1f"),
                "min": st.column_config.NumberColumn("Min", format="%.1f"),
                "max": st.column_config.NumberColumn("Max", format="%.1f"),
                "intraday_range": st.column_config.NumberColumn("Range", format="%.1f"),
                "negative_hours": st.column_config.NumberColumn("Neg. hours", format="%d"),
                "residual_demand_gw": st.column_config.NumberColumn("Resid. demand GW", format="%.1f"),
                "wind_gw": st.column_config.NumberColumn("Wind GW", format="%.1f"),
                "solar_gw": st.column_config.NumberColumn("Solar GW", format="%.1f"),
                "load_gw": st.column_config.NumberColumn("Load GW", format="%.1f"),
                "hours": st.column_config.NumberColumn("Hours", format="%d"),
            },
        )
