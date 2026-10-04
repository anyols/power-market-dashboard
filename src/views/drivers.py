"""Page 2 - Price Driver Analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import charts
from src.feature_engineering import correlation_frame
from src.views.common import Ctx, insight, page_header, require_rows


def render(ctx: Ctx) -> None:
    page_header(ctx, "Price Driver Analysis", "Which fundamentals explain the price?")
    v = ctx.view
    if not require_rows(v):
        return
    C = charts.C

    insight("Residual demand is often one of the most important short-term power price drivers: it determines "
            "where on the merit order the marginal (price-setting) plant sits.")

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.driver_scatter(v, "residual_demand", "price", "Residual demand (GW)", "€/MWh", C["residual"],
                                          title="Residual demand vs price"), key="sc_rd")
    c2.plotly_chart(charts.driver_scatter(v, "wind_actual", "price", "Wind generation (GW)", "€/MWh", C["wind"],
                                          title="Wind generation vs price"), key="sc_wind")
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.driver_scatter(v, "solar_actual", "price", "Solar generation (GW)", "€/MWh", C["solar"],
                                          title="Solar generation vs price"), key="sc_solar")
    c4.plotly_chart(charts.driver_scatter(v, "load_error", "price_change_24h", "Load forecast error (GW, actual − forecast)",
                                          "Δ price vs same hour yesterday (€/MWh)", C["load"], fit=False,
                                          title="Load forecast error vs price change"), key="sc_lerr")
    insight("Renewable forecast errors matter because they can change the balance of the system close to delivery. "
            "Note that day-ahead prices are fixed before the error is known, so a contemporaneous relationship "
            "here reflects how well the market's own forecasts anticipated the published TSO forecast's miss.",
            icon="🎯")

    st.subheader("Merit-order view")
    coef, r2 = charts.quadratic_fit(v["residual_demand"] / charts.GW, v["price"])
    m1, m2, m3 = st.columns(3)
    m1.metric("R² of price on residual demand (quadratic)", f"{r2:.2f}" if pd.notna(r2) else "n/a", border=True,
              help="Share of hourly price variance explained by a convex function of residual demand alone.")
    slope = np.polyval(np.polyder(coef), np.nanmedian(v["residual_demand"]) / charts.GW) if np.isfinite(coef).all() else np.nan
    m2.metric("Price sensitivity at median residual demand", f"{slope:,.2f} €/MWh per GW" if pd.notna(slope) else "n/a",
              border=True, help="Slope of the fitted curve at the median residual demand: how much 1 GW of extra "
                                "residual demand moves the price around typical conditions.")
    m3.metric("Correlation: residual demand vs price", f"{v['residual_demand'].corr(v['price']):.2f}", border=True)

    c5, c6 = st.columns([3, 2])
    c5.plotly_chart(charts.empirical_merit_order(v), key="merit")
    with c6:
        st.markdown("**Reading the curve**")
        st.caption(
            "Each point is the median price for a residual-demand bucket. The shape is the market's *revealed* "
            "supply stack: flat where baseload and mid-merit plants are marginal, steep at the top where scarce "
            "peaking capacity sets the price, and dropping below zero at the bottom where renewables must be "
            "curtailed. Price spikes tend to occur when demand is high and renewable output is low."
        )
        st.caption(
            "Points far from the curve are where *other* drivers dominated: fuel and carbon prices, plant outages, "
            "interconnector congestion, hydro strategy or scarcity bidding."
        )
    if np.isfinite(coef).all():
        fitted = pd.Series(np.polyval(coef, v["residual_demand"] / charts.GW), index=v.index)
        st.plotly_chart(charts.fit_residual_chart(v, fitted), key="fit_resid")

    st.subheader("Correlations")
    c7, c8 = st.columns([1, 1])
    c7.plotly_chart(charts.correlation_heatmap(correlation_frame(v)), key="corr")
    with c8:
        window_days = st.select_slider("Rolling window", options=[3, 7, 14, 30], value=7, format_func=lambda d: f"{d} days")
        rc = v["residual_demand"].rolling(window_days * 24, min_periods=window_days * 12).corr(v["price"])
        st.plotly_chart(charts.rolling_correlation_chart(rc, f"{window_days}d"), key="rollcorr")
        st.caption("A falling correlation is a hint that something other than residual demand is driving prices - "
                   "for example a gas-price move, a large outage or cross-border constraints.")
