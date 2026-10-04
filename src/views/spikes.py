"""Page 4 - Spike & Volatility Analyzer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.market_commentary import explain_spikes
from src.views.common import Ctx, eur, gw, page_header, pct, require_rows


def render(ctx: Ctx) -> None:
    page_header(ctx, "Spike & Volatility Analyzer", "What does the system look like when prices blow out?")
    v = ctx.view
    if not require_rows(v):
        return
    spikes = ctx.spike_mask.reindex(v.index).fillna(False).astype(bool)
    groups = pd.Series("Normal", index=v.index)
    groups[spikes] = "Spike"
    groups[v["is_negative"]] = "Negative"

    st.caption(f"Spike definition: {ctx.spike_label}. Negative price: price < 0 €/MWh. Change it in the sidebar.")
    k = st.columns(5)
    k[0].metric("Spike threshold", eur(ctx.spike_level), border=True)
    k[1].metric("Spike hours", f"{int(spikes.sum()):,}", delta=pct(spikes.mean(), 1) + " of hours", delta_color="off", border=True)
    k[2].metric("Avg spike price", eur(v.loc[spikes, "price"].mean()), border=True)
    k[3].metric("Negative hours", f"{int(v['is_negative'].sum()):,}", border=True)
    k[4].metric("Resid. demand: spike vs normal",
                gw(v.loc[spikes, "residual_demand"].mean() - v.loc[groups == "Normal", "residual_demand"].mean()),
                help="Average residual demand in spike hours minus normal hours.", border=True)

    st.info(explain_spikes(v, spikes), icon="🗒️")

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.price_distribution(v, ctx.spike_level), key="dist")
    c2.plotly_chart(charts.spikes_by_hour(v, spikes), key="byhour")

    st.subheader("Fundamentals during spikes vs normal hours")
    b1, b2 = st.columns(2)
    b1.plotly_chart(charts.regime_box(v, groups, "residual_demand", "Residual demand"), key="box_rd")
    b2.plotly_chart(charts.regime_box(v, groups, "wind_actual", "Wind generation"), key="box_wind")
    b3, b4 = st.columns(2)
    b3.plotly_chart(charts.regime_box(v, groups, "solar_actual", "Solar generation"), key="box_solar")
    b4.plotly_chart(charts.regime_box(v, groups, "residual_demand_error", "Residual-demand forecast error"), key="box_err")

    comparison = v.groupby(groups).agg(
        hours=("price", "size"),
        price=("price", "mean"),
        residual_demand=("residual_demand", "mean"),
        load=("load_actual", "mean"),
        wind=("wind_actual", "mean"),
        solar=("solar_actual", "mean"),
        load_error=("load_error", "mean"),
        renewable_error=("renewable_error", "mean"),
        peak_share=("is_peak", "mean"),
    ).reindex(["Normal", "Spike", "Negative"]).dropna(how="all")
    for col in ["residual_demand", "load", "wind", "solar", "load_error", "renewable_error"]:
        comparison[col] = comparison[col] / 1000
    gw_col = lambda label: st.column_config.NumberColumn(label, format="%.2f")  # noqa: E731
    st.dataframe(
        comparison,
        column_config={
            "hours": st.column_config.NumberColumn("hours", format="%d"),
            "price": st.column_config.NumberColumn("avg price €/MWh", format="%.1f"),
            "residual_demand": gw_col("resid. demand GW"), "load": gw_col("load GW"), "wind": gw_col("wind GW"),
            "solar": gw_col("solar GW"), "load_error": gw_col("load error GW"),
            "renewable_error": gw_col("RES error GW"),
            "peak_share": st.column_config.ProgressColumn("share in peak block", min_value=0, max_value=1, format="percent"),
        },
    )

    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.spike_frequency_heatmap(v, spikes), key="spike_heat")
    with c4:
        st.plotly_chart(charts.volatility_chart(v), key="vol")
        regime_share = v["vol_regime"].value_counts(normalize=True).reindex(["low", "normal", "high"]).fillna(0)
        st.caption("Share of hours by volatility regime (thresholds from the series' own past): "
                   + ", ".join(f"{k} {val:.0%}" for k, val in regime_share.items()))

    st.subheader("Spike table")
    top = v[spikes].sort_values("price", ascending=False).head(50)
    st.dataframe(
        pd.DataFrame({
            "timestamp": top.index.strftime("%Y-%m-%d %H:%M"),
            "price": top["price"],
            "residual_demand_gw": top["residual_demand"] / 1000,
            "wind_gw": top["wind_actual"] / 1000,
            "solar_gw": top["solar_actual"] / 1000,
            "load_error_gw": top["load_error"] / 1000,
            "wind_error_gw": top["wind_error"] / 1000,
        }),
        hide_index=True,
        column_config={
            "price": st.column_config.NumberColumn("price €/MWh", format="%.2f"),
            "residual_demand_gw": gw_col("resid. demand GW"), "wind_gw": gw_col("wind GW"),
            "solar_gw": gw_col("solar GW"), "load_error_gw": gw_col("load error GW"),
            "wind_error_gw": gw_col("wind error GW"),
        },
    )
