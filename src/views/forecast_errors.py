"""Page 3 - Forecast Error / Surprise Monitor."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.market_commentary import surprise_comment
from src.views.common import Ctx, gw, insight, page_header, require_rows

ERROR_OPTIONS = {
    "residual_demand_error": "Residual-demand error",
    "load_error": "Load error",
    "wind_error": "Wind error",
    "solar_error": "Solar error",
    "renewable_error": "Renewable (wind + solar) error",
}


def render(ctx: Ctx) -> None:
    page_header(ctx, "Forecast Error / Surprise Monitor", "Where did the day-ahead forecasts miss?")
    v = ctx.view
    if not require_rows(v):
        return

    st.markdown(
        "All errors are **actual − day-ahead forecast**. A positive *residual-demand error* means the system was "
        "**shorter** than the auction assumed (more load and/or less renewables), which typically supports intraday "
        "and imbalance prices relative to day-ahead."
    )
    k = st.columns(5)
    for col, (name, label) in zip(k, list(ERROR_OPTIONS.items())[:4] + [("renewable_error", "RES error")]):
        col.metric(f"{label} MAE", gw(v[name].abs().mean(), 2), delta=f"bias {v[name].mean() / 1000:+.2f} GW",
                   delta_color="off", border=True, help="Mean absolute error; the delta shows the mean (signed) error.")

    st.plotly_chart(charts.error_timeseries(v), key="err_ts")
    st.plotly_chart(charts.error_histograms(v), key="err_hist")

    c1, c2 = st.columns([1, 1])
    with c1:
        pick = st.selectbox("Heatmap variable", list(ERROR_OPTIONS), format_func=ERROR_OPTIONS.get, key="err_heat")
        st.plotly_chart(charts.hour_weekday_heatmap(v, pick, f"Mean {ERROR_OPTIONS[pick].lower()} by hour and weekday"),
                        key="err_heatmap")
    with c2:
        pick_z = st.selectbox("z-score variable", list(ERROR_OPTIONS), format_func=ERROR_OPTIONS.get, key="err_z")
        zcol = f"{pick_z}_z"
        st.plotly_chart(charts.zscore_chart(v, zcol, ERROR_OPTIONS[pick_z]), key="err_zchart")
        share = (v[zcol].abs() > 2).mean()
        st.caption(f"{share:.1%} of hours sit beyond ±2σ of the trailing 30-day distribution "
                   f"(≈4.6% if errors were normally distributed). Fat tails are typical for wind.")
        insight("Wind errors cluster in time (weather fronts arrive early or late), so a large miss in one hour "
                "is usually followed by more in the same direction - exactly what intraday desks trade.", icon="🌬️")

    st.subheader("Top 20 fundamental surprise hours")
    st.caption("Ranked by the absolute residual-demand forecast error. Price change is versus the same delivery hour on the previous day.")
    top = v.reindex(v["residual_demand_error"].abs().sort_values(ascending=False).index[:20])
    table = pd.DataFrame(
        {
            "timestamp": top.index.strftime("%Y-%m-%d %H:%M"),
            "country": ctx.zone,
            "price": top["price"],
            "load_error": top["load_error"] / 1000,
            "wind_error": top["wind_error"] / 1000,
            "solar_error": top["solar_error"] / 1000,
            "residual_demand_error": top["residual_demand_error"] / 1000,
            "price_change": top["price_change_24h"],
            "comment": [surprise_comment(r) for _, r in top.iterrows()],
        }
    )
    gw_fmt = st.column_config.NumberColumn(format="%+.2f GW")
    st.dataframe(
        table, hide_index=True,
        column_config={
            "price": st.column_config.NumberColumn("price €/MWh", format="%.2f"),
            "load_error": gw_fmt, "wind_error": gw_fmt, "solar_error": gw_fmt, "residual_demand_error": gw_fmt,
            "price_change": st.column_config.NumberColumn("price change €/MWh", format="%+.2f"),
            "comment": st.column_config.TextColumn(width="large"),
        },
    )

    st.subheader("Largest surprise days")
    daily = v.groupby("date").agg(
        residual_demand_error=("residual_demand_error", "mean"),
        abs_error=("residual_demand_error", lambda s: s.abs().mean()),
        wind_error=("wind_error", "mean"),
        load_error=("load_error", "mean"),
        baseload=("price", "mean"),
    ).sort_values("abs_error", ascending=False).head(10)
    daily.index = daily.index.date
    st.dataframe(
        daily / [1000, 1000, 1000, 1000, 1],
        column_config={
            "residual_demand_error": st.column_config.NumberColumn("mean resid. demand error", format="%+.2f GW"),
            "abs_error": st.column_config.NumberColumn("mean |error|", format="%.2f GW"),
            "wind_error": st.column_config.NumberColumn("mean wind error", format="%+.2f GW"),
            "load_error": st.column_config.NumberColumn("mean load error", format="%+.2f GW"),
            "baseload": st.column_config.NumberColumn("baseload €/MWh", format="%.1f"),
        },
    )
