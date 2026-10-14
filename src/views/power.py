"""Power deep dive: European day-ahead prices and what drives them."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.config import border_capacity, neighbours
from src.feature_engineering import border_summary, daily_summary, flag_spikes, net_flows_by_border
from src.market_commentary import explain_spikes, surprise_comment
from src.utils import slice_market_days
from src.views.common import Ctx, eur, gw, power_picker


def render(_=None) -> None:
    st.title("Power")
    st.caption("Day-ahead auction prices for European bidding zones, next to the load, wind and solar behind them.")
    ctx = power_picker()
    if ctx is None or len(ctx.view) < 48:
        return
    tabs = st.tabs(["Prices", "What drives it", "Spikes & surprises", "Interconnectors"])
    with tabs[0]:
        _prices(ctx)
    with tabs[1]:
        _drivers(ctx)
    with tabs[2]:
        _spikes(ctx)
    with tabs[3]:
        _flows(ctx)


def _prices(ctx: Ctx) -> None:
    v = ctx.view
    k = st.columns(4)
    k[0].metric("Average", eur(v["price"].mean(), 1) + "/MWh", border=True)
    k[1].metric("Low / high", f"{v['price'].min():,.0f} / {v['price'].max():,.0f}".replace("-", "−"), border=True)
    k[2].metric("Hours below zero", f"{int(v['is_negative'].sum())}", border=True)
    k[3].metric("Avg residual load", gw(v["residual_demand"].mean()), border=True,
                help="Load minus wind minus solar - what the gas, coal and nuclear fleet (plus imports) has to cover.")
    st.plotly_chart(charts.fundamentals_stack(v, ctx.spike_mask), key="pw_stack")
    st.plotly_chart(charts.daily_range_chart(daily_summary(v)), key="pw_daily")


def _drivers(ctx: Ctx) -> None:
    v = ctx.view
    st.markdown("Residual load (load − wind − solar) decides which power plant sets the price. Plot one against the "
                "other and you get the merit order back out of the data.")
    a, b = st.columns(2)
    a.plotly_chart(charts.driver_scatter(v, "residual_demand", "price", "Residual load (GW)", "€/MWh",
                                         charts.C["residual"], title="Residual load vs price"), key="pw_sc_rd")
    b.plotly_chart(charts.empirical_merit_order(v), key="pw_merit")
    c, d = st.columns(2)
    c.plotly_chart(charts.driver_scatter(v, "wind_actual", "price", "Wind (GW)", "€/MWh", charts.C["wind"],
                                         title="Wind vs price"), key="pw_sc_wind")
    d.plotly_chart(charts.driver_scatter(v, "solar_actual", "price", "Solar (GW)", "€/MWh", charts.C["solar"],
                                         title="Solar vs price"), key="pw_sc_solar")


def _spikes(ctx: Ctx) -> None:
    v = ctx.view
    pctile = st.slider("Call it a spike above this percentile", 80, 99, 90, key="pw_spike_pct")
    spikes, level = flag_spikes(v["price"], percentile=pctile / 100)
    st.markdown(explain_spikes(v, spikes))
    a, b = st.columns(2)
    a.plotly_chart(charts.spikes_by_hour(v, spikes), key="pw_spk_hour")
    groups = pd.Series("Normal", index=v.index)
    groups[spikes] = "Spike"
    groups[v["is_negative"]] = "Negative"
    b.plotly_chart(charts.regime_box(v, groups, "residual_demand", "Residual load in spike vs normal hours"),
                   key="pw_spk_box")

    st.markdown("#### Biggest forecast misses")
    st.caption("Actual minus the day-ahead forecast. The auction clears on the forecast, so these are the hours "
               "intraday traders had to fix.")
    top = v.reindex(v["residual_demand_error"].abs().sort_values(ascending=False).index[:12])
    st.dataframe(pd.DataFrame({
        "hour": top.index.strftime("%a %d %b %H:00"),
        "price": top["price"],
        "residual load miss (GW)": top["residual_demand_error"] / 1000,
        "wind miss (GW)": top["wind_error"] / 1000,
        "what happened": [surprise_comment(r) for _, r in top.iterrows()],
    }), hide_index=True, column_config={
        "price": st.column_config.NumberColumn("€/MWh", format="%.1f"),
        "residual load miss (GW)": st.column_config.NumberColumn(format="%+.1f"),
        "wind miss (GW)": st.column_config.NumberColumn(format="%+.1f"),
        "what happened": st.column_config.TextColumn(width="large"),
    })


def _flows(ctx: Ctx) -> None:
    if not ctx.data.has_flows:
        st.info("No cross-border flow data for this market yet.")
        return
    zones = [ctx.zone] + neighbours(ctx.zone)
    flows = ctx.data.flows
    flows = flows[flows["from_zone"].isin(zones) & flows["to_zone"].isin(zones)]
    net = slice_market_days(net_flows_by_border(flows, ctx.zone), ctx.start, ctx.end)
    if net.empty:
        st.info("No cross-border flow data for this period.")
        return
    prices = slice_market_days(ctx.data.zone_prices, ctx.start, ctx.end)
    caps = {}
    for nb in net.columns:
        caps[f"{nb}->{ctx.zone}"] = border_capacity(nb, ctx.zone)
        caps[f"{ctx.zone}->{nb}"] = border_capacity(ctx.zone, nb)
    st.plotly_chart(charts.border_flow_chart(net, ctx.zone), key="pw_borders")
    summary = border_summary(net, prices, ctx.zone, caps)
    if not summary.empty:
        st.dataframe(summary, column_config={
            "avg net import (MW)": st.column_config.NumberColumn(format="%+,.0f"),
            "avg spread nb - zone (EUR/MWh)": st.column_config.NumberColumn("avg spread (€/MWh)", format="%+.2f"),
            "price convergence": st.column_config.ProgressColumn("same price", min_value=0, max_value=1, format="percent"),
            "flow with spread": st.column_config.ProgressColumn("flows cheap → dear", min_value=0, max_value=1,
                                                                format="percent"),
            "near capacity": st.column_config.ProgressColumn("line full", min_value=0, max_value=1, format="percent"),
        })
    options = [nb for nb in net.columns if nb in prices]
    if options:
        nb = st.selectbox("Border", options, key="pw_flow_nb")
        spread = (prices[nb] - prices[ctx.zone]).reindex(net.index)
        st.plotly_chart(charts.flow_vs_spread(spread, -net[nb], ctx.zone, nb, border_capacity(ctx.zone, nb)),
                        key="pw_fvs")
        st.caption("When the line has room, coupling pulls both prices together (spread ≈ 0). Once it's full, "
                   "the spread opens up.")
