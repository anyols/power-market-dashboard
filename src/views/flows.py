"""Page 5 - Cross-Border Flow Monitor."""

from __future__ import annotations

import streamlit as st

from src import charts
from src.config import border_capacity, neighbours
from src.feature_engineering import border_summary, net_flows_by_border
from src.utils import slice_market_days
from src.views.common import Ctx, gw, page_header, pct

UNAVAILABLE = (
    "Cross-border flow data unavailable for this configuration. "
    "Add ENTSO-E flow endpoint or cached data to enable this page."
)


def render(ctx: Ctx) -> None:
    page_header(ctx, "Cross-Border Flow Monitor", "Imports, exports and regional price dislocations")
    flows = ctx.data.flows
    if not ctx.data.has_flows:
        st.warning(UNAVAILABLE, icon="🔌")
        st.caption(
            "The module is wired up: `EntsoeClient.get_cross_border_flows` (document type A11) feeds "
            "`data_loader.load_entsoe_flows`, and borders are configured in `src/config.py`."
        )
        return

    flows = flows[flows["from_zone"].isin([ctx.zone] + neighbours(ctx.zone)) & flows["to_zone"].isin([ctx.zone] + neighbours(ctx.zone))]
    net = slice_market_days(net_flows_by_border(flows, ctx.zone), ctx.start, ctx.end)
    if net.empty:
        st.warning(UNAVAILABLE, icon="🔌")
        return
    prices = slice_market_days(ctx.data.zone_prices, ctx.start, ctx.end)
    caps = {}
    for nb in net.columns:
        caps[f"{nb}->{ctx.zone}"] = border_capacity(nb, ctx.zone)
        caps[f"{ctx.zone}->{nb}"] = border_capacity(ctx.zone, nb)
    summary = border_summary(net, prices, ctx.zone, caps)
    total = net.sum(axis=1)

    k = st.columns(4)
    k[0].metric("Avg net position", gw(total.mean(), 2), help="+ = net importer across the modelled borders.", border=True)
    k[1].metric("Hours as net importer", pct((total > 0).mean()), border=True)
    if not summary.empty:
        most = summary["near capacity"].idxmax() if summary["near capacity"].notna().any() else None
        k[2].metric("Most congested border", f"{ctx.zone}–{most}" if most else "n/a",
                    delta=pct(summary.loc[most, "near capacity"]) + " of hours near capacity" if most else None,
                    delta_color="off", border=True)
        best = summary["price convergence"].idxmax()
        k[3].metric("Most coupled neighbour", best, delta=pct(summary.loc[best, "price convergence"]) + " hours within €1",
                    delta_color="off", border=True)

    daily_net = total.groupby(total.index.tz_localize(None).normalize()).mean()
    st.plotly_chart(charts.net_position_chart(daily_net), key="netpos")
    st.caption("Daily average net position. Modelled borders only - flows with other neighbours are not shown.")
    st.plotly_chart(charts.border_flow_chart(net, ctx.zone), key="borders")

    st.subheader("Border summary")
    if not summary.empty:
        st.dataframe(
            summary,
            column_config={
                "avg net import (MW)": st.column_config.NumberColumn(format="%+,.0f"),
                "avg spread nb - zone (EUR/MWh)": st.column_config.NumberColumn(format="%+.2f"),
                "price convergence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                "flow with spread": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                "near capacity": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            },
        )
        st.caption("*Flow with spread*: share of hours with a >€1 spread where power moved from the cheaper to the "
                   "dearer zone. *Near capacity*: |flow| above 90% of an indicative capacity - a congestion proxy, since "
                   "real Core capacities are flow-based and change every hour.")

    st.subheader("Spread vs flow by border")
    options = [nb for nb in net.columns if nb in prices]
    if not options:
        st.info("Neighbour prices unavailable; spreads cannot be computed.")
        return
    nb = st.selectbox("Neighbour", options, key="flow_nb")
    spread = (prices[nb] - prices[ctx.zone]).reindex(net.index)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.spread_chart(spread, ctx.zone, nb), key="spread")
    c2.plotly_chart(charts.flow_vs_spread(spread, -net[nb], ctx.zone, nb, border_capacity(ctx.zone, nb)), key="fvs")

    with st.expander("Trading interpretation", expanded=True):
        st.markdown(
            f"""
- **Imports can relieve tightness.** When {ctx.zone} imports, neighbours' cheaper generation substitutes local
  peakers - which is why residual demand alone overstates local tightness when interconnectors are open.
- **Exports can tighten the local market.** Exporting adds to the generation the zone must provide, pulling its
  price up towards its neighbours'.
- **Interconnector constraints create price spreads.** While capacity is available, market coupling equalises prices;
  once a line congests the spread opens and the congestion rent goes to the TSOs.
- **Flow direction matters for regional dislocations.** A spread *with* flows at capacity is structural congestion;
  a spread with flows running the "wrong" way points to loop flows or flow-based effects worth investigating.
"""
        )
