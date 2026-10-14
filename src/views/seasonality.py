"""Seasonality: does the calendar matter for a commodity?"""

from __future__ import annotations

import streamlit as st

from src import charts
from src.charts import MONTHS
from src.markets import ASSETS, COMMODITY_GROUPS, month_stats, monthly_returns, seasonal_paths
from src.views.common import cached_markets, pct


def render(_=None) -> None:
    st.title("Seasonality")
    st.caption("Heating season for gas, driving season for gasoline, Brazilian frost season for coffee, "
               "harvest for grains. Some of it shows up in prices, a lot of it is already priced in.")
    mp = cached_markets()
    p = mp.prices
    if p.empty:
        st.error("No price data available right now.")
        return

    keys = [a.key for a in ASSETS.values() if a.group in COMMODITY_GROUPS and p[a.key].notna().any()]
    c1, c2 = st.columns([2, 1])
    k = c1.selectbox("Commodity", keys, index=keys.index("natgas") if "natgas" in keys else 0,
                     format_func=lambda x: ASSETS[x].name, key="season_asset")
    years = c2.select_slider("Years", [5, 10, 15], value=10, key="season_years")

    s = p[k].dropna()
    current = s.index.max().year
    table = monthly_returns(s)
    table = table[table.index >= current - years]
    stats = month_stats(table, exclude_year=current)

    best, worst = stats["mean"].idxmax(), stats["mean"].idxmin()
    st.markdown(
        f"Over {int(stats['years'].max())} years, **{MONTHS[best - 1]}** has been the best month for "
        f"{ASSETS[k].name.lower()} ({pct(stats.loc[best, 'mean'], 1, signed=True)} on average, up in "
        f"{pct(stats.loc[best, 'share_up'])} of years) and **{MONTHS[worst - 1]}** the worst "
        f"({pct(stats.loc[worst, 'mean'], 1, signed=True)}, up in {pct(stats.loc[worst, 'share_up'])})."
    )
    st.plotly_chart(charts.seasonal_path_chart(seasonal_paths(s, years), current), key="season_path")
    a, b = st.columns([3, 2])
    a.plotly_chart(charts.seasonality_heatmap(table), key="season_heat")
    b.plotly_chart(charts.month_bars(stats), key="season_bars")
    st.caption(f"Monthly changes in the continuous front-month price ({ASSETS[k].ticker}), not the return of holding "
               f"a contract: the jumps at contract rolls (large in gas) are in here. {current} is excluded from the "
               "averages.")
