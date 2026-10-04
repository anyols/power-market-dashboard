"""Crude page 5 - deterministic weekly brief on the latest EIA report."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.crude.commentary import generate_crude_brief
from src.crude.features import seasonal_profile
from src.views.common import CrudeCtx, crude_header, crude_params_from_state
from src.views.crude_signal_lab import live_reading


def render(ctx: CrudeCtx) -> None:
    crude_header(ctx, "Weekly Crude Brief", "Rule-based desk note on the EIA Weekly Petroleum Status Report - no LLM")
    weeks = ctx.weekly["crude_stocks"].dropna().index.sort_values(ascending=False)
    picked = st.selectbox("Report (week ending)", weeks[:156], format_func=lambda d: f"{d:%a %d %b %Y}", key="cb_week")
    is_latest = picked == weeks[0]

    signal_row = live_reading(ctx, crude_params_from_state()[0]) if is_latest else None

    brief = generate_crude_brief(ctx.weekly, ctx.daily, ctx.cot, picked, live=ctx.live if is_latest else None,
                                 signal_row=signal_row)
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown(f"### {brief.title}")
            st.markdown(f"**{brief.headline}**")
            for i, (name, text) in enumerate(brief.sections.items(), start=1):
                st.markdown(f"**{i}. {name}**")
                st.markdown(text)
        st.download_button("Download brief (.md)", brief.to_markdown(), file_name=f"crude_brief_{picked:%Y%m%d}.md",
                           mime="text/markdown")
    with right:
        year = picked.isocalendar().year
        hist = ctx.weekly[ctx.weekly.index <= picked]
        st.plotly_chart(charts.seasonal_band_chart(seasonal_profile(hist["crude_stocks"], year),
                                                   "US crude stocks vs 5-year range", "mb", year, scale=1000),
                        key="cb_band")
        st.plotly_chart(charts.seasonal_band_chart(seasonal_profile(hist["cushing_stocks"], year),
                                                   "Cushing stocks vs 5-year range", "mb", year, scale=1000),
                        key="cb_cush")
        st.caption("Written as of the release date: balances to the week ending, prices to release day, and CFTC "
                   "reports already published. Historical briefs never use the live Yahoo layer.")
    if not is_latest:
        st.info(f"Historical brief: written as of {picked + pd.Timedelta(days=5):%a %d %b %Y} with official data only.")
