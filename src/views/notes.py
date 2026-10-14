"""Auto-written market notes: weekly oil, daily power."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.crude.commentary import generate_crude_brief
from src.market_commentary import generate_daily_brief
from src.views.common import build_crude_context, crude_params_from_state, power_picker, synthetic_note


def render(_=None) -> None:
    st.title("Notes")
    st.caption("Short write-ups built from the data with plain if/else rules, roughly how a desk note reads.")
    oil, power = st.tabs(["Oil, weekly", "Power, daily"])
    with oil:
        _oil()
    with power:
        _power()


def _note(title: str, headline: str, sections: dict[str, str], markdown: str, file_name: str) -> None:
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(f"*{headline}*")
        for name, text in sections.items():
            st.markdown(f"**{name}**")
            st.markdown(text)
    st.download_button("Download as markdown", markdown, file_name=file_name, mime="text/markdown")


def _oil() -> None:
    from src.views.backtests import oil_reading

    ctx = build_crude_context(5)
    if ctx.data.is_sample:
        synthetic_note("Oil")
    weeks = ctx.weekly["crude_stocks"].dropna().index.sort_values(ascending=False)
    picked = st.selectbox("EIA report, week ending", weeks[:156], format_func=lambda d: f"{d:%d %b %Y}",
                          key="note_week")
    latest = picked == weeks[0]
    row = oil_reading(ctx, crude_params_from_state()[0]) if latest else None
    brief = generate_crude_brief(ctx.weekly, ctx.daily, ctx.cot, picked, live=ctx.live if latest else None,
                                 signal_row=row)
    _note(brief.title, brief.headline, brief.sections, brief.to_markdown(), f"oil_{picked:%Y%m%d}.md")


def _power() -> None:
    ctx = power_picker()
    if ctx is None:
        return
    dates = sorted(ctx.view["date"].unique())
    picked = st.date_input("Delivery day", value=pd.Timestamp(dates[-1]).date(), min_value=pd.Timestamp(dates[0]).date(),
                           max_value=pd.Timestamp(dates[-1]).date(), format="YYYY-MM-DD", key="note_day")
    day = pd.Timestamp(picked)
    brief = generate_daily_brief(ctx.feat, ctx.zone, day, ctx.data.zone_prices, ctx.data.flows)
    _note(brief.title, brief.headline, brief.sections, brief.to_markdown(), f"power_{ctx.zone}_{picked}.md")
    feat = ctx.feat
    st.plotly_chart(charts.day_profile(feat[feat["date"] == day], feat[feat["date"] == day - pd.Timedelta(days=1)]),
                    key="note_profile")
