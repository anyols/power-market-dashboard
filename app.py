"""Spot & Spread - a commodities dashboard.  Run with:  streamlit run app.py"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Spot & Spread", page_icon=":material/show_chart:", layout="wide")

from src import charts  # noqa: E402
from src.views import about, backtests, board, exposure, notes, oil, power, ratios, seasonality  # noqa: E402
from src.views.common import persist_crude_signal_state, persist_signal_state  # noqa: E402

persist_signal_state()
persist_crude_signal_state()
theme = getattr(st.context, "theme", None)
charts.set_dark_mode(getattr(theme, "type", None) == "dark")

pages = {
    "": [st.Page(board.render, title="Board", icon=":material/dashboard:", url_path="board", default=True)],
    "Across markets": [
        st.Page(ratios.render, title="Ratios & spreads", icon=":material/balance:", url_path="ratios"),
        st.Page(exposure.render, title="Stocks vs commodities", icon=":material/compare_arrows:", url_path="stocks"),
        st.Page(seasonality.render, title="Seasonality", icon=":material/calendar_month:", url_path="seasonality"),
    ],
    "Deep dives": [
        st.Page(power.render, title="Power", icon=":material/bolt:", url_path="power"),
        st.Page(oil.render, title="Oil", icon=":material/oil_barrel:", url_path="oil"),
        st.Page(notes.render, title="Notes", icon=":material/edit_note:", url_path="notes"),
        st.Page(backtests.render, title="Backtests", icon=":material/science:", url_path="backtests"),
    ],
    " ": [st.Page(about.render, title="About", icon=":material/info:", url_path="about")],
}
nav = st.navigation(pages)
with st.sidebar:
    st.markdown("**Spot & Spread**  \n<small>power · oil · metals · ags</small>", unsafe_allow_html=True)
nav.run()
