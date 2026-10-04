"""Energy Market Trading Dashboard - European power and crude oil. Streamlit entry point.

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Energy Market Trading Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src import charts  # noqa: E402
from src.config import ZONES, get_settings  # noqa: E402
from src.data_loader import available_zones, resolve_source, store_period  # noqa: E402
from src.views import (  # noqa: E402
    crude_brief,
    crude_curve,
    crude_inventories,
    crude_overview,
    crude_signal_lab,
    daily_brief,
    data_status,
    drivers,
    flows,
    forecast_errors,
    methodology,
    overview,
    risk,
    signal_lab,
    spikes,
)
from src.views.common import (  # noqa: E402
    build_context,
    build_crude_context,
    cached_sample_period,
    persist_crude_signal_state,
    persist_signal_state,
)

persist_signal_state()
persist_crude_signal_state()
theme = getattr(st.context, "theme", None)
charts.set_dark_mode(getattr(theme, "type", None) == "dark")

STATE: dict = {}
KIND: dict[str, str] = {}


def _page(module, title: str, icon: str, url: str, kind: str = "power", default: bool = False) -> st.Page:
    KIND[url] = kind
    return st.Page(lambda: module.render(STATE.get(kind)), title=title, icon=icon, url_path=url, default=default)


pages = {
    "Power · Market": [
        _page(overview, "Market Overview", "📈", "overview", default=True),
        _page(drivers, "Price Drivers", "🧭", "drivers"),
        _page(forecast_errors, "Forecast Surprises", "🎯", "forecast-errors"),
        _page(spikes, "Spikes & Volatility", "⚡", "spikes"),
        _page(flows, "Cross-Border Flows", "🔀", "flows"),
    ],
    "Power · Trading": [
        _page(signal_lab, "Signal Lab", "🧪", "signal-lab"),
        _page(daily_brief, "Daily Brief", "📝", "daily-brief"),
        _page(risk, "Risk Dashboard", "🛡️", "risk"),
    ],
    "Crude Oil": [
        _page(crude_overview, "Crude Overview", "🛢️", "crude-overview", kind="crude"),
        _page(crude_inventories, "Inventories & Balances", "🏭", "crude-inventories", kind="crude"),
        _page(crude_curve, "Curve & Positioning", "📐", "crude-curve", kind="crude"),
        _page(crude_signal_lab, "Crude Signal Lab", "🧪", "crude-signal-lab", kind="crude"),
        _page(crude_brief, "Weekly Crude Brief", "📝", "crude-brief", kind="crude"),
    ],
    "About": [
        _page(data_status, "Data Pipeline Status", "🔄", "data-status", kind="status"),
        _page(methodology, "Methodology & Data", "📚", "methodology"),
    ],
}
nav = st.navigation(pages, expanded=True)
kind = KIND.get(nav.url_path, "power")

with st.sidebar:
    st.markdown("### ⚡ Energy Trading Dashboard")


def power_sidebar() -> None:
    source = resolve_source(get_settings())
    with st.sidebar:
        badge = {"sample": ("Synthetic sample data", "orange", "🧪"), "store": ("ENTSO-E · refreshed daily", "green", "🔄"),
                 "entsoe": ("ENTSO-E live API", "green", "📡")}[source]
        st.badge(badge[0], color=badge[1], icon=badge[2])
        zone = st.selectbox("Bidding zone", available_zones(source), format_func=lambda z: f"{ZONES[z].name} ({z})",
                            key="zone")
        if source == "sample":
            first, last = cached_sample_period()
            default_start = max(first, last - pd.Timedelta(days=59).to_pytimedelta())
        elif source == "store":
            first, last = store_period(zone)
            default_start = max(first, last - pd.Timedelta(days=59).to_pytimedelta())
        else:
            last = (pd.Timestamp.now(tz="Europe/Brussels") + pd.Timedelta(days=1)).date()
            first = pd.Timestamp("2015-01-05").date()
            default_start = last - pd.Timedelta(days=30).to_pytimedelta()
        picked = st.date_input("Delivery period", value=(default_start, last), min_value=first, max_value=last,
                               format="YYYY-MM-DD", key=f"period_{source}")
        if not isinstance(picked, tuple) or len(picked) != 2:
            st.info("Select a start and an end date.")
            st.stop()
        start, end = picked
        if (end - start).days < 6:
            st.warning("Select at least 7 days for meaningful statistics.")
            st.stop()
        with st.expander("Spike definition", expanded=False):
            spike_mode = st.radio("Method", ["Percentile", "Absolute threshold"], horizontal=True, key="spike_mode")
            spike_pct = st.slider("Percentile of selected period", 75, 99, 90, key="spike_pct",
                                  disabled=spike_mode != "Percentile")
            spike_abs = st.number_input("Threshold (€/MWh)", value=150.0, step=10.0, key="spike_abs",
                                        disabled=spike_mode == "Percentile")
    try:
        ctx = build_context(zone, start, end, source, spike_mode, float(spike_pct), float(spike_abs))
    except Exception as exc:  # surface data problems clearly instead of a stack trace
        st.error(f"Could not load data for {zone}: {exc}")
        st.stop()
    STATE["power"] = ctx
    with st.sidebar:
        q = ctx.data.quality
        with st.expander("Data quality" + (" ⚠️" if q.warnings else " ✅"), expanded=False):
            for w in q.warnings or ["No issues detected in the loaded window."]:
                st.caption(f"• {w}")
            st.caption(f"{q.rows:,} hourly rows loaded (incl. warm-up history for trailing statistics).")


def crude_sidebar() -> None:
    with st.sidebar:
        window = st.selectbox("Chart window", ["1 year", "2 years", "5 years", "10 years", "Max"], index=1, key="crude_window")
    ctx = build_crude_context(window)
    STATE["crude"] = ctx
    with st.sidebar:
        if ctx.data.is_sample:
            st.badge("Synthetic sample data", color="orange", icon="🧪")
        else:
            st.badge("EIA + CFTC · refreshed daily", color="green", icon="🔄")
            st.badge("Live prices: Yahoo (unofficial)" if ctx.live.ok else "Live prices unavailable",
                     color="blue" if ctx.live.ok else "gray")


if kind == "power":
    power_sidebar()
elif kind == "crude":
    crude_sidebar()

with st.sidebar:
    st.caption("Educational research tool. Not investment advice. Signals and backtests are illustrative and use "
               "simplified assumptions.")

nav.run()
