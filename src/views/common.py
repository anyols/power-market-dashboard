"""Shared caching, context objects and small UI helpers for the pages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from src.config import ZONES, BacktestParams, SignalParams, get_settings
from src.data_loader import MarketData, available_zones, load_market_data, resolve_source, sample_period, store_period
from src.feature_engineering import build_features, flag_spikes
from src.utils import slice_market_days


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def eur(x: float, decimals: int = 0) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{'−' if x < 0 else ''}€{abs(x):,.{decimals}f}"


def usd(x: float, decimals: int = 2, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    sign = "+" if signed and x > 0 else ("−" if x < 0 else "")
    return f"{sign}${abs(x):,.{decimals}f}"


def pct(x: float, decimals: int = 0, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    sign = "+" if signed and x > 0 else ("−" if x < 0 else "")
    return f"{sign}{abs(x) * 100:.{decimals}f}%"


def gw(mw: float, decimals: int = 1) -> str:
    return "n/a" if mw is None or pd.isna(mw) else f"{mw / 1000:,.{decimals}f} GW"


def mb(kb: float, signed: bool = False) -> str:
    if kb is None or pd.isna(kb):
        return "n/a"
    sign = "+" if signed and kb > 0 else ("−" if kb < 0 else "")
    return f"{sign}{abs(kb) / 1000:,.1f} mb"


def synthetic_note(what: str) -> None:
    st.caption(f":orange[●] {what} is running on the synthetic sample data (see About).")


# ---------------------------------------------------------------------------
# Cross-market prices (Yahoo)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching prices…")
def cached_markets():
    from src.markets import load_prices

    return load_prices()


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    zone: str
    zone_name: str
    start: date
    end: date
    data: MarketData
    feat: pd.DataFrame  # full history incl. warm-up for trailing statistics
    view: pd.DataFrame  # selected window only
    spike_mask: pd.Series
    spike_level: float
    spike_label: str


@st.cache_data(show_spinner=False)
def cached_sample_period() -> tuple[date, date]:
    return sample_period()


@st.cache_data(show_spinner="Loading power data…", max_entries=16)
def cached_market_data(zone: str, start: date, end: date, source_key: str) -> tuple[MarketData, pd.DataFrame]:
    data = load_market_data(zone, start, end)
    return data, build_features(data.frame)


def build_context(zone: str, start: date, end: date, source_key: str, spike_mode: str = "Percentile",
                  spike_pct: float = 90.0, spike_abs: float = 150.0) -> Ctx:
    data, feat = cached_market_data(zone, start, end, source_key)
    view = slice_market_days(feat, start, end)
    if spike_mode == "Percentile":
        mask, level = flag_spikes(view["price"], percentile=spike_pct / 100)
        label = f"above the {spike_pct:.0f}th percentile of the window ({level:,.0f} €/MWh)"
    else:
        mask, level = flag_spikes(view["price"], threshold=spike_abs)
        label = f"above {spike_abs:,.0f} €/MWh"
    return Ctx(zone, ZONES[zone].name, start, end, data, feat, view, mask, level, label)


def power_picker(spike_pct: float = 90.0) -> Ctx | None:
    """Zone + date range, inline at the top of the page."""
    source = resolve_source(get_settings())
    c1, c2, _ = st.columns([2, 3, 3])
    zone = c1.selectbox("Market", available_zones(source), format_func=lambda z: ZONES[z].name, key="zone")
    if source == "sample":
        first, last = cached_sample_period()
    elif source == "store":
        first, last = store_period(zone)
    else:
        last = (pd.Timestamp.now(tz="Europe/Brussels") + pd.Timedelta(days=1)).date()
        first = pd.Timestamp("2015-01-05").date()
    default_start = max(first, last - pd.Timedelta(days=59).to_pytimedelta())
    picked = c2.date_input("Period", value=(default_start, last), min_value=first, max_value=last,
                           format="YYYY-MM-DD", key=f"period_{source}")
    if not isinstance(picked, tuple) or len(picked) != 2 or (picked[1] - picked[0]).days < 6:
        st.info("Pick a period of at least a week.")
        return None
    try:
        ctx = build_context(zone, picked[0], picked[1], source, "Percentile", spike_pct)
    except Exception as exc:
        st.error(f"Couldn't load {zone}: {exc}")
        return None
    if ctx.data.is_sample:
        synthetic_note("Power")
    return ctx


SIGNAL_DEFAULTS = {
    "sig_horizon": "hourly", "sig_rd_mode": "change", "sig_rd_thr": 1.0, "sig_surprise_thr": 0.5,
    "sig_window": 30, "sig_entry": 2, "sig_momentum": False, "sig_cost": 1.0,
}


def persist_signal_state() -> None:
    """Keep widget values when switching pages."""
    for key, default in SIGNAL_DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, default)
    for key in ("zone",):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def signal_controls() -> tuple[SignalParams, BacktestParams]:
    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("Horizon", ["hourly", "daily"], key="sig_horizon",
                 format_func=lambda h: "hour vs same hour yesterday" if h == "hourly" else "baseload vs yesterday")
    c2.selectbox("Residual load input", ["change", "level"], key="sig_rd_mode",
                 format_func=lambda m: "change vs yesterday" if m == "change" else "level vs 30d mean")
    c3.slider("Residual load threshold (z)", 0.25, 2.5, step=0.25, key="sig_rd_thr")
    c4.slider("Surprise threshold (z)", 0.0, 2.0, step=0.25, key="sig_surprise_thr")
    c5, c6, c7, c8 = st.columns(4)
    c5.select_slider("Trade when |score| ≥", options=[1, 2, 3], key="sig_entry")
    c6.slider("Lookback (days)", 14, 90, step=1, key="sig_window")
    c7.number_input("Cost per MWh (€)", 0.0, 10.0, step=0.25, key="sig_cost")
    c8.toggle("Only with momentum", key="sig_momentum")
    s = st.session_state
    sp = SignalParams(horizon=s["sig_horizon"], rd_mode=s["sig_rd_mode"], rd_z_threshold=s["sig_rd_thr"],
                      surprise_z_threshold=s["sig_surprise_thr"], zscore_window_days=s["sig_window"],
                      entry_threshold=s["sig_entry"], use_momentum_filter=s["sig_momentum"])
    return sp, BacktestParams(cost_per_mwh=s["sig_cost"])


# ---------------------------------------------------------------------------
# Oil
# ---------------------------------------------------------------------------
@dataclass
class CrudeCtx:
    data: object  # CrudeData
    daily: pd.DataFrame
    weekly: pd.DataFrame
    cot: pd.DataFrame
    live: object  # LiveCrude
    start: pd.Timestamp


@st.cache_data(ttl=3600, show_spinner="Loading oil data…")
def cached_crude():
    from src.crude.data import load_crude_data
    from src.crude.features import add_cot_features, add_daily_features, add_weekly_features

    data = load_crude_data()
    return data, add_daily_features(data.daily), add_weekly_features(data.weekly), add_cot_features(data.cot)


@st.cache_data(ttl=3600, show_spinner="Fetching futures…")
def cached_live(is_sample: bool):
    from src.crude.data import load_crude_data
    from src.crude.live import fetch_live_crude, sample_live

    return sample_live(load_crude_data()) if is_sample else fetch_live_crude()


def build_crude_context(years: int | None = 3) -> CrudeCtx:
    data, daily, weekly, cot = cached_crude()
    live = cached_live(data.is_sample)
    last = max(daily.index.max(), weekly.index.max())
    start = last - pd.DateOffset(years=years) if years else daily.index.min()
    return CrudeCtx(data, daily, weekly, cot, live, pd.Timestamp(start))


CRUDE_SIGNAL_DEFAULTS = {
    "csig_target": "futures", "csig_curve": True, "csig_positioning": True, "csig_surprise": 1.0,
    "csig_cushing": 1.0, "csig_curve_thr": 0.10, "csig_crowd": 0.85, "csig_entry": 2, "csig_cost": 0.03,
}


def persist_crude_signal_state() -> None:
    for key, default in CRUDE_SIGNAL_DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, default)


def crude_signal_controls():
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.selectbox("Price", ["futures", "spot"], key="csig_target",
                 format_func=lambda t: "front-month futures" if t == "futures" else "WTI spot",
                 help="EIA stopped publishing futures in April 2024, so the futures test ends there.")
    c2.toggle("Curve vote", key="csig_curve")
    c3.toggle("Positioning vote", key="csig_positioning")
    c4.select_slider("Trade when |score| ≥", options=[1, 2, 3, 4], key="csig_entry")
    c5.number_input("Cost ($/bbl)", 0.0, 0.5, step=0.01, key="csig_cost", format="%.2f")
    c6, c7, c8, c9 = st.columns(4)
    c6.slider("Inventory surprise (z)", 0.25, 2.5, step=0.25, key="csig_surprise")
    c7.slider("Cushing (z)", 0.25, 2.5, step=0.25, key="csig_cushing")
    c8.slider("Curve dead-band ($)", 0.0, 1.0, step=0.05, key="csig_curve_thr")
    c9.slider("Crowded above pctile", 0.6, 0.95, step=0.05, key="csig_crowd")
    return crude_params_from_state()


def crude_params_from_state():
    from src.crude.signals import CrudeSignalParams

    s = {k: st.session_state.get(k, v) for k, v in CRUDE_SIGNAL_DEFAULTS.items()}  # read-only
    params = CrudeSignalParams(target=s["csig_target"], use_curve=s["csig_curve"], use_positioning=s["csig_positioning"],
                               surprise_z=s["csig_surprise"], cushing_z=s["csig_cushing"],
                               curve_threshold=s["csig_curve_thr"], crowding_pctile=s["csig_crowd"],
                               entry_threshold=s["csig_entry"])
    return params, BacktestParams(cost_per_mwh=s["csig_cost"])
