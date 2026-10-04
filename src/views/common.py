"""Shared state, caching and UI helpers for the dashboard pages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from src.config import ZONES, BacktestParams, SignalParams
from src.data_loader import MarketData, load_market_data, sample_period
from src.feature_engineering import build_features, flag_spikes
from src.market_commentary import RegimeNote
from src.utils import slice_market_days

TONE_BADGE = {"bullish": ("red", "▲"), "bearish": ("blue", "▼"), "risk": ("violet", "⚠"), "neutral": ("gray", "•")}


@dataclass
class Ctx:
    zone: str
    zone_name: str
    start: date
    end: date
    data: MarketData
    feat: pd.DataFrame  # full history incl. warm-up (for trailing statistics)
    view: pd.DataFrame  # selected window only
    spike_mask: pd.Series
    spike_level: float
    spike_label: str

    @property
    def source_label(self) -> str:
        return "Synthetic sample data" if self.data.is_sample else "ENTSO-E Transparency Platform"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_sample_period() -> tuple[date, date]:
    return sample_period()


@st.cache_data(show_spinner="Loading market data…", max_entries=16)
def cached_market_data(zone: str, start: date, end: date, source_key: str) -> tuple[MarketData, pd.DataFrame]:
    """Load + feature-engineer once per (zone, window, source)."""
    data = load_market_data(zone, start, end)
    return data, build_features(data.frame)


def build_context(zone: str, start: date, end: date, source_key: str, spike_mode: str,
                  spike_pct: float, spike_abs: float) -> Ctx:
    data, feat = cached_market_data(zone, start, end, source_key)
    view = slice_market_days(feat, start, end)
    if spike_mode == "Percentile":
        mask, level = flag_spikes(view["price"], percentile=spike_pct / 100)
        label = f"price above the {spike_pct:.0f}th percentile of the selected period ({level:,.0f} €/MWh)"
    else:
        mask, level = flag_spikes(view["price"], threshold=spike_abs)
        label = f"price above {spike_abs:,.0f} €/MWh"
    return Ctx(zone, ZONES[zone].name, start, end, data, feat, view, mask, level, label)


# ---------------------------------------------------------------------------
# Strategy settings shared by the Signal Lab and Risk pages
# ---------------------------------------------------------------------------
SIGNAL_DEFAULTS = {
    "sig_horizon": "hourly",
    "sig_rd_mode": "change",
    "sig_rd_thr": 1.0,
    "sig_surprise_thr": 0.5,
    "sig_window": 30,
    "sig_entry": 2,
    "sig_momentum": False,
    "sig_cost": 1.0,
}


def persist_signal_state() -> None:
    """Keep strategy widgets' values when navigating between pages."""
    for key, default in SIGNAL_DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, default)


def signal_controls() -> tuple[SignalParams, BacktestParams]:
    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("Horizon", ["hourly", "daily"], key="sig_horizon",
                 format_func=lambda h: "Each hour vs same hour yesterday" if h == "hourly" else "Baseload vs yesterday's baseload")
    c2.selectbox("Residual-demand input", ["change", "level"], key="sig_rd_mode",
                 format_func=lambda m: "Change vs reference day" if m == "change" else "Level vs trailing mean",
                 help="'Change' compares tomorrow's forecast residual demand with the forecast that yesterday's price "
                      "was set on. 'Level' compares it with its trailing mean.")
    c3.slider("Residual-demand z threshold", 0.25, 2.5, step=0.25, key="sig_rd_thr")
    c4.slider("Surprise z threshold", 0.0, 2.0, step=0.25, key="sig_surprise_thr",
              help="Dead-band applied to the (lagged) load and renewable forecast surprises.")
    c5, c6, c7, c8 = st.columns(4)
    c5.select_slider("Entry: |score| ≥", options=[1, 2, 3], key="sig_entry")
    c6.slider("Trailing window (days)", 14, 90, step=1, key="sig_window")
    c7.number_input("Cost + slippage (€/MWh)", 0.0, 10.0, step=0.25, key="sig_cost",
                    help="Charged on every traded MWh. Covers exchange fees, bid/ask and slippage.")
    c8.toggle("Require momentum confirmation", key="sig_momentum",
              help="Only trade when the 3-day vs 7-day baseload momentum (known at decision time) agrees.")
    return current_params()


def current_params() -> tuple[SignalParams, BacktestParams]:
    s = st.session_state
    sp = SignalParams(
        horizon=s["sig_horizon"], rd_mode=s["sig_rd_mode"], rd_z_threshold=s["sig_rd_thr"],
        surprise_z_threshold=s["sig_surprise_thr"], zscore_window_days=s["sig_window"],
        entry_threshold=s["sig_entry"], use_momentum_filter=s["sig_momentum"],
    )
    return sp, BacktestParams(cost_per_mwh=s["sig_cost"])


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def page_header(ctx: Ctx, title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(f"{ctx.zone_name} · {ctx.start:%d %b %Y} – {ctx.end:%d %b %Y} · {subtitle}")
    if ctx.data.is_sample:
        st.warning("**Synthetic sample data** — generated to mimic ENTSO-E data structure and market behaviour. "
                   "Not real market data; see *Methodology*.", icon="🧪")
    for notice in ctx.data.notices:
        if not notice.startswith("SYNTHETIC"):
            st.info(notice)


def insight(text: str, icon: str = "💡") -> None:
    st.info(text, icon=icon)


def regime_cards(notes: list[RegimeNote]) -> None:
    cols = st.columns(len(notes))
    for col, note in zip(cols, notes):
        color, symbol = TONE_BADGE.get(note.tone, ("gray", "•"))
        with col.container(border=True):
            st.markdown(f"**{note.title}**")
            st.badge(f"{symbol} {note.state}", color=color)
            st.caption(note.text)


def eur(x: float, decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{'−' if x < 0 else ''}€{abs(x):,.{decimals}f}"


def pct(x: float, decimals: int = 0) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{decimals}f}%"


def gw(mw: float, decimals: int = 1) -> str:
    return "n/a" if mw is None or pd.isna(mw) else f"{mw / 1000:,.{decimals}f} GW"


def require_rows(view: pd.DataFrame, n: int = 48) -> bool:
    if len(view) < n:
        st.warning("Not enough data in the selected window for this analysis. Widen the date range.")
        return False
    return True


# ---------------------------------------------------------------------------
# Crude oil context
# ---------------------------------------------------------------------------
@dataclass
class CrudeCtx:
    data: object  # CrudeData
    daily: pd.DataFrame  # daily features (EIA)
    weekly: pd.DataFrame  # weekly features (EIA)
    cot: pd.DataFrame  # COT features
    live: object  # LiveCrude
    start: pd.Timestamp  # first date shown in time-series charts
    window_label: str

    @property
    def source_label(self) -> str:
        return "Synthetic sample data" if self.data.is_sample else "EIA + CFTC (refreshed daily)"


@st.cache_data(ttl=3600, show_spinner="Loading crude data…")
def cached_crude():
    from src.crude.data import load_crude_data
    from src.crude.features import add_cot_features, add_daily_features, add_weekly_features

    data = load_crude_data()
    return data, add_daily_features(data.daily), add_weekly_features(data.weekly), add_cot_features(data.cot)


@st.cache_data(ttl=3600, show_spinner="Fetching indicative prices…")
def cached_live(is_sample: bool):
    from src.crude.data import load_crude_data
    from src.crude.live import fetch_live_crude, sample_live

    return sample_live(load_crude_data()) if is_sample else fetch_live_crude()


def build_crude_context(window: str) -> CrudeCtx:
    data, daily, weekly, cot = cached_crude()
    live = cached_live(data.is_sample)
    last = max(daily.index.max(), weekly.index.max())
    years = {"1 year": 1, "2 years": 2, "5 years": 5, "10 years": 10}.get(window)
    start = last - pd.DateOffset(years=years) if years else daily.index.min()
    return CrudeCtx(data, daily, weekly, cot, live, pd.Timestamp(start), window)


def crude_header(ctx: CrudeCtx, title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(f"WTI / Brent · {ctx.source_label} · {subtitle}")
    if ctx.data.is_sample:
        st.warning("**Synthetic sample data** — the data store has no real crude data yet. Run the morning update "
                   "(GitHub Action or `scripts/daily_update.py`) with an `EIA_API_KEY` to switch to real data.", icon="🧪")
    for notice in ctx.data.notices:
        if not notice.startswith("SYNTHETIC"):
            st.info(notice)


CRUDE_SIGNAL_DEFAULTS = {
    "csig_target": "futures",
    "csig_curve": True,
    "csig_positioning": True,
    "csig_surprise": 1.0,
    "csig_cushing": 1.0,
    "csig_curve_thr": 0.10,
    "csig_crowd": 0.85,
    "csig_entry": 2,
    "csig_cost": 0.03,
}


def persist_crude_signal_state() -> None:
    for key, default in CRUDE_SIGNAL_DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, default)


def crude_signal_controls():
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.selectbox("Target", ["futures", "spot"], key="csig_target",
                 format_func=lambda t: "Front-month futures (roll-adjusted)" if t == "futures" else "WTI Cushing spot",
                 help="Official futures history (EIA) ends in April 2024; spot runs to the latest data.")
    c2.toggle("Curve vote", key="csig_curve", help="Needs futures prices: no official data after April 2024.")
    c3.toggle("Positioning vote", key="csig_positioning")
    c4.select_slider("Entry: |score| ≥", options=[1, 2, 3, 4], key="csig_entry")
    c5.number_input("Cost + slippage ($/bbl)", 0.0, 0.5, step=0.01, key="csig_cost", format="%.2f",
                    help="Round trip per barrel; ~$0.02-0.03 covers a TAS tick plus fees on one CL contract.")
    c6, c7, c8, c9 = st.columns(4)
    c6.slider("Inventory surprise z", 0.25, 2.5, step=0.25, key="csig_surprise")
    c7.slider("Cushing z", 0.25, 2.5, step=0.25, key="csig_cushing")
    c8.slider("Curve dead-band ($/bbl)", 0.0, 1.0, step=0.05, key="csig_curve_thr")
    c9.slider("Crowding percentile", 0.6, 0.95, step=0.05, key="csig_crowd")
    return crude_params_from_state()


def crude_params_from_state():
    from src.crude.signals import CrudeSignalParams

    s = {k: st.session_state.get(k, v) for k, v in CRUDE_SIGNAL_DEFAULTS.items()}  # read-only
    params = CrudeSignalParams(target=s["csig_target"], use_curve=s["csig_curve"], use_positioning=s["csig_positioning"],
                               surprise_z=s["csig_surprise"], cushing_z=s["csig_cushing"],
                               curve_threshold=s["csig_curve_thr"], crowding_pctile=s["csig_crowd"],
                               entry_threshold=s["csig_entry"])
    return params, BacktestParams(cost_per_mwh=s["csig_cost"])


def usd(x: float, decimals: int = 2, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    sign = "+" if signed and x > 0 else ("−" if x < 0 else "")
    return f"{sign}${abs(x):,.{decimals}f}"


def mb(kb: float, signed: bool = False) -> str:
    if kb is None or pd.isna(kb):
        return "n/a"
    sign = "+" if signed and kb > 0 else ("−" if kb < 0 else "")
    return f"{sign}{abs(kb) / 1000:,.1f} mb"
