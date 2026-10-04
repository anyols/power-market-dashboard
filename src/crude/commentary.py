"""Deterministic weekly crude brief, written as of the EIA report release.

Look-ahead discipline for the brief on week ending F (release R = F + 5 days):
weekly balances up to F, prices up to R, CFTC reports published by R. The
live Yahoo layer is only used when the brief is for the latest report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.crude.live import LiveCrude

SECTIONS = ["Price action", "Inventories", "Refining & supply", "Curve", "Positioning", "Signal & what to watch"]
CUSHING_OPERATIONAL_FLOOR_MB = 20.0


@dataclass
class CrudeBrief:
    week_ending: pd.Timestamp
    release_date: pd.Timestamp
    title: str
    headline: str
    sections: dict[str, str]
    stats: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        parts = [f"# {self.title}", f"**{self.headline}**", ""]
        for i, (name, text) in enumerate(self.sections.items(), start=1):
            parts += [f"## {i}. {name}", text, ""]
        return "\n".join(parts).strip() + "\n"


def _mb(kb: float, signed: bool = False) -> str:
    if kb is None or pd.isna(kb):
        return "n/a"
    sign = ("+" if kb > 0 else "−" if kb < 0 else "") if signed else ("−" if kb < 0 else "")
    return f"{sign}{abs(kb) / 1000:,.1f} mb"


def _usd(x: float, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    sign = ("+" if x > 0 else "−" if x < 0 else "") if signed else ("−" if x < 0 else "")
    return f"{sign}${abs(x):,.2f}"


def _pct(x: float, signed: bool = True) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    sign = ("+" if x > 0 else "−" if x < 0 else "") if signed else ""
    return f"{sign}{abs(x) * 100:.1f}%"


def _flow(kbd: float) -> str:
    return "n/a" if pd.isna(kbd) else f"{kbd / 1000:,.2f} mb/d"


def _last(s: pd.Series, asof: pd.Timestamp):
    s = s[s.index <= asof].dropna()
    return (s.iloc[-1], s.index[-1]) if len(s) else (np.nan, None)


def generate_crude_brief(weekly_feat: pd.DataFrame, daily_feat: pd.DataFrame, cot_feat: pd.DataFrame | None,
                         week_ending=None, live: LiveCrude | None = None,
                         signal_row: pd.Series | None = None) -> CrudeBrief:
    weeks = weekly_feat["crude_stocks"].dropna().index
    F = pd.Timestamp(week_ending) if week_ending is not None else weeks.max()
    if F not in weeks:
        F = weeks[weeks <= F].max()
    R = F + pd.Timedelta(days=5)
    wk = weekly_feat.loc[F]
    prev4 = weekly_feat[weekly_feat.index <= F - pd.Timedelta(weeks=4)]
    is_latest = F == weeks.max()
    use_live = live is not None and live.ok and is_latest

    s: dict = {"F": F, "R": R}
    sections = {
        "Price action": _price(daily_feat, R, live if use_live else None, s),
        "Inventories": _inventories(wk, s),
        "Refining & supply": _refining(wk, prev4, F, s),
        "Curve": _curve(daily_feat, R, live if use_live else None, s),
        "Positioning": _positioning(cot_feat, R, s),
        "Signal & what to watch": _watch(signal_row, wk, F, R, s),
    }
    return CrudeBrief(F, R, f"Weekly Crude Brief — week ending {F.date()} (EIA release {R.date()})",
                      _headline(s), sections, s)


def _price(daily: pd.DataFrame, R: pd.Timestamp, live: LiveCrude | None, s: dict) -> str:
    if live is not None:
        f = live.fronts.dropna(subset=["wti_front"])
        last_d = f.index.max()
        wti = f["wti_front"].iloc[-1]
        wk_ago = f["wti_front"][f.index <= last_d - pd.Timedelta(days=7)]
        chg = wti - wk_ago.iloc[-1] if len(wk_ago) else np.nan
        brent = f["brent_front"].dropna().iloc[-1] if f["brent_front"].notna().any() else np.nan
        s.update(wti=wti, wti_chg=chg, brent=brent)
        text = (f"WTI front-month futures closed at {_usd(wti)}/bbl on {last_d:%a %d %b} ({live.source}), "
                f"{_usd(chg, True)} on the week. Brent front-month stood at {_usd(brent)}, a Brent–WTI spread of "
                f"{_usd(brent - wti)}.")
    else:
        wti, d = _last(daily["wti_spot"], R)
        wti_prev, _ = _last(daily["wti_spot"], R - pd.Timedelta(days=7))
        brent, _ = _last(daily["brent_spot"], R)
        s.update(wti=wti, wti_chg=wti - wti_prev, brent=brent)
        text = (f"WTI Cushing spot was {_usd(wti)}/bbl on {d:%d %b %Y} (EIA), {_usd(wti - wti_prev, True)} on the week; "
                f"Brent spot {_usd(brent)}, a Brent–WTI spread of {_usd(brent - wti)}.") if d is not None else \
            "No price data available for this week."
    vol, _ = _last(daily["realized_vol_20d"], R)
    if pd.notna(vol):
        text += f" Twenty-day realised volatility on WTI spot is {vol * 100:.0f}% annualised."
    return text


def _inventories(wk: pd.Series, s: dict) -> str:
    chg, norm, surprise = wk["crude_stocks_chg"], wk["crude_stocks_chg_5y_avg"], wk["crude_stocks_surprise"]
    s.update(surprise=surprise, crude_vs_5y_pct=wk["crude_stocks_vs_5y_pct"], cushing=wk["cushing_stocks"],
             cushing_vs_5y_pct=wk["cushing_stocks_vs_5y_pct"])
    verb = "drew" if chg < 0 else "built"
    norm_txt = f"a 5-year average {'draw' if norm < 0 else 'build'} of {_mb(abs(norm))}" if pd.notna(norm) else "no seasonal norm"
    tone = "bullish" if surprise < 0 else "bearish"
    out = [
        f"US commercial crude stocks {verb} {_mb(abs(chg))} in the week to {s['F']:%d %b}, versus {norm_txt} for the same "
        f"week — a {tone} {_mb(abs(surprise))} deviation from seasonal norms (not analyst consensus, which is not freely available).",
        f"Stocks stand at {_mb(wk['crude_stocks'])}, {_mb(abs(wk['crude_stocks_vs_5y']))} "
        f"({_pct(wk['crude_stocks_vs_5y_pct'])}) {'below' if wk['crude_stocks_vs_5y'] < 0 else 'above'} the 5-year average.",
    ]
    cush_mb = wk["cushing_stocks"] / 1000
    cush = (f"At Cushing, the WTI delivery hub, inventories {'fell' if wk['cushing_stocks_chg'] < 0 else 'rose'} "
            f"{_mb(abs(wk['cushing_stocks_chg']))} to {_mb(wk['cushing_stocks'])} ({_pct(wk['cushing_stocks_vs_5y_pct'])} vs the 5-year average)")
    if cush_mb < CUSHING_OPERATIONAL_FLOOR_MB + 5:
        cush += f", uncomfortably close to operational lows (~{CUSHING_OPERATIONAL_FLOOR_MB:.0f} mb) - a classic setup for front-month squeezes"
    out.append(cush + ".")
    out.append(
        f"Gasoline stocks moved {_mb(wk['gasoline_stocks_chg'], True)} (5-year norm {_mb(wk['gasoline_stocks_chg_5y_avg'], True)}) "
        f"and distillates {_mb(wk['distillate_stocks_chg'], True)} (norm {_mb(wk['distillate_stocks_chg_5y_avg'], True)}); "
        f"total commercial crude and product stocks are {_pct(wk['total_commercial_stocks_vs_5y_pct'])} vs the 5-year average."
    )
    return " ".join(out)


def _refining(wk: pd.Series, prev4: pd.DataFrame, F: pd.Timestamp, s: dict) -> str:
    util, util5 = wk["refinery_utilization"], wk["refinery_utilization_5y_avg"]
    month = F.month
    season = ""
    if month in (2, 3, 4):
        season = " as spring maintenance season weighs on runs"
    elif month in (9, 10):
        season = " during autumn maintenance season"
    elif month in (6, 7, 8):
        season = " in peak summer run season"
    prod_prev = prev4["crude_production"].dropna().iloc[-1] if len(prev4) and prev4["crude_production"].notna().any() else np.nan
    demand = wk["product_supplied_4w"]
    demand_5y = wk["product_supplied_5y_avg"]
    s.update(util=util)
    return (
        f"Refinery utilisation was {util:.1f}% ({util - util5:+.1f} pp vs the 5-year average){season}; refiners ran "
        f"{_flow(wk['refinery_crude_input'])} of crude, giving {wk['days_of_cover']:.1f} days of crude cover. "
        f"Crude production is estimated at {_flow(wk['crude_production'])} ({(wk['crude_production'] - prod_prev):+,.0f} kb/d "
        f"vs four weeks earlier) and net crude imports at {_flow(wk['crude_net_imports'])}. Implied product demand averaged "
        f"{_flow(demand)} over four weeks, {_pct((demand - demand_5y) / demand_5y)} vs the 5-year average for the week."
    )


def _curve(daily: pd.DataFrame, R: pd.Timestamp, live: LiveCrude | None, s: dict) -> str:
    if live is not None and len(live.curve) >= 2:
        m12 = live.m1_m2
        m1n = live.curve.iloc[0] - live.curve.iloc[-1]
        n = len(live.curve)
        src = f"the live curve on {live.curve_date:%d %b} ({live.source})"
    else:
        m12, d = _last(daily["m1_m2"], R)
        m1n, _ = _last(daily["m1_m4"], R)
        n = 4
        if d is None or d < R - pd.Timedelta(days=7):
            s["m1_m2"] = np.nan
            return ("No current futures curve is available: EIA stopped publishing NYMEX futures prices after April 2024 "
                    "and the live (Yahoo) curve is only used for the latest report.")
        src = f"EIA settlements on {d:%d %b %Y}"
    s["m1_m2"] = m12
    shape = "backwardation" if m12 > 0.05 else ("contango" if m12 < -0.05 else "a flat structure")
    meaning = {
        "backwardation": "Prompt barrels command a premium - the market is paying for oil now, a sign of tight balances, "
                         "and holders of the front contract earn roll yield.",
        "contango": "Deferred barrels trade above prompt - the market is paying to store oil, a sign of surplus.",
        "a flat structure": "The curve offers neither a clear scarcity premium nor a storage incentive.",
    }[shape]
    return (f"The WTI curve is in {shape}: M1–M2 at {_usd(m12, True)} and M1–M{n} at {_usd(m1n, True)}, per {src}. {meaning}")


def _positioning(cot: pd.DataFrame | None, R: pd.Timestamp, s: dict) -> str:
    if cot is None or cot.empty:
        s["mm_pctile"] = np.nan
        return "CFTC positioning data is not available."
    known = cot[cot["available_date"] <= R]
    if known.empty:
        s["mm_pctile"] = np.nan
        return "No CFTC report had been published by the release date."
    row = known.iloc[-1]
    T = known.index[-1]
    pctile = row["mm_net_pctile"]
    s["mm_pctile"] = pctile
    stance = "net long" if row["mm_net"] >= 0 else "net short"
    crowd = ""
    if pd.notna(pctile):
        if pctile > 0.85:
            crowd = " Length is crowded: new buyers are scarce, which skews risk towards liquidation on bearish news."
        elif pctile < 0.15:
            crowd = " Positioning is unusually short, leaving the market exposed to short-covering rallies."
        else:
            crowd = " Positioning is not at an extreme."
    head = f"Managed money was {stance} {abs(row['mm_net']) / 1000:,.0f}k contracts as of {T:%d %b}"
    if pd.isna(pctile):
        return head + "."
    change = f"{row['mm_net_change'] / 1000:+,.0f}k".replace("-", "−")
    return (f"{head} ({row['mm_net_pct_oi'] * 100:.1f}% of open interest, {change} w/w), "
            f"the {pctile * 100:.0f}th percentile of the past three years.{crowd}")


def _watch(signal_row: pd.Series | None, wk: pd.Series, F: pd.Timestamp, R: pd.Timestamp, s: dict) -> str:
    items = []
    if signal_row is not None and pd.notna(signal_row.get("score")):
        votes = {"inventory surprise": signal_row.get("comp_inventory"), "Cushing": signal_row.get("comp_cushing"),
                 "curve": signal_row.get("comp_curve"), "positioning": signal_row.get("comp_positioning")}
        parts = [f"{k} {int(v):+d}" for k, v in votes.items() if pd.notna(v)]
        s["score"] = signal_row["score"]
        items.append(f"Rule-based signal score {int(signal_row['score']):+d} ({', '.join(parts)}) - a research indicator, not advice.")
    items.append(f"The next EIA Weekly Petroleum Status Report on Wed {R + pd.Timedelta(days=7):%d %b}, and Friday's CFTC positioning update.")
    if wk["cushing_stocks"] / 1000 < CUSHING_OPERATIONAL_FLOOR_MB + 5:
        items.append("Cushing draws: further declines towards operational lows would pressure the front spreads.")
    m = F.month
    if m in (6, 7, 8, 9, 10, 11):
        items.append("Atlantic hurricane season: Gulf Coast production and refining outages can swing balances quickly.")
    if m in (2, 3, 4, 9, 10):
        items.append("Refinery maintenance: lower runs typically mean crude builds and product draws.")
    if m in (5, 6, 7, 8):
        items.append("US driving season gasoline demand.")
    if m in (11, 12, 1, 2):
        items.append("Winter distillate demand and heating-oil cracks.")
    pct = s.get("mm_pctile", np.nan)
    if pd.notna(pct) and (pct > 0.85 or pct < 0.15):
        items.append("Positioning extremes: crowded trades unwind abruptly on surprises.")
    items.append("OPEC+ policy and geopolitical supply news, which this model does not capture.")
    return "For the coming week, watch:\n" + "\n".join(f"- {i}" for i in items[:6])


def _headline(s: dict) -> str:
    parts = []
    if pd.notna(s.get("surprise")):
        parts.append(("Bigger-than-normal draw" if s["surprise"] < 0 else "Bigger-than-normal build")
                     if abs(s["surprise"]) > 1000 else "Stocks in line with seasonal norms")
    cp = s.get("cushing_vs_5y_pct")
    if pd.notna(cp):
        parts.append("Cushing tight" if cp < -0.1 else ("Cushing comfortable" if cp > 0.1 else "Cushing near normal"))
    m12 = s.get("m1_m2")
    if pd.notna(m12):
        parts.append("curve backwardated" if m12 > 0.05 else ("curve in contango" if m12 < -0.05 else "flat curve"))
    px = f" | WTI {_usd(s['wti'])}" if pd.notna(s.get("wti")) else ""
    return "; ".join(parts) + px
