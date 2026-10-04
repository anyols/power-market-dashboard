"""Deterministic, rule-based market commentary.

No language model is involved: every sentence is produced by explicit rules
from the data, so the same inputs always give the same note and every claim
can be traced back to a number. The style imitates an internal desk note.

Look-ahead discipline for the daily brief on day D:
* realised data for D itself (it is an end-of-day brief),
* trailing comparisons over D-30 .. D-1 only,
* day-ahead *forecasts* for D+1, which are published on D,
* nothing else from the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import ZONES
from src.utils import describe_hours, fmt_eur, fmt_gw, fmt_pct, period_of_day, safe_div

LOOKBACK_DAYS = 30
SECTION_TITLES = [
    "Price action",
    "Demand and residual load",
    "Renewables",
    "Forecast errors",
    "Volatility / spike risk",
    "What to watch next",
]


@dataclass
class DailyBrief:
    zone: str
    date: pd.Timestamp
    title: str
    headline: str
    sections: dict[str, str]
    stats: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        parts = [f"# {self.title}", f"**{self.headline}**", ""]
        for i, (name, text) in enumerate(self.sections.items(), start=1):
            parts += [f"## {i}. {name}", text, ""]
        return "\n".join(parts).strip() + "\n"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _hh(hour: int) -> str:
    return f"{int(hour):02d}:00"


def _direction(x: float, up: str, down: str, flat: str = "unchanged", tol: float = 0.0) -> str:
    if pd.isna(x) or abs(x) <= tol:
        return flat
    return up if x > 0 else down


def _signed_eur(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{'+' if x >= 0 else '−'}€{abs(x):,.0f}"


def _signed_gw(mw: float) -> str:
    if pd.isna(mw):
        return "n/a"
    return f"{'+' if mw >= 0 else '−'}{abs(mw) / 1000:,.1f} GW"


def _strongest_window(err: pd.Series, width: int = 4) -> tuple[float, list[int]]:
    """Mean and hours of the contiguous `width`-hour window with the largest |mean| error."""
    err = err.dropna()
    if len(err) < width:
        return (float(err.mean()) if len(err) else np.nan), list(err.index)
    rolled = err.rolling(width).mean()
    end = rolled.abs().idxmax()
    pos = err.index.get_loc(end)
    window = err.iloc[pos - width + 1 : pos + 1]
    return float(window.mean()), list(window.index)


# ---------------------------------------------------------------------------
# Daily brief
# ---------------------------------------------------------------------------
def generate_daily_brief(
    feat: pd.DataFrame,
    zone: str,
    date,
    zone_prices: pd.DataFrame | None = None,
    flows: pd.DataFrame | None = None,
) -> DailyBrief:
    """Build the internal-style market note for `zone` on delivery day `date`."""
    date = pd.Timestamp(date).normalize()
    day = feat[feat["date"] == date].copy()
    if day.empty:
        raise ValueError(f"No data for {zone} on {date.date()}.")
    hist = feat[(feat["date"] < date) & (feat["date"] >= date - pd.Timedelta(days=LOOKBACK_DAYS))]
    prev = feat[feat["date"] == date - pd.Timedelta(days=1)]
    nxt = feat[feat["date"] == date + pd.Timedelta(days=1)]
    nxt = nxt[[c for c in nxt.columns if c.endswith("_forecast") or c in ("date", "hour")]]  # forecasts only

    day = day.set_index("hour", drop=False)
    zone_name = ZONES[zone].name if zone in ZONES else zone
    s = _brief_stats(day, hist, prev, nxt)

    sections = {
        "Price action": _price_section(zone_name, day, s),
        "Demand and residual load": _demand_section(day, s, zone, date, zone_prices, flows),
        "Renewables": _renewables_section(day, s),
        "Forecast errors": _errors_section(day, s),
        "Volatility / spike risk": _volatility_section(day, s),
        "What to watch next": _watch_section(day, s, date),
    }
    return DailyBrief(
        zone=zone,
        date=date,
        title=f"Daily Power Market Brief — {zone_name} — {date.date().isoformat()}",
        headline=_headline(s),
        sections=sections,
        stats=s,
    )


def _brief_stats(day: pd.DataFrame, hist: pd.DataFrame, prev: pd.DataFrame, nxt: pd.DataFrame) -> dict:
    hist_daily = hist.groupby("date").agg(
        base=("price", "mean"),
        rng=("price", lambda x: x.max() - x.min()),
        rd=("residual_demand", "mean"),
        load=("load_actual", "mean"),
        wind=("wind_actual", "mean"),
        solar=("solar_actual", "mean"),
        res_share=("renewable_share", "mean"),
    )
    s = {
        "base": day["price"].mean(),
        "prev_base": prev["price"].mean() if len(prev) else np.nan,
        "avg30_base": hist_daily["base"].mean() if len(hist_daily) else np.nan,
        "peak": day.loc[day["is_peak"], "price"].mean() if day["is_peak"].any() else np.nan,
        "offpeak": day.loc[~day["is_peak"], "price"].mean(),
        "max": day["price"].max(),
        "max_hour": int(day["price"].idxmax()),
        "min": day["price"].min(),
        "min_hour": int(day["price"].idxmin()),
        "neg_hours": int((day["price"] < 0).sum()),
        "range": day["price"].max() - day["price"].min(),
        "avg30_range": hist_daily["rng"].mean() if len(hist_daily) else np.nan,
        "rd": day["residual_demand"].mean(),
        "rd_max": day["residual_demand"].max(),
        "rd_max_hour": int(day["residual_demand"].idxmax()) if day["residual_demand"].notna().any() else 0,
        "avg30_rd": hist_daily["rd"].mean() if len(hist_daily) else np.nan,
        "std30_rd": hist_daily["rd"].std() if len(hist_daily) > 2 else np.nan,
        "load": day["load_actual"].mean(),
        "avg30_load": hist_daily["load"].mean() if len(hist_daily) else np.nan,
        "wind": day["wind_actual"].mean(),
        "avg30_wind": hist_daily["wind"].mean() if len(hist_daily) else np.nan,
        "solar": day["solar_actual"].mean(),
        "solar_max": day["solar_actual"].max(),
        "solar_max_hour": int(day["solar_actual"].idxmax()) if day["solar_actual"].notna().any() else 12,
        "avg30_solar": hist_daily["solar"].mean() if len(hist_daily) else np.nan,
        "res_share": day["renewable_share"].mean(),
        "avg30_res_share": hist_daily["res_share"].mean() if len(hist_daily) else np.nan,
        "load_err": day["load_error"].mean(),
        "wind_err": day["wind_error"].mean(),
        "solar_err": day["solar_error"].mean(),
        "rd_err": day["residual_demand_error"].mean(),
        "sd30_load_err": hist["load_error"].std() if len(hist) > 24 else np.nan,
        "sd30_wind_err": hist["wind_error"].std() if len(hist) > 24 else np.nan,
        "sd30_rd_err": hist["residual_demand_error"].std() if len(hist) > 24 else np.nan,
        "p90_30d": hist["price"].quantile(0.9) if len(hist) > 24 else np.nan,
        "vol_regime": day["vol_regime"].iloc[-1] if "vol_regime" in day else "n/a",
        "is_weekend": bool(day["is_weekend"].iloc[0]),
        "has_next": len(nxt) > 0 and nxt["load_forecast"].notna().any(),
    }
    s["spike_hours"] = (
        sorted(day.loc[day["price"] > s["p90_30d"], "hour"].astype(int).tolist()) if pd.notna(s["p90_30d"]) else []
    )
    s["rd_z"] = safe_div(s["rd"] - s["avg30_rd"], s["std30_rd"])
    if s["has_next"]:
        s["next_rd_fc"] = nxt["residual_demand_forecast"].mean() if "residual_demand_forecast" in nxt else (
            nxt["load_forecast"] - nxt["wind_forecast"] - nxt["solar_forecast"]
        ).mean()
        s["next_wind_fc"] = nxt["wind_forecast"].mean()
        s["next_solar_fc_max"] = nxt["solar_forecast"].max()
        s["next_load_fc"] = nxt["load_forecast"].mean()
        s["next_is_weekend"] = (pd.Timestamp(nxt["date"].iloc[0]).dayofweek >= 5)
    return s


def _headline(s: dict) -> str:
    if pd.notna(s["rd_z"]) and s["rd_z"] > 0.75:
        regime = "Tighter system"
    elif pd.notna(s["rd_z"]) and s["rd_z"] < -0.75:
        regime = "Looser system"
    else:
        regime = "Balanced system"
    wind_dev = s["wind"] - s["avg30_wind"]
    driver = ""
    if pd.notna(wind_dev) and abs(wind_dev) > 0.25 * max(s["avg30_wind"], 1):
        low_wind = wind_dev < 0
        consistent = (low_wind and regime == "Tighter system") or (not low_wind and regime == "Looser system")
        driver = f" {'on' if consistent else 'despite'} {'low' if low_wind else 'strong'} wind"
    elif s["neg_hours"] > 0:
        driver = " with negative midday prices"
    dd = s["base"] - s["prev_base"]
    dd_txt = f" ({_signed_eur(dd)} d/d)" if pd.notna(dd) else ""
    return f"{regime}{driver}: baseload {fmt_eur(s['base'])}{dd_txt}"


def _price_section(zone_name: str, day: pd.DataFrame, s: dict) -> str:
    out = [f"{zone_name} day-ahead baseload averaged {fmt_eur(s['base'])}"]
    comps = []
    if pd.notna(s["prev_base"]):
        dd = s["base"] - s["prev_base"]
        comps.append(f"{_direction(dd, 'up', 'down', 'flat', 0.5)} {_signed_eur(dd).lstrip('+−')} on the previous day"
                     if abs(dd) > 0.5 else "flat on the previous day")
    if pd.notna(s["avg30_base"]):
        d30 = s["base"] - s["avg30_base"]
        if abs(d30) < 1:
            comps.append("in line with its 30-day average")
        else:
            comps.append(f"{_signed_eur(d30).lstrip('+−')} {'above' if d30 >= 0 else 'below'} its 30-day average")
    out[0] += (", " + " and ".join(comps) if comps else "") + "."

    if pd.notna(s["peak"]):
        out.append(f"Peak hours averaged {fmt_eur(s['peak'])} against {fmt_eur(s['offpeak'])} off-peak.")
    else:
        out.append("As a weekend day there was no standard peak block; demand-driven shape was flatter than on weekdays.")

    out.append(
        f"The daily high of {fmt_eur(s['max'])} printed at {_hh(s['max_hour'])} ({period_of_day(s['max_hour'])}), "
        f"while the low of {fmt_eur(s['min'])} came at {_hh(s['min_hour'])} ({period_of_day(s['min_hour'])})."
    )
    if s["neg_hours"]:
        neg_hours = day.loc[day["price"] < 0, "hour"].astype(int).tolist()
        solar_driven = s["solar_max"] > 0 and any(10 <= h <= 16 for h in neg_hours)
        reason = (
            "as solar output overwhelmed midday demand and inflexible generation had to pay to stay online"
            if solar_driven
            else "as renewable output exceeded what demand and export capacity could absorb"
        )
        out.append(
            f"Prices turned negative for {s['neg_hours']} hour{'s' if s['neg_hours'] > 1 else ''} during "
            f"{describe_hours(neg_hours)}, {reason}."
        )
    return " ".join(out)


def _demand_section(day, s, zone, date, zone_prices, flows) -> str:
    out = []
    if pd.notna(s["avg30_rd"]):
        d = s["rd"] - s["avg30_rd"]
        rel = safe_div(d, abs(s["avg30_rd"]))
        out.append(
            f"Residual demand averaged {fmt_gw(s['rd'])}, {fmt_gw(abs(d))} "
            f"{'above' if d >= 0 else 'below'} its 30-day average"
            + (f" ({fmt_pct(rel, 0, signed=True)})" if pd.notna(rel) and abs(s['avg30_rd']) > 1000 else "")
        )
        # Attribute the deviation: d(rd) = d(load) - d(wind) - d(solar)
        contrib = {
            "load": s["load"] - s["avg30_load"],
            "wind": -(s["wind"] - s["avg30_wind"]),
            "solar": -(s["solar"] - s["avg30_solar"]),
        }
        same_sign = {k: v for k, v in contrib.items() if np.sign(v) == np.sign(d) and abs(v) > 300}
        ranked = sorted(same_sign, key=lambda k: -abs(contrib[k]))
        phrases = {
            "load": ("firm load", "softer load"),
            "wind": ("lower wind output", "stronger wind output"),
            "solar": ("weaker solar", "stronger solar"),
        }
        if ranked:
            words = [phrases[k][0] if contrib[k] > 0 else phrases[k][1] for k in ranked[:2]]
            out[-1] += ", mainly driven by " + " and ".join(words)
        out[-1] += "."
        if s["rd_z"] > 0.75:
            out.append(
                "That leaves the system tighter than usual: more expensive dispatchable units are needed at the margin, "
                "which is bullish for price levels."
            )
        elif s["rd_z"] < -0.75:
            out.append(
                "That leaves the system looser than usual, pushing cheaper units onto the margin and weighing on prices."
            )
        else:
            out.append("Overall, system tightness was close to recent norms.")
    else:
        out.append(f"Residual demand averaged {fmt_gw(s['rd'])} (insufficient history for a 30-day comparison).")
    out.append(f"The residual-load peak of {fmt_gw(s['rd_max'])} came at {_hh(s['rd_max_hour'])}.")

    # Cross-border context (uses only same-day data)
    if flows is not None and not flows.empty:
        f = _same_day_flows(flows, date)
        if not f.empty:
            imp = f.loc[f["to_zone"] == zone, "flow_mw"].groupby(f["timestamp"]).sum()
            exp = f.loc[f["from_zone"] == zone, "flow_mw"].groupby(f["timestamp"]).sum()
            net = imp.sub(exp, fill_value=0).mean()
            if pd.notna(net) and abs(net) > 200:
                out.append(
                    f"The zone was a net {'importer' if net > 0 else 'exporter'} on the day (average {fmt_gw(abs(net))}), "
                    + ("with imports helping to relieve local tightness." if net > 0 else "which adds to local demand for generation.")
                )
    if zone_prices is not None and zone in zone_prices:
        zp = zone_prices[pd.DatetimeIndex(zone_prices.index).tz_localize(None).normalize() == date]
        spreads = {nb: (zp[nb] - zp[zone]).mean() for nb in zp.columns if nb != zone and zp[nb].notna().any()}
        if spreads:
            nb, sp = max(spreads.items(), key=lambda kv: abs(kv[1]))
            if abs(sp) > 3:
                out.append(
                    f"The widest average spread was against {nb} ({_signed_eur(sp)}/MWh {nb} minus {zone}), "
                    "pointing to congested interconnection on that border for part of the day."
                )
    return " ".join(out)


def _same_day_flows(flows: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    local_day = pd.DatetimeIndex(flows["timestamp"]).tz_localize(None).normalize()
    return flows[local_day == date]


def _renewables_section(day: pd.DataFrame, s: dict) -> str:
    out = []
    wind_txt = f"Wind averaged {fmt_gw(s['wind'])}"
    if pd.notna(s["avg30_wind"]):
        wind_txt += f" against a 30-day average of {fmt_gw(s['avg30_wind'])}"
    if s["solar_max"] > 500:
        out.append(f"{wind_txt}, while solar peaked at {fmt_gw(s['solar_max'])} at {_hh(s['solar_max_hour'])}.")
    else:
        out.append(f"{wind_txt}; solar contribution was negligible.")
    if pd.notna(s["res_share"]):
        share = f"Wind and solar covered {fmt_pct(s['res_share'])} of load"
        if pd.notna(s["avg30_res_share"]):
            share += f", versus {fmt_pct(s['avg30_res_share'])} on average over the past month"
        out.append(share + ".")
    # Solar cannibalisation: the cheapest hours coincide with the solar peak
    if s["solar_max"] > 500 and 10 <= s["min_hour"] <= 16:
        out.append(
            f"The daily price low at {_hh(s['min_hour'])} coincided with peak solar output, the familiar "
            "cannibalisation pattern that compresses midday capture prices."
        )
    elif pd.notna(s["avg30_wind"]) and s["wind"] < 0.6 * s["avg30_wind"]:
        out.append("Weak wind removed a large block of zero-marginal-cost supply, especially felt in the evening hours.")
    elif pd.notna(s["avg30_wind"]) and s["wind"] > 1.4 * s["avg30_wind"]:
        out.append("Strong wind displaced thermal generation across most hours, flattening the price curve.")
    return " ".join(out)


def _materiality(x: float, sd: float) -> str:
    if pd.isna(x) or pd.isna(sd) or sd == 0:
        return ""
    z = abs(x) / sd
    if z > 1.0:
        return "materially "
    if z < 0.35:
        return "only marginally "
    return ""


def _errors_section(day: pd.DataFrame, s: dict) -> str:
    out = []
    w_mean, w_hours = _strongest_window(day["wind_error"])
    if pd.notna(w_mean) and w_hours and abs(w_mean) > 300:
        out.append(
            f"Wind came in {_materiality(w_mean, s['sd30_wind_err'])}{'below' if w_mean < 0 else 'above'} forecast "
            f"during {describe_hours(w_hours)} ({_signed_gw(w_mean)} on average between {_hh(w_hours[0])} and "
            f"{_hh(w_hours[-1] + 1)}; {_signed_gw(s['wind_err'])} over the day)"
            + (", increasing system tightness into those hours." if w_mean < 0 else ", easing the balance in those hours.")
        )
    else:
        out.append(f"Wind output was broadly in line with the day-ahead forecast ({_signed_gw(s['wind_err'])} on the day).")
    if pd.notna(s["load_err"]):
        out.append(
            f"Load was {_materiality(s['load_err'], s['sd30_load_err'])}"
            f"{'above' if s['load_err'] >= 0 else 'below'} forecast ({_signed_gw(s['load_err'])} on average)"
            + (f" and solar {('over' if s['solar_err'] >= 0 else 'under')}-delivered by {fmt_gw(abs(s['solar_err']))}."
               if s["solar_max"] > 500 and pd.notna(s["solar_err"]) else ".")
        )
    if pd.notna(s["rd_err"]):
        if abs(s["rd_err"]) > 0.35 * (s["sd30_rd_err"] if pd.notna(s["sd30_rd_err"]) else np.inf):
            short = s["rd_err"] > 0
            out.append(
                f"Net, residual demand ran {fmt_gw(abs(s['rd_err']))} {'above' if short else 'below'} the day-ahead "
                f"forecast: the system was {'shorter' if short else 'longer'} than the auction assumed, a setup that "
                f"typically {'supports' if short else 'weighs on'} intraday and imbalance prices relative to day-ahead."
            )
        else:
            out.append("Net, the residual-demand forecast error was small, so the day-ahead schedule needed little re-balancing.")
    return " ".join(out)


def _volatility_section(day: pd.DataFrame, s: dict) -> str:
    out = []
    if pd.notna(s["avg30_range"]):
        ratio = safe_div(s["range"], s["avg30_range"])
        level = "elevated" if ratio > 1.25 else ("subdued" if ratio < 0.75 else "in line with recent norms")
        out.append(
            f"Price volatility was {level}: the intraday range of {fmt_eur(s['range'])} compares with a 30-day "
            f"average of {fmt_eur(s['avg30_range'])}."
        )
    else:
        out.append(f"The intraday range was {fmt_eur(s['range'])}.")
    if pd.notna(s["p90_30d"]):
        n = len(s["spike_hours"])
        if n:
            out.append(
                f"{n} hour{'s' if n > 1 else ''} cleared above the 90th percentile of the previous 30 days "
                f"({fmt_eur(s['p90_30d'])}), concentrated in {describe_hours(s['spike_hours'])}."
            )
        else:
            out.append(f"No hour cleared above the 90th percentile of the previous 30 days ({fmt_eur(s['p90_30d'])}).")
    if s["vol_regime"] in ("low", "normal", "high"):
        out.append(f"The trailing 7-day volatility regime is classified as {s['vol_regime']}.")
    return " ".join(out)


def _watch_section(day: pd.DataFrame, s: dict, date: pd.Timestamp) -> str:
    items: list[str] = []
    if s.get("has_next"):
        d_rd = s["next_rd_fc"] - s["rd"]
        d_wind = s["next_wind_fc"] - s["wind"]
        bias = "supportive" if d_rd > 1000 else ("bearish" if d_rd < -1000 else "neutral")
        wind_clause = ""
        if abs(d_wind) > 1000:
            wind_clause = f" as wind is forecast to {'recover' if d_wind > 0 else 'drop'} to {fmt_gw(s['next_wind_fc'])}"
        items.append(
            f"Day-ahead forecasts published for {(date + pd.Timedelta(days=1)).date()} point to residual demand of "
            f"{fmt_gw(s['next_rd_fc'])} ({_signed_gw(d_rd)} vs today){wind_clause} — {bias} for price levels relative "
            "to today."
        )
        if s.get("next_is_weekend") and not s["is_weekend"]:
            items.append("Lower weekend demand, which typically softens the peak and raises negative-price risk if solar is strong.")
    wind_volatile = pd.notna(s["sd30_wind_err"]) and abs(s["wind_err"]) > 0.5 * s["sd30_wind_err"]
    if wind_volatile or (pd.notna(s["avg30_wind"]) and s["wind"] < 0.6 * s["avg30_wind"]):
        items.append("Wind forecast revisions into the next auction - recent forecast errors have been large.")
    if 17 <= s["max_hour"] <= 20 or s["rd_z"] > 0.75:
        items.append("Evening-ramp demand and import availability from neighbouring markets, which set the marginal price when solar fades.")
    midday_negatives = any(10 <= h <= 16 for h in day.loc[day["price"] < 0, "hour"].astype(int))
    if midday_negatives or s["solar_max"] > 0.5 * max(s["load"], 1):
        items.append("Midday solar output and negative-price risk, particularly on low-demand days.")
    if s["spike_hours"]:
        items.append("Scarcity-pricing risk in the tightest hours if renewables under-deliver again.")
    items.append("Unplanned outage notices (REMIT UMMs), which this dataset does not capture.")
    seen, unique = set(), []
    for it in items:
        if it not in seen:
            unique.append(it)
            seen.add(it)
    return "For the next session, the main variables to monitor are:\n" + "\n".join(f"- {it}" for it in unique[:5])


# ---------------------------------------------------------------------------
# Period-level notes (Market Overview / Spike pages)
# ---------------------------------------------------------------------------
@dataclass
class RegimeNote:
    title: str
    state: str  # short label, e.g. "Tight"
    tone: str  # "bullish" | "bearish" | "neutral" | "risk"
    text: str


def market_regime_notes(view: pd.DataFrame) -> list[RegimeNote]:
    """Interpretation boxes for the selected window (last 7 days vs whole window)."""
    notes = []
    last7 = view[view["date"] > view["date"].max() - pd.Timedelta(days=7)]

    rd_z = last7["residual_demand_z"].mean()
    if pd.notna(rd_z) and rd_z > 0.4:
        notes.append(RegimeNote("Residual demand", "Tight", "bullish",
                                f"Residual demand over the last 7 days sits {rd_z:.1f}σ above its trailing 30-day mean. "
                                "A tighter system needs more expensive dispatchable plants: bullish pressure."))
    elif pd.notna(rd_z) and rd_z < -0.4:
        notes.append(RegimeNote("Residual demand", "Loose", "bearish",
                                f"Residual demand over the last 7 days sits {abs(rd_z):.1f}σ below its trailing 30-day mean. "
                                "A looser system pushes cheaper units onto the margin: bearish pressure."))
    else:
        notes.append(RegimeNote("Residual demand", "Balanced", "neutral",
                                "Residual demand is close to its trailing 30-day mean; no strong fundamental bias."))

    res_mae = last7["renewable_error"].abs().mean()
    res_mae_all = view["renewable_error"].abs().mean()
    ratio = safe_div(res_mae, res_mae_all)
    if pd.notna(ratio) and ratio > 1.2:
        notes.append(RegimeNote("Renewable forecast error", "Large", "risk",
                                f"Mean absolute wind+solar forecast error over the last 7 days is {fmt_gw(res_mae)}, "
                                f"{ratio:.1f}x the window average. Large errors are a key intraday/imbalance driver."))
    else:
        notes.append(RegimeNote("Renewable forecast error", "Normal", "neutral",
                                f"Mean absolute wind+solar forecast error over the last 7 days is {fmt_gw(res_mae)}, "
                                "in line with the window average."))

    regime = view["vol_regime"].iloc[-1] if "vol_regime" in view and len(view) else "n/a"
    vol_txt = {
        "high": ("High", "risk", "Hour-to-hour price volatility is in the top third of its history: more opportunity, but wider risk limits and worse fills."),
        "low": ("Low", "neutral", "Volatility is in the bottom third of its history: tighter ranges and less to trade around."),
        "normal": ("Normal", "neutral", "Volatility is within its typical range."),
    }.get(regime, ("n/a", "neutral", "Not enough history to classify volatility yet."))
    notes.append(RegimeNote("Volatility", *vol_txt))

    neg_share = view["is_negative"].mean()
    if neg_share > 0.02:
        notes.append(RegimeNote("Negative prices", f"{fmt_pct(neg_share, 1)} of hours", "bearish",
                                "Frequent negative prices signal renewable oversupply relative to flexible demand and export capacity."))
    else:
        notes.append(RegimeNote("Negative prices", "Rare", "neutral",
                                "Negative prices were rare in this window; inflexibility is not a dominant theme."))
    return notes


def explain_spikes(view: pd.DataFrame, spike_mask: pd.Series) -> str:
    """Natural-language comparison of spike vs normal hours."""
    spike_mask = spike_mask.reindex(view.index).fillna(False).astype(bool)
    n = int(spike_mask.sum())
    if n == 0:
        return "No price spikes were detected under the current definition."
    spikes, normal = view[spike_mask], view[~spike_mask]
    conditions = []
    rd_gap = spikes["residual_demand"].mean() - normal["residual_demand"].mean()
    if rd_gap > 0:
        conditions.append(f"residual demand was high ({fmt_gw(rd_gap, signed=True)} vs normal hours)")
    wind_ratio = safe_div(spikes["wind_actual"].mean(), normal["wind_actual"].mean())
    if pd.notna(wind_ratio) and wind_ratio < 0.9:
        conditions.append(f"wind output was below normal ({fmt_pct(wind_ratio - 1, 0)})")
    elif pd.notna(wind_ratio) and wind_ratio > 1.1:
        conditions.append(f"wind output was, unusually, above normal ({fmt_pct(wind_ratio - 1, 0, signed=True)})")
    le = spikes["load_error"].mean()
    if pd.notna(le):
        conditions.append(f"load forecast errors were {'positive' if le > 0 else 'negative'} on average ({_signed_gw(le)})")
    top_hours = spikes["hour"].value_counts().index[:3].tolist()
    when = describe_hours(spikes["hour"].tolist())
    text = (
        f"During the selected period, {n} hours qualified as spikes. Spikes were most common in {when} "
        f"(most frequent delivery hours: {', '.join(_hh(h) for h in sorted(top_hours))})"
    )
    if conditions:
        text += ", and typically occurred when " + ", ".join(conditions[:-1]) + (
            f" and {conditions[-1]}" if len(conditions) > 1 else conditions[0]
        )
    text += "."
    res_err_spike = spikes["residual_demand_error"].mean()
    if pd.notna(res_err_spike) and res_err_spike > 0:
        text += (
            f" On average the system was {fmt_gw(res_err_spike)} shorter than forecast during spike hours, "
            "although day-ahead prices are set before these errors are known — the link runs through the market's own forecasts."
        )
    return text


def surprise_comment(row: pd.Series) -> str:
    """One-line desk comment for a single surprise hour."""
    parts = []
    if abs(row.get("wind_error", 0)) > 1000:
        parts.append(f"wind {'shortfall' if row['wind_error'] < 0 else 'overshoot'} {fmt_gw(abs(row['wind_error']))}")
    if abs(row.get("solar_error", 0)) > 1000:
        parts.append(f"solar {'miss' if row['solar_error'] < 0 else 'beat'} {fmt_gw(abs(row['solar_error']))}")
    if abs(row.get("load_error", 0)) > 500:
        parts.append(f"load {'above' if row['load_error'] > 0 else 'below'} forecast by {fmt_gw(abs(row['load_error']))}")
    rde = row.get("residual_demand_error", np.nan)
    head = "System short vs forecast" if rde > 0 else "System long vs forecast"
    detail = "; ".join(parts) if parts else "broad-based error"
    pc = row.get("price_change_24h", np.nan)
    px = f"; DA price {_signed_eur(pc)} vs same hour yesterday" if pd.notna(pc) else ""
    return f"{head}: {detail}{px}"
