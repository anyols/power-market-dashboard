"""Plotly figure builders with one consistent visual language.

Conventions
-----------
* Each entity keeps its colour on every page (price = blue, load = violet,
  wind = aqua, solar = amber, residual demand = orange). Forecasts are drawn
  as a dashed neutral line next to the actual.
* Different units never share a y-axis: stacked panels instead of dual axes.
* Volumes are shown in GW, prices in EUR/MWh.
* Thin lines, hairline grids, transparent backgrounds so the Streamlit light
  and dark themes both work. Dark mode uses its own validated colour steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_LIGHT = {
    "price": "#2a78d6", "load": "#4a3aa7", "wind": "#1baf7a", "solar": "#eda100",
    "residual": "#eb6834", "spike": "#e34948", "negative": "#e87ba4",
    "forecast": "#898781", "gain": "#2a78d6", "loss": "#e34948",
    "muted": "#898781", "ink": "#0b0b0b", "ink2": "#52514e",
    "grid": "rgba(137,135,129,0.20)", "axis": "#c3c2b7", "mid": "#f0efec",
    "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
}
_DARK = {
    "price": "#3987e5", "load": "#9085e9", "wind": "#199e70", "solar": "#c98500",
    "residual": "#d95926", "spike": "#e66767", "negative": "#d55181",
    "forecast": "#898781", "gain": "#3987e5", "loss": "#e66767",
    "muted": "#898781", "ink": "#ffffff", "ink2": "#c3c2b7",
    "grid": "rgba(137,135,129,0.25)", "axis": "#383835", "mid": "#383835",
    "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
}
C = dict(_LIGHT)
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
GW = 1000.0


def set_dark_mode(dark: bool) -> None:
    """Switch the palette in place (called once per Streamlit run)."""
    C.clear()
    C.update(_DARK if dark else _LIGHT)


def _seq_scale() -> list:
    steps = C["seq"]
    return [[i / (len(steps) - 1), c] for i, c in enumerate(steps)]


def _ordinal_scale() -> list:
    """Sequential ramp for ordered categories (e.g. years): the lightest step still reads on the surface."""
    steps = C["seq"][2:] if C["seq"][0] == _LIGHT["seq"][0] else C["seq"][:-2]
    return [[i / (len(steps) - 1), c] for i, c in enumerate(steps)]


def _div_scale() -> list:
    return [[0.0, C["price"]], [0.5, C["mid"]], [1.0, C["spike"]]]


def _style(fig: go.Figure, height: int = 380, title: str | None = None, hover: str = "x unified",
           legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=48 if title else 24, b=8),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12),
        hovermode=hover,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=C["axis"], ticks="outside", tickcolor=C["axis"], zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], gridwidth=1, zeroline=False, linecolor=C["axis"])
    return fig


def _line(x, y, name, color, dash=None, width=2, group=None, show=True, fmt=",.1f", unit=""):
    return go.Scatter(
        x=x, y=y, name=name, mode="lines", line=dict(color=color, width=width, dash=dash),
        legendgroup=group or name, showlegend=show,
        hovertemplate=f"{name}: %{{y:{fmt}}}{unit}<extra></extra>",
    )


def empty_figure(message: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=C["muted"]), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height=height, legend=False)


# ---------------------------------------------------------------------------
# Market overview
# ---------------------------------------------------------------------------
def fundamentals_stack(view: pd.DataFrame, spike_mask: pd.Series | None = None) -> go.Figure:
    """Price and its fundamental drivers on aligned, separate panels."""
    titles = ["Day-ahead price (€/MWh)", "Load (GW)", "Wind (GW)", "Solar (GW)", "Residual demand = load − wind − solar (GW)"]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035, subplot_titles=titles)
    x = view.index
    fig.add_trace(_line(x, view["price"], "Day-ahead price", C["price"], width=1.5, fmt=",.2f", unit=" €/MWh"), 1, 1)
    if spike_mask is not None and spike_mask.any():
        s = view.loc[spike_mask.reindex(view.index).fillna(False).astype(bool), "price"]
        fig.add_trace(go.Scatter(x=s.index, y=s, mode="markers", name="Spike hours",
                                 marker=dict(color=C["spike"], size=6, line=dict(color="rgba(0,0,0,0)")),
                                 hovertemplate="Spike: %{y:,.2f} €/MWh<extra></extra>"), 1, 1)
    neg = view.loc[view["price"] < 0, "price"]
    if len(neg):
        fig.add_trace(go.Scatter(x=neg.index, y=neg, mode="markers", name="Negative hours",
                                 marker=dict(color=C["negative"], size=6),
                                 hovertemplate="Negative: %{y:,.2f} €/MWh<extra></extra>"), 1, 1)
    pairs = [
        (2, "load_actual", "load_forecast", "Load", C["load"]),
        (3, "wind_actual", "wind_forecast", "Wind", C["wind"]),
        (4, "solar_actual", "solar_forecast", "Solar", C["solar"]),
        (5, "residual_demand", "residual_demand_forecast", "Residual demand", C["residual"]),
    ]
    for i, (row, act, fc, name, color) in enumerate(pairs):
        fig.add_trace(_line(x, view[fc] / GW, "Day-ahead forecast", C["forecast"], dash="dot", width=1.5,
                            group="fc", show=i == 0, unit=" GW"), row, 1)
        fig.add_trace(_line(x, view[act] / GW, name, color, width=1.5, unit=" GW"), row, 1)
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1), row=5, col=1)
    _style(fig, height=980)
    fig.update_annotations(font=dict(size=13), x=0, xanchor="left")
    fig.update_layout(legend=dict(y=1.04))
    return fig


def daily_range_chart(daily: pd.DataFrame) -> go.Figure:
    """Daily min-max range (floating bars) with the baseload average."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily.index, y=daily["max"] - daily["min"], base=daily["min"], name="Daily min–max",
                         marker=dict(color=C["price"], opacity=0.25, cornerradius=4),
                         customdata=np.stack([daily["min"], daily["max"]], axis=1),
                         hovertemplate="Range: %{customdata[0]:,.0f} – %{customdata[1]:,.0f} €/MWh<extra></extra>"))
    fig.add_trace(_line(daily.index, daily["baseload"], "Baseload average", C["price"], fmt=",.1f", unit=" €/MWh"))
    if daily["peak"].notna().any():
        fig.add_trace(go.Scatter(x=daily.index, y=daily["peak"], mode="markers", name="Peak average",
                                 marker=dict(color=C["residual"], size=7, symbol="diamond"),
                                 hovertemplate="Peak: %{y:,.1f} €/MWh<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(title_text="€/MWh")
    return _style(fig, height=360, title="Daily price range")


# ---------------------------------------------------------------------------
# Price drivers
# ---------------------------------------------------------------------------
def quadratic_fit(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, float]:
    """Least-squares quadratic fit (merit orders are convex) and its R²."""
    m = x.notna() & y.notna()
    if m.sum() < 10:
        return np.array([np.nan, np.nan, np.nan]), np.nan
    coef = np.polyfit(x[m], y[m], 2)
    pred = np.polyval(coef, x[m])
    ss_res = ((y[m] - pred) ** 2).sum()
    ss_tot = ((y[m] - y[m].mean()) ** 2).sum()
    return coef, float(1 - ss_res / ss_tot) if ss_tot else np.nan


def driver_scatter(view: pd.DataFrame, x: str, y: str, x_label: str, y_label: str, color: str,
                   x_scale: float = GW, fit: bool = True, title: str | None = None) -> go.Figure:
    xs = view[x] / x_scale
    ys = view[y]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers", name="Hourly observations",
        marker=dict(color=color, size=5, opacity=0.35),
        customdata=view.index.strftime("%Y-%m-%d %H:00"),
        hovertemplate=f"%{{customdata}}<br>{x_label}: %{{x:,.1f}}<br>{y_label}: %{{y:,.1f}}<extra></extra>",
    ))
    if fit:
        coef, r2 = quadratic_fit(xs, ys)
        if np.isfinite(coef).all():
            grid = np.linspace(xs.min(), xs.max(), 100)
            fig.add_trace(go.Scatter(x=grid, y=np.polyval(coef, grid), mode="lines", name=f"Quadratic fit (R² = {r2:.2f})",
                                     line=dict(color=C["ink2"], width=2), hoverinfo="skip"))
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return _style(fig, height=380, title=title, hover="closest")


def empirical_merit_order(view: pd.DataFrame, bins: int = 20) -> go.Figure:
    """Median price (and inter-quartile band) by residual-demand bucket."""
    d = view[["residual_demand", "price"]].dropna()
    if len(d) < bins * 5:
        return empty_figure("Not enough observations for the binned curve")
    d["bucket"] = pd.qcut(d["residual_demand"], bins, duplicates="drop")
    g = d.groupby("bucket", observed=True).agg(
        rd=("residual_demand", "median"), p25=("price", lambda s: s.quantile(0.25)),
        p50=("price", "median"), p75=("price", lambda s: s.quantile(0.75)), n=("price", "size"),
    )
    x = g["rd"] / GW
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=g["p75"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=g["p25"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(235,104,52,0.15)", name="Inter-quartile range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=g["p50"], mode="lines+markers", name="Median price",
                             line=dict(color=C["residual"], width=2), marker=dict(size=8, line=dict(width=2, color="white")),
                             customdata=np.stack([g["p25"], g["p75"], g["n"]], axis=1),
                             hovertemplate="Residual demand %{x:,.1f} GW<br>Median %{y:,.1f} €/MWh"
                                           "<br>IQR %{customdata[0]:,.0f} – %{customdata[1]:,.0f}"
                                           "<br>n = %{customdata[2]}<extra></extra>"))
    fig.update_xaxes(title_text="Residual demand (GW, bucket median)")
    fig.update_yaxes(title_text="€/MWh")
    return _style(fig, height=380, title="Empirical supply curve: price by residual-demand bucket", hover="closest")


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    z = corr.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=_div_scale(),
        text=np.round(z, 2), texttemplate="%{text:.2f}", textfont=dict(size=11),
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>", xgap=2, ygap=2,
        colorbar=dict(thickness=10, outlinewidth=0),
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _style(fig, height=430, title="Correlation matrix (hourly)", hover="closest", legend=False)


def rolling_correlation_chart(rc: pd.Series, window_label: str) -> go.Figure:
    fig = go.Figure(_line(rc.index, rc, f"Rolling correlation ({window_label})", C["residual"], fmt=".2f"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(range=[-1, 1], title_text="Correlation")
    return _style(fig, height=320, title=f"Rolling correlation: residual demand vs price ({window_label})", legend=False)


def fit_residual_chart(view: pd.DataFrame, fitted: pd.Series) -> go.Figure:
    resid = view["price"] - fitted
    fig = go.Figure(go.Bar(x=view.index, y=resid, name="Price − fundamental fit",
                           marker=dict(color=np.where(resid >= 0, C["spike"], C["price"])),
                           hovertemplate="%{y:+,.1f} €/MWh vs fit<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(title_text="€/MWh")
    return _style(fig, height=300, title="What fundamentals do not explain: price minus quadratic residual-demand fit",
                  legend=False)


# ---------------------------------------------------------------------------
# Forecast errors
# ---------------------------------------------------------------------------
ERROR_SERIES = [
    ("load_error", "Load error", "load"),
    ("wind_error", "Wind error", "wind"),
    ("solar_error", "Solar error", "solar"),
    ("residual_demand_error", "Residual-demand error", "residual"),
]


def error_timeseries(view: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[f"{label} (GW, actual − forecast)" for _, label, _ in ERROR_SERIES])
    for i, (col, label, ckey) in enumerate(ERROR_SERIES, start=1):
        fig.add_trace(_line(view.index, view[col] / GW, label, C[ckey], width=1.2, unit=" GW"), i, 1)
        fig.add_hline(y=0, line=dict(color=C["axis"], width=1), row=i, col=1)
    _style(fig, height=760, legend=False)
    fig.update_annotations(font=dict(size=13), x=0, xanchor="left")
    return fig


def error_histograms(view: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=4, subplot_titles=[label for _, label, _ in ERROR_SERIES], horizontal_spacing=0.06)
    for i, (col, label, ckey) in enumerate(ERROR_SERIES, start=1):
        fig.add_trace(go.Histogram(x=view[col] / GW, name=label, nbinsx=40, marker=dict(color=C[ckey], opacity=0.85),
                                   hovertemplate=f"{label}: %{{x}} GW<br>Hours: %{{y}}<extra></extra>"), 1, i)
        fig.add_vline(x=0, line=dict(color=C["axis"], width=1), row=1, col=i)
        fig.update_xaxes(title_text="GW", row=1, col=i)
    _style(fig, height=300, legend=False, hover="closest")
    fig.update_annotations(font=dict(size=13))
    fig.update_layout(bargap=0.05)
    return fig


def hour_weekday_heatmap(view: pd.DataFrame, col: str, title: str, scale: float = GW, unit: str = "GW",
                         diverging: bool = True, agg: str = "mean") -> go.Figure:
    pivot = view.pivot_table(index="hour", columns="weekday", values=col, aggfunc=agg) / scale
    pivot = pivot.reindex(index=range(24), columns=range(7))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    z = pivot.to_numpy()
    lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1
    fig = go.Figure(go.Heatmap(
        z=z, x=days, y=[f"{h:02d}:00" for h in range(24)],
        colorscale=_div_scale() if diverging else _seq_scale(),
        zmin=-lim if diverging else None, zmax=lim if diverging else None,
        xgap=2, ygap=2, colorbar=dict(thickness=10, outlinewidth=0, title=dict(text=unit)),
        hovertemplate="%{x} %{y}: %{z:,.2f} " + unit + "<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _style(fig, height=520, title=title, hover="closest", legend=False)


def zscore_chart(view: pd.DataFrame, col: str, label: str) -> go.Figure:
    fig = go.Figure(_line(view.index, view[col], f"{label} z-score", C["residual"], width=1.2, fmt=".2f"))
    for y in (-2, 2):
        fig.add_hline(y=y, line=dict(color=C["muted"], width=1), annotation_text=f"{y:+d}σ",
                      annotation_position="top left", annotation_font=dict(color=C["muted"], size=11))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(title_text="σ vs trailing 30 days")
    return _style(fig, height=320, title=f"{label}: z-score vs trailing 30-day distribution", legend=False)


# ---------------------------------------------------------------------------
# Spikes & volatility
# ---------------------------------------------------------------------------
def spikes_by_hour(view: pd.DataFrame, spike_mask: pd.Series) -> go.Figure:
    counts = view.loc[spike_mask, "hour"].value_counts().reindex(range(24), fill_value=0)
    fig = go.Figure(go.Bar(x=[f"{h:02d}" for h in counts.index], y=counts, name="Spike hours",
                           marker=dict(color=C["spike"], cornerradius=4),
                           hovertemplate="Hour %{x}:00 — %{y} spikes<extra></extra>"))
    fig.update_xaxes(title_text="Delivery hour (local)")
    fig.update_yaxes(title_text="Spike count")
    return _style(fig, height=320, title="When do spikes happen?", legend=False, hover="closest")


def price_distribution(view: pd.DataFrame, level: float) -> go.Figure:
    fig = go.Figure(go.Histogram(x=view["price"], nbinsx=60, marker=dict(color=C["price"], opacity=0.85),
                                 name="Hourly prices", hovertemplate="%{x} €/MWh: %{y} hours<extra></extra>"))
    fig.add_vline(x=level, line=dict(color=C["spike"], width=2), annotation_text=f"Spike threshold {level:,.0f}",
                  annotation_position="top right", annotation_font=dict(color=C["ink2"]))
    fig.add_vline(x=0, line=dict(color=C["axis"], width=1))
    fig.update_xaxes(title_text="€/MWh")
    fig.update_yaxes(title_text="Hours")
    return _style(fig, height=320, title="Price distribution", legend=False, hover="closest")


def regime_box(view: pd.DataFrame, groups: pd.Series, col: str, label: str, scale: float = GW, unit: str = "GW") -> go.Figure:
    order = ["Normal", "Spike", "Negative"]
    colors = {"Normal": C["muted"], "Spike": C["spike"], "Negative": C["negative"]}
    fig = go.Figure()
    for g in order:
        vals = view.loc[groups == g, col] / scale
        if vals.notna().sum() == 0:
            continue
        fig.add_trace(go.Box(y=vals, name=f"{g} ({vals.notna().sum()} h)", marker=dict(color=colors[g], size=3),
                             line=dict(width=1.5), boxpoints=False,
                             hovertemplate=f"{g}<br>%{{y:,.1f}} {unit}<extra></extra>"))
    fig.update_yaxes(title_text=unit)
    return _style(fig, height=340, title=label, legend=False, hover="closest")


def spike_frequency_heatmap(view: pd.DataFrame, spike_mask: pd.Series) -> go.Figure:
    d = view.assign(spike=spike_mask.astype(float))
    return hour_weekday_heatmap(d, "spike", "Spike frequency by hour and weekday (share of hours)", scale=0.01,
                                unit="%", diverging=False)


def volatility_chart(view: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(_line(view.index, view["price_vol_24h"], "24h volatility", C["price"], width=1.2, unit=" €/MWh"))
    fig.add_trace(_line(view.index, view["price_vol_7d"], "7-day volatility", C["residual"], unit=" €/MWh"))
    fig.update_yaxes(title_text="Std of hourly price changes (€/MWh)")
    return _style(fig, height=320, title="Rolling volatility (absolute €/MWh changes — prices can be negative)")


# ---------------------------------------------------------------------------
# Cross-border flows
# ---------------------------------------------------------------------------
def net_position_chart(net: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(x=net.index, y=net / GW, name="Net imports",
                           marker=dict(color=np.where(net >= 0, C["price"], C["residual"]), line=dict(width=0)),
                           hovertemplate="%{y:+,.2f} GW<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(title_text="GW (+ import / − export)")
    fig.update_layout(bargap=0)
    return _style(fig, height=320, title="Net position across modelled borders (+ = net import)", legend=False)


def border_flow_chart(net_by_border: pd.DataFrame, zone: str) -> go.Figure:
    palette = [C["price"], C["residual"], C["wind"], C["solar"], C["negative"], C["load"]]
    fig = go.Figure()
    for i, nb in enumerate(net_by_border.columns):
        fig.add_trace(_line(net_by_border.index, net_by_border[nb] / GW, f"{nb} → {zone}", palette[i % len(palette)],
                            width=1.3, fmt="+,.2f", unit=" GW"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(title_text=f"Net flow into {zone} (GW)")
    return _style(fig, height=360, title="Net flow by border (+ = into the selected zone)")


def spread_chart(spread: pd.Series, zone: str, nb: str) -> go.Figure:
    fig = go.Figure(_line(spread.index, spread, f"{nb} − {zone}", C["price"], width=1.2, fmt="+,.2f", unit=" €/MWh"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(title_text="€/MWh")
    return _style(fig, height=300, title=f"Day-ahead price spread: {nb} minus {zone}", legend=False)


def flow_vs_spread(spread: pd.Series, flow: pd.Series, zone: str, nb: str, capacity: float | None) -> go.Figure:
    d = pd.DataFrame({"spread": spread, "flow": flow}).dropna()
    fig = go.Figure(go.Scattergl(
        x=d["spread"], y=d["flow"] / GW, mode="markers", name="Hours",
        marker=dict(color=C["price"], size=5, opacity=0.35),
        hovertemplate=f"Spread {nb}−{zone}: %{{x:+,.1f}} €/MWh<br>Flow {zone}→{nb}: %{{y:+,.2f}} GW<extra></extra>",
    ))
    if capacity:
        fig.add_hline(y=capacity / GW, line=dict(color=C["muted"], width=1),
                      annotation_text="indicative capacity", annotation_font=dict(color=C["muted"], size=11))
    fig.add_vline(x=0, line=dict(color=C["axis"], width=1))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_xaxes(title_text=f"Price spread {nb} − {zone} (€/MWh)")
    fig.update_yaxes(title_text=f"Net flow {zone} → {nb} (GW)")
    return _style(fig, height=380, title="Flow direction vs price spread", hover="closest", legend=False)


# ---------------------------------------------------------------------------
# Signals, backtest, risk
# ---------------------------------------------------------------------------
def signal_overview(signals: pd.DataFrame, horizon: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                        subplot_titles=["Delivered day-ahead price (€/MWh)" if horizon == "hourly" else "Daily baseload (€/MWh)",
                                        "Signal score (−3 bearish … +3 bullish)"])
    x = signals.index
    fig.add_trace(_line(x, signals["delivered_price"], "Price", C["price"], width=1.3, fmt=",.2f", unit=" €/MWh"), 1, 1)
    score = signals["score"]
    fig.add_trace(go.Bar(x=x, y=score, name="Score", marker=dict(color=np.where(score >= 0, C["spike"], C["price"]),
                                                                  line=dict(width=0)),
                         hovertemplate="Score %{y:+.0f}<extra></extra>"), 2, 1)
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1), row=2, col=1)
    fig.update_yaxes(range=[-3.5, 3.5], dtick=1, row=2, col=1)
    _style(fig, height=520, legend=False)
    fig.update_annotations(font=dict(size=13), x=0, xanchor="left")
    fig.update_layout(bargap=0)
    return fig


def target_by_score(signals: pd.DataFrame, unit: str = "€/MWh") -> go.Figure:
    d = signals.dropna(subset=["score", "target"])
    fig = go.Figure()
    for sc in range(-4, 5):
        vals = d.loc[d["score"] == sc, "target"]
        if len(vals) < 3:
            continue
        color = C["spike"] if sc > 0 else (C["price"] if sc < 0 else C["muted"])
        fig.add_trace(go.Box(y=vals, name=f"{sc:+d}", marker=dict(color=color), boxpoints=False, line=dict(width=1.5),
                             hovertemplate=f"Score {sc:+d}<br>%{{y:+,.2f}} {unit}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_xaxes(title_text="Signal score")
    fig.update_yaxes(title_text=f"Realised price change vs reference ({unit})")
    return _style(fig, height=380, title="Does the score line up with the realised move?", legend=False, hover="closest")


def equity_curve(daily_pnl: pd.Series, compare: dict[str, pd.Series] | None = None, ccy: str = "€",
                 clip: str = "1 MW clip") -> go.Figure:
    cum = daily_pnl.cumsum()
    dd = cum - cum.cummax().clip(lower=0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.68, 0.32],
                        subplot_titles=[f"Cumulative PnL proxy ({ccy}, {clip})", f"Drawdown ({ccy})"])
    fig.add_trace(_line(cum.index, cum, "Strategy", C["price"], fmt=",.0f", unit=f" {ccy}"), 1, 1)
    palette = [C["residual"], C["wind"]]
    for i, (name, series) in enumerate((compare or {}).items()):
        fig.add_trace(_line(series.index, series.cumsum(), name, palette[i % 2], width=1.5, dash="dot", fmt=",.0f", unit=f" {ccy}"), 1, 1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", fill="tozeroy", name="Drawdown",
                             line=dict(color=C["loss"], width=1), fillcolor="rgba(227,73,72,0.12)",
                             hovertemplate=f"Drawdown: %{{y:,.0f}} {ccy}<extra></extra>", showlegend=False), 2, 1)
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1), row=1, col=1)
    _style(fig, height=480, legend=bool(compare))
    fig.update_annotations(font=dict(size=13), x=0, xanchor="left")
    return fig


def daily_pnl_bars(daily_pnl: pd.Series, ccy: str = "€", title: str = "Daily PnL proxy") -> go.Figure:
    fig = go.Figure(go.Bar(x=daily_pnl.index, y=daily_pnl, name=title,
                           marker=dict(color=np.where(daily_pnl >= 0, C["gain"], C["loss"]), cornerradius=3),
                           hovertemplate=f"%{{x|%Y-%m-%d}}: %{{y:+,.0f}} {ccy}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(title_text=ccy)
    return _style(fig, height=320, title=title, legend=False, hover="closest")


def confusion_heatmap(confusion: pd.DataFrame) -> go.Figure:
    if confusion.empty:
        return empty_figure("No trades")
    z = confusion.to_numpy()
    fig = go.Figure(go.Heatmap(z=z, x=confusion.columns, y=confusion.index, colorscale=_seq_scale(), showscale=False,
                               text=z, texttemplate="%{text}", textfont=dict(size=14), xgap=3, ygap=3,
                               hovertemplate="Predicted %{y}, actual %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _style(fig, height=300, title="Direction confusion matrix (counts)", hover="closest", legend=False)


def exposure_chart(exposure: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{h:02d}" for h in exposure.index], y=exposure["Long"], name="Long",
                         marker=dict(color=C["spike"], cornerradius=3), hovertemplate="Hour %{x}: %{y} long<extra></extra>"))
    fig.add_trace(go.Bar(x=[f"{h:02d}" for h in exposure.index], y=-exposure["Short"], name="Short",
                         marker=dict(color=C["price"], cornerradius=3),
                         customdata=exposure["Short"], hovertemplate="Hour %{x}: %{customdata} short<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_layout(barmode="relative")
    fig.update_xaxes(title_text="Delivery hour (local)")
    fig.update_yaxes(title_text="Positions (long up / short down)")
    return _style(fig, height=340, title="Exposure by delivery hour", hover="closest")


def pnl_by_regime(ledger: pd.DataFrame) -> go.Figure:
    if "vol_regime" not in ledger or ledger.empty:
        return empty_figure("No regime information")
    g = ledger.groupby("vol_regime")["pnl"].agg(["sum", "count"]).reindex(["low", "normal", "high"]).dropna()
    fig = go.Figure(go.Bar(x=g.index.str.capitalize(), y=g["sum"], name="PnL",
                           marker=dict(color=np.where(g["sum"] >= 0, C["gain"], C["loss"]), cornerradius=4),
                           customdata=g["count"], hovertemplate="%{x} volatility: %{y:+,.0f} € over %{customdata} obs.<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(title_text="€")
    return _style(fig, height=300, title="PnL proxy by volatility regime (regime known at decision time)", legend=False,
                  hover="closest")


# ---------------------------------------------------------------------------
# Daily brief
# ---------------------------------------------------------------------------
def day_profile(day: pd.DataFrame, prev: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=["Hourly price vs previous day (€/MWh)", "Residual demand: actual vs forecast (GW)"])
    fig.add_trace(_line(day["hour"], day["price"], "Price (day)", C["price"], fmt=",.1f", unit=" €/MWh"), 1, 1)
    if len(prev):
        fig.add_trace(_line(prev["hour"], prev["price"], "Price (previous day)", C["muted"], dash="dot", width=1.5,
                            fmt=",.1f", unit=" €/MWh"), 1, 1)
    fig.add_trace(_line(day["hour"], day["residual_demand"] / GW, "Residual demand", C["residual"], unit=" GW"), 1, 2)
    fig.add_trace(_line(day["hour"], day["residual_demand_forecast"] / GW, "Residual demand forecast", C["forecast"],
                        dash="dot", width=1.5, unit=" GW"), 1, 2)
    fig.update_xaxes(title_text="Delivery hour", dtick=3)
    _style(fig, height=380)
    fig.update_annotations(font=dict(size=13), x=0, xanchor="left")
    fig.layout.annotations[1].update(x=0.54)
    fig.update_layout(legend=dict(y=-0.25, yanchor="top", x=0, xanchor="left"), margin=dict(b=40))
    return fig


# ---------------------------------------------------------------------------
# Crude oil
# ---------------------------------------------------------------------------
def seasonal_band_chart(profile: pd.DataFrame, title: str, unit: str, year: int, scale: float = 1.0) -> go.Figure:
    """Classic inventory chart: previous-5-year range and average vs this year and last year."""
    wk = profile.index
    p = profile / scale
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wk, y=p["max"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=wk, y=p["min"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(137,135,129,0.18)", name="Previous 5-year range",
                             hovertemplate=f"5y range low: %{{y:,.1f}} {unit}<extra></extra>"))
    fig.add_trace(_line(wk, p["avg"], "5-year average", C["muted"], dash="dot", width=1.5, unit=f" {unit}"))
    if p["last_year"].notna().any():
        fig.add_trace(_line(wk, p["last_year"], str(year - 1), C["residual"], width=1.5, unit=f" {unit}"))
    fig.add_trace(_line(wk, p["current"], str(year), C["price"], width=2.5, unit=f" {unit}"))
    fig.update_xaxes(title_text="Week of year", range=[1, 52], dtick=4)
    fig.update_yaxes(title_text=unit)
    return _style(fig, height=340, title=title)


def price_lines(df: pd.DataFrame, columns: dict[str, str], title: str, unit: str = "$/bbl", height: int = 360) -> go.Figure:
    palette = [C["price"], C["residual"], C["wind"], C["load"]]
    fig = go.Figure()
    for i, (col, label) in enumerate(columns.items()):
        if col in df and df[col].notna().any():
            fig.add_trace(_line(df.index, df[col], label, palette[i % len(palette)], width=1.6, fmt=",.2f",
                                unit=f" {unit}"))
    fig.update_yaxes(title_text=unit)
    return _style(fig, height=height, title=title, legend=len(columns) > 1)


def zero_line_chart(s: pd.Series, title: str, label: str, unit: str = "$/bbl", height: int = 300) -> go.Figure:
    fig = go.Figure(_line(s.index, s, label, C["residual"], width=1.4, fmt="+,.2f", unit=f" {unit}"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(title_text=unit)
    return _style(fig, height=height, title=title, legend=False)


def forward_curve_chart(curve: pd.Series, title: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=list(curve.index), y=curve.to_numpy(), mode="lines+markers", name="WTI futures",
                               line=dict(color=C["price"], width=2),
                               marker=dict(size=9, line=dict(width=2, color="white")),
                               hovertemplate="%{x}: $%{y:,.2f}/bbl<extra></extra>"))
    fig.update_yaxes(title_text="$/bbl")
    fig.update_xaxes(title_text="Contract month")
    return _style(fig, height=320, title=title, legend=False, hover="closest")


def storage_vs_spread(x: pd.Series, y: pd.Series, years: pd.Series) -> go.Figure:
    """Theory of storage: inventories vs the prompt time spread, coloured by year."""
    d = pd.DataFrame({"x": x, "y": y, "year": years}).dropna()
    fig = go.Figure(go.Scattergl(
        x=d["x"] * 100, y=d["y"], mode="markers", name="Weeks",
        marker=dict(color=d["year"], colorscale=_ordinal_scale(), size=7, opacity=0.85,
                    colorbar=dict(title=dict(text="Year"), thickness=10, outlinewidth=0)),
        customdata=d.index.strftime("%Y-%m-%d"),
        hovertemplate="%{customdata}<br>Cushing vs 5y: %{x:+.0f}%<br>M1–M2: $%{y:+.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.add_vline(x=0, line=dict(color=C["axis"], width=1))
    fig.update_xaxes(title_text="Cushing stocks vs 5-year average (%)")
    fig.update_yaxes(title_text="WTI M1–M2 spread ($/bbl, + = backwardation)")
    return _style(fig, height=420, title="Theory of storage: low inventories, steeper backwardation", hover="closest",
                  legend=False)


def positioning_chart(cot: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45],
                        subplot_titles=["Managed-money net position (% of open interest)",
                                        "Percentile vs previous 3 years"])
    fig.add_trace(_line(cot.index, cot["mm_net_pct_oi"] * 100, "Net % of OI", C["price"], fmt="+.1f", unit="%"), 1, 1)
    fig.add_trace(go.Scatter(x=cot.index, y=cot["mm_net_pctile"] * 100, mode="lines", name="Percentile",
                             line=dict(color=C["residual"], width=1.5),
                             hovertemplate="%{y:.0f}th percentile<extra></extra>"), 2, 1)
    for y in (15, 85):
        fig.add_hline(y=y, line=dict(color=C["muted"], width=1), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1), row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    _style(fig, height=460, legend=False)
    fig.update_annotations(font=dict(size=13), x=0, xanchor="left")
    return fig


def weekly_signal_chart(signals: pd.DataFrame, price_label: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                        subplot_titles=[price_label, "Weekly signal score (− bearish … + bullish)"])
    fig.add_trace(_line(signals.index, signals["reference_price"], "Price", C["price"], width=1.3, fmt=",.2f",
                        unit=" $/bbl"), 1, 1)
    score = signals["score"]
    fig.add_trace(go.Bar(x=signals.index, y=score, name="Score",
                         marker=dict(color=np.where(score >= 0, C["spike"], C["price"]), line=dict(width=0)),
                         hovertemplate="%{x|%Y-%m-%d}: score %{y:+.0f}<extra></extra>"), 2, 1)
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1), row=2, col=1)
    fig.update_yaxes(range=[-4.5, 4.5], dtick=1, row=2, col=1)
    _style(fig, height=500, legend=False)
    fig.update_annotations(font=dict(size=13), x=0, xanchor="left")
    fig.update_layout(bargap=0.1)
    return fig
