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
    titles = ["Day-ahead price (€/MWh)", "Load (GW)", "Wind (GW)", "Solar (GW)", "Residual load = load − wind − solar (GW)"]
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


# ---------------------------------------------------------------------------
# Forecast errors
# ---------------------------------------------------------------------------
ERROR_SERIES = [
    ("load_error", "Load error", "load"),
    ("wind_error", "Wind error", "wind"),
    ("solar_error", "Solar error", "solar"),
    ("residual_demand_error", "Residual-demand error", "residual"),
]


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


# ---------------------------------------------------------------------------
# Cross-border flows
# ---------------------------------------------------------------------------
def border_flow_chart(net_by_border: pd.DataFrame, zone: str) -> go.Figure:
    palette = [C["price"], C["residual"], C["wind"], C["solar"], C["negative"], C["load"]]
    fig = go.Figure()
    for i, nb in enumerate(net_by_border.columns):
        fig.add_trace(_line(net_by_border.index, net_by_border[nb] / GW, f"{nb} → {zone}", palette[i % len(palette)],
                            width=1.3, fmt="+,.2f", unit=" GW"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(title_text=f"Net flow into {zone} (GW)")
    return _style(fig, height=360, title="Net flow by border (+ = into the selected zone)")


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


# ---------------------------------------------------------------------------
# Cross-market (board, ratios, stocks vs commodities, seasonality)
# ---------------------------------------------------------------------------
def _updown_scale() -> list:
    return [[0, C["spike"]], [0.5, C["mid"]], [1, C["wind"]]]


def returns_heatmap(perf: pd.DataFrame, cols: list[str]) -> go.Figure:
    z = perf[cols].to_numpy(dtype=float) * 100
    lim = max(5.0, float(np.nanpercentile(np.abs(z), 90))) if np.isfinite(z).any() else 5.0
    text = [[f"{v:+.1f}%" if np.isfinite(v) else "" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=perf["name"], zmin=-lim, zmax=lim, colorscale=_updown_scale(),
        text=text, texttemplate="%{text}", textfont=dict(size=11), xgap=2, ygap=2, showscale=False,
        hovertemplate="%{y} %{x}: %{z:+.1f}%<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    return _style(fig, height=max(320, 26 * len(perf) + 60), hover="closest", legend=False)


def range_strip(perf: pd.DataFrame) -> go.Figure:
    """Where each asset sits between its 52-week low (0) and high (100)."""
    d = perf.dropna(subset=["range_pos"]).sort_values("range_pos")
    fig = go.Figure()
    for name in d["name"]:
        fig.add_trace(go.Scatter(x=[0, 100], y=[name, name], mode="lines", line=dict(color=C["grid"], width=6),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=d["range_pos"] * 100, y=d["name"], mode="markers", name="Now",
        marker=dict(size=11, color=np.where(d["range_pos"] > 0.5, C["wind"], C["spike"])),
        customdata=np.stack([d["52w_low"], d["52w_high"], d["last"]], axis=1),
        hovertemplate="%{y}: %{x:.0f}% of the range<br>low %{customdata[0]:,.2f} · high %{customdata[1]:,.2f}"
                      "<br>now %{customdata[2]:,.2f}<extra></extra>",
    ))
    fig.update_xaxes(range=[-3, 103], tickvals=[0, 50, 100], ticktext=["52w low", "middle", "52w high"])
    return _style(fig, height=max(300, 24 * len(d) + 40), hover="closest", legend=False)


def ratio_chart(s: pd.Series, title: str, unit: str, companion: pd.Series | None = None,
                companion_label: str = "") -> go.Figure:
    s = s.dropna()
    rows = 2 if companion is not None else 1
    head = title if unit in ("", "ratio") else f"{title} ({unit})"
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.68, 0.32] if rows == 2 else [1.0],
                        subplot_titles=[head, companion_label] if rows == 2 else None)
    fig.add_trace(_line(s.index, s, title, C["price"], width=1.6, fmt=",.2f"), 1, 1)
    for q, label in [(0.1, "10th pct"), (0.5, "median"), (0.9, "90th pct")]:
        fig.add_hline(y=s.quantile(q), line=dict(color=C["muted"], width=1, dash="dot"), row=1, col=1,
                      annotation_text=label, annotation_position="right", annotation_font=dict(size=10, color=C["muted"]))
    fig.add_trace(go.Scatter(x=[s.index[-1]], y=[s.iloc[-1]], mode="markers", marker=dict(size=9, color=C["spike"]),
                             hovertemplate="latest: %{y:,.2f}<extra></extra>", showlegend=False), 1, 1)
    if companion is not None:
        c = companion.dropna()
        fig.add_trace(_line(c.index, c, companion_label, C["residual"], width=1.3, fmt=",.2f"), 2, 1)
        fig.update_annotations(font=dict(size=13), x=0, xanchor="left", selector=dict(text=head))
        fig.update_annotations(font=dict(size=13), x=0, xanchor="left", selector=dict(text=companion_label))
    _style(fig, height=470 if rows == 2 else 380, legend=False)
    fig.update_layout(margin=dict(r=70))
    return fig


def indexed_performance(df: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    d = df.dropna()
    idx = 100 * d / d.iloc[0]
    fig = go.Figure()
    for col, color in zip(d.columns, [C["price"], C["residual"]]):
        fig.add_trace(_line(idx.index, idx[col], labels[col], color, width=1.6, fmt=",.0f"))
    fig.add_hline(y=100, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(title_text=f"{d.index[0]:%b %Y} = 100 (log scale)", type="log")
    return _style(fig, height=360)


def rolling_pair_chart(rb: pd.DataFrame, window_label: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=[f"Correlation ({window_label} rolling, weekly returns)", "Beta"])
    fig.add_trace(_line(rb.index, rb["corr"], "correlation", C["price"], width=1.5, fmt="+.2f"), 1, 1)
    fig.add_trace(_line(rb.index, rb["beta"], "beta", C["residual"], width=1.5, fmt="+.2f"), 2, 1)
    for r in (1, 2):
        fig.add_hline(y=0, line=dict(color=C["axis"], width=1), row=r, col=1)
    fig.update_yaxes(range=[-1, 1], row=1, col=1)
    _style(fig, height=430, legend=False)
    fig.update_annotations(font=dict(size=13), x=0, xanchor="left")
    return fig


def lead_lag_chart(ll: pd.Series, commodity: str, stock: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=ll.index, y=ll.to_numpy(), marker=dict(color=np.where(ll.index == 0, C["muted"], C["price"]), cornerradius=3),
        hovertemplate="lag %{x} wk: corr %{y:+.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_xaxes(title_text=f"← {stock} moves first   ·   weeks   ·   {commodity} moves first →", dtick=1)
    fig.update_yaxes(title_text="correlation")
    return _style(fig, height=300, hover="closest", legend=False)


def weekly_scatter(rc: pd.Series, rs: pd.Series, xlab: str, ylab: str) -> go.Figure:
    d = pd.concat({"c": rc, "s": rs}, axis=1).dropna()
    fig = go.Figure(go.Scattergl(x=d["c"] * 100, y=d["s"] * 100, mode="markers",
                                 marker=dict(color=C["price"], size=5, opacity=0.4),
                                 customdata=d.index.strftime("%d %b %Y"),
                                 hovertemplate="week to %{customdata}<br>" + xlab + " %{x:+.1f}%<br>" + ylab
                                               + " %{y:+.1f}%<extra></extra>"))
    if len(d) > 10:
        b, a = np.polyfit(d["c"], d["s"], 1)
        xs = np.linspace(d["c"].min(), d["c"].max(), 50)
        fig.add_trace(go.Scatter(x=xs * 100, y=(a + b * xs) * 100, mode="lines", line=dict(color=C["ink2"], width=1.5),
                                 hoverinfo="skip"))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.add_vline(x=0, line=dict(color=C["axis"], width=1))
    fig.update_xaxes(title_text=f"{xlab}, weekly %")
    fig.update_yaxes(title_text=f"{ylab}, weekly %")
    return _style(fig, height=360, hover="closest", legend=False)


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def seasonality_heatmap(table: pd.DataFrame) -> go.Figure:
    z = table.to_numpy(dtype=float) * 100
    lim = max(5.0, float(np.nanpercentile(np.abs(z), 90))) if np.isfinite(z).any() else 5.0
    text = [[f"{v:+.0f}" if np.isfinite(v) else "" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=MONTHS, y=[str(y) for y in table.index], zmin=-lim, zmax=lim, colorscale=_updown_scale(),
        text=text, texttemplate="%{text}", textfont=dict(size=10), xgap=2, ygap=2, showscale=False,
        hovertemplate="%{x} %{y}: %{z:+.1f}%<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(side="top", showgrid=False)
    return _style(fig, height=max(320, 22 * len(table) + 60), hover="closest", legend=False)


def month_bars(stats: pd.DataFrame) -> go.Figure:
    m = stats["mean"] * 100
    fig = go.Figure(go.Bar(
        x=MONTHS, y=m.to_numpy(), marker=dict(color=np.where(m >= 0, C["wind"], C["spike"]), cornerradius=3),
        customdata=np.stack([stats["median"] * 100, stats["share_up"] * 100, stats["years"]], axis=1),
        hovertemplate="%{x}: mean %{y:+.1f}% · median %{customdata[0]:+.1f}%<br>up in %{customdata[1]:.0f}% of "
                      "%{customdata[2]} years<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    fig.update_yaxes(title_text="average monthly return, %")
    return _style(fig, height=300, hover="closest", legend=False)


def seasonal_path_chart(paths: pd.DataFrame, current_year: int) -> go.Figure:
    past = paths.drop(columns=[current_year], errors="ignore")
    fig = go.Figure()
    for y in past.columns:
        fig.add_trace(go.Scatter(x=past.index, y=past[y], mode="lines", line=dict(color=C["grid"], width=1),
                                 name=str(y), hovertemplate=f"{y}: %{{y:.0f}}<extra></extra>", showlegend=False))
    fig.add_trace(_line(past.index, past.mean(axis=1), f"average {past.columns.min()}–{past.columns.max()}",
                        C["muted"], width=2.2, dash="dot", fmt=".1f"))
    if current_year in paths:
        cur = paths[current_year].dropna()
        fig.add_trace(_line(cur.index, cur, str(current_year), C["price"], width=2.6, fmt=".1f"))
    fig.add_hline(y=100, line=dict(color=C["axis"], width=1))
    ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    fig.update_xaxes(tickvals=ticks, ticktext=MONTHS, range=[1, 366])
    fig.update_yaxes(title_text="first close of the year = 100")
    return _style(fig, height=380, hover="x unified")
