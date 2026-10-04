"""Honest, minimal backtester for the fundamental signal.

Design choices (all deliberate, all documented in the UI):

* Positions come from `signal_engine.build_signal_frame`, which only uses
  information available before the D-1 day-ahead gate closure.
* PnL proxy = position x (delivered price - reference price) x MW x hours,
  minus a cost per traded MWh that covers fees, bid/ask and slippage.
* Every eligible hour/day counts, including days with no trade, when
  computing the Sharpe-like ratio, so inactivity is not hidden.
* No compounding, no position sizing, no capital base: results are in EUR
  for a fixed 1 MW clip, which is how a desk would sanity-check an idea.

This is an educational research backtest - not evidence of a tradeable edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import BacktestParams

DAYS_PER_YEAR = 365  # power delivers every day, weekends included
PERIODS_PER_YEAR = {"D": DAYS_PER_YEAR, "W-THU": 52}


@dataclass
class BacktestResult:
    ledger: pd.DataFrame  # every eligible row (traded or not) with PnL
    trades: pd.DataFrame  # rows with a non-zero position
    daily_pnl: pd.Series  # PnL per period - delivery day or week - with zeros on idle periods
    metrics: dict = field(default_factory=dict)
    confusion: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def cumulative_pnl(self) -> pd.Series:
        return self.daily_pnl.cumsum()


def max_drawdown(cumulative: pd.Series) -> float:
    """Largest peak-to-trough fall of a cumulative PnL curve (<= 0, EUR)."""
    if cumulative.empty:
        return 0.0
    curve = pd.concat([pd.Series([0.0]), cumulative.reset_index(drop=True)])
    return float((curve - curve.cummax()).min())


def sharpe_like(daily_pnl: pd.Series, periods_per_year: int = DAYS_PER_YEAR) -> float:
    """Annualised mean/std of per-period PnL. No capital base, so 'Sharpe-like' only."""
    if len(daily_pnl) < 2 or daily_pnl.std() == 0:
        return np.nan
    return float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(periods_per_year))


def direction_confusion(ledger: pd.DataFrame) -> pd.DataFrame:
    predicted = ledger["position"].map({1.0: "Long", -1.0: "Short", 0.0: "No trade"})
    actual = np.sign(ledger["target"]).map({1.0: "Price up", -1.0: "Price down", 0.0: "Unchanged"})
    table = pd.crosstab(predicted, actual)
    rows = [r for r in ["Long", "Short", "No trade"] if r in table.index]
    cols = [c for c in ["Price up", "Price down", "Unchanged"] if c in table.columns]
    return table.loc[rows, cols]


def compute_metrics(ledger: pd.DataFrame, trades: pd.DataFrame, daily: pd.Series,
                    periods_per_year: int = DAYS_PER_YEAR) -> dict:
    n_trades = len(trades)
    traded_mwh = float(trades["mwh"].sum()) if n_trades else 0.0
    hits = np.sign(trades["target"]) == trades["position"]
    longs, shorts = trades[trades["position"] > 0], trades[trades["position"] < 0]
    gains, losses = trades.loc[trades["pnl"] > 0, "pnl"].sum(), trades.loc[trades["pnl"] < 0, "pnl"].sum()
    return {
        "eligible_rows": int(len(ledger)),
        "n_trades": int(n_trades),
        "trade_frequency": n_trades / len(ledger) if len(ledger) else np.nan,
        "n_days": int(len(daily)),
        "hit_rate": float(hits.mean()) if n_trades else np.nan,
        "hit_rate_long": float((np.sign(longs["target"]) == 1).mean()) if len(longs) else np.nan,
        "hit_rate_short": float((np.sign(shorts["target"]) == -1).mean()) if len(shorts) else np.nan,
        "base_rate_up": float((ledger["target"] > 0).mean()) if len(ledger) else np.nan,
        "n_long": int(len(longs)),
        "n_short": int(len(shorts)),
        "gross_pnl": float(trades["gross_pnl"].sum()) if n_trades else 0.0,
        "total_cost": float(trades["cost"].sum()) if n_trades else 0.0,
        "total_pnl": float(trades["pnl"].sum()) if n_trades else 0.0,
        "avg_pnl_per_trade": float(trades["pnl"].mean()) if n_trades else np.nan,
        "avg_pnl_per_mwh": float(trades["pnl"].sum() / traded_mwh) if traded_mwh else np.nan,
        "profit_factor": float(gains / -losses) if losses < 0 else np.nan,
        "sharpe_like": sharpe_like(daily, periods_per_year),
        "max_drawdown": max_drawdown(daily.cumsum()),
        "worst_trade": float(trades["pnl"].min()) if n_trades else np.nan,
        "best_trade": float(trades["pnl"].max()) if n_trades else np.nan,
    }


def run_backtest(
    signals: pd.DataFrame,
    params: BacktestParams | None = None,
    exclude: pd.Series | None = None,
    period_freq: str = "D",
) -> BacktestResult:
    """Evaluate a signal frame (power: build_signal_frame, crude: build_crude_signals).

    `exclude` is an optional boolean mask aligned to `signals.index`; masked
    rows are dropped before evaluation (used by the spike stress test).
    `period_freq` is the decision frequency: "D" (power) or "W-THU" (weekly crude).
    The `hours` column holds the units traded per position (MWh per MW, or barrels).
    """
    params = params or BacktestParams()
    ledger = signals[signals["target"].notna() & signals["reference_price"].notna()].copy()
    if exclude is not None:
        ledger = ledger[~exclude.reindex(ledger.index).fillna(False).astype(bool)]

    ledger["mwh"] = ledger["position"].abs() * params.volume_mw * ledger["hours"]
    ledger["gross_pnl"] = ledger["position"] * ledger["target"] * params.volume_mw * ledger["hours"]
    ledger["cost"] = ledger["mwh"] * params.cost_per_mwh
    ledger["pnl"] = ledger["gross_pnl"] - ledger["cost"]
    trades = ledger[ledger["position"] != 0].copy()

    daily = ledger.groupby("date")["pnl"].sum()
    if not daily.empty:
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq=period_freq), fill_value=0.0)
    daily.index.name = "date"

    return BacktestResult(
        ledger=ledger,
        trades=trades,
        daily_pnl=daily,
        metrics=compute_metrics(ledger, trades, daily, PERIODS_PER_YEAR.get(period_freq, DAYS_PER_YEAR)),
        confusion=direction_confusion(ledger) if len(ledger) else pd.DataFrame(),
    )


def exposure_by_hour(trades: pd.DataFrame) -> pd.DataFrame:
    """Count of long and short positions per delivery hour (hourly horizon)."""
    if trades.empty or "hour" not in trades:
        return pd.DataFrame(columns=["Long", "Short"])
    side = trades["position"].map({1.0: "Long", -1.0: "Short"})
    table = pd.crosstab(trades["hour"], side).reindex(range(24), fill_value=0)
    for col in ("Long", "Short"):
        if col not in table:
            table[col] = 0
    return table[["Long", "Short"]]


def stress_spike_hours(signals: pd.DataFrame, spike_mask: pd.Series, params: BacktestParams | None = None,
                       period_freq: str = "D") -> pd.DataFrame:
    """Compare results with and without the extreme-price rows.

    If most of the PnL comes from a handful of spike hours, the 'edge' is
    really tail exposure - which is useful to know, because tails are
    exactly where liquidity, limits and slippage bite hardest.
    """
    scenarios = {
        "All observations": run_backtest(signals, params, period_freq=period_freq),
        "Excluding spike rows": run_backtest(signals, params, exclude=spike_mask, period_freq=period_freq),
        "Spike rows only": run_backtest(signals, params, exclude=~spike_mask.astype(bool), period_freq=period_freq),
    }
    keys = ["n_trades", "hit_rate", "total_pnl", "avg_pnl_per_trade", "sharpe_like", "max_drawdown"]
    return pd.DataFrame({name: {k: r.metrics[k] for k in keys} for name, r in scenarios.items()}).T
