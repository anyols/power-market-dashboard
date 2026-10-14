"""Cross-commodity prices (Yahoo Finance) and the analytics behind the Board,
Ratios, Stocks-vs-commodities and Seasonality pages.

Prices are continuous front-month futures, so they jump when contracts roll
(natural gas is the worst offender). Good enough for a board and for weekly
correlations; the oil page does proper roll-adjusted WTI returns.

Yahoo data is fetched live and cached by the app, never written to the repo.
Set MARKETS_SOURCE=synthetic to work offline (tests do this).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.market_prices import MarketPriceError, fetch_yahoo_closes


@dataclass(frozen=True)
class Asset:
    key: str
    name: str
    ticker: str
    group: str  # Energy | Metals | Agriculture | Stocks | Macro
    unit: str
    decimals: int = 2


UNIVERSE: list[Asset] = [
    Asset("wti", "WTI crude", "CL=F", "Energy", "$/bbl"),
    Asset("brent", "Brent crude", "BZ=F", "Energy", "$/bbl"),
    Asset("natgas", "US natural gas", "NG=F", "Energy", "$/MMBtu", 3),
    Asset("ttf", "Dutch TTF gas", "TTF=F", "Energy", "€/MWh"),
    Asset("gasoline", "RBOB gasoline", "RB=F", "Energy", "$/gal", 3),
    Asset("heating_oil", "Heating oil", "HO=F", "Energy", "$/gal", 3),
    Asset("gold", "Gold", "GC=F", "Metals", "$/oz", 0),
    Asset("silver", "Silver", "SI=F", "Metals", "$/oz"),
    Asset("copper", "Copper", "HG=F", "Metals", "$/lb", 3),
    Asset("platinum", "Platinum", "PL=F", "Metals", "$/oz", 0),
    Asset("palladium", "Palladium", "PA=F", "Metals", "$/oz", 0),
    Asset("aluminium", "Aluminium", "ALI=F", "Metals", "$/t", 0),
    Asset("wheat", "Wheat", "ZW=F", "Agriculture", "¢/bu", 1),
    Asset("corn", "Corn", "ZC=F", "Agriculture", "¢/bu", 1),
    Asset("soybeans", "Soybeans", "ZS=F", "Agriculture", "¢/bu", 1),
    Asset("coffee", "Coffee (arabica)", "KC=F", "Agriculture", "¢/lb", 1),
    Asset("cocoa", "Cocoa", "CC=F", "Agriculture", "$/t", 0),
    Asset("sugar", "Sugar", "SB=F", "Agriculture", "¢/lb"),
    Asset("cotton", "Cotton", "CT=F", "Agriculture", "¢/lb"),
    Asset("oj", "Orange juice", "OJ=F", "Agriculture", "¢/lb", 1),
    Asset("cattle", "Live cattle", "LE=F", "Agriculture", "¢/lb", 1),
    Asset("xom", "ExxonMobil", "XOM", "Stocks", "$"),
    Asset("shel", "Shell", "SHEL", "Stocks", "$"),
    Asset("xle", "US energy stocks (XLE)", "XLE", "Stocks", "$"),
    Asset("eqt", "EQT (US gas producer)", "EQT", "Stocks", "$"),
    Asset("rwe", "RWE", "RWE.DE", "Stocks", "€"),
    Asset("eon", "E.ON", "EOAN.DE", "Stocks", "€"),
    Asset("fcx", "Freeport-McMoRan", "FCX", "Stocks", "$"),
    Asset("nem", "Newmont", "NEM", "Stocks", "$"),
    Asset("gdx", "Gold miners (GDX)", "GDX", "Stocks", "$"),
    Asset("hsy", "Hershey", "HSY", "Stocks", "$"),
    Asset("sbux", "Starbucks", "SBUX", "Stocks", "$"),
    Asset("deere", "Deere", "DE", "Stocks", "$"),
    Asset("adm", "ADM", "ADM", "Stocks", "$"),
    Asset("spx", "S&P 500", "^GSPC", "Macro", "pts", 0),
    Asset("dxy", "US dollar index", "DX-Y.NYB", "Macro", "pts"),
    Asset("us10y", "US 10y yield", "^TNX", "Macro", "%"),
    Asset("eurusd", "EUR/USD", "EURUSD=X", "Macro", "", 4),
]
ASSETS = {a.key: a for a in UNIVERSE}
COMMODITY_GROUPS = ["Energy", "Metals", "Agriculture"]


@dataclass
class MarketPrices:
    prices: pd.DataFrame  # columns = asset keys, daily closes
    source: str  # "Yahoo Finance" | "synthetic"
    errors: list[str] = field(default_factory=list)

    @property
    def asof(self) -> pd.Timestamp | None:
        return None if self.prices.empty else self.prices.dropna(how="all").index.max()

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"


def load_prices(start: str = "2010-01-01", source: str | None = None) -> MarketPrices:
    source = (source or os.getenv("MARKETS_SOURCE", "yahoo")).lower()
    if source == "synthetic":
        return MarketPrices(synthetic_prices(start=start), "synthetic")
    try:
        prices = fetch_yahoo_closes({a.key: a.ticker for a in UNIVERSE}, start=start)
    except MarketPriceError as exc:
        return MarketPrices(pd.DataFrame(columns=list(ASSETS)), "Yahoo Finance", [str(exc)])
    missing = [k for k in ASSETS if k not in prices or prices[k].dropna().empty]
    errors = [f"No data for: {', '.join(missing)}"] if missing else []
    return MarketPrices(prices.reindex(columns=list(ASSETS)), "Yahoo Finance", errors)


def synthetic_prices(start: str = "2015-01-01", end: str = "2026-06-30", seed: int = 7) -> pd.DataFrame:
    """Deterministic fake prices with sector-level correlation (offline mode and tests only)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    groups = {g: rng.normal(0, 0.008, len(idx)) for g in ["Energy", "Metals", "Agriculture", "Stocks", "Macro"]}
    market = rng.normal(0.0002, 0.006, len(idx))
    out = {}
    for i, a in enumerate(UNIVERSE):
        vol = {"Energy": 0.022, "Metals": 0.015, "Agriculture": 0.018, "Stocks": 0.016, "Macro": 0.006}[a.group]
        shock = 0.6 * groups[a.group] + 0.3 * market + rng.normal(0, vol, len(idx))
        season = 0.002 * np.sin(2 * np.pi * (idx.dayofyear / 365.25 + i / 7)) if a.group != "Stocks" else 0
        out[a.key] = 50 * (1 + i % 5) * np.exp(np.cumsum(shock + season))
    df = pd.DataFrame(out, index=idx)
    df["hsy"] = df["hsy"] * np.exp(-0.25 * np.log(df["cocoa"] / df["cocoa"].iloc[0]))  # built-in negative link
    return df


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns; undefined across non-positive prices (WTI, 20 Apr 2020)."""
    p = prices.where(prices > 0)
    return np.log(p / p.shift(1))


def performance_table(prices: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    keys = keys or [k for k in prices.columns if prices[k].notna().any()]
    rows = []
    for k in keys:
        s = prices[k].dropna()
        if len(s) < 30:
            continue
        last, d = s.iloc[-1], s.index[-1]

        def ret(n: int) -> float:
            return s.iloc[-1] / s.iloc[-1 - n] - 1 if len(s) > n and s.iloc[-1 - n] > 0 else np.nan

        ytd_base = s[s.index < pd.Timestamp(d.year, 1, 1)]
        yr = s[s.index > d - pd.Timedelta(days=365)]
        lo, hi = yr.min(), yr.max()
        rows.append({
            "key": k, "name": ASSETS[k].name if k in ASSETS else k, "group": ASSETS[k].group if k in ASSETS else "",
            "last": last, "date": d, "1D": ret(1), "1W": ret(5), "1M": ret(21), "1Y": ret(252),
            "YTD": last / ytd_base.iloc[-1] - 1 if len(ytd_base) and ytd_base.iloc[-1] > 0 else np.nan,
            "52w_low": lo, "52w_high": hi, "range_pos": (last - lo) / (hi - lo) if hi > lo else np.nan,
            "vol_1m": log_returns(s.to_frame()).iloc[-21:, 0].std() * np.sqrt(252),
        })
    return pd.DataFrame(rows).set_index("key")


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Ratio:
    key: str
    name: str
    unit: str
    needs: tuple[str, ...]
    why: str

    def compute(self, p: pd.DataFrame) -> pd.Series:
        f = RATIO_FORMULAS[self.key]
        return f(p).replace([np.inf, -np.inf], np.nan).rename(self.key)


MMBTU_PER_MWH = 3.412
GALLONS_PER_BARREL = 42

RATIO_FORMULAS = {
    "gold_silver": lambda p: p["gold"] / p["silver"],
    "gold_oil": lambda p: p["gold"] / p["wti"].where(p["wti"] > 0),
    "copper_gold": lambda p: 1000 * p["copper"] / p["gold"],
    "oil_gas": lambda p: p["wti"].where(p["wti"] > 0) / p["natgas"],
    "crack_321": lambda p: (2 * p["gasoline"] * GALLONS_PER_BARREL + p["heating_oil"] * GALLONS_PER_BARREL - 3 * p["wti"]) / 3,
    "brent_wti": lambda p: p["brent"] - p["wti"],
    "soy_corn": lambda p: p["soybeans"] / p["corn"],
    "ttf_hh": lambda p: p["ttf"] * p["eurusd"] / MMBTU_PER_MWH - p["natgas"],
    "platinum_gold": lambda p: p["platinum"] / p["gold"],
}

RATIOS: dict[str, Ratio] = {r.key: r for r in [
    Ratio("gold_silver", "Gold / silver", "oz of silver per oz of gold", ("gold", "silver"),
          "The classic metals ratio. It blows out in panics (silver is half industrial metal, gold is pure "
          "safe haven) and compresses in reflation rallies."),
    Ratio("gold_oil", "Gold / oil", "barrels of WTI per oz of gold", ("gold", "wti"),
          "How many barrels an ounce of gold buys. A very high reading means oil is cheap in 'real money' terms "
          "- it went vertical in 2020 when WTI collapsed."),
    Ratio("copper_gold", "Copper / gold (x1000)", "ratio", ("copper", "gold"),
          "Growth metal over fear metal. People watch it as a read on growth expectations and it tends to move "
          "with US long-end yields (see the panel underneath)."),
    Ratio("oil_gas", "Oil / US gas", "$/bbl per $/MMBtu", ("wti", "natgas"),
          "On pure energy content a barrel holds ~5.8 MMBtu, so ~6 is 'parity'. US gas has traded far below parity "
          "for most of the shale era - a ratio in the 20s-40s says gas is cheap relative to oil."),
    Ratio("crack_321", "3-2-1 crack spread", "$/bbl", ("wti", "gasoline", "heating_oil"),
          "Rough US refining margin: 3 barrels of crude in, 2 of gasoline and 1 of distillate out. "
          "High cracks pull refiners to run hard, which drains crude stocks."),
    Ratio("brent_wti", "Brent - WTI", "$/bbl", ("brent", "wti"),
          "Seaborne vs inland US crude. A wide spread makes US exports attractive; it was huge in 2011-13 "
          "before the US export ban was lifted at the end of 2015."),
    Ratio("soy_corn", "Soybeans / corn", "ratio", ("soybeans", "corn"),
          "US farmers choose between the two each spring. Around 2.3-2.5 is the rough break-even; well above "
          "favours planting soy, well below favours corn."),
    Ratio("ttf_hh", "TTF - Henry Hub", "$/MMBtu", ("ttf", "eurusd", "natgas"),
          "European vs US gas, both in $/MMBtu. This gap is what pays for shipping US LNG across the Atlantic; "
          "it exploded in 2022 when Russian pipeline gas stopped."),
    Ratio("platinum_gold", "Platinum / gold", "ratio", ("platinum", "gold"),
          "Platinum used to trade above gold. It has sat at a deep discount since 2015 - diesel's decline "
          "after Dieselgate didn't help."),
]}


def ratio_stats(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 60:
        return {"last": np.nan, "pctile": np.nan, "median": np.nan, "z_1y": np.nan, "chg_1m": np.nan}
    last = s.iloc[-1]
    yr = s.iloc[-252:]
    return {
        "last": last,
        "pctile": float((s < last).mean()),
        "median": float(s.median()),
        "z_1y": float((last - yr.mean()) / yr.std()) if yr.std() else np.nan,
        "chg_1m": float(last - s.iloc[-22]) if len(s) > 22 else np.nan,
    }


# ---------------------------------------------------------------------------
# Stocks vs commodities
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pair:
    commodity: str
    stock: str
    story: str


PAIRS: list[Pair] = [
    Pair("cocoa", "hsy", "Cocoa is chocolate's biggest input. The 2024 cocoa squeeze was a margin problem for every chocolate maker."),
    Pair("coffee", "sbux", "Green coffee is a small slice of a latte's price, so Starbucks should barely care. Does it?"),
    Pair("wti", "xom", "The obvious one: an oil major against oil."),
    Pair("brent", "shel", "Shell prices off Brent rather than WTI, and also runs a huge LNG book."),
    Pair("natgas", "eqt", "EQT is one of the largest US natural gas producers - about as pure a gas bet as listed equity gets."),
    Pair("ttf", "rwe", "RWE generates and trades power in Europe; TTF gas sets the marginal power price much of the time."),
    Pair("copper", "fcx", "Freeport is one of the biggest listed copper miners."),
    Pair("gold", "nem", "Newmont, the largest gold miner. Miners are often a levered bet on gold - check the beta."),
    Pair("gold", "gdx", "The gold-miners ETF against gold itself."),
    Pair("corn", "deere", "Farm income drives tractor demand, with a lag. A good one to try the lead/lag chart on."),
    Pair("corn", "adm", "ADM buys, stores and processes grain - it earns on volumes and spreads more than on the price level."),
]


def weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Friday-to-Friday log returns (side-steps US/EU closing-time mismatches)."""
    w = prices.where(prices > 0).resample("W-FRI").last()
    return np.log(w / w.shift(1))


def pair_frame(prices: pd.DataFrame, commodity: str, stock: str) -> pd.DataFrame:
    df = prices[[commodity, stock]].dropna()
    df.columns = ["commodity", "stock"]
    return df


def rolling_beta_corr(r_stock: pd.Series, r_comm: pd.Series, window: int) -> pd.DataFrame:
    """Rolling correlation and beta of stock returns on commodity returns."""
    df = pd.concat({"s": r_stock, "c": r_comm}, axis=1).dropna()
    cov = df["s"].rolling(window).cov(df["c"])
    var = df["c"].rolling(window).var()
    return pd.DataFrame({"corr": df["s"].rolling(window).corr(df["c"]), "beta": cov / var})


def lead_lag(r_stock: pd.Series, r_comm: pd.Series, max_lag: int = 6) -> pd.Series:
    """corr(stock_t, commodity_{t-k}). Positive k: the commodity moved first."""
    out = {}
    for k in range(-max_lag, max_lag + 1):
        out[k] = r_stock.corr(r_comm.shift(k))
    return pd.Series(out, name="corr")


def decoupling(r_stock: pd.Series, r_comm: pd.Series, beta_window: int = 52, horizon: int = 13) -> pd.DataFrame:
    """How far the stock has drifted from what its commodity exposure implies.

    Residual = stock return over `horizon` weeks minus (trailing beta x commodity
    return over the same weeks). The beta uses data *before* the window, so a
    big residual is a genuine divergence, not a fitted one.
    """
    df = pd.concat({"s": r_stock, "c": r_comm}, axis=1).dropna()
    beta = (df["s"].rolling(beta_window).cov(df["c"]) / df["c"].rolling(beta_window).var()).shift(horizon)
    s_h = df["s"].rolling(horizon).sum()
    c_h = df["c"].rolling(horizon).sum()
    resid = s_h - beta * c_h
    z = (resid - resid.expanding(52).mean()) / resid.expanding(52).std()
    return pd.DataFrame({"beta": beta, "residual": resid, "z": z})


def correlation_matrix(prices: pd.DataFrame, keys: list[str], weeks: int) -> pd.DataFrame:
    r = weekly_returns(prices[keys]).iloc[-weeks:]
    c = r.corr(min_periods=max(10, weeks // 3))
    names = {k: ASSETS[k].name for k in keys}
    return c.rename(index=names, columns=names)


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------
def monthly_returns(s: pd.Series) -> pd.DataFrame:
    """Year x month table of calendar-month returns (month-end to month-end)."""
    m = s.where(s > 0).resample("ME").last()
    r = m / m.shift(1) - 1
    table = pd.DataFrame({"year": r.index.year, "month": r.index.month, "ret": r.to_numpy()}).dropna()
    return table.pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))


def month_stats(table: pd.DataFrame, exclude_year: int | None = None) -> pd.DataFrame:
    t = table.drop(index=exclude_year, errors="ignore") if exclude_year else table
    return pd.DataFrame({"mean": t.mean(), "median": t.median(), "share_up": (t > 0).sum() / t.notna().sum(),
                         "years": t.notna().sum()})


def seasonal_paths(s: pd.Series, years: int = 10) -> pd.DataFrame:
    """Each calendar year's path indexed to 100 at its first close, by day of year."""
    s = s.where(s > 0).dropna()
    out = {}
    last_year = s.index.max().year
    for y in range(last_year - years, last_year + 1):
        ys = s[s.index.year == y]
        if len(ys) < 20:
            continue
        path = 100 * ys / ys.iloc[0]
        path.index = ys.index.dayofyear
        out[y] = path.groupby(level=0).last()
    return pd.DataFrame(out).sort_index().ffill(limit=5)
