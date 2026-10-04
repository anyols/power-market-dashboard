"""Central configuration: bidding zones, interconnectors, paths and model parameters.

Everything that a user might reasonably want to tweak lives here, so the
analytics modules stay free of magic numbers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
CACHE_DB_PATH = RAW_DIR / "entsoe_cache.sqlite"

load_dotenv(PROJECT_ROOT / ".env")

# All CWE / Iberian / Italian / Nordic zones covered here trade on CET/CEST.
# The *market day* (delivery day) is defined in this timezone, so daily
# aggregation, hour-of-day features and peak definitions all use it.
MARKET_TZ = "Europe/Brussels"

# SDAC harmonised clearing price limits (EUR/MWh). Used for validation.
PRICE_FLOOR = -500.0
PRICE_CAP = 4000.0


# ---------------------------------------------------------------------------
# Bidding zones
# ---------------------------------------------------------------------------
class BiddingZone(BaseModel):
    """A day-ahead bidding zone and its ENTSO-E identifiers."""

    code: str
    name: str
    eic: str
    has_offshore_wind: bool = False
    in_sample: bool = True  # included in the synthetic sample dataset


ZONES: dict[str, BiddingZone] = {
    z.code: z
    for z in [
        BiddingZone(code="DE_LU", name="Germany/Luxembourg", eic="10Y1001A1001A82H", has_offshore_wind=True),
        BiddingZone(code="FR", name="France", eic="10YFR-RTE------C", has_offshore_wind=True),
        BiddingZone(code="NL", name="Netherlands", eic="10YNL----------L", has_offshore_wind=True),
        BiddingZone(code="BE", name="Belgium", eic="10YBE----------2", has_offshore_wind=True),
        BiddingZone(code="ES", name="Spain", eic="10YES-REE------0"),
        BiddingZone(code="IT_NORD", name="Italy (North)", eic="10Y1001A1001A73I"),
        # Nordic zones: available through the API only (not in the sample data).
        BiddingZone(code="DK_1", name="Denmark (West)", eic="10YDK-1--------W", has_offshore_wind=True, in_sample=False),
        BiddingZone(code="SE_3", name="Sweden (SE3)", eic="10Y1001A1001A46L", in_sample=False),
        BiddingZone(code="NO_1", name="Norway (NO1)", eic="10YNO-1--------2", in_sample=False),
    ]
}


class Border(BaseModel):
    """An interconnected pair of bidding zones.

    `capacity_ab_mw` / `capacity_ba_mw` are indicative transfer capacities used
    only by the synthetic data generator and for congestion heuristics. Real
    capacities are flow-based in Core and change hourly.
    """

    zone_a: str
    zone_b: str
    capacity_ab_mw: float
    capacity_ba_mw: float


BORDERS: list[Border] = [
    Border(zone_a="DE_LU", zone_b="NL", capacity_ab_mw=4500, capacity_ba_mw=4500),
    Border(zone_a="DE_LU", zone_b="FR", capacity_ab_mw=3000, capacity_ba_mw=3000),
    Border(zone_a="DE_LU", zone_b="BE", capacity_ab_mw=1000, capacity_ba_mw=1000),
    Border(zone_a="FR", zone_b="BE", capacity_ab_mw=3300, capacity_ba_mw=2500),
    Border(zone_a="NL", zone_b="BE", capacity_ab_mw=2400, capacity_ba_mw=2400),
    Border(zone_a="FR", zone_b="ES", capacity_ab_mw=3300, capacity_ba_mw=2800),
    Border(zone_a="FR", zone_b="IT_NORD", capacity_ab_mw=4000, capacity_ba_mw=1100),
    # Nordic links (API only)
    Border(zone_a="DK_1", zone_b="DE_LU", capacity_ab_mw=2500, capacity_ba_mw=2500),
    Border(zone_a="DK_1", zone_b="NL", capacity_ab_mw=700, capacity_ba_mw=700),
    Border(zone_a="SE_3", zone_b="DK_1", capacity_ab_mw=700, capacity_ba_mw=700),
    Border(zone_a="NO_1", zone_b="SE_3", capacity_ab_mw=2100, capacity_ba_mw=2100),
]


def neighbours(zone: str) -> list[str]:
    """Zones directly interconnected with `zone` in the configured border list."""
    out = []
    for b in BORDERS:
        if b.zone_a == zone:
            out.append(b.zone_b)
        elif b.zone_b == zone:
            out.append(b.zone_a)
    return out


def border_capacity(zone_from: str, zone_to: str) -> float | None:
    """Indicative capacity in the direction zone_from -> zone_to (MW)."""
    for b in BORDERS:
        if (b.zone_a, b.zone_b) == (zone_from, zone_to):
            return b.capacity_ab_mw
        if (b.zone_b, b.zone_a) == (zone_from, zone_to):
            return b.capacity_ba_mw
    return None


# ---------------------------------------------------------------------------
# Market definitions
# ---------------------------------------------------------------------------
# Standard (EEX-style) peak: Monday-Friday, delivery hours 08:00-20:00 local.
PEAK_START_HOUR = 8
PEAK_END_HOUR = 20  # exclusive


# ---------------------------------------------------------------------------
# Runtime settings (.env)
# ---------------------------------------------------------------------------
DataSource = Literal["auto", "sample", "entsoe", "store"]


class Settings(BaseModel):
    entsoe_api_token: str | None = None
    data_source: DataSource = "auto"
    request_timeout_s: int = 60
    cache_enabled: bool = True

    @property
    def has_token(self) -> bool:
        return bool(self.entsoe_api_token)


def get_settings() -> Settings:
    token = os.getenv("ENTSOE_API_TOKEN", "").strip() or None
    source = os.getenv("DATA_SOURCE", "auto").strip().lower() or "auto"
    if source not in ("auto", "sample", "entsoe", "store"):
        source = "auto"
    return Settings(
        entsoe_api_token=token,
        data_source=source,  # type: ignore[arg-type]
        cache_enabled=os.getenv("ENTSOE_CACHE", "1") != "0",
    )


# ---------------------------------------------------------------------------
# Analytics parameters
# ---------------------------------------------------------------------------
class SignalParams(BaseModel):
    """Parameters of the transparent fundamental signal (see signal_engine.py)."""

    horizon: Literal["hourly", "daily"] = Field(
        "hourly",
        description="hourly = each delivery hour vs the same hour on the previous day; "
        "daily = baseload average vs previous day's baseload.",
    )
    rd_mode: Literal["change", "level"] = Field(
        "change",
        description="change = forecast residual demand vs the reference day; "
        "level = forecast residual demand vs its trailing mean.",
    )
    rd_z_threshold: float = Field(1.0, gt=0)
    surprise_z_threshold: float = Field(0.5, ge=0)
    zscore_window_days: int = Field(30, ge=7, le=180)
    entry_threshold: int = Field(2, ge=1, le=3)
    use_momentum_filter: bool = False

    @field_validator("entry_threshold")
    @classmethod
    def _entry_in_range(cls, v: int) -> int:
        if v not in (1, 2, 3):
            raise ValueError("entry_threshold must be 1, 2 or 3")
        return v


class BacktestParams(BaseModel):
    """Execution assumptions for the research backtest."""

    cost_per_mwh: float = Field(
        1.0,
        ge=0,
        description="Round-trip transaction cost + slippage in EUR/MWh charged on every traded MWh.",
    )
    volume_mw: float = Field(1.0, gt=0)


# Descriptive analysis defaults
SPIKE_PERCENTILE = 0.90
ROLLING_WINDOW_DAYS = 30
