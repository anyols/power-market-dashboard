"""Morning data refresh - run by the scheduled GitHub Action (or by hand).

    python scripts/daily_update.py            # incremental update of everything configured
    python scripts/daily_update.py --full     # re-download full history
    python scripts/daily_update.py --only crude

Sources and keys (environment variables or .env):
* EIA API v2        EIA_API_KEY       crude prices, futures curve, weekly balances
* CFTC COT          (no key)          managed-money positioning in WTI
* ENTSO-E           ENTSOE_API_TOKEN  power: prices, load, wind, solar, flows

Each source is fetched incrementally (with an overlap window so revisions are
picked up), merged into data/store/*.csv and recorded in data/store/status.json.
One failing source never blocks the others; the exit code is non-zero if any
source failed, so the workflow run is flagged while successful data is still
committed. Yahoo prices are intentionally NOT stored (see src/market_prices.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cftc_client import WTI_CODE, get_disaggregated  # noqa: E402
from src.config import BORDERS, get_settings  # noqa: E402
from src.crude.config import EIA_DAILY, EIA_WEEKLY, HISTORY_START, T_COT, T_DAILY, T_WEEKLY  # noqa: E402
from src.data_store import merge_update, read_table, update_status, write_table  # noqa: E402
from src.eia_client import EiaClient  # noqa: E402

POWER_HISTORY_DAYS = 400
POWER_OVERLAP_DAYS = 7
EIA_OVERLAP = {T_DAILY: pd.Timedelta(days=60), T_WEEKLY: pd.Timedelta(weeks=26)}


EIA_INCREMENTAL_ROWS = {T_DAILY: 90, T_WEEKLY: 30}  # newest rows re-fetched on routine runs (revisions)


def _start(table: str, overlap: pd.Timedelta, full: bool) -> str:
    last = None if full else (read_table(table).index.max() if read_table(table) is not None else None)
    return HISTORY_START if last is None else (last - overlap).strftime("%Y-%m-%d")


def _has_history(table: str, column: str) -> bool:
    df = read_table(table)
    return df is not None and column in df and df[column].notna().sum() > 50


def update_eia(full: bool) -> bool:
    key = os.getenv("EIA_API_KEY", "").strip()
    if not key:
        # The public DEMO_KEY allows ~10 requests per several hours - not enough for one refresh.
        print("[EIA] EIA_API_KEY not set - skipping EIA. Register for a free key at "
              "https://www.eia.gov/opendata/register.php")
        for table in (T_DAILY, T_WEEKLY):
            update_status(table, ok=False, source="EIA API v2", message="skipped: EIA_API_KEY not configured")
        return False
    client = EiaClient(key)
    all_ok = True
    for table, series in ((T_DAILY, EIA_DAILY), (T_WEEKLY, EIA_WEEKLY)):
        start = _start(table, EIA_OVERLAP[table], full)
        cols, errors = {}, []
        for name, sid in series.items():
            try:
                if not full and _has_history(table, name):
                    cols[name] = client.get_series(sid, last_n=EIA_INCREMENTAL_ROWS[table])
                else:
                    cols[name] = client.get_series(sid, start=HISTORY_START)
            except Exception as exc:  # keep going; report at the end
                errors.append(f"{name}: {exc}")
                if "rate limit" in str(exc).lower():
                    errors.append("remaining series skipped (rate limit)")
                    break
        if cols:
            new = pd.DataFrame(cols)
            merged = merge_update(read_table(table), new)
            merged = merged[merged.index >= pd.Timestamp(HISTORY_START)]
            write_table(table, merged)
            latest = merged.dropna(how="all").index.max()
            print(f"[EIA] {table}: {len(new)} rows fetched since {start}; latest {latest.date()}")
        else:
            merged, latest = read_table(table), None
        ok = not errors
        all_ok &= ok
        update_status(table, ok=ok, source="EIA API v2", message="; ".join(errors)[:500],
                      latest=str(latest.date()) if latest is not None else None,
                      rows=len(merged) if merged is not None else 0)
        for e in errors:
            print(f"[EIA] ERROR {e}", file=sys.stderr)
    return all_ok


def update_cot(full: bool) -> bool:
    start = _start(T_COT, pd.Timedelta(weeks=8), full)
    try:
        new = get_disaggregated(WTI_CODE, start)
        merged = merge_update(read_table(T_COT), new)
        write_table(T_COT, merged)
        latest = merged.index.max()
        print(f"[CFTC] {len(new)} reports fetched since {start}; latest {latest.date()}")
        update_status(T_COT, ok=True, source="CFTC Disaggregated COT", latest=str(latest.date()), rows=len(merged))
        return True
    except Exception as exc:
        print(f"[CFTC] ERROR {exc}", file=sys.stderr)
        update_status(T_COT, ok=False, source="CFTC Disaggregated COT", message=str(exc)[:500])
        return False


def update_power(full: bool) -> bool:
    settings = get_settings()
    if not settings.has_token:
        print("[ENTSO-E] ENTSOE_API_TOKEN not set - skipping power (the app keeps using the synthetic sample).")
        return True
    from src.data_loader import POWER_STORE_ZONES, load_entsoe_frame, make_client
    from src.utils import market_day_bounds

    client = make_client(settings)
    today = pd.Timestamp.now(tz="Europe/Brussels").normalize().tz_localize(None)
    horizon_end = (today + pd.Timedelta(days=1)).date()  # day-ahead prices for tomorrow
    floor = (today - pd.Timedelta(days=POWER_HISTORY_DAYS)).date()
    all_ok = True
    prices = {}

    def window(table: str):
        existing = None if full else read_table(table, utc=True)
        if existing is None or existing.empty:
            return floor
        return max(floor, (existing.index.max().tz_convert("Europe/Brussels") - pd.Timedelta(days=POWER_OVERLAP_DAYS)).date())

    for zone in POWER_STORE_ZONES:
        table = f"power/{zone}"
        try:
            start, end = market_day_bounds(window(table), horizon_end)
            frame = load_entsoe_frame(client, zone, start, end)
            frame.index = frame.index.tz_convert("UTC")
            merged = merge_update(read_table(table, utc=True), frame)
            merged = merged[merged.index >= pd.Timestamp(floor, tz="UTC")]
            write_table(table, merged, float_format="%.2f")
            prices[zone] = merged["price"]
            latest = merged["price"].dropna().index.max()
            print(f"[ENTSO-E] {zone}: {len(frame)} hours fetched; latest price {latest}")
            update_status(table, ok=True, source="ENTSO-E Transparency Platform", latest=str(latest), rows=len(merged))
        except Exception as exc:
            all_ok = False
            print(f"[ENTSO-E] ERROR {zone}: {exc}", file=sys.stderr)
            update_status(table, ok=False, source="ENTSO-E Transparency Platform", message=str(exc)[:500])

    # Cross-border physical flows for borders between stored zones (one column per direction)
    try:
        start, end = market_day_bounds(window("power/flows"), horizon_end)
        cols = {}
        for b in BORDERS:
            if b.zone_a in POWER_STORE_ZONES and b.zone_b in POWER_STORE_ZONES:
                for a, z in ((b.zone_a, b.zone_b), (b.zone_b, b.zone_a)):
                    s = client.get_cross_border_flows(a, z, start, end)
                    if not s.empty:
                        cols[f"{a}->{z}"] = s.resample("h").mean()
        if cols:
            merged = merge_update(read_table("power/flows", utc=True), pd.DataFrame(cols))
            merged = merged[merged.index >= pd.Timestamp(floor, tz="UTC")]
            write_table("power/flows", merged, float_format="%.0f")
            update_status("power/flows", ok=True, source="ENTSO-E Transparency Platform",
                          latest=str(merged.index.max()), rows=len(merged))
    except Exception as exc:
        all_ok = False
        print(f"[ENTSO-E] ERROR flows: {exc}", file=sys.stderr)
        update_status("power/flows", ok=False, source="ENTSO-E Transparency Platform", message=str(exc)[:500])
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="re-download full history")
    parser.add_argument("--only", choices=["crude", "power"], help="update a single commodity")
    args = parser.parse_args()

    results = {}
    steps = {"eia": update_eia, "cot": update_cot} if args.only != "power" else {}
    if args.only != "crude":
        steps["power"] = update_power
    for name, fn in steps.items():
        try:
            results[name] = fn(args.full)
        except Exception:  # never let one source kill the run
            traceback.print_exc()
            results[name] = False
    print("Summary:", ", ".join(f"{k}={'ok' if v else 'FAILED'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
