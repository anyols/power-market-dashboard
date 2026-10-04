"""Download ENTSO-E data for one or more zones and export clean hourly CSVs.

    python scripts/fetch_entsoe_data.py --zones DE_LU FR --start 2024-01-01 --end 2024-03-31

Requires ENTSOE_API_TOKEN in .env. Raw responses are cached in
data/raw/entsoe_cache.sqlite, so re-running is cheap. Outputs go to
data/processed/ (git-ignored):

    <zone>_<start>_<end>_market.csv   hourly prices + fundamentals (market time)
    <zone>_<start>_<end>_flows.csv    hourly physical flows per border direction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, ZONES, get_settings  # noqa: E402
from src.data_loader import load_market_data  # noqa: E402
from src.entsoe_client import EntsoeError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zones", nargs="+", default=["DE_LU"], choices=sorted(ZONES))
    parser.add_argument("--start", required=True, help="First delivery day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Last delivery day, YYYY-MM-DD")
    args = parser.parse_args()

    settings = get_settings().model_copy(update={"data_source": "entsoe"})
    if not settings.has_token:
        print("ENTSOE_API_TOKEN is not set. Copy .env.template to .env and add your token.", file=sys.stderr)
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for zone in args.zones:
        print(f"[{zone}] downloading {args.start} .. {args.end}")
        try:
            data = load_market_data(zone, args.start, args.end, warmup_days=0, settings=settings)
        except EntsoeError as exc:
            print(f"[{zone}] failed: {exc}", file=sys.stderr)
            continue
        stem = f"{zone}_{args.start}_{args.end}"
        data.frame.to_csv(PROCESSED_DIR / f"{stem}_market.csv")
        data.flows.to_csv(PROCESSED_DIR / f"{stem}_flows.csv", index=False)
        print(f"[{zone}] {len(data.frame):,} hours written; coverage:")
        print(data.quality.to_frame()[["still missing", "coverage"]].to_string())
        for warning in data.quality.warnings:
            print(f"[{zone}] warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
