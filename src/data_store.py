"""File-based data store refreshed by the scheduled job (scripts/daily_update.py).

The dashboard never has to call an API at page load: a scheduled GitHub
Action fetches whatever is new each morning, merges it into the CSV tables
under data/store/ and commits them. Streamlit Cloud then serves the updated
repository. CSV is deliberate - small, diff-able in git, readable anywhere.

A `status.json` file records, per dataset, when it was last refreshed, the
latest observation it contains and whether the last attempt succeeded, so
the UI can always say how fresh each number is.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

STORE_DIR = DATA_DIR / "store"
STATUS_PATH = STORE_DIR / "status.json"


def table_path(name: str, root: Path | None = None) -> Path:
    return (root or STORE_DIR) / f"{name}.csv"


def read_table(name: str, root: Path | None = None, index_col: str | None = None, utc: bool = False) -> pd.DataFrame | None:
    """Read a stored table; the first column is parsed as the (datetime) index."""
    path = table_path(name, root)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col = index_col or df.columns[0]
    df[col] = pd.to_datetime(df[col], utc=utc)
    return df.set_index(col).sort_index()


def write_table(name: str, df: pd.DataFrame, root: Path | None = None, float_format: str | None = None) -> Path:
    path = table_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_csv(path, float_format=float_format)
    return path


def merge_update(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Combine stored and freshly downloaded rows; fresh values win (revisions)."""
    if old is None or old.empty:
        return new.sort_index()
    if new is None or new.empty:
        return old.sort_index()
    combined = new.combine_first(old)
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def last_index(name: str, root: Path | None = None) -> pd.Timestamp | None:
    df = read_table(name, root)
    return None if df is None or df.empty else df.index.max()


# ---------------------------------------------------------------------------
# Status bookkeeping
# ---------------------------------------------------------------------------
def read_status(path: Path = STATUS_PATH) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def update_status(dataset: str, *, ok: bool, source: str, message: str = "", latest: str | None = None,
                  rows: int | None = None, path: Path = STATUS_PATH) -> dict:
    status = read_status(path)
    entry = status.get(dataset, {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry.update({"source": source, "last_attempt_utc": now, "ok": ok, "message": message})
    if ok:
        entry["last_success_utc"] = now
        if latest is not None:
            entry["latest_observation"] = latest
        if rows is not None:
            entry["rows"] = rows
    status[dataset] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True))
    return status


def status_frame(status: dict) -> pd.DataFrame:
    if not status:
        return pd.DataFrame()
    df = pd.DataFrame(status).T
    df.index.name = "dataset"
    cols = [c for c in ["source", "ok", "latest_observation", "last_success_utc", "last_attempt_utc", "rows", "message"] if c in df]
    return df[cols].sort_index()
