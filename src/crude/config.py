"""Crude-oil dataset definitions (EIA series IDs, units, store layout)."""

from __future__ import annotations

HISTORY_START = "2010-01-01"  # 5 years of seasonal history before the 2015+ analysis window
BARRELS_PER_CONTRACT = 1_000  # NYMEX CL contract size

# Daily prices ($/bbl). RCLC1..4 = NYMEX light sweet crude futures, contracts 1-4
# (contract 1 = nearest expiry; it rolls the day after last trade).
EIA_DAILY = {
    "wti_spot": "PET.RWTC.D",
    "brent_spot": "PET.RBRTE.D",
    "cl1": "PET.RCLC1.D",
    "cl2": "PET.RCLC2.D",
    "cl3": "PET.RCLC3.D",
    "cl4": "PET.RCLC4.D",
}

# Weekly Petroleum Status Report. Stocks in thousand barrels, flows in kb/d.
EIA_WEEKLY = {
    "crude_stocks": "PET.WCESTUS1.W",                   # commercial crude, excl. SPR
    "cushing_stocks": "PET.W_EPC0_SAX_YCUOK_MBBL.W",     # Cushing, OK - WTI delivery hub
    "spr_stocks": "PET.WCSSTUS1.W",
    "gasoline_stocks": "PET.WGTSTUS1.W",
    "distillate_stocks": "PET.WDISTUS1.W",
    "refinery_utilization": "PET.WPULEUS3.W",            # % of operable capacity
    "crude_production": "PET.WCRFPUS2.W",
    "crude_net_imports": "PET.WCRNTUS2.W",
    "refinery_crude_input": "PET.WCRRIUS2.W",
    "product_supplied": "PET.WRPUPUS2.W",                # implied demand, total products
}

LABELS = {
    "crude_stocks": "US commercial crude stocks",
    "cushing_stocks": "Cushing stocks",
    "spr_stocks": "Strategic Petroleum Reserve",
    "gasoline_stocks": "Gasoline stocks",
    "distillate_stocks": "Distillate stocks",
    "refinery_utilization": "Refinery utilisation",
    "crude_production": "US crude production",
    "crude_net_imports": "Crude net imports",
    "refinery_crude_input": "Refinery crude runs",
    "product_supplied": "Implied product demand",
}
UNITS = {k: ("%" if k == "refinery_utilization" else ("kb/d" if k in
         ("crude_production", "crude_net_imports", "refinery_crude_input", "product_supplied") else "mb"))
         for k in EIA_WEEKLY}

# Store / sample table names (relative to data/store or data/sample)
T_DAILY = "crude/eia_daily"
T_WEEKLY = "crude/eia_weekly"
T_COT = "crude/cot_wti"
T_YAHOO = "crude/yahoo_daily"  # sample only; live Yahoo data is never committed
