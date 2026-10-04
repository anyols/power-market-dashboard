# European Power & Crude Oil Trading Dashboard

A Python/Streamlit analytics platform that turns energy-market fundamentals into trading-style reasoning.

- **European power.** Day-ahead prices, load, wind, solar, residual demand, forecast errors, spikes,
  volatility and cross-border flows.
- **Crude oil (WTI/Brent).** Prices, inventories against seasonal norms, Cushing, the futures curve,
  speculative positioning and the Brent–WTI spread.

Each market has a transparent signal lab with an honest backtest and a rule-based market brief. A
scheduled GitHub Action refreshes the data every morning.

> **This project is designed to connect market fundamentals to trading-style decision-making.
> It is not a black-box price prediction model.**

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red) ![Tests](https://img.shields.io/badge/tests-pytest-green) ![Data](https://img.shields.io/badge/data-refreshed%20daily-orange)

> ⚠️ **Data notice.** Until the morning job has run with API keys, the app uses **synthetic sample
> datasets** (`data/sample/`). They copy the structure and broad behaviour of the real sources, but they are
> **not real market data**, and every page says so.

---

## 1. Why this matters for trading

Desks trading fundamental energy markets start the day with the same questions.

**Power**
- How tight is the system tomorrow? Where does residual demand land on the merit order?
- What did the forecasts miss, and how will that be re-traded intraday and in imbalance?
- Where are the spike and negative-price risks? Are interconnectors congested?

**Crude**
- Did this week's EIA report draw more or less than normal for the season?
- How close is Cushing to tank bottoms?
- Is the curve signalling scarcity (backwardation) or surplus (contango)?
- Are speculators already crowded on one side?

The dashboard answers these with transparent, reproducible analytics. Every number can be traced
back to a source, and every rule is written down.

## 2. Market intuition

### Power

**Residual demand.** Residual demand = load − wind − solar. Renewables have near-zero marginal
cost, so the residual must be covered by dispatchable plants or imports.
- High residual demand pushes pricier units onto the margin: a tighter system and bullish pressure.
- Low or negative residual demand is where negative prices come from.

**Renewables.** Wind sets multi-day regimes. Solar shapes the day: it depresses midday prices
(*cannibalisation*) and creates a steep evening ramp, where many spikes sit.

**Forecast errors.** The day-ahead auction clears on forecasts. Misses are re-balanced intraday or
through balancing energy. A system shorter than forecast supports intraday and imbalance prices.

**Spikes and flows.**
- Spikes cluster where high demand meets low renewables. Because prices can go negative,
  volatility is measured in €/MWh, not % returns.
- Market coupling moves power from cheap to expensive zones until prices converge or a line
  congests. Imports relieve tightness; exports add to it.

### Crude oil

**Inventories.** Stocks are the market's shock absorber. Below-normal stocks leave less buffer, so
prompt barrels command a premium.
- Seasonality is large: crude builds in spring and draws in summer. Every level and weekly change
  is therefore compared with the *same week* in the previous five years.

**Cushing.** Cushing, Oklahoma is the delivery point of the NYMEX WTI contract.
- Low Cushing stocks tighten the front of the WTI curve directly.
- Near tank bottoms (~20 mb), squeezes become possible.

**The curve (theory of storage).**
- Scarce inventories give physical oil a *convenience yield*, so the curve backwardates (front
  above deferred) and holders of the front contract earn roll yield.
- Surplus inventories put the curve in contango, which pays for storage.

**Positioning.** When managed money is already very long, new buyers are scarce and risk skews
towards liquidation (and vice versa). Extremes are judged against a trailing three-year
distribution.

**Brent–WTI.** The spread between the seaborne and the inland US benchmark reflects export
economics, pipeline logistics and the regional supply–demand balance.

## 3. Features

| Section | Page | What it shows |
|---|---|---|
| Power | **Market Overview** | Aligned panels of price, load, wind, solar and residual demand (actual vs forecast), daily ranges, volatility, spike and negative hours, and a rule-based trader read-out |
| Power | **Price Drivers** | Driver scatters, quadratic merit-order fit (R², €/MWh per GW), empirical supply curve, correlation matrix and rolling correlation |
| Power | **Forecast Surprises** | Error time series and distributions, hour × weekday heatmaps, z-scores, top 20 surprise hours with auto-comments |
| Power | **Spikes & Volatility** | Configurable spike definition, timing, fundamentals in spike vs normal vs negative hours, natural-language explanation |
| Power | **Cross-Border Flows** | Net position, flows by border, spreads, flow vs spread, convergence and congestion statistics |
| Power | **Signal Lab / Daily Brief / Risk** | −3…+3 fundamental score with honest backtest, deterministic daily note, PnL/drawdown/exposure/stress tests |
| Crude | **Crude Overview** | WTI and Brent (live indicative and official), Brent–WTI, current forward curve, KPIs, data-freshness table |
| Crude | **Inventories & Balances** | 5-year seasonal bands for crude, Cushing, gasoline and distillate; refinery runs; demand; supply; latest report vs seasonal norms |
| Crude | **Curve & Positioning** | Theory-of-storage scatter (Cushing vs M1–M2), time spreads, live curve, CFTC managed-money positioning and percentile |
| Crude | **Crude Signal Lab** | Weekly four-vote score (inventory surprise, Cushing, curve carry, positioning), roll-adjusted backtest, current reading |
| Crude | **Weekly Crude Brief** | Rule-based desk note on the latest EIA report, plus any historical week |
| About | **Data Pipeline Status** | What the morning job refreshed, the latest observation per dataset, errors |

**Engineering highlights**

- **Morning pipeline.** `scripts/daily_update.py` runs as a scheduled GitHub Action.
  - It fetches incrementally, with an overlap window to pick up revisions, and merges into CSV
    tables under `data/store/`.
  - It validates the data with `tests/test_store_integrity.py` before committing.
  - One failing source never blocks the others.
- **API clients.** ENTSO-E (XML, SQLite cache, 15/60-min resolutions), EIA API v2
  (newest-first paging, one request per series per day), CFTC Socrata, and Yahoo Finance.
- **NYMEX CL contract calendar.** Implements the CME last-trade rule and is tested against
  published 2024 expiries. It is used for **roll-adjusted** front-month returns, so contract rolls
  never book fake PnL.
- **Release-calendar-aware information sets.**
  - EIA weekly data is used only from the Thursday after the week ends (released Wednesday).
  - CFTC data is used only from Friday (positions as of Tuesday).
  - Power signals use only what is known before the 12:00 CET day-ahead gate.
  - Perturbation tests scramble everything not yet published and check that signals don't move.
    Deliberately injected leaks were verified to fail these tests.
- **Timezone-correct power pipeline.** It handles 23- and 25-hour DST days, matches the same hour
  by local date, and validates against SDAC price limits.

## 4. Screenshots

> Placeholders. Run the app and save screenshots into `docs/screenshots/` with these names.

| | |
|---|---|
| ![Power overview](docs/screenshots/overview.png) | ![Crude inventories](docs/screenshots/crude_inventories.png) |
| *Power: Market Overview* | *Crude: Inventories vs 5-year range* |
| ![Signal lab](docs/screenshots/signal_lab.png) | ![Crude brief](docs/screenshots/crude_brief.png) |
| *Power: Signal Lab* | *Crude: Weekly Brief* |

## 5. Installation

Requires Python 3.11+.

```bash
git clone <your-fork-url> power-market-dashboard
cd power-market-dashboard
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Tested with Python 3.11.9, pandas 3.0, numpy 2.4, Streamlit 1.65, Plotly 7.1, pydantic 2.13 and yfinance 1.7.

## 6. API keys

| Key | Needed for | How to get it |
|---|---|---|
| `EIA_API_KEY` | Crude prices and weekly balances | Free, instant: [eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php) |
| `ENTSOE_API_TOKEN` | Power data | Free: register on the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/), then request REST API access. At the time of writing, you email `transparency@entsoe.eu` with the subject "Restful API access". Then generate a token in your account settings |
| none | CFTC positioning, Yahoo prices | — |

For local use, copy `.env.template` to `.env` and fill in the keys:

```bash
cp .env.template .env
```

The public dashboard itself needs **no keys**: it reads the refreshed data committed to the repository.

## 7. Running

```bash
streamlit run app.py
```

Refresh the data store by hand (does the same as the morning job):

```bash
python scripts/daily_update.py
```

Run the tests:

```bash
pytest
```

The synthetic samples are deterministic and can be regenerated at any time:

```bash
python scripts/generate_sample_data.py
```

```bash
python scripts/generate_sample_crude.py
```

## 8. Automated morning refresh: GitHub Actions + Streamlit Cloud

```
 05:15 UTC daily                     commits data/store/*.csv            redeploys on push
┌──────────────────────┐   EIA     ┌──────────────────────────┐        ┌─────────────────────┐
│ GitHub Action        │──CFTC───▶│ GitHub repo (main)        │──────▶│ Streamlit Community │
│ daily-update.yml     │  ENTSO-E  │ data/store + status.json  │        │ Cloud: app.py       │
└──────────────────────┘           └──────────────────────────┘        └─────────┬───────────┘
                                                                                  │ live, cached 1 h
                                                                        Yahoo Finance (never stored)
```

One-time setup:

1. **Push the project to GitHub.** Create an empty repository, then:
   ```bash
   git init -b main
   ```
   ```bash
   git add . && git commit -m "Initial commit"
   ```
   ```bash
   git remote add origin https://github.com/<you>/power-market-dashboard.git && git push -u origin main
   ```
2. **Add secrets.** In the repository, go to *Settings → Secrets and variables → Actions → New
   repository secret* and add `EIA_API_KEY`, plus `ENTSOE_API_TOKEN` once you have it.
3. **Allow the workflow to commit.** In *Settings → Actions → General → Workflow permissions*,
   choose "Read and write permissions".
4. **Backfill history.** In *Actions → Daily data refresh → Run workflow*, tick "Re-download full
   history" for the first run. After that it runs every morning at 05:15 UTC on its own.
5. **Deploy.** On [share.streamlit.io](https://share.streamlit.io), create an app from your repo:
   branch `main`, main file `app.py`, Python 3.11. No secrets are needed. Every data commit
   redeploys the app.

Operational notes:

- **Keep-alive.** GitHub disables scheduled workflows in repositories with no activity for 60 days.
  The daily data commits normally keep the repository active. If the schedule ever stops,
  re-enable it in the Actions tab.
- **Failure handling.** A run with a failed source is flagged red, but successful sources are
  still committed.
- **Status page.** The *Data Pipeline Status* page shows what happened on each run.
- **CI.** `.github/workflows/ci.yml` runs ruff and the test suite on every push and pull request.

## 9. Data sources and their limits

| Source | Data | Rhythm | Notes |
|---|---|---|---|
| ENTSO-E | Power prices, load, wind/solar (actual and forecast), physical flows | Day-ahead prices ~13:00 CET for the next day | 15-minute day-ahead MTU since 1 Oct 2025 (averaged to hourly here) |
| EIA API v2 | WTI/Brent spot; weekly stocks, runs, production, trade, demand | Weekly report Wed 10:30 ET; daily prices lag several days | Public domain |
| EIA API v2 | NYMEX futures, contracts 1–4 | **Discontinued after 5 Apr 2024** | History only; used for the curve and the roll-adjusted backtest |
| CFTC | Disaggregated COT, WTI-Physical (NYMEX 067651) | Fri 15:30 ET, positions as of Tuesday | Public |
| Yahoo Finance | CL=F, BZ=F and individual CL contracts | Live | Unofficial and indicative; its terms restrict redistribution, so the app fetches it live and **never stores or commits it** |

The project never mixes real and synthetic data on the same page. Each commodity switches to real
data only when its core tables are in the store.

## 10. Methodology

### Power signal

The decision for delivery day **D** is taken on **D-1 at ~11:00 CET**, before the 12:00 SDAC gate.
- **Inputs.** Day-ahead forecasts for D, actuals up to D-2, and prices up to D-1.
- **Target.** Price(D, h) − price(D-1, h). "Next-hour day-ahead change" is not a valid target,
  because all 24 prices are set in one auction.
- **Score.** Three votes: the residual-demand change z-score, the lagged load surprise and the
  lagged renewable surprise.
- **PnL.** A *proxy*: yesterday's price is not a tradeable entry level. A cost is charged on every MWh.

### Crude signal (weekly)

Decided every **Thursday at settlement**. Each input is used only after its official release:
- EIA weekly data from Thursday;
- CFTC data from Friday;
- prices at D's settlement.

| Vote | Input | Bullish when |
|---|---|---|
| Inventory surprise | Weekly crude-stock change minus the 5-year average change for that week, standardised | Bigger draw than normal |
| Cushing tightness | Cushing stocks vs the 5-year average, standardised | Unusually low |
| Curve (carry) | Front minus second-month futures | Backwardation |
| Positioning (contrarian) | Managed-money net length percentile, 3 years | Crowded short (bearish when crowded long) |

- **Target.** Next Thursday's settlement minus this Thursday's, on a **roll-adjusted** front-month
  series (official EIA futures, to Apr 2024) or on WTI spot (to the latest data).
- **Execution.** Settlement-to-settlement entry and exit is achievable with NYMEX **Trade-at-Settlement**
  orders, so this PnL is much closer to executable than the power proxy.
- **Costs.** Default $0.03/bbl round trip; PnL is per 1,000-bbl contract.

### Briefs

Both briefs are deterministic and rule-based. Each uses only what was public when it would have
been written, and a unit test enforces this.

### Illustrative results on the synthetic samples

| Market | Vote | Hit rate when active |
|---|---|---|
| Power (DE_LU) | Residual-demand change | ~98% |
| Power | Lagged load / renewable surprise | ~48% / ~51% |
| Crude | Curve (carry) | ~54% |
| Crude | Cushing tightness | ~55% |
| Crude | Inventory surprise / positioning | ~48% / ~49% |
| Crude | Combined weekly score | ~56% on 262 trades, Sharpe-like ≈ 0.5 |

These numbers show the method, not real-market evidence. Each generator documents the
relationships it builds in:
- In power, day-ahead prices follow forecast residual demand. This is real, but the market knows the
  forecast too, so beating yesterday's price is not beating the market.
- In crude, the curve earns roll yield, while weekly inventory surprises are priced instantly and so
  carry no information for the following week.

The two coin-flip results are useful negative findings, and the dashboard shows them instead of
hiding them.

## 11. Limitations

- **Proxy PnL (power).** Yesterday's price is not a tradeable reference.
- **No analyst consensus (crude).** "Surprises" are measured against seasonal norms, because
  consensus surveys are not free.
- **No official futures curve after April 2024.** The live curve comes from an unofficial source.
- **Missing drivers.**
  - Power: TTF gas, EUA carbon and REMIT outage data.
  - Crude: OPEC+ policy, refinery outages and product cracks.
- **Execution not modelled.** No market impact, position limits, margin or intraday re-trading.
- **Short or synthetic samples.** Treat every metric as descriptive, not statistically proven.

## 12. Future improvements

- **New commodities:** natural gas (Henry Hub storage, EU AGSI+ storage) and refined products
  (crack spreads, using EIA product prices and stocks).
- **Power:** target the day-ahead vs intraday/imbalance spread, where forecast errors carry value.
- **Statistics:** walk-forward evaluation with bootstrap confidence intervals.
- **Data:** a paid futures feed (e.g. Databento) for an official current curve and full contract
  histories; weather-forecast ingestion (Open-Meteo); hydro and reservoir data.

## 13. CV bullets

**European Power & Crude Oil Trading Dashboard**

- Built a Python/Streamlit dashboard tracking European power prices, load, renewable generation and residual demand across selected markets
- Analysed price spikes, volatility and forecast errors to identify short-term supply-demand imbalances and market drivers
- Developed transparent trading-style signals and daily market briefs linking fundamentals, weather-driven generation and price behaviour
- Extended the platform to crude oil: EIA inventories vs 5-year seasonal norms, Cushing, futures curve (theory of storage) and CFTC positioning, with a release-calendar-aware weekly signal and roll-adjusted backtest
- Automated a daily data pipeline (GitHub Actions → validated CSV store → Streamlit Cloud) integrating ENTSO-E, EIA and CFTC APIs with incremental updates, failure isolation and data-freshness monitoring

---

*Educational and research project. Nothing here is investment advice. Signals and backtests are illustrative and rely on simplified assumptions.*
