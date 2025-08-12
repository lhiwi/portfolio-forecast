# Portfolio Forecast (Week 11 · GMF Investments)

Time series forecasting and portfolio optimization for **TSLA**, **BND**, and **SPY** (2015-07-01 → 2025-07-31).
This repo uses a **notebooks-first** workflow. 

Task-1 is complete and reproducible in Colab.

## Data & Source

* **Assets:** TSLA (Tesla), BND (Vanguard Total Bond Market ETF), SPY (S\&P 500 ETF proxy).
* **Window:** 2015-07-01 → 2025-07-31 (business days).
* **Vendor:** **Stooq** via `pandas_datareader` for stability in Colab (symbols used: `TSLA.US`, `BND.US`, `SPY.US`, with `VOO.US` / `IVV.US` as backups for SPY).
* **Canonical labels:** In the notebook we map back to **TSLA / BND / SPY** so downstream code stays consistent.

> Why Stooq? Colab hit intermittent `yfinance` errors (timezone / “possibly delisted”). Stooq is stable and good for daily OHLCV.


## Repository Structure

```
.github/
  └─ workflows/
     └─ ci.yml               # lightweight CI (smoke checks)
.venv/                        # local virtual environment (ignored)
data/
  └─ raw/                     # optional local cache (ignored)
notebooks/
  └─ preprocessing.ipynb      # Task 1: end-to-end preprocessing & EDA
scripts/
  └─ __init__.py              # (placeholder for future)
src/
  └─ __init__.py              # (placeholder for future)
tests/
  ├─ __init__.py
  └─ test_smoke.py            # simple import smoke (optional)
.gitignore
README.md
requirements.txt
```

## Task 1 

* **Ingestion & Canonicalization:** Fetch OHLCV for TSLA/BND/SPY, map underlying Stooq symbols back to canonical tickers to keep all downstream code consistent.
* **Calendar Alignment:** Pivot to wide (Date × Ticker), reindex to **business days**, and forward-fill per asset to handle non-overlapping holidays.
* **Returns & Volatility:** Daily **log returns** (stationary proxy) and **21-day rolling std** (monthly volatility).
* **Stationarity (ADF):**

  * Prices typically **non-stationary** → differencing required for ARIMA on prices.
  * Returns generally **closer to stationary** → good targets for classical models or as features.
* **Risk Metrics:**

  * **Sharpe (annualized)** for risk-adjusted performance.
  * **1-day 95% VaR** (historical) as a loss threshold.
* **Outliers:** Simple $|z|>3$ flags for major return shocks—useful for robust modeling and risk commentary.


## CI (Continuous Integration)

We use a lightweight CI that:

* Sets up Python 3.11 on Ubuntu,
* Installs minimal dependencies,
* Runs a basic smoke check (and pytest only if `tests/` exists).

If you don’t need CI, you can ignore it; it won’t block your notebook workflow.

## Next Steps

**Task 2 — Modeling (ARIMA/SARIMA vs LSTM)**

* Split chronologically: train ≤ 2023-12-31, test 2024-01-01 → 2025-07-31.
* **ARIMA/SARIMA:** Choose differencing `d` via ADF (likely `d=1` for prices) or model returns with `d=0`.
* **LSTM:** Scale series, create supervised windows (e.g., 60-step lookback), train baseline LSTM.
* **Compare:** MAE / RMSE / MAPE and interpretation (intervals vs flexibility).

**Task 3 — Forecast (6–12 months)**

* Use the better model to produce a 6–12 month forecast.
* Plot forecast with **confidence intervals**; discuss interval widening and uncertainty.

**Task 4 — Portfolio Optimization (MPT)**

* **Expected returns:** Use model’s forward view for TSLA; use historical annualized means for BND & SPY.
* **Covariance:** From historical daily returns (annualized).
* Plot **Efficient Frontier**; mark **Tangency (Max Sharpe)** and **Min-Vol**; pick a recommendation.

**Task 5 — Backtesting**

* Backtest period: **2024-08-01 → 2025-07-31**.
* Compare chosen strategy vs **60/40 SPY/BND** benchmark: cumulative return & Sharpe.
* State limitations (no fees/slippage, rebalancing simplifications).


## Troubleshooting

* **Ticker fetch errors:** Task-1 uses **Stooq** (`pandas_datareader`), which is stable in Colab. The notebook prints which symbols were used (e.g., `SPY.US`).
* **Colab push conflicts:** If “Save a copy in GitHub” complains about non–fast-forward, either pick a new branch name (e.g., `task-1a`) or use a small push cell that does `git pull --rebase` before push.
* **Python version:** Use **Python 3.11** locally for smooth installs and compatibility.

