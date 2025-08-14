# Portfolio Forecasting & Optimization (TSLA, BND, SPY)

> 10 Academy — Week 11 Challenge
> Time Series Forecasting for Portfolio Management Optimization (GMF Investments)

**TL;DR**
We forecast **Tesla (TSLA)** using an LSTM model, convert that view into an expected return, then optimize a **TSLA/BND/SPY** portfolio with Modern Portfolio Theory and **backtest** it versus a 60/40 benchmark.


## Project Goal

1. **Forecast TSLA** over 6–12 months with uncertainty bands.
2. **Translate** that forecast into an expected return for TSLA.
3. **Optimize** a 3-asset portfolio **(TSLA, BND, SPY)** using MPT.
4. **Backtest** the chosen allocation against a **60/40 SPY/BND** benchmark.

We treat the forecast as **one input** to decision-making, consistent with the **Efficient Market Hypothesis (EMH).**

---

## Data

* **Assets:** TSLA (equity), BND (US bond ETF), SPY (US equity ETF)
* **Frequency:** daily business days
* **Window:** **2015-07-01 → 2025-07-31**
* **Source:** `yfinance` with automatic **fallback to Stooq** if Yahoo endpoints fail
* **Saved datasets:**

  * `data/processed/prices.csv` — Close prices (note: **not Adjusted Close**)
  * `data/processed/returns.csv` — Simple daily returns
  * `data/processed/meta.json` — Provenance (tickers, date range, source used)

> If you prefer Adjusted Close, switch in the notebooks and be consistent throughout; current pipeline uses **Close**.

---

## Results (1-page)

**Modeling (Task 2, test 2024–2025)**

* **ARIMA(2,1,2):** MAE **63.73**, RMSE **78.87**, MAPE **24.22%**
* **LSTM (TF):** **MAE 15.13**, **RMSE 19.31**, **MAPE 5.99%** → **Selected**

**Forecast (Task 3, as of 2025-07-31, last price \$308.27)**

* **6-month mean:** **\$248.26** (**−19.47%**), 95% CI widens ×**1.33**
* **12-month mean:** **\$247.87** (**−19.59%**), 95% CI widens ×**1.30**

**MPT (Task 4, long-only, rf=0%)**

* **Max Sharpe:** μ **11.52%**, σ **14.82%**, Sharpe **0.78**

  * Weights: **TSLA 0.0%**, **BND \~18.0%**, **SPY \~82.0%**
* **Min Vol:** μ 1.48%, σ 5.31%, Sharpe 0.28

**Backtest (Task 5, 2024-08-01 → 2025-07-31)**

* **Strategy (Max-Sharpe, Hold):** **Total 15.80%**, Ann **15.27%**, σ **16.33%**, **Sharpe 0.955**
* **Strategy (Monthly Rebal):** Total 15.68%, Ann 15.17%, σ 16.18%, Sharpe 0.957
* **Benchmark 60/40 (Hold):** Total 11.04%, Ann 10.68%, σ 12.25%, Sharpe 0.892

**Conclusion:** The LSTM-informed **Max-Sharpe** allocation **outperformed** 60/40 on return and Sharpe over the backtest, with slightly higher volatility.

---

## Repository Structure

```
portfolio-forecast/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/                  # (ignored) source pulls / scratch
│   └── processed/            # prices.csv, returns.csv, meta.json
├── models/                   # saved models (kept local or LFS; optional)
├── notebooks/
│   ├── preprocessing.ipynb           # Task 1
│   ├── modeling.ipynb                # Task 2 (ARIMA vs LSTM)
│   ├── forecasting.ipynb             # Task 3 (LSTM + MC-dropout bands)
│   ├── portfolio_optimization.ipynb  # Task 4 (NumPy/SciPy MPT)
│   └── backtest.ipynb                # Task 5 (strategy vs 60/40)
├── scripts/
│   └── __init__.py
├── src/
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
├── requirements.txt   # (for local use; Colab recommended)
├── .gitignore
└── README.md
```

> Large files under `data/` are ignored by Git. Use the notebooks to regenerate.

---

## How to Run (Colab-first)

**Open any notebook in Colab:**
GitHub → the notebook → **Open in Colab** (or use the Colab extension) → **File → Save a copy to GitHub** to commit changes.

**Local (optional)**
If you must run locally, use Python **3.11** and `pip install -r requirements.txt`. Some packages (TF on Windows) may require extra toolchains; Colab is smoother.

---

## Repro Steps by Task

**Task 1 — Preprocessing & EDA**
Run `notebooks/preprocessing.ipynb`

* Builds `prices.csv`, `returns.csv`, `meta.json`
* EDA: close curves, log returns, rolling vol, stationarity tests

**Task 2 — Modeling**
Run `notebooks/modeling.ipynb`

* Chronological split (train ≤ 2023-12-31; test ≥ 2024-01-01)
* ARIMA vs LSTM; print **MAE/RMSE/MAPE**
* Saves **`models/tsla_lstm_tf.keras`** and **`models/tsla_scaler.pkl`**

**Task 3 — Forecasts**
Run `notebooks/forecasting.ipynb`

* Loads saved LSTM + scaler
* 6/12-month rollout + **MC-dropout 95% bands**
* Optionally writes CSVs/PNGs to **Drive**: `/content/drive/MyDrive/preprocessed_data`

**Task 4 — Optimization**
Run `notebooks/portfolio_optimization.ipynb`

* TSLA expected μ from 12m forecast; BND/SPY μ from historical means
* Annualized covariance from history
* **NumPy/SciPy** efficient frontier, **Max-Sharpe**/**Min-Vol**
* Recommends Max-Sharpe (TSLA 0%, BND \~18%, SPY \~82%)

**Task 5 — Backtest**
Run `notebooks/backtest.ipynb`

* Window: **2024-08-01 → 2025-07-31**
* Simulates **buy-and-hold** and **monthly rebal**
* Plots cumulative value & prints **Total/Ann Return, Ann Vol, Sharpe**

---

## Models & Artifacts

* `models/tsla_lstm_tf.keras` — Trained LSTM (TensorFlow/Keras)
* `models/tsla_scaler.pkl` — MinMaxScaler for inverse transforms
* `models/tsla_arima.pkl` — ARIMA reference (optional)
* `models/summary.json` — Chosen model + metrics + split date

> If you don’t commit `models/`, keep them in Drive and load from there.

---

## Design Choices & Notes

* **Price basis:** **Close** (not Adjusted Close). Be explicit in figure captions.
* **Stationarity:** ARIMA applied on differenced series; LSTM worked on scaled levels.
* **Uncertainty:** ARIMA provides CIs natively; for LSTM we used **MC-dropout** (200 sims).
* **MPT engine:** Implemented with **NumPy + SciPy (SLSQP)** to avoid heavy deps and environment issues.
* **Constraints:** Long-only, fully invested, rf=0%. Shorting/leverage can be enabled if needed.
* **EMH stance:** Forecasts are **inputs** to allocation; uncertainty grows with horizon.

---

## Figures to Export

1. **Figure 1:** TSLA/BND/SPY **Close** prices (2015–2025).
   *Caption:* “Close prices (not adjusted for dividends/splits).”
2. **Figure 2:** **Log returns** and **21-day rolling annualized volatility**.
   *Caption:* “Daily log returns $r_t=\ln(P_t/P_{t-1})$ and 21-day rolling σ × √252.”
3. **Figure 3:** ARIMA vs test (with 95% CI).
4. **Figure 4:** LSTM vs test (forecast overlay).
5. **Figure 5:** 6-month LSTM mean forecast with 95% MC-dropout band.
6. **Figure 6:** 12-month LSTM mean forecast with 95% MC-dropout band.
7. **Figure 7:** Efficient Frontier (assets + **Max-Sharpe** and **Min-Vol** markers).
8. **Figure 8:** Backtest — cumulative portfolio value (strategy variants vs 60/40).

---

## Next Steps

* Add **exogenous features** (rates, macro, earnings) and test **ARIMA-GARCH** / **hybrids**.
* Use **shrinkage** or **EWMA** covariance; report sensitivity.
* Introduce **risk-free > 0%**, **turnover budgets**, and **trading-cost** modeling.
* Add **max drawdown** and **rolling Sharpe** to the backtest report.
* Automate a **monthly refresh** loop: refetch → refit → re-optimize → report.

---

## Disclaimer

This repository is for **educational purposes** only and **not investment advice**. Markets are risky; past or simulated performance does not guarantee future results.

---

**Maintainer:** GMF Investments — Week 11 challenge
**Branches used:** `task-1`, `task-2`, `task-3`, `task-4`, `task-5` (Colab-first workflow)
