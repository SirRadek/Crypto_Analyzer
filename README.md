# Crypto Analyzer

[![CI](https://github.com/SirRadek/Crypto_Analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/SirRadek/Crypto_Analyzer/actions/workflows/ci.yml)

A modular Python project for **cryptocurrency price analysis and prediction** using both rule-based strategies and machine learning.

## Features

- Imports OHLCV data for any crypto pair (default: BTC/USDT) directly from Binance to SQLite.
- Calculates technical indicators (SMA, EMA, RSI, etc.).
- Feature engineering for ML models.
- Supports both rule-based and machine learning (RandomForest) predictions.
- Aggregates multiple ML models using usage-based weighting.
- Forecast loop ensembles all regression models via usage-based weights.
- Combines signals for final trading decision.
- Bound-model pipeline outputs price interval `p_low` ≤ `p_hat` ≤ `p_high`.
- Fully modular and easy to expand.

---

## Project Structure

```

crypto\_analyzer/
│
├── db/
│   ├── data/crypto_data.sqlite # Database and raw data storage
│   ├── db_connector.py       # Load data from SQLite
│   └── btc_import.py         # Download OHLCV data from Binance
├── analysis/
│   ├── indicators.py         # Technical indicators
│   ├── feature_engineering.py # Feature creation for ML
│   └── rules.py              # Rule-based signals
├── ml/
│   ├── train.py              # ML model training
│   ├── predict.py            # ML prediction
│   └── model_utils.py        # Save/load/evaluate models
├── prediction/
│   └── predictor.py          # Signal combination logic
├── utils/
│   └── helpers.py            # Helper utilities
├── main.py                   # Project entry point
├── requirements.txt
└── README.md

````

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/SirRadek/crypto_analyzer.git
cd crypto_analyzer
````

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download OHLCV data from Binance

Edit settings in `db/btc_import.py` (symbol, interval, date range if needed), then run:

```bash
python db/btc_import.py
```

This creates the `db/data/crypto_data.sqlite` file with price data.

### 4. Optional on-chain data

On-chain metrics can be merged from public APIs:

* [mempool.space](https://mempool.space/api/) – mempool stats (no key required; fetched automatically with `--use_onchain`)
* [Glassnode](https://glassnode.com/) – exchange flows (API key optional)
* [Whale Alert](https://developer.whale-alert.io/) – large transfers/mints (API key optional)

Store the resulting metrics alongside price data in the SQLite `prices` table
with column names prefixed by `onch_`.

### On-chain backfill + live logging

Historical mempool and difficulty data can be imported into a dedicated table
and continuously updated:

```bash
python api/backfill_onchain_history.py --start 2020-07-22 --end 2020-07-23 --db db/data/crypto_data.sqlite
python -m api.mempool_ws_logger --db db/data/crypto_data.sqlite
```

The backfill aligns snapshots to a 5‑minute UTC grid and uses Jochen Hoenicke
and Blockchain.com datasets as the mempool.space REST API does not expose
historical mempool snapshots. The WebSocket logger keeps the table current and
is intended to be scheduled via `cron`.

### 4. Run analysis and prediction

```bash
python main.py
```

By default, this trains a RandomForest model and outputs the latest signals.

### 5. Run the training pipeline

Example commands for 120‑minute horizon:

```bash
# Classification
python main.py --task clf --horizon 120 --split_params '{"test_size":0.2}' --out_dir outputs --use_onchain

# Regression
python main.py --task reg --horizon 120 --split_params '{"test_size":0.2}' --out_dir outputs --use_onchain
```

The pipeline writes metrics and predictions to CSV files and a simple PNG plot
into the chosen `outputs/` directory.  Run configuration is logged as
`outputs/run_config.json`.

> **Note:** When adding on-chain signals, resample them to the candle interval
> (5 min) before merging to prevent look‑ahead leakage.

---

## Training on Historical Data

The project can be trained end‑to‑end on past data to evaluate strategies or to
produce models for later inference.  The following example demonstrates a
typical workflow for a 2‑hour classification horizon:

1. **Download candles** – adjust the symbol and date range in
   `db/btc_import.py` and run:

   ```bash
   python db/btc_import.py
   ```

2. **(Optional) Merge additional on‑chain metrics** – mempool stats are
   fetched automatically, but other metrics (e.g. exchange flows) can be
   retrieved via `api/onchain.py` and stored with the `onch_` prefix in the
   SQLite `prices` table.

3. **Feature engineering & target creation** – invoke the training pipeline
   which automatically builds all features and targets:

   ```bash
   python main.py --task clf --horizon 120 --out_dir outputs --use_onchain
   ```

   The command writes predictions, metrics and diagnostic plots into the
   `outputs/` directory.  Additional artefacts such as permutation and SHAP
   importances are also exported for further analysis.

4. **Inspect results** – review the generated CSV files and graphs in
   `outputs/` to understand model performance and feature behaviour.  The file
   `run_config.json` captures the exact parameters used for the run, ensuring
   full reproducibility.

---

## Customization

* Add or change features in `analysis/feature_engineering.py`.
* Adjust or create new rules in `analysis/rules.py`.
* Tune ML model or use another classifier in `ml/train.py`.
* Combine rule and ML signals as you wish in `prediction/predictor.py`.

---

## Requirements

* Python 3.13
* [See `requirements.txt`](./requirements.txt) – notable pin: `xgboost>=2.1,<4`

## Determinism & Repro

Training routines default to ``random_state=42`` as defined in
``ml/train.py``. The chosen value, together with other parameters, is logged to
``outputs/run_config.json`` to ensure runs are repeatable.

## Troubleshooting

XGBoost and PyTorch automatically fall back to CPU if no compatible CUDA wheel
is available.

## Development: pre-commit

Install hooks and run them locally before committing:

```bash
pip install pre-commit
pre-commit install
pre-commit run -a
```

## No financial advice

The project is provided for research and educational purposes only and does not
constitute financial advice.
