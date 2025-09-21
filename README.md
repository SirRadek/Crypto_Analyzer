# Crypto Analyzer

[![CI](https://github.com/SirRadek/Crypto_Analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/SirRadek/Crypto_Analyzer/actions/workflows/ci.yml)

A modular Python project for **cryptocurrency price analysis and prediction** focused on probabilistic classification of short-term ±0.5 % moves.

## Features

- Imports OHLCV data for any crypto pair (default: BTC/USDT) directly from Binance to SQLite.
- Calculates technical indicators (SMA, EMA, RSI, etc.).
- Feature engineering for ML models.
- Supports both rule-based and machine learning classifiers (RandomForest/XGBoost) tuned for directional moves.
- Outputs calibrated probabilities for "touch ±0.5 %" style targets over configurable horizons.
- Lightweight signal aggregation keeps the default pipeline purely classification-based.
- Fully modular and easy to expand.

## Rozšířené rysy

Advanced data sources can be merged into the default feature table without
introducing look-ahead bias. All auxiliary loaders align their inputs using
**left-label resampling**: external snapshots are resampled to the candle
frequency with the *label* aligned to the left edge of the interval and then
forward-filled so that the feature timestamp never exceeds the candle open.

### Derivátová data

Enable the derivative enrichments while generating features:

```bash
python scripts/make_features.py --use_derivatives
```

The loader expects either the columns to be present in the raw OHLCV source or
external tables configured via `config/app.yaml`. The following columns are
produced after left-label alignment:

| Sloupec | Popis |
| --- | --- |
| `funding_z` | Z-skóre posledního funding rate s ohledem na historický průměr a směrodatnou odchylku. |
| `basis_bp` | Perpetual basis přepočtený na basis points (bp) a zarovnaný na mřížku svíček. |
| `oi_change_rate` | Relativní změna (pct change) open interestu mezi jednotlivými snapshoty. |

### Order book signály

Depth snapshots a event stream lze zapojit přes praktický přepínač:

```bash
python scripts/make_features.py --use_orderbook
```

Při zpracování se používá stejná left-label resampling logika – order book depth
je agregován k času otevření svíčky a nikdy nevidí budoucí tick. Výstup obsahuje
tyto metriky:

| Sloupec | Popis |
| --- | --- |
| `depth_imbalance` | Poměr kumulativních bid/ask velikostí `(bid - ask) / (bid + ask)` v rámci sledovaných hladin. |
| `spread` | Quoted spread v basis points spočítaný z nejlepšího bidu a asku. |
| `ofi` | Order flow imbalance agregovaný ze streamu obchodů / objednávek podle timestampu. |

---

## Project Structure

```

.
├── src/
│   └── crypto_analyzer/
│       ├── data/            # Data access (SQLite, ingestion helpers)
│       ├── features/        # Feature engineering pipelines
│       ├── labeling/        # Target generation utilities
│       ├── models/          # Training, prediction & ensembles
│       ├── eval/            # Backtests, CV utilities, metrics
│       ├── utils/           # Shared configuration and helpers
│       └── legacy/          # Backwards-compatible regression / usage ensembles
├── scripts/
│   ├── make_features.py     # CLI for feature engineering
│   ├── train.py             # CLI for model training
│   └── backtest.py          # CLI for quick backtests
├── archive/                 # Historical experiments & shell scripts
├── config/
├── tests/
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

### 3. Development tooling

Activate the virtual environment and install the development extras to get the linting,
typing and security tooling used in CI:

```bash
pip install -e .[dev]
pip install pip-audit bandit detect-secrets
```

Install the Git hooks so that Ruff, Black, isort, Pyupgrade and the formatting / secret
checks run automatically before each commit:

```bash
pre-commit install
pre-commit run --all-files  # run the full suite manually
```

Run the same commands as the CI workflow when preparing a change set:

```bash
# Formatting & linting
ruff check .
black --check .
isort . --check-only

# Typing
mypy src

# Tests
pytest -q --cov=src/crypto_analyzer --cov-report=term-missing --cov-fail-under=80
# (Coverage excludes integration/CLI modules defined in pyproject.toml to focus on unit-tested code.)

# Security
pip-audit
bandit -r src
```

### 4. Download OHLCV data from Binance

Edit settings in `src/crypto_analyzer/data/binance_import.py` (symbol, interval, date
range if needed), then run:

```bash
python -m crypto_analyzer.data.binance_import
```

This creates the `data/crypto_data.sqlite` file with price data.

### 5. Configure the application

Configuration is centralised in `config/app.yaml`. Copy
`config/app.example.yaml`, adjust the sections to match your environment (e.g.
database paths, feature toggles, on-chain credentials) and optionally point the
`APP_CONFIG_FILE` environment variable to your custom file. Every option has a
documented default so the application still runs without manual changes.

### 6. Optional on-chain data

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
python api/backfill_onchain_history.py --start 2020-07-22 --end 2020-07-23 --db data/crypto_data.sqlite
python -m api.mempool_ws_logger --db data/crypto_data.sqlite
```

The backfill aligns snapshots to a 5‑minute UTC grid and uses Jochen Hoenicke
and Blockchain.com datasets as the mempool.space REST API does not expose
historical mempool snapshots. The WebSocket logger keeps the table current and
is intended to be scheduled via `cron`.

### 7. Run analysis and prediction

```bash
python scripts/make_features.py --output data/features.parquet
python scripts/train.py --features data/features.parquet --model-path artifacts/meta_model.joblib
python scripts/backtest.py data/predictions.csv --equity-output reports/equity.csv
```

Cost-aware backtest with latency and probability gates:

```bash
python scripts/backtest.py data/predictions.csv --fee_bps 1 --slip_bps 1 --latency_min 1 --p_touch_thr 0.6 --p_up_thr 0.55
```

To explore how execution costs interact with the probability gates you can run
an EV sweep and heatmap export:

```bash
python scripts/optimize_thresholds.py data/predictions.csv --fee_bps 1 --slip_bps 1 --latency_min 1
```

The CLI entry points can be combined with your own data source by pointing
`scripts/make_features.py` to a CSV/Parquet file (`--source file --input ...`). The
trained model stores calibrated probabilities for the default ±0.5 % "touch"
target at the path provided via `--model-path` and the backtest command writes
both metrics (`backtest_metrics.json`) and the equity curve to CSV.

---

## Default classification target

The current production pipeline predicts whether price will touch ±0.5 % from
the open within the selected horizon.  Feature engineering, training and
backtesting CLIs therefore operate on classification labels and expose
probabilities that can be thresholded for position sizing.  Legacy regression or
usage-based ensembles remain available under `src/crypto_analyzer/legacy/`, but
are no longer wired into the default command sequence above.

### 8. Legacy pipeline

Example commands for 120‑minute horizon:

```bash
python main.py --task clf --horizon 120 --split_params '{"test_size":0.2}' --out_dir outputs --use_onchain
```

The pipeline writes metrics, predictions, explainability artefacts and trained
models into a timestamped directory under the requested output root.  Each run
is stored as `outputs/run_id=.../` and contains `metadata.json`,
`config_snapshot.yaml`, `metrics.json`, the trained model and generated
artefacts so that the experiment can be reproduced.

Reliability diagnostics and equity curves are exported to `reports/` for quick
inspection, for example:

![Reliability Curve](reports/reliability_20240101_120000.png)

![Equity Curve](reports/equity_20240101_120000.png)

> **Note:** When adding on-chain signals, resample them to the candle interval
> (5 min) before merging to prevent look‑ahead leakage.

---

## Training on Historical Data

The project can be trained end‑to‑end on past data to evaluate strategies or to
produce models for later inference.  The following example demonstrates a
typical workflow for a 2‑hour classification horizon:

1. **Download candles** – adjust the symbol and date range in
   `src/crypto_analyzer/data/binance_import.py` and run:

   ```bash
   python -m crypto_analyzer.data.binance_import
   ```

2. **(Optional) Merge additional on-chain metrics** – mempool stats are
   fetched automatically, but other metrics (e.g. exchange flows) can be
   retrieved via `api/onchain.py` and stored with the `onch_` prefix in the
   SQLite `prices` table.

3. **Feature engineering** – export engineered features to disk:

   ```bash
   python scripts/make_features.py --output data/features.parquet
   ```

4. **Model training** – train the gradient boosted classifier for the ±0.5 %
   target:

   ```bash
python scripts/train.py --features data/features.parquet --model-path artifacts/meta_model.joblib
```

Walk-forward cross-validation with isotonic calibration:

```bash
python scripts/train.py --features data/features.parquet --cv purged-wf --embargo_min 360 --calibration isotonic
```

5. **Backtest predictions** – evaluate the resulting forecasts on a hold-out
   set or historical predictions:

   ```bash
   python scripts/backtest.py data/predictions.csv --equity-output reports/equity.csv
   ```

6. **Inspect results** – review the generated CSV files and graphs in
   `reports/` (or your chosen output directory) to understand model performance
   and feature behaviour.

---

## Customization

* Add or change features in `src/crypto_analyzer/features/engineering.py`.
* Adjust or create new rules in `src/crypto_analyzer/labeling/rules.py`.
* Tune ML models in `src/crypto_analyzer/models/train.py`.
* Combine rule and ML signals in `src/crypto_analyzer/models/predictor.py`.

---

## Requirements

* Python 3.13
* [See `requirements.txt`](./requirements.txt) – all runtime dependencies are
  version pinned for reproducibility.

## Determinism & Repro

Training routines default to the seed defined in the configuration
(``models.random_seed``).  Every pipeline execution records the effective seed,
configuration snapshot and metrics in ``outputs/run_id=.../metadata.json`` and
``config_snapshot.yaml`` to ensure runs are repeatable.

## Troubleshooting

XGBoost and PyTorch automatically fall back to CPU if no compatible CUDA wheel
is available.

## Testing & Quality Checks

The repository ships with both unit tests and static analysis helpers to keep
pipelines reproducible.  After setting up the virtual environment, run the test
suite and linters from the project root:

```bash
pytest
make lint  # runs the default quality gates defined in the Makefile
```

Many CI checks depend on cached feature files.  When the required fixtures are
missing, regenerate them via `scripts/make_features.py` using the configuration
outlined in the _Quick Start_ section above.

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
