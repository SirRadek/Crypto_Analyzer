# Crypto Analyzer

A modular Python project for **cryptocurrency price analysis and prediction** using both rule-based strategies and machine learning.

## Features

- Imports OHLCV data for any crypto pair (default: BTC/USDT) directly from Binance to SQLite.
- Calculates technical indicators (SMA, EMA, RSI, etc.).
- Feature engineering for ML models.
- Supports both rule-based and machine learning (RandomForest) predictions.
- Trains a single meta-level RandomForest for classification and regression.
- Combines rule-based and meta-model signals for the final trading decision.
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

### 4. Run analysis and prediction

```bash
python main.py
```

By default, this trains a RandomForest model and outputs the latest signals.

---

## Customization

* Add or change features in `analysis/feature_engineering.py`.
* Adjust or create new rules in `analysis/rules.py`.
* Tune ML model or use another classifier in `ml/train.py`.
* Combine rule and ML signals as you wish in `prediction/predictor.py`.

---

## Meta-only Random Forest

The project now uses a single meta-level RandomForest instead of large ensembles.
Utilities in `ml.meta` handle training, calibration, and batched inference.

### GPU training

```bash
python - <<'PY'
from ml.train import train_model
from ml.train_regressor import train_regressor
# X, y = ... load your training data ...
train_model(X, y, use_gpu=True)
train_regressor(X, y, use_gpu=True)
PY
```

Training falls back to the CPU path when CUDA or `cuml` is unavailable.

### Multi-output regression

```bash
python - <<'PY'
from ml.meta import fit_meta_regressor, predict_meta
# X, Y = ... features and multi-horizon targets ...
fit_meta_regressor(X, Y, FEATURE_COLUMNS, multi_output=True)
preds = predict_meta(X, FEATURE_COLUMNS, 'ml/meta_model_reg.joblib', multi_output=True)
PY
```

### Quantile prediction intervals

```bash
python - <<'PY'
from ml.meta import predict_meta
# df = ... features for inference ...
preds, intervals = predict_meta(
    df,
    FEATURE_COLUMNS,
    'ml/meta_model_reg.joblib',
    return_pi=True,
    quantiles=(0.05, 0.95),
)
PY
```

### Benchmarks

Synthetic 5k-row datasets were used to gauge baseline performance:

```bash
python - <<'PY'
import time, resource, pandas as pd
from sklearn.datasets import make_classification, make_regression
from ml.meta import fit_meta_classifier, fit_meta_regressor

n = 5000
Xc, yc = make_classification(n_samples=n, n_features=20, random_state=0)
start = time.perf_counter();
_, f1 = fit_meta_classifier(pd.DataFrame(Xc), pd.Series(yc), range(20),
    n_splits=3, gap=1, n_estimators=50, model_path='ml/tmp_cls.joblib',
    feature_list_path='ml/tmp_feat.json', version_path='ml/tmp_ver.json',
    threshold_path='ml/tmp_thr.json');
cls_time = time.perf_counter() - start
cls_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

Xr, yr = make_regression(n_samples=n, n_features=20, random_state=0)
start = time.perf_counter();
_, mae = fit_meta_regressor(pd.DataFrame(Xr), pd.Series(yr), range(20),
    n_splits=3, gap=1, n_estimators=50, model_path='ml/tmp_reg.joblib',
    feature_list_path='ml/tmp_feat.json', version_path='ml/tmp_ver.json');
reg_time = time.perf_counter() - start
reg_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - cls_mem

print(f"CLASSIFIER time={cls_time:.2f}s memory={cls_mem}KB F1={f1:.3f}")
print(f"REGRESSOR time={reg_time:.2f}s memory={reg_mem}KB MAE={mae:.3f}")
PY
```

| model      | latency (s) | peak RAM (MB) | score    |
|------------|-------------|---------------|----------|
| classifier | 4.24        | 167           | F1 = 0.973 |
| regressor  | 3.67        | 43            | MAE = 40.33 |

---

## Requirements

* Python 3.13
* [See `requirements.txt`](./requirements.txt)

## Performance notes

* **Lazy imports** keep the cold-start time low by deferring heavy
  dependencies (pandas, scikit-learn, etc.) until the functions that need
  them are called.
* **GPU fallback**: GPU libraries (`cuml`, `cudf`) are imported only when
  `use_gpu=True` and CUDA is available. Otherwise the code logs a warning
  and falls back to the CPU implementation without importing those
  modules.
* **Thread control**: inference can limit BLAS and OpenMP threads via
  ``threadpoolctl`` for more predictable performance.
* **Batching**: large inference jobs should use batch sizes of around
  200k rows for best throughput.
* To measure import performance run ``tools/check_import_time.sh``. The
  JSON flame graph is written to ``ml/importtime.json`` and uploaded as a
  CI artifact.
