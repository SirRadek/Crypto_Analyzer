# Crypto Analyzer

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

## Requirements

* Python 3.13
* [See `requirements.txt`](./requirements.txt)
