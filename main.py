import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from analysis.compare_predictions import backfill_actuals_and_errors
from analysis.feature_engineering import FEATURE_COLUMNS, assign_feature_groups, create_features
from db.db_connector import get_price_data
from db.predictions_store import create_predictions_table, save_predictions
from deleting_SQL import delete_old_records
from ml.model_utils import match_model_features
from ml.train import load_model, train_model
from ml.xgb_price import clip_inside, to_price
from utils.config import CONFIG
from utils.helpers import ensure_dir_exists, get_logger, set_cpu_limit
from utils.progress import p, step, timed

logger = get_logger(__name__)

SYMBOL = CONFIG.symbol
DB_PATH = CONFIG.db_path
FEATURE_COLS = FEATURE_COLUMNS
TABLE_PRED = CONFIG.table_pred
INTERVAL = CONFIG.interval
# number of future five-minute steps to predict, e.g. 24 -> next 2 hours
FORWARD_STEPS = CONFIG.forward_steps
CPU_LIMIT = CONFIG.cpu_limit
REPEAT_COUNT = CONFIG.repeat_count
INTERVAL_TO_MIN = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": 1440,
}
PRAGUE_TZ = ZoneInfo("Europe/Prague")
FEATURES_VERSION = "ext_v1"

CLS_MODEL_COUNT = 1
REG_MODEL_COUNT = 1
CLS_ACC_PATH = Path("ml/backtest_acc_cls.json")
REG_ACC_PATH = Path("ml/backtest_acc_reg.json")


def _created_at_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def _delete_future_predictions(db_path: str, symbol: str, from_ms: int, table_name: str) -> int:
    """Smaž predikce s NULL y_true_hat od času >= from_ms (včetně) pro daný symbol."""
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""DELETE FROM {table_name}
                WHERE symbol = ?
                  AND y_true_hat IS NULL""",
            (symbol, int(from_ms)),
        )
        deleted = cur.rowcount if cur.rowcount not in (None, -1) else 0
        conn.commit()
        cur.execute("PRAGMA optimize;")
        conn.commit()
    return int(deleted)

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            f"DELETE FROM {table_name} WHERE symbol = ? AND y_true IS NULL",
            (symbol,),
        )
        deleted = cur.rowcount or 0
        cur.execute("PRAGMA optimize;")
        conn.commit()
    return int(deleted)


def list_model_paths(pattern: str, count: int) -> tuple[list[str], list[int]]:
    """Return existing model paths following a numbered pattern."""

    paths: list[str] = []
    indices: list[int] = []
    for i in range(1, count + 1):
        path = pattern.format(i=i)
        if Path(path).exists():
            paths.append(path)
            indices.append(i)
        else:
            p(f"Missing model {path}")
    return paths, indices


def load_accuracy_weights(path: Path, indices: Iterable[int]) -> list[float]:
    """Load accuracy weights for the given model ``indices``."""

    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    return [float(data.get(str(i), 1.0)) for i in indices]


def prepare_targets(df: pd.DataFrame, forward_steps: int = 1) -> pd.DataFrame:
    """Create classification and regression targets for ``forward_steps`` ahead."""

    df = df.copy(deep=False)
    df["target_cls"] = (df["close"].shift(-forward_steps) > df["close"]).astype(int)
    df["target_reg"] = df["close"].shift(-forward_steps)
    df = df.dropna(subset=["target_cls", "target_reg"])
    return df


def run_pipeline(
    task: str,
    horizon: int,
    use_onchain: bool,
    txn_cost_bps: float,
    split_params: dict[str, Any],
    gpu: bool,
    out_dir: str,
) -> None:
    """Minimal end-to-end training pipeline.

    Parameters
    ----------
    task:
        "clf" for classification or "reg" for regression.
    horizon:
        Prediction horizon in minutes.
    use_onchain:
        Whether to include ``onch_`` features.
    txn_cost_bps:
        Transaction cost in basis points.
    split_params:
        Parameters forwarded to :func:`sklearn.model_selection.train_test_split`.
    gpu:
        If ``True`` use GPU-accelerated XGBoost when available.
    out_dir:
        Directory where artefacts are written.
    """

    ensure_dir_exists(out_dir)

    df = get_price_data(SYMBOL, db_path=DB_PATH)
    if use_onchain:
        from api.onchain import fetch_mempool_5m

        start = pd.to_datetime(df["timestamp"].min(), utc=True)
        end = pd.to_datetime(df["timestamp"].max(), utc=True)
        try:
            onch = fetch_mempool_5m(start, end)
            onch.index = onch.index.tz_localize(None)
            df = df.merge(onch, left_on="timestamp", right_index=True, how="left")
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("mempool fetch failed: %s", exc)
    df = create_features(df)
    if not use_onchain:
        df = df.loc[:, ~df.columns.str.startswith("onch_")]

    horizon_steps = horizon // 5
    if task == "clf":
        df["target"] = (df["close"].shift(-horizon_steps) > df["close"]).astype(np.int32)
    else:
        df["target"] = (df["close"].shift(-horizon_steps) - df["close"]).astype(np.float32)
    df = df.dropna(subset=["target"])

    base_cols = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
        "number_of_trades",
    }
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    drop_cols = base_cols.union(delta_cols).union({"target"})
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].astype(np.float32)
    y = df["target"]

    from sklearn.model_selection import train_test_split

    split_cfg = {"test_size": 0.2, "shuffle": False}
    split_cfg.update(split_params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_cfg)

    import xgboost as xgb

    tree_method = "gpu_hist" if gpu else "hist"
    if task == "clf":
        model = xgb.XGBClassifier(
            n_estimators=50,
            tree_method=tree_method,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=50,
            tree_method=tree_method,
            random_state=42,
        )
    model.fit(X_train, y_train)

    if task == "clf":
        preds = model.predict_proba(X_test)[:, 1]
    else:
        preds = model.predict(X_test)

    from sklearn.metrics import accuracy_score, mean_squared_error

    if task == "clf":
        metric_name = "accuracy"
        metric_val = accuracy_score(y_test, (preds > 0.5).astype(int))
    else:
        metric_name = "mse"
        metric_val = mean_squared_error(y_test, preds)

    metrics = {metric_name: float(metric_val)}
    with open(Path(out_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    pred_df = pd.DataFrame(
        {
            "timestamp": df.loc[y_test.index, "timestamp"].reset_index(drop=True),
            "y_true": y_test.reset_index(drop=True),
            "y_pred": preds,
        }
    )
    pred_df.to_csv(Path(out_dir) / f"predictions_{task}.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        import shap
        from sklearn.inspection import permutation_importance

        fig, ax = plt.subplots()
        ax.plot(pred_df["y_true"].to_numpy(), label="true")
        ax.plot(pred_df["y_pred"].to_numpy(), label="pred")
        ax.legend()
        fig.savefig(Path(out_dir) / f"pred_vs_actual_{task}.png")
        plt.close(fig)

        # permutation importance -------------------------------------------------
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        perm_df = pd.DataFrame(
            {"feature": feature_cols, "importance": perm.importances_mean}
        ).sort_values("importance", ascending=False)
        perm_df.to_csv(Path(out_dir) / f"perm_importance_{task}.csv", index=False)
        fig, ax = plt.subplots()
        perm_df.plot.barh(x="feature", y="importance", ax=ax)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"perm_importance_{task}.png")
        plt.close(fig)

        # SHAP values -----------------------------------------------------------
        explainer = shap.Explainer(model, X_train)
        shap_vals = explainer(X_test).values
        shap_df = pd.DataFrame(shap_vals, columns=feature_cols)
        shap_mean = (
            shap_df.abs()
            .mean()
            .reset_index()
            .rename(columns={"index": "feature", 0: "mean_abs_shap"})
        )
        shap_mean.sort_values("mean_abs_shap", ascending=False).to_csv(
            Path(out_dir) / f"shap_values_{task}.csv", index=False
        )
        shap.summary_plot(shap_vals, X_test, show=False)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"shap_summary_{task}.png")
        plt.close()

        # Group SHAP & permutation ----------------------------------------------
        groups = assign_feature_groups(feature_cols)
        shap_mean["group"] = shap_mean["feature"].map(groups)
        group_shap = shap_mean.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
        group_shap.to_csv(Path(out_dir) / f"shap_group_{task}.csv")
        fig, ax = plt.subplots()
        group_shap.plot.barh(ax=ax)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"shap_group_{task}.png")
        plt.close(fig)

        perm_df["group"] = perm_df["feature"].map(groups)
        group_perm = perm_df.groupby("group")["importance"].sum().sort_values(ascending=False)
        group_perm.to_csv(Path(out_dir) / f"perm_group_{task}.csv")
        fig, ax = plt.subplots()
        group_perm.plot.barh(ax=ax)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"perm_group_{task}.png")
        plt.close(fig)

        # simple drift ----------------------------------------------------------
        train_mean = pd.DataFrame(X_train.mean(), columns=["train_mean"])
        test_mean = pd.DataFrame(X_test.mean(), columns=["test_mean"])
        drift_df = train_mean.join(test_mean)
        drift_df["mean_diff"] = drift_df["test_mean"] - drift_df["train_mean"]
        drift_df.reset_index(inplace=True)
        drift_df.rename(columns={"index": "feature"}, inplace=True)
        drift_df.to_csv(Path(out_dir) / f"drift_{task}.csv", index=False)
        fig, ax = plt.subplots()
        drift_df.sort_values("mean_diff").plot.barh(x="feature", y="mean_diff", ax=ax)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"drift_{task}.png")
        plt.close(fig)

        drift_df["group"] = drift_df["feature"].map(groups)
        group_drift = drift_df.groupby("group")["mean_diff"].mean().sort_values(ascending=False)
        group_drift.to_csv(Path(out_dir) / f"drift_group_{task}.csv")
        fig, ax = plt.subplots()
        group_drift.plot.barh(ax=ax)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"drift_group_{task}.png")
        plt.close(fig)

    except Exception:
        pass

    model.save_model(Path(out_dir) / f"{task}_model.json")

    run_cfg = {
        "task": task,
        "horizon": horizon,
        "use_onchain": use_onchain,
        "txn_cost_bps": txn_cost_bps,
        "split_params": split_params,
        "gpu": gpu,
        "out_dir": out_dir,
        "random_state": 42,
    }
    with open(Path(out_dir) / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)


def predict_price(
    model_dir: str = "models/xgb_price", target_kind: str = "log"
) -> tuple[pd.Timestamp, float, float, float]:
    """Predict next price interval using pre-trained models."""

    df = get_price_data(SYMBOL, db_path=DB_PATH)
    df = create_features(df)
    last_row = df.iloc[[-1]]
    last_close = float(last_row["close"].iloc[0])
    X_last = last_row[FEATURE_COLS].astype("float32")
    reg = joblib.load(Path(model_dir) / "reg.joblib", mmap_mode="r")
    lowm = joblib.load(Path(model_dir) / "low.joblib", mmap_mode="r")
    highm = joblib.load(Path(model_dir) / "high.joblib", mmap_mode="r")
    delta = reg.predict(X_last)[0]
    low = lowm.predict(X_last)[0]
    high = highm.predict(X_last)[0]
    p_hat = to_price(last_close, delta, kind=target_kind)
    p_low = to_price(last_close, low, kind=target_kind)
    p_high = to_price(last_close, high, kind=target_kind)
    p_low, p_high = np.minimum(p_low, p_high), np.maximum(p_low, p_high)
    p_hat = clip_inside(p_hat, p_low, p_high)
    ts = last_row["timestamp"].iloc[0]
    return ts, float(p_low), float(p_hat), float(p_high)


def main(train: bool = True) -> None:
    from ml.train_regressor import load_regressor, train_regressor

    ensure_dir_exists("db/data")
    start = time.perf_counter()

    step(1, 8, "Import latest data")
    try:
        from db.btc_import import import_latest_data

        import_latest_data()
    except Exception as exc:
        p(f"btc_import failed: {exc}")

    step(2, 8, "Loading data from DB")
    with timed("Load"):
        start_ts = int((pd.Timestamp.utcnow() - pd.Timedelta(days=5 * 365)).timestamp() * 1000)
        df = get_price_data(SYMBOL, start_ts=start_ts, db_path=DB_PATH)
        p(f"  -> rows={len(df)}, cols={len(df.columns)}")

    step(3, 8, "Feature engineering")
    with timed("Features"):
        df = create_features(df)
        p(f"  -> rows after features={len(df)}")

    full_df = df.copy(deep=False)

    step(4, 8, "Backfill predictions & train/load models")
    create_predictions_table(DB_PATH, TABLE_PRED)
    backfill_actuals_and_errors(db_path=DB_PATH, table_pred=TABLE_PRED, symbol=SYMBOL)
    last_ts = full_df["timestamp"].max()

    cls_paths, cls_indices = list_model_paths("ml/meta_model_cls.joblib", CLS_MODEL_COUNT)
    reg_paths, reg_indices = list_model_paths("ml/meta_model_reg.joblib", REG_MODEL_COUNT)
    cls_weights = load_accuracy_weights(CLS_ACC_PATH, cls_indices)
    reg_weights = load_accuracy_weights(REG_ACC_PATH, reg_indices)

    # Sestavení trénovací tabulky přes více horizontů
    horizon_dfs = []
    for horizon in range(1, FORWARD_STEPS + 1):
        df_h = prepare_targets(full_df, forward_steps=horizon)
        df_h["horizon"] = horizon
        horizon_dfs.append(df_h)
    train_df = pd.concat(horizon_dfs, ignore_index=True)

    # Základní feature set
    base_feats = train_df[FEATURE_COLS].astype("float32")

    # Predikce bázových modelů → vstup do meta-modelu
    for path, idx, w in zip(cls_paths, cls_indices, cls_weights, strict=False):
        model = load_model(model_path=path)
        feats = match_model_features(base_feats, model).astype("float32")
        preds = model.predict_proba(feats)[:, 1] * w
        train_df[f"cls_pred_{idx}"] = preds
        del model
    for path, idx, w in zip(reg_paths, reg_indices, reg_weights, strict=False):
        model = load_regressor(model_path=path)
        feats = match_model_features(base_feats, model).astype("float32")
        preds = model.predict(feats) * w
        train_df[f"reg_pred_{idx}"] = preds
        del model

    cls_pred_cols = [f"cls_pred_{i}" for i in cls_indices]
    reg_pred_cols = [f"reg_pred_{i}" for i in reg_indices]
    feature_cols_meta = FEATURE_COLS + ["horizon"] + cls_pred_cols + reg_pred_cols

    X_all = train_df[feature_cols_meta].astype("float32")
    y_cls_all = train_df["target_cls"].astype("int32")
    y_reg_all = train_df["target_reg"].astype("float32")

    model_path_cls = "ml/meta_model_cls.joblib"
    model_path_reg = "ml/meta_model_reg.joblib"

    if train:
        with timed("Train meta-classifier"):
            # GPU preferováno, fallback řeší uvnitř train_model
            cls_model = train_model(X_all, y_cls_all, model_path=model_path_cls, use_gpu=True)
        with timed("Train meta-regressor"):
            reg_model = train_regressor(X_all, y_reg_all, model_path=model_path_reg, use_gpu=True)
    else:
        with timed("Load meta-classifier"):
            cls_model = load_model(model_path=model_path_cls)
        with timed("Load meta-regressor"):
            reg_model = load_regressor(model_path=model_path_reg)

    rows_to_save = []
    last_row = full_df.iloc[[-1]]
    pred_time = last_row["timestamp"].iloc[0]

    # Base model predictions for the latest row
    base_last = last_row[FEATURE_COLS].astype("float32")
    last_base_preds = {}
    for path, idx, w in zip(cls_paths, cls_indices, cls_weights, strict=False):
        model = load_model(model_path=path)
        feats = match_model_features(base_last, model).astype("float32")
        last_base_preds[f"cls_pred_{idx}"] = float(model.predict_proba(feats)[:, 1][0] * w)
        del model
    for path, idx, w in zip(reg_paths, reg_indices, reg_weights, strict=False):
        model = load_regressor(model_path=path)
        feats = match_model_features(base_last, model).astype("float32")
        last_base_preds[f"reg_pred_{idx}"] = float(model.predict(feats)[0] * w)
        del model

    step(5, 8, "Predict horizons")
    for horizon in range(1, FORWARD_STEPS + 1):
        with timed("Predict"):
            X_last = base_last.copy()
            X_last["horizon"] = horizon
            for name, val in last_base_preds.items():
                X_last[name] = val
            X_last = X_last[feature_cols_meta].astype("float32")
            prob_up = float(cls_model.predict_proba(X_last)[:, 1][0])
            reg_pred = float(reg_model.predict(X_last)[0])
            last_close = float(last_row["close"].iloc[0])
            combined_price = last_close + (reg_pred - last_close) * prob_up

        target_time = pred_time + pd.Timedelta(minutes=horizon * INTERVAL_TO_MIN[INTERVAL])

        row = (
            SYMBOL,
            INTERVAL,
            int(target_time.value // 1_000_000),
            float(combined_price),
            float(combined_price),
            float(combined_price),
        )
        rows_to_save.append(row)

    step(6, 8, "Save predictions")
    save_predictions(rows_to_save, DB_PATH, TABLE_PRED)
    p(f"Saved {len(rows_to_save)} predictions to DB.")

    total = time.perf_counter() - start
    p(f"Done in {total:.2f}s")

    p("Waiting 20 minutes before next run…")
    time.sleep(20 * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--task", choices=["clf", "reg"], default=None)
    parser.add_argument("--horizon", type=int, choices=[120, 240], default=120)
    parser.add_argument("--use_onchain", action="store_true")
    parser.add_argument("--txn_cost_bps", type=float, default=0.0)
    parser.add_argument("--split_params", type=str, default="{}")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    set_cpu_limit(CPU_LIMIT)
    if args.task is not None:
        run_pipeline(
            task=args.task,
            horizon=args.horizon,
            use_onchain=args.use_onchain,
            txn_cost_bps=args.txn_cost_bps,
            split_params=json.loads(args.split_params),
            gpu=args.gpu,
            out_dir=args.out_dir,
        )
    elif args.predict:
        ts, p_low, p_hat, p_high = predict_price()
        logger.info("%s,%.2f,%.2f,%.2f", ts, p_low, p_hat, p_high)
    else:
        for _ in range(REPEAT_COUNT):
            main()
