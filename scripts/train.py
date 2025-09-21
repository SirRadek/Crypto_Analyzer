#!/usr/bin/env python
"""Train entry point for the gradient boosted meta-classifier."""
from __future__ import annotations

#!/usr/bin/env python
"""Training entry-point with optional calibration and conformal outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics

from crypto_analyzer.data.db_connector import get_price_data
from crypto_analyzer.features.engineering import (
    FEATURE_COLUMNS,
    create_features,
    get_feature_columns,
)
from crypto_analyzer.features.engineering import make_targets as make_default_targets
from crypto_analyzer.eval.cv import purged_walkforward_splits
from crypto_analyzer.models.calibration import (
    brier_score,
    fit_isotonic,
    fit_platt,
    log_loss as log_loss_metric,
    plot_reliability,
    reliability_curve,
)
from crypto_analyzer.models.conformal import conformal_interval
from crypto_analyzer.utils.config import CONFIG, FeatureSettings, override_feature_settings


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _prepare_settings(args: argparse.Namespace) -> FeatureSettings:
    settings = CONFIG.features
    overrides: dict[str, Any] = {}
    if args.include_onchain is not None:
        overrides["include_onchain"] = args.include_onchain
    if args.include_orderbook is not None:
        overrides["include_orderbook"] = args.include_orderbook
    if args.include_derivatives is not None:
        overrides["include_derivatives"] = args.include_derivatives
    if overrides:
        settings = override_feature_settings(settings, **overrides)

    if args.forward_fill_limit is not None or args.fillna_value is not None:
        settings = FeatureSettings(
            include_onchain=settings.include_onchain,
            include_orderbook=settings.include_orderbook,
            include_derivatives=settings.include_derivatives,
            forward_fill_limit=(
                args.forward_fill_limit
                if args.forward_fill_limit is not None
                else settings.forward_fill_limit
            ),
            fillna_value=(
                args.fillna_value if args.fillna_value is not None else settings.fillna_value
            ),
        )
    return settings


def _load_features(args: argparse.Namespace, settings: FeatureSettings) -> pd.DataFrame:
    if args.features is not None:
        df = _read_table(args.features)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df

    raw = get_price_data(args.symbol, db_path=args.db_path)
    return create_features(raw, settings=settings)


def _ensure_label(df: pd.DataFrame, horizon: int, label: str) -> tuple[pd.DataFrame, str]:
    if label in df.columns:
        return df, label

    labeled = make_default_targets(df, horizon=horizon)
    target_col = f"cls_sign_{horizon}m"
    if target_col not in labeled.columns:
        raise ValueError(
            "Unable to infer training targets. Provide a --label column or ensure the input data "
            "contains OHLC prices so targets can be generated."
        )
    return labeled, target_col


def _compute_realized_volatility(df: pd.DataFrame) -> pd.Series:
    """Estimate realized volatility using available price features."""

    if "vol_realized_1h" in df.columns:
        realized = pd.to_numeric(df["vol_realized_1h"], errors="coerce")
    else:
        if "close" not in df.columns:
            raise KeyError(
                "close price column missing; unable to compute realized volatility"
            )
        close_prices = pd.to_numeric(df["close"], errors="coerce")
        log_returns = np.log(close_prices).diff()
        realized = log_returns.pow(2).rolling(window=60, min_periods=10).sum().pow(0.5)

    return realized.astype(float)


def _chronological_split(
    X: pd.DataFrame, y: pd.Series, timestamps: pd.Series, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must lie in (0, 1)")
    split_idx = max(1, int(round(len(X) * (1 - test_size))))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    ts_train = timestamps.iloc[:split_idx]
    ts_test = timestamps.iloc[split_idx:]
    return X_train, X_test, y_train, y_test, ts_train, ts_test


def _split_calibration(
    X: pd.DataFrame, y: pd.Series, timestamps: pd.Series, fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    if len(X) < 2 or fraction <= 0:
        return X, None, y, None, timestamps, None
    cal_size = max(1, int(round(len(X) * fraction)))
    if cal_size >= len(X):
        cal_size = len(X) - 1
    if cal_size <= 0:
        return X, None, y, None, timestamps, None
    X_cal = X.iloc[-cal_size:]
    y_cal = y.iloc[-cal_size:]
    ts_cal = timestamps.iloc[-cal_size:]
    X_train = X.iloc[:-cal_size]
    y_train = y.iloc[:-cal_size]
    ts_train = timestamps.iloc[:-cal_size]
    return X_train, X_cal, y_train, y_cal, ts_train, ts_cal


def _export_metrics_by_volatility(
    y_true: pd.Series,
    probs_raw: np.ndarray,
    realized_vol: pd.Series,
    metrics_path: Path,
    reliability_path: Path,
    *,
    calibrated_probs: np.ndarray | None = None,
    calibration_label: str | None = None,
    n_quantiles: int = 5,
) -> None:
    """Persist evaluation metrics grouped by realized volatility regimes."""

    columns = [
        "vol_quantile",
        "count",
        "vol_min",
        "vol_max",
        "brier_raw",
        "auc_raw",
        "hit_rate_raw",
        "brier_calibrated",
        "auc_calibrated",
        "hit_rate_calibrated",
    ]

    data = pd.DataFrame(
        {
            "target": y_true.to_numpy(dtype=int),
            "prob_raw": np.asarray(probs_raw, dtype=float),
            "volatility": realized_vol.reindex(y_true.index).astype(float),
        },
        index=y_true.index,
    )

    if calibrated_probs is not None:
        data["prob_calibrated"] = np.asarray(calibrated_probs, dtype=float)

    data = data.dropna(subset=["volatility"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if data.empty:
        pd.DataFrame(columns=columns).to_csv(metrics_path, index=False)
        reliability_path.parent.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt  # deferred import for hygiene tests

        plt.figure(figsize=(6, 6))
        plt.text(0.5, 0.5, "No volatility data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(reliability_path)
        plt.close()
        return

    quantiles = pd.qcut(
        data["volatility"],
        q=n_quantiles,
        labels=False,
        retbins=False,
        duplicates="drop",
    )
    data = data.assign(vol_bin=quantiles)

    metrics_rows: list[dict[str, Any]] = []
    reliability_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for bin_id in sorted(data["vol_bin"].dropna().unique()):
        subset = data[data["vol_bin"] == bin_id]
        if subset.empty:
            continue

        y_bin = subset["target"].to_numpy(dtype=int)
        proba_raw = subset["prob_raw"].to_numpy(dtype=float)
        labels_raw = (proba_raw >= 0.5).astype(int)

        hit_rate_raw = float(np.mean(labels_raw == y_bin)) if len(y_bin) else float("nan")
        brier_raw = brier_score(y_bin, proba_raw)
        try:
            auc_raw = float(metrics.roc_auc_score(y_bin, proba_raw))
        except ValueError:
            auc_raw = float("nan")

        row: dict[str, Any] = {
            "vol_quantile": int(bin_id) + 1,
            "count": int(len(subset)),
            "vol_min": float(subset["volatility"].min()),
            "vol_max": float(subset["volatility"].max()),
            "brier_raw": float(brier_raw),
            "auc_raw": auc_raw,
            "hit_rate_raw": hit_rate_raw,
            "brier_calibrated": float("nan"),
            "auc_calibrated": float("nan"),
            "hit_rate_calibrated": float("nan"),
        }

        if calibrated_probs is not None and calibration_label:
            proba_cal = subset["prob_calibrated"].to_numpy(dtype=float)
            labels_cal = (proba_cal >= 0.5).astype(int)
            row["brier_calibrated"] = float(brier_score(y_bin, proba_cal))
            try:
                row["auc_calibrated"] = float(metrics.roc_auc_score(y_bin, proba_cal))
            except ValueError:
                row["auc_calibrated"] = float("nan")
            row["hit_rate_calibrated"] = (
                float(np.mean(labels_cal == y_bin)) if len(y_bin) else float("nan")
            )

        _, obs, exp, _, _ = reliability_curve(y_bin, proba_raw, n_bins=10)
        reliability_series[f"Q{int(bin_id) + 1}"] = (exp, obs)

        metrics_rows.append(row)

    if not metrics_rows:
        pd.DataFrame(columns=columns).to_csv(metrics_path, index=False)
        reliability_path.parent.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt  # deferred import for hygiene tests

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        plt.title("Reliability by realized volatility quantile")
        plt.tight_layout()
        plt.savefig(reliability_path)
        plt.close()
        return

    metrics_df = pd.DataFrame(metrics_rows).sort_values("vol_quantile")
    metrics_df.to_csv(metrics_path, index=False)

    reliability_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt  # deferred import for hygiene tests

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for label, (exp, obs) in reliability_series.items():
        mask = ~np.isnan(exp) & ~np.isnan(obs)
        if not np.any(mask):
            continue
        plt.plot(exp[mask], obs[mask], marker="o", label=label)

    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability by realized volatility quantile")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(reliability_path)
    plt.close()


def _fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    use_gpu: bool,
    random_state: int,
) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="gpu_hist" if use_gpu else "hist",
        predictor="gpu_predictor" if use_gpu else "cpu_predictor",
        n_jobs=-1,
        eval_metric="logloss",
        random_state=random_state,
        use_label_encoder=False,
    )
    model = xgb.XGBClassifier(**params)
    try:
        model.fit(X_train, y_train)
    except xgb.core.XGBoostError:
        if use_gpu:
            model.set_params(tree_method="hist", predictor="cpu_predictor")
            model.fit(X_train, y_train)
        else:
            raise
    return model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Crypto Analyzer meta-model")
    parser.add_argument(
        "--features",
        type=Path,
        help="Optional engineered feature table (CSV/Parquet). If omitted data is pulled from the DB.",
    )
    parser.add_argument(
        "--symbol",
        default=CONFIG.symbol,
        help="Trading symbol used when sourcing data from the database.",
    )
    parser.add_argument(
        "--db-path",
        default=CONFIG.db_path,
        help="SQLite database file to read raw price data from.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=120,
        help="Target horizon in minutes for label generation when not provided in the dataset.",
    )
    parser.add_argument(
        "--label",
        help="Existing label column to use. Defaults to cls_sign_<horizon>m.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/meta_model.joblib"),
        help="Output path for the trained model.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("artifacts/oob_metrics.json"),
        help="Where to store evaluation metrics gathered during training.",
    )
    parser.add_argument(
        "--split",
        choices=("holdout", "walkforward"),
        default="holdout",
        help="Evaluation split used during training.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction used for the validation split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for model training.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration even when available.",
    )
    parser.add_argument(
        "--include-onchain",
        dest="include_onchain",
        action="store_true",
        help="Force-enable on-chain features regardless of config defaults.",
    )
    parser.add_argument(
        "--exclude-onchain",
        dest="include_onchain",
        action="store_false",
        help="Force-disable on-chain features regardless of config defaults.",
    )
    parser.add_argument(
        "--include-orderbook",
        dest="include_orderbook",
        action="store_true",
        help="Force-enable orderbook features regardless of config defaults.",
    )
    parser.add_argument(
        "--exclude-orderbook",
        dest="include_orderbook",
        action="store_false",
        help="Force-disable orderbook features regardless of config defaults.",
    )
    parser.add_argument(
        "--include-derivatives",
        dest="include_derivatives",
        action="store_true",
        help="Force-enable derivative features regardless of config defaults.",
    )
    parser.add_argument(
        "--exclude-derivatives",
        dest="include_derivatives",
        action="store_false",
        help="Force-disable derivative features regardless of config defaults.",
    )
    parser.add_argument(
        "--forward-fill-limit",
        type=int,
        help="Override forward-fill window for NaN handling.",
    )
    parser.add_argument(
        "--fillna-value",
        type=float,
        help="Override fallback value used when forward fill runs out.",
    )
    parser.add_argument(
        "--wfs-train-days",
        type=int,
        help="Training window size in days for walk-forward evaluation.",
    )
    parser.add_argument(
        "--wfs-test-days",
        type=int,
        help="Test window size in days for walk-forward evaluation.",
    )
    parser.add_argument(
        "--wfs-step-days",
        type=int,
        help="Step size in days when rolling the walk-forward window.",
    )
    parser.add_argument(
        "--wfs-min-train-days",
        type=int,
        help="Minimal amount of training data required for walk-forward evaluation.",
    )
    parser.add_argument(
        "--cv",
        choices=("purged-wf",),
        help="Optional cross-validation strategy to evaluate during training.",
    )
    parser.add_argument(
        "--embargo_min",
        type=int,
        default=CONFIG.cv.embargo_min,
        help=(
            "Embargo window in minutes used for purged walk-forward cross-validation."
            " Defaults to 360 minutes."
        ),
    )
    parser.add_argument(
        "--calibration",
        choices=("none", "isotonic", "platt"),
        default=CONFIG.calibration.method,
        help="Probability calibration method applied on the validation split.",
    )
    parser.add_argument(
        "--conformal_alpha",
        type=float,
        default=0.1,
        help="Enable conformal prediction intervals at the specified miscoverage level.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier used when storing artefacts.",
    )
    parser.set_defaults(include_onchain=None, include_orderbook=None, include_derivatives=None)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.split != "holdout":
        raise NotImplementedError("Only holdout split is supported in the CLI")

    settings = _prepare_settings(args)
    df = _load_features(args, settings)

    label = args.label or f"cls_sign_{args.horizon}m"
    df, label_col = _ensure_label(df, args.horizon, label)
    df = df.dropna(subset=[label_col]).sort_values("timestamp")

    feature_cols = get_feature_columns(settings)
    if not feature_cols:
        feature_cols = FEATURE_COLUMNS

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(
            "Feature columns missing from dataset: " + ", ".join(sorted(missing))
        )

    X = df[feature_cols].astype(np.float32)
    y = df[label_col].astype(int)
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    realized_volatility = _compute_realized_volatility(df)

    run_id = args.run_id or pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / f"run_id={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.conformal_alpha is not None and not (0.0 < float(args.conformal_alpha) < 1.0):
        raise ValueError("conformal_alpha must lie in (0, 1)")

    if args.cv == "purged-wf":
        purged_walkforward_splits(
            pd.DatetimeIndex(timestamps),
            n_splits=CONFIG.cv.n_splits,
            embargo_min=args.embargo_min,
            run_id=run_id,
            reports_dir=reports_dir,
        )

    X_train_full, X_test, y_train_full, y_test, ts_train_full, ts_test = _chronological_split(
        X, y, timestamps, args.test_size
    )
    X_train, X_cal, y_train, y_cal, ts_train, ts_cal = _split_calibration(
        X_train_full, y_train_full, ts_train_full
    )

    model_path_default = parser.get_default("model_path")
    model_output = args.model_path if args.model_path != model_path_default else run_dir / "model.joblib"
    model_output.parent.mkdir(parents=True, exist_ok=True)

    model = _fit_model(X_train, y_train, use_gpu=not args.no_gpu, random_state=args.random_state)
    joblib.dump(model, model_output)

    proba_test = model.predict_proba(X_test)[:, 1]
    labels_test = (proba_test >= 0.5).astype(int)

    metrics_raw = {
        "accuracy": float(metrics.accuracy_score(y_test, labels_test)),
        "f1": float(metrics.f1_score(y_test, labels_test)),
        "precision": float(metrics.precision_score(y_test, labels_test)),
        "recall": float(metrics.recall_score(y_test, labels_test)),
        "roc_auc": float(metrics.roc_auc_score(y_test, proba_test)),
        "brier": brier_score(y_test, proba_test),
        "log_loss": log_loss_metric(y_test, proba_test),
    }

    reliability_data = {"raw": metrics_raw}
    prob_series = {"Raw": proba_test}

    calibrated_metrics = None
    calibrated_probs = None
    calibration_method = args.calibration
    if calibration_method != "none" and X_cal is not None and len(X_cal) > 0:
        cal_probs = model.predict_proba(X_cal)[:, 1]
        if calibration_method == "isotonic":
            calibrator = fit_isotonic(cal_probs, y_cal)
        else:
            calibrator = fit_platt(cal_probs, y_cal)
        calibrated_probs = calibrator.predict(proba_test)
        prob_series[f"Calibrated ({calibration_method})"] = calibrated_probs
        calibrated_labels = (calibrated_probs >= 0.5).astype(int)
        calibrated_metrics = {
            "accuracy": float(metrics.accuracy_score(y_test, calibrated_labels)),
            "f1": float(metrics.f1_score(y_test, calibrated_labels)),
            "precision": float(metrics.precision_score(y_test, calibrated_labels)),
            "recall": float(metrics.recall_score(y_test, calibrated_labels)),
            "roc_auc": float(metrics.roc_auc_score(y_test, calibrated_probs)),
            "brier": brier_score(y_test, calibrated_probs),
            "log_loss": log_loss_metric(y_test, calibrated_probs),
        }
        reliability_data[f"calibrated_{calibration_method}"] = calibrated_metrics

    reliability_path = reports_dir / f"reliability_{run_id}.png"
    plot_reliability(y_test, prob_series, reliability_path)

    metrics_by_vol_path = reports_dir / f"metrics_by_vol_{run_id}.csv"
    reliability_by_vol_path = reports_dir / f"reliability_by_vol_{run_id}.png"
    _export_metrics_by_volatility(
        y_test,
        proba_test,
        realized_volatility.reindex(y_test.index),
        metrics_by_vol_path,
        reliability_by_vol_path,
        calibrated_probs=calibrated_probs,
        calibration_label=(
            f"Calibrated ({calibration_method})" if calibrated_probs is not None else None
        ),
    )

    metrics_report = {
        "run_id": run_id,
        "raw": metrics_raw,
        "calibration": calibration_method,
        "calibrated": calibrated_metrics,
    }

    metrics_path = reports_dir / f"metrics_{run_id}.json"
    metrics_payload = json.dumps(metrics_report, indent=2)
    metrics_path.write_text(metrics_payload, encoding="utf-8")
    (run_dir / "metrics.json").write_text(metrics_payload, encoding="utf-8")

    reliability_copy = run_dir / "reliability.png"
    if not reliability_copy.exists():
        reliability_copy.write_bytes(reliability_path.read_bytes())

    if args.conformal_alpha is not None and X_cal is not None and len(X_cal) > 0:
        cal_probs = model.predict_proba(X_cal)[:, 1]
        conformal = conformal_interval(
            y_cal,
            cal_probs,
            (y_test, proba_test),
            float(args.conformal_alpha),
        )
        conformal_path = reports_dir / f"conformal_{run_id}.json"
        conformal_json = json.dumps(conformal, indent=2)
        conformal_path.write_text(conformal_json, encoding="utf-8")
        (run_dir / "conformal.json").write_text(conformal_json, encoding="utf-8")

    config_dump = {
        "config": CONFIG.config_path.as_posix() if CONFIG.config_path else None,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "features": feature_cols,
    }
    (run_dir / "config_dump.json").write_text(json.dumps(config_dump, indent=2), encoding="utf-8")

    print(f"Model trained and stored at {model_output}")
    return model_output


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
