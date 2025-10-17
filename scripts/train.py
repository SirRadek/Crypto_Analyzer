#!/usr/bin/env python
"""Train entry point for the gradient boosted meta-classifier."""
from __future__ import annotations

#!/usr/bin/env python
"""Training entry-point with optional calibration and conformal outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics

import typer

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
from crypto_analyzer.utils.cli import run_cli
from crypto_analyzer.utils.config import CONFIG, FeatureSettings, override_feature_settings
from crypto_analyzer.utils.errors import DataValidationError, ModelError
from crypto_analyzer.utils.io import (
    build_path,
    initialize_run,
    save_json,
    save_model,
)
from crypto_analyzer.utils.logging import get_logger


app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = get_logger(__name__)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(f"Feature file '{path}' does not exist")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _prepare_settings(
    *,
    include_onchain: Optional[bool],
    include_orderbook: Optional[bool],
    include_derivatives: Optional[bool],
    include_sentiment: Optional[bool],
    forward_fill_limit: Optional[int],
    fillna_value: Optional[float],
) -> FeatureSettings:
    settings = CONFIG.features
    overrides: dict[str, Any] = {}
    if include_onchain is not None:
        overrides["include_onchain"] = include_onchain
    if include_orderbook is not None:
        overrides["include_orderbook"] = include_orderbook
    if include_derivatives is not None:
        overrides["include_derivatives"] = include_derivatives
    if include_sentiment is not None:
        overrides["include_sentiment"] = include_sentiment
    if overrides:
        settings = override_feature_settings(settings, **overrides)

    if forward_fill_limit is not None or fillna_value is not None:
        settings = FeatureSettings(
            include_onchain=settings.include_onchain,
            include_orderbook=settings.include_orderbook,
            include_derivatives=settings.include_derivatives,
            include_sentiment=settings.include_sentiment,
            forward_fill_limit=(
                forward_fill_limit
                if forward_fill_limit is not None
                else settings.forward_fill_limit
            ),
            fillna_value=(
                fillna_value if fillna_value is not None else settings.fillna_value
            ),
        )
    return settings


def _load_features(
    *,
    features: Optional[Path],
    symbol: str,
    db_path: Path,
    settings: FeatureSettings,
) -> pd.DataFrame:
    if features is not None:
        df = _read_table(features)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df

    raw = get_price_data(symbol, db_path=db_path)
    return create_features(raw, settings=settings)


def _ensure_label(df: pd.DataFrame, horizon: int, label: str) -> tuple[pd.DataFrame, str]:
    if label in df.columns:
        return df, label

    labeled = make_default_targets(df, horizon=horizon)
    target_col = f"cls_sign_{horizon}m"
    if target_col not in labeled.columns:
        raise DataValidationError(
            "Unable to infer training targets. Provide a --label column or ensure the input data "
            "contains OHLC prices so targets can be generated."
        )
    return labeled, target_col


def _compute_realized_volatility(df: pd.DataFrame) -> pd.Series:
    """Estimate realized volatility using available price features."""

    if "vol_realized_7d" in df.columns:
        realized = pd.to_numeric(df["vol_realized_7d"], errors="coerce")
    elif "vol_realized_1d" in df.columns:
        realized = pd.to_numeric(df["vol_realized_1d"], errors="coerce")
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
        raise DataValidationError("--test-size must lie in (0, 1)")
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


DEFAULT_MODEL_PATH = Path("artifacts/meta_model.joblib")


def _run_training(
    *,
    features: Optional[Path],
    symbol: str,
    db_path: Path,
    horizon: int,
    label: Optional[str],
    model_path: Path,
    split: str,
    test_size: float,
    random_state: int,
    use_gpu: bool,
    include_onchain: Optional[bool],
    include_orderbook: Optional[bool],
    include_derivatives: Optional[bool],
    include_sentiment: Optional[bool],
    forward_fill_limit: Optional[int],
    fillna_value: Optional[float],
    cv_strategy: Optional[str],
    embargo_min: int,
    calibration: str,
    conformal_alpha: Optional[float],
    run_id: Optional[str],
    dump_cv: bool,
    by_vol_bins: int,
    dry_run: bool,
) -> Path:
    if horizon <= 0:
        raise DataValidationError("--horizon must be positive")
    if split.lower() != "holdout":
        raise DataValidationError("Only holdout split is supported in the CLI")
    if cv_strategy not in (None, "purged-wf"):
        raise DataValidationError("Only 'purged-wf' cross-validation is supported")
    if conformal_alpha is not None and not (0.0 < float(conformal_alpha) < 1.0):
        raise DataValidationError("--conformal-alpha must lie in (0, 1)")
    if by_vol_bins <= 0:
        raise DataValidationError("--by-vol-bins must be a positive integer")

    settings = _prepare_settings(
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
        include_sentiment=include_sentiment,
        forward_fill_limit=forward_fill_limit,
        fillna_value=fillna_value,
    )
    df = _load_features(
        features=features,
        symbol=symbol,
        db_path=db_path,
        settings=settings,
    )

    label_name = label or f"cls_sign_{horizon}m"
    df, label_col = _ensure_label(df, horizon, label_name)
    df = df.dropna(subset=[label_col]).sort_values("timestamp")

    feature_cols = get_feature_columns(settings) or FEATURE_COLUMNS
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        joined = ", ".join(sorted(missing))
        raise DataValidationError(f"Feature columns missing from dataset: {joined}")

    X = df[feature_cols].astype(np.float32)
    y = df[label_col].astype(int)
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    realized_volatility = _compute_realized_volatility(df)

    run_id_value, run_dir, reports_dir = initialize_run(
        run_id,
        deterministic_torch=True,
    )
    logger.info(
        "Prepared training artefact directories",
        extra={
            "event": "initialised",
            "run_id": run_id_value,
            "horizon": horizon,
            "rows": int(len(df)),
        },
    )

    if cv_strategy == "purged-wf" and not dry_run:
        purged_walkforward_splits(
            pd.DatetimeIndex(timestamps),
            n_splits=CONFIG.cv.n_splits,
            embargo_min=embargo_min,
            run_id=run_id_value if dump_cv else None,
            reports_dir=reports_dir,
        )

    X_train_full, X_test, y_train_full, y_test, ts_train_full, ts_test = _chronological_split(
        X, y, timestamps, test_size
    )
    X_train, X_cal, y_train, y_cal, ts_train, ts_cal = _split_calibration(
        X_train_full, y_train_full, ts_train_full
    )

    default_model_target = run_dir / "model.joblib"
    model_output = model_path if model_path != DEFAULT_MODEL_PATH else default_model_target

    try:
        model = _fit_model(X_train, y_train, use_gpu=use_gpu, random_state=random_state)
    except xgb.core.XGBoostError as exc:  # pragma: no cover - defensive
        raise ModelError(f"Training failed: {exc}") from exc

    if not dry_run:
        run_model_path = save_model(model, run_id=run_id_value)
        if model_path != DEFAULT_MODEL_PATH:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            model_output = model_path
        else:
            model_output = run_model_path

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
    calibration_method = (calibration or "none").lower()
    if calibration_method not in {"none", "isotonic", "platt"}:
        raise DataValidationError("Unsupported calibration method")
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

    coverage_value: float | None = None
    ev_value: float | None = None

    if not dry_run:
        reliability_path = build_path(
            f"reliability_{run_id_value}.png", run_id=run_id_value, location="reports"
        )
        plot_reliability(y_test, prob_series, reliability_path)

        metrics_by_vol_path = build_path(
            f"metrics_by_vol_{run_id_value}.csv", run_id=run_id_value, location="reports"
        )
        reliability_by_vol_path = build_path(
            f"reliability_by_vol_{run_id_value}.png", run_id=run_id_value, location="reports"
        )

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
            n_quantiles=int(by_vol_bins),
        )

        if conformal_alpha is not None and X_cal is not None and len(X_cal) > 0:
            cal_probs = model.predict_proba(X_cal)[:, 1]
            conformal = conformal_interval(
                y_cal,
                cal_probs,
                (y_test, proba_test),
                float(conformal_alpha),
            )
            coverage_raw = conformal.get("test_coverage")
            if coverage_raw is not None:
                coverage_value = float(coverage_raw)
            conformal_path = save_json(
                conformal,
                f"conformal_{run_id_value}.json",
                run_id=run_id_value,
                location="reports",
            )
            save_json(conformal, "conformal.json", run_id=run_id_value)
            logger.info(
                "Stored conformal diagnostics",
                extra={"event": "conformal", "run_id": run_id_value, "path": str(conformal_path)},
            )

        metrics_payload = {
            "horizon": int(horizon),
            "brier_raw": float(metrics_raw["brier"]),
            "brier_cal": float(calibrated_metrics["brier"]) if calibrated_metrics else None,
            "auc": float(
                calibrated_metrics["roc_auc"]
                if calibrated_metrics and calibrated_metrics.get("roc_auc") is not None
                else metrics_raw["roc_auc"]
            ),
            "logloss": float(
                calibrated_metrics["log_loss"]
                if calibrated_metrics and calibrated_metrics.get("log_loss") is not None
                else metrics_raw["log_loss"]
            ),
            "coverage": coverage_value,
            "ev": ev_value,
        }
        metrics_report_path = save_json(
            metrics_payload,
            f"metrics_{run_id_value}.json",
            run_id=run_id_value,
            location="reports",
        )
        save_json(metrics_payload, "metrics.json", run_id=run_id_value)

        reliability_copy = build_path("reliability.png", run_id=run_id_value)
        if not reliability_copy.exists():
            reliability_copy.write_bytes(Path(reliability_path).read_bytes())

        args_snapshot = {
            "features": features,
            "symbol": symbol,
            "db_path": db_path,
            "horizon": horizon,
            "label": label,
            "model_path": model_path,
            "split": split,
            "test_size": test_size,
            "random_state": random_state,
            "use_gpu": use_gpu,
            "include_onchain": include_onchain,
            "include_orderbook": include_orderbook,
            "include_derivatives": include_derivatives,
            "include_sentiment": include_sentiment,
            "forward_fill_limit": forward_fill_limit,
            "fillna_value": fillna_value,
            "cv_strategy": cv_strategy,
            "embargo_min": embargo_min,
            "calibration": calibration,
            "conformal_alpha": conformal_alpha,
            "run_id": run_id_value,
            "dump_cv": dump_cv,
            "by_vol_bins": by_vol_bins,
        }
        config_dump = {
            "config": CONFIG.config_path.as_posix() if CONFIG.config_path else None,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in args_snapshot.items()},
            "features": feature_cols,
        }
        save_json(config_dump, "config_dump.json", run_id=run_id_value)

        logger.info(
            "Saved training artefacts",
            extra={
                "event": "artefacts",
                "run_id": run_id_value,
                "metrics": str(metrics_report_path),
            },
        )
    else:
        typer.echo("Dry run requested; metrics and artefacts will not be written.")

    typer.echo(f"Model trained{' (dry run)' if dry_run else ''} and stored at {model_output}")
    return model_output


@app.command()
def main(
    features: Optional[Path] = typer.Option(
        None,
        "--features",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Optional engineered feature table (CSV/Parquet).",
    ),
    symbol: str = typer.Option(CONFIG.symbol, "--symbol", help="Trading symbol used when sourcing data."),
    db_path: Path = typer.Option(
        CONFIG.db_path,
        "--db-path",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="SQLite database file to read raw price data from.",
    ),
    horizon: int = typer.Option(120, "--horizon", help="Target horizon in minutes for label generation."),
    label: Optional[str] = typer.Option(None, "--label", help="Existing label column to use."),
    model_path: Path = typer.Option(
        DEFAULT_MODEL_PATH,
        "--model-path",
        resolve_path=True,
        help="Output path for the trained model.",
    ),
    split: str = typer.Option("holdout", "--split", help="Evaluation split used during training."),
    test_size: float = typer.Option(0.2, "--test-size", help="Hold-out fraction used for the validation split."),
    random_state: int = typer.Option(42, "--random-state", help="Random seed used for model training."),
    use_gpu: bool = typer.Option(True, "--use-gpu/--no-gpu", help="Toggle GPU acceleration."),
    include_onchain: Optional[bool] = typer.Option(
        None,
        "--include-onchain/--exclude-onchain",
        help="Override on-chain features regardless of config defaults.",
    ),
    include_orderbook: Optional[bool] = typer.Option(
        None,
        "--include-orderbook/--exclude-orderbook",
        help="Override orderbook features regardless of config defaults.",
    ),
    include_derivatives: Optional[bool] = typer.Option(
        None,
        "--include-derivatives/--exclude-derivatives",
        help="Override derivative features regardless of config defaults.",
    ),
    include_sentiment: Optional[bool] = typer.Option(
        None,
        "--include-sentiment/--exclude-sentiment",
        help="Override sentiment features regardless of config defaults.",
    ),
    forward_fill_limit: Optional[int] = typer.Option(
        None, "--forward-fill-limit", help="Override forward-fill window for NaN handling."
    ),
    fillna_value: Optional[float] = typer.Option(
        None, "--fillna-value", help="Override fallback value used when forward fill runs out."
    ),
    cv_strategy: Optional[str] = typer.Option(
        None, "--cv", help="Optional cross-validation strategy to evaluate during training."
    ),
    embargo_min: int = typer.Option(
        CONFIG.cv.embargo_min,
        "--embargo-min",
        help="Embargo window in minutes used for purged walk-forward cross-validation.",
    ),
    calibration: str = typer.Option(
        CONFIG.calibration.method,
        "--calibration",
        help="Probability calibration method applied on the validation split.",
    ),
    conformal_alpha: Optional[float] = typer.Option(
        0.1, "--conformal-alpha", help="Enable conformal prediction intervals at the specified miscoverage level."
    ),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Optional run identifier."),
    dump_cv: bool = typer.Option(
        False, "--dump-cv", help="Export purged walk-forward CV splits to reports/cv_<run_id>.json."
    ),
    by_vol_bins: int = typer.Option(
        5, "--by-vol-bins", help="Number of realised volatility quantiles used for metrics by regime."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing."),
) -> None:
    cv_normalised = cv_strategy.lower() if cv_strategy else None
    calibration_normalised = calibration.lower()
    _run_training(
        features=features,
        symbol=symbol,
        db_path=db_path,
        horizon=horizon,
        label=label,
        model_path=model_path,
        split=split,
        test_size=test_size,
        random_state=random_state,
        use_gpu=use_gpu,
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
        include_sentiment=include_sentiment,
        forward_fill_limit=forward_fill_limit,
        fillna_value=fillna_value,
        cv_strategy=cv_normalised,
        embargo_min=embargo_min,
        calibration=calibration_normalised,
        conformal_alpha=conformal_alpha,
        run_id=run_id,
        dump_cv=dump_cv,
        by_vol_bins=by_vol_bins,
        dry_run=dry_run,
    )


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    run_cli(app)
