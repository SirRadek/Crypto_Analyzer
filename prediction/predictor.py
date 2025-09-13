from __future__ import annotations

from analysis.rules import combined_signal
from typing import Iterable
import numpy as np
import pandas as pd


def _predict_ml_proba(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.Series:
    """
    Vrať ML pravděpodobnost UP v intervalu [0,1].
    Preferuje predict_ml_proba, jinak fallback na tvrdé 0/1.
    Lazy import kvůli import_hygiene.
    """
    from ml.predict import predict_ml  # lazy
    try:
        from ml.predict import predict_ml_proba  # type: ignore
        proba = predict_ml_proba(df, feature_cols)
        return pd.Series(np.asarray(proba, dtype=np.float32), index=df.index)
    except Exception:
        preds = predict_ml(df, feature_cols)
        return pd.Series(np.asarray(preds, dtype=np.float32), index=df.index)


def combine_predictions(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    model_paths: list[str] | None = None,
    method: str = "weighted",
    use_meta_only: bool = True,
    usage_path: str = "ml/model_usage.json",
    # Ladění pro BTCUSDT 2h:
    weight_ml: float = 0.7,
    weight_rules: float = 0.3,
    prob_threshold: float = 0.55,
    hysteresis: float = 0.02,
) -> pd.Series:
    """
    Kombinace pravidel a ML. Defaulty laděné na 2h BTCUSDT.

    method:
      - "weighted": w_ml*P_ml + w_rules*P_rules → threshold + hysteresis
      - "majority": 1 pokud aspoň jeden dává 1
      - "strict"  : 1 jen když oba dávají 1
    """
    # Pravidla jako „pravděpodobnost“ {0,1}
    rule_preds = combined_signal(df).astype("float32")

    # ML pravděpodobnost
    if use_meta_only or not model_paths:
        ml_prob = _predict_ml_proba(df, feature_cols)  # [0,1]
    else:
        from ml.predict import predict_weighted  # lazy
        ml_cls = predict_weighted(df, feature_cols, model_paths, usage_path=usage_path)
        ml_prob = pd.Series(np.asarray(ml_cls, dtype=np.float32), index=df.index)

    if method == "majority":
        return ((rule_preds + (ml_prob >= 0.5).astype("float32")) > 0).astype("int8")
    if method == "strict":
        return ((rule_preds + (ml_prob >= 0.5).astype("float32")) == 2).astype("int8")
    if method != "weighted":
        raise ValueError("Unknown combination method")

    # Vážený blend + hysteréze proti flip-flopům
    s = float(weight_ml + weight_rules) or 1.0
    w_ml = float(weight_ml) / s
    w_rules = float(weight_rules) / s
    blended = (w_ml * ml_prob.astype("float32")) + (w_rules * rule_preds)

    up = np.zeros(len(blended), dtype=np.int8)
    state = 0
    hi = float(prob_threshold)
    lo = float(max(0.0, prob_threshold - hysteresis))
    for i, p in enumerate(np.asarray(blended)):
        if state == 0 and p >= hi:
            state = 1
        elif state == 1 and p <= lo:
            state = 0
        up[i] = state
    return pd.Series(up, index=df.index)