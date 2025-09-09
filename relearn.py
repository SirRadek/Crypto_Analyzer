import predict_window as pw  # re-use your window config
from ml.retrain_from_errors import retrain_with_error_weights

if __name__ == "__main__":
    # 1) learn from errors observed so far
    retrain_with_error_weights(
        db_path=pw.DB_PATH,
        symbol=pw.SYMBOL,
        forward_steps=pw.FORWARD_STEPS,
        alpha=1.0,          # stronger focus on mistakes -> increase
        max_weight=5.0,
        cutoff_to_latest_backfilled=True
    )
    # 2) try again on the same window but DO NOT retrain inside predict_window
    pw.main(train=False)
