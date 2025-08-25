from analysis.compare_predictions import backfill_actuals_and_errors

if __name__ == "__main__":
    backfill_actuals_and_errors(
        db_path="db/data/crypto_data.sqlite",
        table_pred="prediction",
        symbol="BTCUSDT"
    )