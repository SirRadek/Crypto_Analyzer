import pandas as pd

from crypto_analyzer.labeling.survival import HazardConfig, hazard_touch, time_to_event


def test_hazard_touch_returns_expected_columns():
    prices = pd.Series([100, 101, 102, 101, 103], index=pd.date_range("2024-01-01", periods=5, freq="5min"))
    hazards = hazard_touch(prices, config=HazardConfig(horizon=3, barrier=0.005))
    assert list(hazards.columns) == ["hazard_step_1", "hazard_step_2", "hazard_step_3"]
    assert hazards.iloc[0].sum() >= 0.0


def test_time_to_event_extracts_first_hit():
    data = pd.DataFrame(
        {
            "hazard_step_1": [0.0, 0.0, 1.0],
            "hazard_step_2": [1.0, 0.0, 0.0],
            "hazard_step_3": [0.0, 1.0, 0.0],
        }
    )
    data.index = pd.date_range("2024-01-01", periods=len(data), freq="5min")
    tte = time_to_event(data)
    assert list(tte.values) == [2.0, 3.0, 1.0]

