import pytest
import pandas as pd
import numpy as np
import sys
from unittest.mock import MagicMock

# --- Mocking MetaTrader5 before it's imported by signal_generator ---
# This prevents ModuleNotFoundError during test collection
sys.modules['MetaTrader5'] = MagicMock()

from signal_generator import analyze_tf_opportunity

# Mock config dictionary for weights
MOCK_WEIGHTS = {
    "BULLISH_BOS": 3.0,
    "FVG_BULLISH": 2.0,
    "BEARISH_BOS": -3.0,
    "FVG_BEARISH": -2.0,
    "HH": 1.0,
    "LL": -1.0,
}

@pytest.fixture
def mock_bullish_data():
    """
    Creates a mock pandas DataFrame with a clear bullish FVG and BOS.
    This version is carefully crafted to satisfy the centered rolling window
    logic in the indicator detection functions.
    """
    # Base data for stable indicator calculation
    base_data = {
        'time': pd.date_range(start='2023-01-01', periods=40, freq='5min'),
        'open': np.linspace(100, 105, 40), 'high': np.linspace(101, 106, 40),
        'low': np.linspace(99, 104, 40), 'close': np.linspace(100, 105, 40),
        'tick_volume': [10]*40
    }
    base_df = pd.DataFrame(base_data)

    # Specific pattern construction
    pattern_data = {
        'time': pd.date_range(start=base_df['time'].iloc[-1] + pd.Timedelta(minutes=5), periods=15, freq='5min'),
        'open':  [105, 106, 108, 107, 106, 105, 106, 107, 115, 116, 117, 118, 119, 120, 121],
        'high':  [107, 115, 109, 108, 107, 106, 108, 109, 116, 117, 118, 119, 120, 121, 122], # Swing High at index 1 (115)
        'low':   [104, 105, 107, 106, 105, 104, 105, 106, 114, 115, 116, 117, 118, 119, 120],
        'close': [106, 112, 108, 107, 105, 105, 107, 108, 118, 117, 118, 119, 120, 121, 122], # Breakout at index 8 (close=118 > high=115)
        'tick_volume': [10]*15,
    }
    # With swing_lookback=5, the window is 11.
    # Swing High at index 1 (val=115) is the max of index 0 to 6.
    # The breakout happens at index 8, well after the swing high is established.
    # A Bullish FVG is also created between candle 7 (high=109) and 9 (low=115).
    pattern_df = pd.DataFrame(pattern_data)

    combined_df = pd.concat([base_df, pattern_df], ignore_index=True)
    return combined_df

def test_analyze_bullish_opportunity(monkeypatch, mock_bullish_data):
    """
    Tests that analyze_tf_opportunity correctly identifies bullish patterns
    from mock data and calculates the score.
    """
    # 1. Mock the get_candlestick_data function to return our crafted data
    def mock_get_data(*args, **kwargs):
        return mock_bullish_data

    monkeypatch.setattr("signal_generator.get_candlestick_data", mock_get_data)

    # 2. Call the function under test
    opportunity = analyze_tf_opportunity(
        symbol="XAUUSD",
        tf="M5",
        mt5_path="", # Not used due to mocking
        gng_model=None,
        gng_feature_stats={},
        confidence_threshold=0, # Set to 0 to always get a result
        min_distance_pips_per_tf={},
        weights=MOCK_WEIGHTS
    )

    # 3. Assert the results
    try:
        assert opportunity is not None
        assert opportunity["signal"] == "BUY"

        # Check that the key bullish components we manufactured are detected
        assert "FVG_BULLISH" in opportunity["score_components"]
        assert "BULLISH_BOS" in opportunity["score_components"]

        # The score should be positive and reflect the contributions
        assert opportunity["score"] >= (MOCK_WEIGHTS["FVG_BULLISH"] + MOCK_WEIGHTS["BULLISH_BOS"])

    except AssertionError:
        print("\n--- DEBUG INFO ---")
        print(f"Signal: {opportunity.get('signal')}")
        print(f"Score: {opportunity.get('score')}")
        print(f"Score Components: {opportunity.get('score_components')}")
        print("--- END DEBUG INFO ---\n")
        raise
