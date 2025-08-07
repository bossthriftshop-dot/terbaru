import pytest
import json
import os
from learning import analyze_and_adapt_profiles

# Mock config dictionary, similar to what's loaded from config.json
MOCK_CONFIG = {
    "global_settings": {},
    "base_weights": {
        "BULLISH_BOS": 2.0,
        "FVG_BULLISH": 3.0,
        "BEARISH_BOS": -2.0,
        "FVG_BEARISH": -3.0,
        "PINBAR_BULL": 1.0
    },
    "strategy_profiles": {
        "test_profile": {
            "enabled": True
        }
    },
    "learning": {
        "enabled": True,
        "feedback_file": "test_feedback.json",
        "lookback_trades_per_profile": 50,
        "min_trades_for_adaptation": 1,
        "win_rate_target": 0.60,
        "weight_adjustment_factor": 0.1  # 10% adjustment
    }
}

# Mock trade feedback data
MOCK_FEEDBACK_DATA = [
    # BULLISH_BOS: 100% win rate (1/1 trades) -> weight should increase
    {"result": "win", "profile_name": "test_profile", "score_components": ["BULLISH_BOS", "FVG_BULLISH"]},
    # FVG_BEARISH: 0% win rate (0/1 trades) -> weight should decrease
    {"result": "loss", "profile_name": "test_profile", "score_components": ["BEARISH_BOS", "FVG_BEARISH"]},
    # FVG_BULLISH: 50% win rate (1/2 trades) -> should not change (target is 0.6, threshold is 0.15)
    {"result": "loss", "profile_name": "test_profile", "score_components": ["FVG_BULLISH"]},
    # PINBAR_BULL: 100% win rate, but not enough trades (1<5) -> should not change
    {"result": "win", "profile_name": "test_profile", "score_components": ["PINBAR_BULL"]},
]

@pytest.fixture
def setup_test_environment(tmp_path):
    """Create a temporary feedback file for testing."""
    feedback_file = tmp_path / "test_feedback.json"
    with open(feedback_file, 'w') as f:
        json.dump(MOCK_FEEDBACK_DATA, f)

    # Update config to use the temporary file path
    config = MOCK_CONFIG.copy()
    config["learning"]["feedback_file"] = str(feedback_file)
    return config

def test_weight_adaptation(setup_test_environment):
    """
    Tests that weights are adapted correctly based on trade feedback.
    """
    config = setup_test_environment

    # Run the adaptation process
    adapted_weights = analyze_and_adapt_profiles(config)

    profile_weights = adapted_weights["test_profile"]
    base_weights = config["base_weights"]

    # 1. Test for weight increase (BULLISH_BOS has 100% win rate)
    # The component needs > 5 trades to be adapted, so we need to adjust the test data
    # For now, let's temporarily modify the condition in learning.py to test the logic
    # Or, let's just assert that it SHOULDN'T change because of the >5 rule.
    # Actually, let's check the code again. The rule is `data['total'] > 5`.
    # I will update the mock data to reflect this for a better test.
    # Let's adjust the test to match the code as-is for now. The component won't be adapted.
    assert profile_weights["BULLISH_BOS"] == base_weights["BULLISH_BOS"]

    # Let's create a more realistic test case where adaptation *does* happen.

# A more realistic test case
MOCK_FEEDBACK_DATA_REALISTIC = [
    # BULLISH_BOS: 6 wins, 0 losses -> 100% win rate -> INCREASE weight
    {"result": "win", "profile_name": "test_profile", "score_components": ["BULLISH_BOS"]} for _ in range(6)
] + [
    # FVG_BEARISH: 0 wins, 6 losses -> 0% win rate -> DECREASE weight
    {"result": "loss", "profile_name": "test_profile", "score_components": ["FVG_BEARISH"]} for _ in range(6)
] + [
    # FVG_BULLISH: 3 wins, 3 losses -> 50% win rate -> NO CHANGE (within 0.60 +/- 0.15 threshold)
    {"result": "win", "profile_name": "test_profile", "score_components": ["FVG_BULLISH"]} for _ in range(3)
] + [
    {"result": "loss", "profile_name": "test_profile", "score_components": ["FVG_BULLISH"]} for _ in range(3)
]

@pytest.fixture
def setup_realistic_test(tmp_path):
    """Create a temporary feedback file with more realistic data."""
    feedback_file = tmp_path / "test_feedback_realistic.json"
    with open(feedback_file, 'w') as f:
        json.dump(MOCK_FEEDBACK_DATA_REALISTIC, f)

    config = MOCK_CONFIG.copy()
    config["learning"]["feedback_file"] = str(feedback_file)
    return config

def test_realistic_weight_adaptation(setup_realistic_test):
    """
    Tests weight adaptation with enough data for the logic to trigger.
    """
    config = setup_realistic_test
    adapted_weights = analyze_and_adapt_profiles(config)

    profile_weights = adapted_weights["test_profile"]
    base_weights = config["base_weights"]
    adj_factor = config["learning"]["weight_adjustment_factor"]

    # 1. BULLISH_BOS (100% WR) -> weight should INCREASE
    expected_weight_bullish = base_weights["BULLISH_BOS"] * (1 + adj_factor)
    assert profile_weights["BULLISH_BOS"] == pytest.approx(expected_weight_bullish)

    # 2. FVG_BEARISH (0% WR) -> weight should DECREASE
    # Note: weight is negative, so decreasing means adding the adjustment
    expected_weight_bearish = base_weights["FVG_BEARISH"] - abs(base_weights["FVG_BEARISH"] * adj_factor)
    assert profile_weights["FVG_BEARISH"] == pytest.approx(expected_weight_bearish)

    # 3. FVG_BULLISH (50% WR) -> weight should NOT CHANGE
    assert profile_weights["FVG_BULLISH"] == base_weights["FVG_BULLISH"]

    # 4. BEARISH_BOS was not in feedback -> weight should NOT CHANGE
    assert profile_weights["BEARISH_BOS"] == base_weights["BEARISH_BOS"]
