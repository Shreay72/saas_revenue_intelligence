"""
Monitoring Module Tests
SaaS Revenue Intelligence System

Tests:
    - drift_check: compute_basic_drift_metrics
    - Edge cases: identical frames, missing columns, empty frames
"""

import pytest
import sys
from pathlib import Path

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.drift_check import compute_basic_drift_metrics


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def baseline_df():
    np.random.seed(42)
    return pd.DataFrame({
        "account_id":        range(1, 101),
        "total_mrr":         np.random.normal(5000, 1000, 100).clip(min=0),
        "churn_probability": np.random.uniform(0.05, 0.40, 100),
        "engagement_score":  np.random.uniform(30, 90, 100),
    })


@pytest.fixture
def current_df_similar(baseline_df):
    """Current data similar to baseline — low drift."""
    np.random.seed(99)
    return pd.DataFrame({
        "account_id":        range(101, 201),
        "total_mrr":         np.random.normal(5100, 1000, 100).clip(min=0),
        "churn_probability": np.random.uniform(0.06, 0.42, 100),
        "engagement_score":  np.random.uniform(28, 92, 100),
    })


@pytest.fixture
def current_df_drifted(baseline_df):
    """Current data significantly different from baseline — high drift."""
    np.random.seed(7)
    return pd.DataFrame({
        "account_id":        range(101, 201),
        "total_mrr":         np.random.normal(9000, 1500, 100).clip(min=0),
        "churn_probability": np.random.uniform(0.50, 0.95, 100),
        "engagement_score":  np.random.uniform(5, 35, 100),
    })


# ─────────────────────────────────────────────
# BASIC STRUCTURE TESTS
# ─────────────────────────────────────────────

class TestDriftCheckStructure:

    def test_returns_dict(self, baseline_df, current_df_similar):
        result = compute_basic_drift_metrics(current_df_similar, baseline_df)
        assert isinstance(result, dict)

    def test_returns_mrr_keys(self, baseline_df, current_df_similar):
        result = compute_basic_drift_metrics(current_df_similar, baseline_df)
        assert "total_mrr_mean_current"  in result
        assert "total_mrr_mean_baseline" in result
        assert "total_mrr_mean_shift"    in result

    def test_returns_churn_probability_keys(self, baseline_df, current_df_similar):
        result = compute_basic_drift_metrics(current_df_similar, baseline_df)
        assert "churn_probability_mean_current"  in result
        assert "churn_probability_mean_baseline" in result
        assert "churn_probability_mean_shift"    in result

    def test_all_values_are_floats(self, baseline_df, current_df_similar):
        result = compute_basic_drift_metrics(current_df_similar, baseline_df)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not a float: {type(val)}"


# ─────────────────────────────────────────────
# CORRECTNESS TESTS
# ─────────────────────────────────────────────

class TestDriftCheckCorrectness:

    def test_zero_shift_for_identical_frames(self, baseline_df):
        """Identical current and baseline must produce zero shift."""
        result = compute_basic_drift_metrics(baseline_df, baseline_df)
        assert result["total_mrr_mean_shift"] == 0.0
        assert result["churn_probability_mean_shift"] == 0.0

    def test_positive_shift_when_mrr_increases(self, baseline_df, current_df_drifted):
        result = compute_basic_drift_metrics(current_df_drifted, baseline_df)
        assert result["total_mrr_mean_shift"] > 0, \
            "Expected positive MRR shift when current MRR is higher"

    def test_positive_shift_when_churn_increases(self, baseline_df, current_df_drifted):
        result = compute_basic_drift_metrics(current_df_drifted, baseline_df)
        assert result["churn_probability_mean_shift"] > 0, \
            "Expected positive churn shift when current churn is higher"

    def test_shift_formula(self, baseline_df, current_df_similar):
        """shift = current_mean - baseline_mean."""
        result = compute_basic_drift_metrics(current_df_similar, baseline_df)
        expected = (
            result["total_mrr_mean_current"] -
            result["total_mrr_mean_baseline"]
        )
        assert abs(result["total_mrr_mean_shift"] - expected) < 1e-6

    def test_current_mean_matches_dataframe(self, baseline_df, current_df_similar):
        result = compute_basic_drift_metrics(current_df_similar, baseline_df)
        expected = float(current_df_similar["total_mrr"].mean())
        assert abs(result["total_mrr_mean_current"] - expected) < 1e-6

    def test_baseline_mean_matches_dataframe(self, baseline_df, current_df_similar):
        result = compute_basic_drift_metrics(current_df_similar, baseline_df)
        expected = float(baseline_df["total_mrr"].mean())
        assert abs(result["total_mrr_mean_baseline"] - expected) < 1e-6


# ─────────────────────────────────────────────
# EDGE CASE TESTS
# ─────────────────────────────────────────────

class TestDriftCheckEdgeCases:

    def test_missing_column_in_both(self, baseline_df, current_df_similar):
        """If column missing in both, it should be skipped gracefully."""
        current_no_mrr  = current_df_similar.drop(columns=["total_mrr"])
        baseline_no_mrr = baseline_df.drop(columns=["total_mrr"])
        result = compute_basic_drift_metrics(current_no_mrr, baseline_no_mrr)
        assert "total_mrr_mean_shift" not in result
        # churn_probability still computed
        assert "churn_probability_mean_shift" in result

    def test_missing_column_in_current_only(self, baseline_df, current_df_similar):
        """If column missing in current only, skip gracefully."""
        current_no_mrr = current_df_similar.drop(columns=["total_mrr"])
        result = compute_basic_drift_metrics(current_no_mrr, baseline_df)
        assert "total_mrr_mean_shift" not in result

    def test_empty_current_dataframe(self, baseline_df):
        """Empty current frame should return empty dict or handle gracefully."""
        empty = pd.DataFrame(columns=baseline_df.columns)
        try:
            result = compute_basic_drift_metrics(empty, baseline_df)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"compute_basic_drift_metrics crashed on empty input: {e}")

    def test_single_row_dataframes(self):
        """Single row frames should compute without errors."""
        current  = pd.DataFrame({"total_mrr": [5000.0], "churn_probability": [0.3]})
        baseline = pd.DataFrame({"total_mrr": [4500.0], "churn_probability": [0.2]})
        result = compute_basic_drift_metrics(current, baseline)
        assert result["total_mrr_mean_shift"] == pytest.approx(500.0, abs=1e-3)

    def test_no_monitored_columns_returns_empty(self):
        """If no monitored columns present, return empty dict."""
        current  = pd.DataFrame({"col_a": [1, 2, 3]})
        baseline = pd.DataFrame({"col_a": [4, 5, 6]})
        result = compute_basic_drift_metrics(current, baseline)
        assert result == {}


# ─────────────────────────────────────────────
# INTEGRATION TEST — Real Data
# ─────────────────────────────────────────────

class TestDriftCheckIntegration:

    def test_on_real_account_features(self):
        """Drift check must work on real processed data."""
        csv_path = Path("data/processed/account_level_features.csv")
        if not csv_path.exists():
            pytest.skip("Processed data not found — run Week 1 pipeline first")

        df = pd.read_csv(csv_path)
        midpoint = len(df) // 2
        baseline = df.iloc[:midpoint].copy()
        current  = df.iloc[midpoint:].copy()

        result = compute_basic_drift_metrics(current, baseline)
        assert isinstance(result, dict)
        assert "total_mrr_mean_shift" in result
        # Shift must be a reasonable number (not NaN, not inf)
        assert np.isfinite(result["total_mrr_mean_shift"])


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"])
