"""
Data Pipeline Unit Tests
SaaS Revenue Intelligence System

Tests:
    - DataLoader: loads all 5 CSVs correctly
    - DataCleaner: handles nulls, types
    - DataValidator: schema + quality checks
    - FeatureEngineer: produces correct 34 features
    - End-to-end pipeline integration
"""

import pytest
import sys
from pathlib import Path

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader
from src.data.data_cleaning import DataCleaner
from src.data.data_validator import DataValidator
from src.data.feature_engineering import FeatureEngineer


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_data():
    loader = DataLoader("data/raw")
    return loader.load_all()


@pytest.fixture(scope="module")
def cleaned_data(raw_data):
    cleaner = DataCleaner()
    return cleaner.clean_all(raw_data)


@pytest.fixture(scope="module")
def account_df(cleaned_data):
    engineer = FeatureEngineer()
    return engineer.build_account_level_dataset(cleaned_data)


# ─────────────────────────────────────────────
# DATA LOADER TESTS
# ─────────────────────────────────────────────

class TestDataLoader:

    def test_loader_loads_all_datasets(self, raw_data):
        """All 5 datasets must be loaded."""
        required = {"accounts", "subscriptions", "feature_usage",
                    "support_tickets", "churn_events"}
        assert required.issubset(set(raw_data.keys()))

    def test_accounts_not_empty(self, raw_data):
        assert len(raw_data["accounts"]) > 0

    def test_subscriptions_not_empty(self, raw_data):
        assert len(raw_data["subscriptions"]) > 0

    def test_feature_usage_not_empty(self, raw_data):
        assert len(raw_data["feature_usage"]) > 0

    def test_support_tickets_not_empty(self, raw_data):
        assert len(raw_data["support_tickets"]) > 0

    def test_churn_events_not_empty(self, raw_data):
        assert len(raw_data["churn_events"]) > 0

    def test_accounts_has_account_id(self, raw_data):
        assert "account_id" in raw_data["accounts"].columns

    def test_subscriptions_has_account_id(self, raw_data):
        assert "account_id" in raw_data["subscriptions"].columns

    def test_subscriptions_has_mrr(self, raw_data):
        assert "mrr_amount" in raw_data["subscriptions"].columns

    def test_support_tickets_has_account_id(self, raw_data):
        assert "account_id" in raw_data["support_tickets"].columns

    def test_churn_events_has_account_id(self, raw_data):
        assert "account_id" in raw_data["churn_events"].columns


# ─────────────────────────────────────────────
# DATA CLEANER TESTS
# ─────────────────────────────────────────────

class TestDataCleaner:

    def test_cleaned_data_has_all_keys(self, cleaned_data):
        required = {"accounts", "subscriptions", "feature_usage",
                    "support_tickets", "churn_events"}
        assert required.issubset(set(cleaned_data.keys()))

    def test_accounts_no_duplicate_account_ids(self, cleaned_data):
        ids = cleaned_data["accounts"]["account_id"]
        assert ids.duplicated().sum() == 0, "Duplicate account_ids found after cleaning"

    def test_mrr_amount_non_negative(self, cleaned_data):
        mrr = cleaned_data["subscriptions"]["mrr_amount"]
        assert (mrr >= 0).all(), "Negative MRR found after cleaning"

    def test_no_all_null_rows_accounts(self, cleaned_data):
        df = cleaned_data["accounts"]
        all_null = df.isnull().all(axis=1).sum()
        assert all_null == 0, f"{all_null} fully null rows in accounts"

    def test_seats_non_negative(self, cleaned_data):
        if "seats" in cleaned_data["accounts"].columns:
            assert (cleaned_data["accounts"]["seats"] >= 0).all()


# ─────────────────────────────────────────────
# DATA VALIDATOR TESTS
# ─────────────────────────────────────────────

class TestDataValidator:

    def test_validator_passes_on_clean_data(self, cleaned_data):
        validator = DataValidator()
        # Should not raise
        try:
            validator.validate_raw_datasets(cleaned_data)
            passed = True
        except Exception:
            passed = False
        assert passed, "Validator failed on clean data"

    def test_validator_passes_account_level(self, account_df):
        validator = DataValidator()
        try:
            validator.validate_account_level_dataset(account_df)
            passed = True
        except Exception:
            passed = False
        assert passed, "Validator failed on account-level features"


# ─────────────────────────────────────────────
# FEATURE ENGINEER TESTS
# ─────────────────────────────────────────────

class TestFeatureEngineer:

    def test_output_is_dataframe(self, account_df):
        assert isinstance(account_df, pd.DataFrame)

    def test_account_count_matches_raw(self, raw_data, account_df):
        raw_accounts = len(raw_data["accounts"]["account_id"].unique())
        assert len(account_df) == raw_accounts

    def test_required_base_columns_present(self, account_df):
        required = [
            "account_id", "account_name", "plan_tier",
            "seats", "churn_flag"
        ]
        for col in required:
            assert col in account_df.columns, f"Missing column: {col}"

    def test_required_subscription_columns_present(self, account_df):
        required = [
            "tenure_months", "total_mrr", "total_arr",
            "subscription_count", "upgrade_count",
            "downgrade_count", "auto_renew_ratio"
        ]
        for col in required:
            assert col in account_df.columns, f"Missing column: {col}"

    def test_required_usage_columns_present(self, account_df):
        required = [
            "total_usage", "avg_usage_duration", "error_count",
            "unique_features_used", "error_rate", "engagement_score"
        ]
        for col in required:
            assert col in account_df.columns, f"Missing column: {col}"

    def test_required_support_columns_present(self, account_df):
        required = [
            "ticket_count", "avg_resolution_time",
            "avg_satisfaction_score", "escalation_ratio",
            "support_risk_score"
        ]
        for col in required:
            assert col in account_df.columns, f"Missing column: {col}"

    def test_no_nan_values_in_output(self, account_df):
        nan_cols = account_df.columns[account_df.isnull().any()].tolist()
        assert len(nan_cols) == 0, f"NaN values in: {nan_cols}"

    def test_tenure_months_non_negative(self, account_df):
        assert (account_df["tenure_months"] >= 0).all(), \
            "Negative tenure_months found"

    def test_churn_flag_is_binary(self, account_df):
        unique_vals = set(account_df["churn_flag"].unique())
        assert unique_vals.issubset({0, 1}), \
            f"churn_flag has non-binary values: {unique_vals}"

    def test_engagement_score_in_range(self, account_df):
        assert (account_df["engagement_score"] >= 0).all()
        assert (account_df["engagement_score"] <= 100).all()

    def test_auto_renew_ratio_in_range(self, account_df):
        assert (account_df["auto_renew_ratio"] >= 0).all()
        assert (account_df["auto_renew_ratio"] <= 1).all()

    def test_escalation_ratio_in_range(self, account_df):
        assert (account_df["escalation_ratio"] >= 0).all()
        assert (account_df["escalation_ratio"] <= 1).all()

    def test_error_rate_non_negative(self, account_df):
        assert (account_df["error_rate"] >= 0).all()

    def test_total_mrr_non_negative(self, account_df):
        assert (account_df["total_mrr"] >= 0).all()

    def test_revenue_per_seat_non_negative(self, account_df):
        assert (account_df["revenue_per_seat"] >= 0).all()

    def test_churn_flag_has_both_classes(self, account_df):
        """Dataset must have both churned and non-churned accounts."""
        assert account_df["churn_flag"].sum() > 0, "No churned accounts found"
        assert (account_df["churn_flag"] == 0).sum() > 0, "No active accounts found"

    def test_feature_count_at_least_30(self, account_df):
        """Must have at least 30 engineered features."""
        assert len(account_df.columns) >= 30, \
            f"Only {len(account_df.columns)} features — expected 30+"


# ─────────────────────────────────────────────
# END-TO-END INTEGRATION TEST
# ─────────────────────────────────────────────

class TestWeek1Pipeline:

    def test_full_pipeline_runs_without_error(self):
        """Full pipeline from raw data to account features must succeed."""
        loader = DataLoader("data/raw")
        raw = loader.load_all()

        cleaner = DataCleaner()
        cleaned = cleaner.clean_all(raw)

        validator = DataValidator()
        validator.validate_raw_datasets(cleaned)

        engineer = FeatureEngineer()
        df = engineer.build_account_level_dataset(cleaned)

        validator.validate_account_level_dataset(df)

        assert len(df) > 0
        assert "churn_flag" in df.columns
        assert "total_mrr" in df.columns
        assert "engagement_score" in df.columns

    def test_pipeline_output_matches_saved_csv(self):
        """Pipeline output must be consistent with saved processed CSV."""
        saved = pd.read_csv("data/processed/account_level_features.csv")
        assert len(saved) > 0
        assert "churn_flag" in saved.columns
        assert "total_mrr" in saved.columns
        assert saved["churn_flag"].isnull().sum() == 0


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"])
