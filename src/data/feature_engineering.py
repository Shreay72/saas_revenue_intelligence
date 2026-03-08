"""
Feature Engineering Module
SaaS Revenue Risk & Retention Intelligence System
Week 1 — Complete version with all 31 features + Week 3 state/signal features

Features created (34 total):
    Base (9):         account_id, account_name, industry, country, referral_source,
                      plan_tier, seats, is_trial, churn_flag
    Subscription (7): tenure_months, total_mrr, total_arr, subscription_count,
                      upgrade_count, downgrade_count, auto_renew_ratio
    Usage (6):        total_usage, avg_usage_duration, error_count,
                      unique_features_used, error_rate, engagement_score
    Support (6):      ticket_count, avg_resolution_time, avg_first_response_time,
                      avg_satisfaction_score, escalation_ratio, support_risk_score
    Revenue (3):      revenue_per_seat, total_refund_amount, is_reactivation
    State/Signal (3): engagement_state, revenue_change_signal, support_pressure_signal
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict


class FeatureEngineer:

    def __init__(self, reference_date=None, log_level=logging.INFO):
        self.reference_date = reference_date or datetime.now()
        self.logger = self._setup_logger(log_level)
        self.logger.info("FeatureEngineer initialized.")

    def _setup_logger(self, log_level):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    # ---------------------------------------------------
    # Detect account creation date column dynamically
    # ---------------------------------------------------
    def _detect_creation_column(self, accounts_df: pd.DataFrame) -> str:
        possible_columns = [
            "created_date",
            "signup_date",
            "account_created_at",
            "registration_date",
            "created_at",
        ]
        for col in possible_columns:
            if col in accounts_df.columns:
                return col
        raise ValueError(
            "No valid account creation date column found in accounts dataset."
        )

    # ---------------------------------------------------
    # MAIN PIPELINE
    # ---------------------------------------------------
    def build_account_level_dataset(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Build complete account-level dataset with all 34 features.
        """

        accounts      = datasets["accounts"].copy()
        subscriptions = datasets["subscriptions"].copy()
        usage         = datasets["feature_usage"].copy()
        support       = datasets["support_tickets"].copy()
        churn         = datasets["churn_events"].copy()

        # -----------------------------------
        # 1. Tenure Calculation
        # -----------------------------------
        self.logger.info("Calculating tenure...")
        creation_col = self._detect_creation_column(accounts)
        accounts[creation_col] = pd.to_datetime(accounts[creation_col], errors="coerce")
        accounts["tenure_months"] = (
            (self.reference_date - accounts[creation_col]).dt.days / 30.44
        )
        accounts["tenure_months"] = accounts["tenure_months"].clip(lower=0)

        # -----------------------------------
        # 2. Subscription Aggregation (Basic)
        # -----------------------------------
        self.logger.info("Aggregating subscription metrics...")
        sub_agg = (
            subscriptions.groupby("account_id")
            .agg(
                total_mrr=("mrr_amount", "sum"),
                total_arr=("arr_amount", "sum"),
                subscription_count=("subscription_id", "count"),
            )
            .reset_index()
        )

        # -----------------------------------
        # 3. Subscription Aggregation (Advanced)
        # -----------------------------------
        self.logger.info("Adding upgrade/downgrade counts...")
        upgrade_agg = subscriptions.groupby("account_id")["upgrade_flag"].sum().reset_index()
        upgrade_agg.columns = ["account_id", "upgrade_count"]

        downgrade_agg = subscriptions.groupby("account_id")["downgrade_flag"].sum().reset_index()
        downgrade_agg.columns = ["account_id", "downgrade_count"]

        self.logger.info("Calculating auto-renew ratio...")
        autorenew_agg = subscriptions.groupby("account_id")["auto_renew_flag"].mean().reset_index()
        autorenew_agg.columns = ["account_id", "auto_renew_ratio"]

        # -----------------------------------
        # 4. Usage Aggregation (JOIN via subscription_id)
        # -----------------------------------
        self.logger.info("Aggregating usage metrics...")
        usage_merged = usage.merge(
            subscriptions[["subscription_id", "account_id"]],
            on="subscription_id",
            how="left",
        )

        usage_agg = (
            usage_merged.groupby("account_id")
            .agg(
                total_usage=("usage_count", "sum"),
                avg_usage_duration=("usage_duration_secs", "mean"),
                error_count=("error_count", "sum"),
            )
            .reset_index()
        )

        self.logger.info("Calculating unique features...")
        if "feature_name" in usage.columns:
            unique_features = (
                usage_merged.groupby("account_id")["feature_name"].nunique().reset_index()
            )
            unique_features.columns = ["account_id", "unique_features_used"]
        else:
            unique_features = pd.DataFrame({"account_id": [], "unique_features_used": []})

        # -----------------------------------
        # 5. Support Aggregation
        # -----------------------------------
        self.logger.info("Aggregating support metrics...")
        support_agg = (
            support.groupby("account_id")
            .agg(
                ticket_count=("ticket_id", "count"),
                avg_resolution_time=("resolution_time_hours", "mean"),
                avg_first_response_time=("first_response_time_minutes", "mean"),
                avg_satisfaction_score=("satisfaction_score", "mean"),
                escalation_ratio=("escalation_flag", "mean"),
            )
            .reset_index()
        )

        # -----------------------------------
        # 6. Churn Flag & Details
        # -----------------------------------
        self.logger.info("Processing churn data...")
        churn_accounts = set(churn["account_id"])
        accounts["churn_flag"] = accounts["account_id"].isin(churn_accounts).astype(int)

        if not churn.empty:
            churn_details = churn.groupby("account_id").agg(
                total_refund_amount=("refund_amount_usd", "sum"),
                is_reactivation=("is_reactivation", "max"),
            ).reset_index()
        else:
            churn_details = pd.DataFrame(
                {
                    "account_id": [],
                    "total_refund_amount": [],
                    "is_reactivation": [],
                }
            )

        # -----------------------------------
        # 7. Merge All Features
        # -----------------------------------
        self.logger.info("Merging all features...")
        df = accounts.merge(sub_agg, on="account_id", how="left")
        df = df.merge(upgrade_agg, on="account_id", how="left")
        df = df.merge(downgrade_agg, on="account_id", how="left")
        df = df.merge(autorenew_agg, on="account_id", how="left")
        df = df.merge(usage_agg, on="account_id", how="left")
        df = df.merge(unique_features, on="account_id", how="left")
        df = df.merge(support_agg, on="account_id", how="left")
        df = df.merge(churn_details, on="account_id", how="left")

        # -----------------------------------
        # 8. Derived Features
        # -----------------------------------
        self.logger.info("Creating derived features...")

        # Error rate
        df["error_rate"] = df["error_count"] / (df["total_usage"] + 1)

        # Revenue per seat
        df["revenue_per_seat"] = df["total_mrr"] / df["seats"].replace(0, 1)

        # Engagement score (0–100)
        if "unique_features_used" in df.columns:
            max_usage = df["total_usage"].max() if df["total_usage"].max() > 0 else 1
            max_features = (
                df["unique_features_used"].max()
                if df["unique_features_used"].max() > 0
                else 1
            )
            df["engagement_score"] = (
                (df["total_usage"] / max_usage) * 50
                + (df["unique_features_used"] / max_features) * 50
            )
        else:
            df["engagement_score"] = 0

        # Support risk score (composite)
        df["support_risk_score"] = (
            df.get("avg_satisfaction_score", 0) * 20
            - df.get("avg_resolution_time", 0) * 2
            - df.get("escalation_ratio", 0) * 100
        )

        # -----------------------------------
        # 8b. STATE / SIGNAL FEATURES (pseudo-trends)
        # -----------------------------------
        self.logger.info("Adding engagement/support/revenue state features...")

        # Engagement state: +1 improving, -1 declining, 0 neutral
        if "engagement_score" in df.columns and "error_rate" in df.columns:
            median_engagement = df["engagement_score"].median()
            median_error = df["error_rate"].median()
            df["engagement_state"] = np.where(
                (df["engagement_score"] >= median_engagement)
                & (df["error_rate"] <= median_error),
                1,
                np.where(
                    (df["engagement_score"] < median_engagement)
                    & (df["error_rate"] > median_error),
                    -1,
                    0,
                ),
            )
        else:
            df["engagement_state"] = 0

        # Revenue change signal: net upgrades vs downgrades (clipped -1..1)
        if "upgrade_count" in df.columns and "downgrade_count" in df.columns:
            net_changes = df["upgrade_count"] - df["downgrade_count"]
            total_changes = df["upgrade_count"] + df["downgrade_count"] + 1
            df["revenue_change_signal"] = (net_changes / total_changes).clip(-1, 1)
        else:
            df["revenue_change_signal"] = 0.0

        # Support pressure signal: +1 worsening, -1 healthy, 0 neutral
        if "escalation_ratio" in df.columns and "ticket_count" in df.columns:
            median_tickets = df["ticket_count"].median()
            df["support_pressure_signal"] = np.where(
                (df["escalation_ratio"] > 0.3) & (df["ticket_count"] >= median_tickets),
                1,
                np.where(
                    (df["escalation_ratio"] == 0)
                    & (df["ticket_count"] < median_tickets),
                    -1,
                    0,
                ),
            )
        else:
            df["support_pressure_signal"] = 0

        # -----------------------------------
        # 9. Final Cleanup
        # -----------------------------------
        df.fillna(0, inplace=True)

        self.logger.info(f"Account-level dataset created. Shape: {df.shape}")
        self.logger.info(f"Total features: {len(df.columns)}")
        self.logger.info(f"Features: {list(df.columns)}")

        return df


def build_account_features(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convenience function to build account-level features."""
    engineer = FeatureEngineer()
    return engineer.build_account_level_dataset(datasets)


if __name__ == "__main__":
    print("Feature Engineering module ready.")
    print("Use: engineer = FeatureEngineer()")
    print("     df = engineer.build_account_level_dataset(datasets)")
