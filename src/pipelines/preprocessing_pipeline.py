"""
Preprocessing Pipeline Module
SaaS Revenue Risk & Retention Intelligence System

Week 1 – Data Preparation Phase
"""

import logging
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[ColumnTransformer, object, object, pd.Series, pd.Series]:
    """
    Build preprocessing pipeline and perform train-test split.

    Parameters:
    -----------
    df : pd.DataFrame
        Account-level dataset containing features + churn_flag
    test_size : float
        Test split ratio (default=0.2)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    preprocessor : ColumnTransformer
    X_train_processed : np.ndarray
    X_test_processed : np.ndarray
    y_train : pd.Series
    y_test : pd.Series
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if "churn_flag" not in df.columns:
        raise ValueError("Target column 'churn_flag' not found in dataset.")

    logger.info("Starting preprocessing pipeline...")

    # ----------------------------
    # Separate features and target
    # ----------------------------
    X = df.drop(columns=["churn_flag"])
    y = df["churn_flag"]

    # Drop account_id (identifier)
    if "account_id" in X.columns:
        X = X.drop(columns=["account_id"])

    # ----------------------------
    # Identify column types
    # ----------------------------
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")

    # ----------------------------
    # Train-Test Split (Stratified)
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Test shape: {X_test.shape}")

    # ----------------------------
    # Preprocessing Pipeline
    # ----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop"
    )

    # Fit only on training data (prevents leakage)
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    logger.info("Preprocessing completed successfully.")

    return preprocessor, X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    print("Preprocessing module ready.")
