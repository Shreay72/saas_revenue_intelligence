"""
Week 3 Revenue Model Tests — Updated for deterministic CLV
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers  import safe_divide, normalize_series, format_currency
from src.utils.metrics  import (
    calculate_clv, calculate_revenue_at_risk,
    calculate_health_score, calculate_portfolio_metrics,
)
from src.models.clv_model import CLVModel


# ─────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'account_id':             ['A-001', 'A-002', 'A-003', 'A-004', 'A-005'],
        'account_name':           ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'],
        'plan_tier':              ['Enterprise', 'Pro', 'Basic', 'Enterprise', 'Pro'],
        'seats':                  [50, 20, 5, 100, 15],
        'total_mrr':              [10000, 5000, 1000, 25000, 3000],
        'total_arr':              [120000, 60000, 12000, 300000, 36000],
        'tenure_months':          [24, 12, 6, 36, 18],
        'churn_flag':             [0, 1, 0, 0, 1],
        'churn_probability':      [0.05, 0.85, 0.10, 0.03, 0.70],
        'engagement_score':       [80.0, 20.0, 60.0, 90.0, 30.0],
        'support_risk_score':     [10.0, 80.0, 30.0, 5.0, 60.0],
        'auto_renew_ratio':       [1.0, 0.5, 0.8, 1.0, 0.6],
        'upgrade_count':          [2, 0, 1, 3, 0],
        'downgrade_count':        [0, 1, 0, 0, 2],
        'ticket_count':           [3.0, 12.0, 5.0, 2.0, 8.0],
        'avg_satisfaction_score': [4.5, 2.0, 3.5, 4.8, 2.5],
        'error_rate':             [0.01, 0.20, 0.05, 0.01, 0.15],
        'unique_features_used':   [15, 3, 8, 20, 5],
        'total_usage':            [500, 50, 200, 800, 80],
        'escalation_ratio':       [0.0, 0.5, 0.1, 0.0, 0.4],
    })


@pytest.fixture
def check_revenue_models_exist():
    """Skip test if models not yet trained."""
    from pathlib import Path
    mrr_path = Path('models/revenue/revenue_model_v1.pkl')
    clv_path = Path('models/revenue/clv_metadata.json')
    if not mrr_path.exists() or not clv_path.exists():
        pytest.skip("Revenue models not found. Run train script first.")


# ─────────────────────────────────────────────────────
# HELPER TESTS
# ─────────────────────────────────────────────────────

class TestHelpers:

    def test_safe_divide_normal(self):
        from src.utils.helpers import safe_divide
        assert safe_divide(10, 2) == 5.0

    def test_safe_divide_zero(self):
        from src.utils.helpers import safe_divide
        assert safe_divide(10, 0) == 0.0

    def test_normalize_series(self):
        from src.utils.helpers import normalize_series
        s = pd.Series([0, 50, 100])
        result = normalize_series(s)
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_normalize_constant_series(self):
        from src.utils.helpers import normalize_series
        s = pd.Series([5, 5, 5])
        result = normalize_series(s)
        assert all(result == 0.5)

    def test_format_currency(self):
        from src.utils.helpers import format_currency
        assert format_currency(1234.5) == "$1,234.50"

    def test_health_score_healthy_customer(self):
        score = calculate_health_score(
            engagement_score=85, support_risk_score=10,
            churn_probability=0.05, tenure_months=24,
            auto_renew_ratio=1.0
        )
        assert score > 70

    def test_health_score_at_risk_customer(self):
        score = calculate_health_score(
            engagement_score=15, support_risk_score=85,
            churn_probability=0.90, tenure_months=2,
            auto_renew_ratio=0.2
        )
        assert score < 40


# ─────────────────────────────────────────────────────
# METRICS TESTS
# ─────────────────────────────────────────────────────

class TestMetrics:

    def test_calculate_clv_basic(self):
        clv = calculate_clv(mrr=10000, churn_probability=0.10)
        assert clv > 0

    def test_calculate_clv_high_churn(self):
        low_clv  = calculate_clv(mrr=10000, churn_probability=0.90)
        high_clv = calculate_clv(mrr=10000, churn_probability=0.05)
        assert high_clv > low_clv

    def test_calculate_clv_zero_mrr(self):
        assert calculate_clv(mrr=0, churn_probability=0.10) == 0.0

    def test_revenue_at_risk(self):
        risk = calculate_revenue_at_risk(mrr=10000, churn_probability=0.80)
        assert risk == pytest.approx(8000.0, rel=0.01)

    def test_revenue_at_risk_zero_churn(self):
        assert calculate_revenue_at_risk(mrr=10000, churn_probability=0.0) == 0.0

    def test_expansion_revenue_starter(self):
        from src.utils.metrics import calculate_expansion_revenue
        exp = calculate_expansion_revenue(mrr=1000, plan_tier='Basic')
        assert exp > 0

    def test_expansion_revenue_enterprise_lower(self):
        from src.utils.metrics import calculate_expansion_revenue
        basic_exp      = calculate_expansion_revenue(mrr=5000, plan_tier='Basic')
        enterprise_exp = calculate_expansion_revenue(mrr=5000, plan_tier='Enterprise')
        assert basic_exp > enterprise_exp

    def test_portfolio_metrics(self, sample_df):
        metrics = calculate_portfolio_metrics(sample_df)
        assert 'total_mrr'       in metrics
        assert 'churn_rate'      in metrics
        assert metrics['total_mrr'] == pytest.approx(44000.0, rel=0.01)

    def test_portfolio_metrics_with_clv(self, sample_df):
        sample_df = sample_df.copy()
        sample_df['clv'] = [100000, 5000, 20000, 250000, 8000]
        metrics = calculate_portfolio_metrics(sample_df)
        assert 'total_clv' in metrics
        assert 'avg_clv'   in metrics


# ─────────────────────────────────────────────────────
# CLV MODEL TESTS — Deterministic
# ─────────────────────────────────────────────────────

class TestCLVModel:

    def test_clv_model_initialization(self):
        model = CLVModel()
        assert model.gross_margin  == 0.70
        assert model.discount_rate == 0.10
        assert model.is_fitted is True  # deterministic → always fitted

    def test_clv_model_fit(self, sample_df):
        model   = CLVModel()
        metrics = model.fit(sample_df)
        assert 'method'   in metrics
        assert metrics['method'] == 'deterministic_formula'
        assert metrics['clv_mean'] > 0

    def test_clv_model_predict(self, sample_df):
        model       = CLVModel()
        predictions = model.predict(sample_df)
        assert len(predictions) == len(sample_df)
        assert all(p >= 0 for p in predictions)

    def test_clv_predict_with_segments(self, sample_df):
        model   = CLVModel()
        results = model.predict_with_segments(sample_df)
        assert 'predicted_clv' in results.columns
        assert 'clv_segment'   in results.columns
        valid_segments = {'Champions', 'Loyalists', 'At-Risk', 'Lost Causes'}
        assert set(results['clv_segment'].unique()).issubset(valid_segments)

    def test_high_mrr_higher_clv(self):
        """Higher MRR with same churn → higher CLV."""
        model = CLVModel()
        df_low  = pd.DataFrame({'total_mrr': [1000],  'churn_probability': [0.10]})
        df_high = pd.DataFrame({'total_mrr': [10000], 'churn_probability': [0.10]})
        assert model.predict(df_high)[0] > model.predict(df_low)[0]

    def test_clv_deterministic_stability(self):
        """Same inputs must always produce same CLV — no randomness."""
        model = CLVModel()
        df    = pd.DataFrame({'total_mrr': [5000], 'churn_probability': [0.20]})
        results = [model.predict(df)[0] for _ in range(5)]
        assert len(set(results)) == 1, "Deterministic model must return identical results"

    def test_clv_formula_correctness(self):
        """Verify CLV matches expected formula output."""
        mrr, churn, margin, discount = 10000, 0.12, 0.70, 0.10
        monthly_churn    = churn    / 12
        monthly_discount = discount / 12
        expected_clv     = (mrr * margin) / (monthly_churn + monthly_discount)
        actual_clv       = calculate_clv(mrr=mrr, churn_probability=churn,
                                         gross_margin=margin, discount_rate=discount)
        assert actual_clv == pytest.approx(expected_clv, rel=0.01)


# ─────────────────────────────────────────────────────
# PIPELINE TESTS
# ─────────────────────────────────────────────────────

class TestRevenuePipeline:

    def test_pipeline_load(self, check_revenue_models_exist):
        from src.pipelines.revenue_pipeline import load_revenue_pipeline
        pipeline = load_revenue_pipeline()
        assert pipeline.is_loaded is True

    def test_predict_mrr(self, check_revenue_models_exist, sample_df):
        from src.pipelines.revenue_pipeline import load_revenue_pipeline
        pipeline = load_revenue_pipeline()
        preds    = pipeline.predict_mrr(sample_df)
        assert len(preds) == len(sample_df)
        assert all(p >= 0 for p in preds)

    def test_predict_clv(self, check_revenue_models_exist, sample_df):
        from src.pipelines.revenue_pipeline import load_revenue_pipeline
        pipeline = load_revenue_pipeline()
        preds    = pipeline.predict_clv(sample_df)
        assert len(preds) == len(sample_df)
        assert all(p >= 0 for p in preds)

    def test_revenue_at_risk(self, check_revenue_models_exist, sample_df):
        from src.pipelines.revenue_pipeline import load_revenue_pipeline
        pipeline = load_revenue_pipeline()
        risk     = pipeline.calculate_revenue_at_risk(sample_df)
        assert len(risk) == len(sample_df)
        assert all(r >= 0 for r in risk)

    def test_health_scores(self, check_revenue_models_exist, sample_df):
        from src.pipelines.revenue_pipeline import load_revenue_pipeline
        pipeline = load_revenue_pipeline()
        scores   = pipeline.calculate_health_scores(sample_df)
        assert len(scores) == len(sample_df)
        assert all(0 <= s <= 100 for s in scores)

    def test_generate_account_intelligence(self, check_revenue_models_exist, sample_df):
        from src.pipelines.revenue_pipeline import load_revenue_pipeline
        pipeline = load_revenue_pipeline()
        results  = pipeline.generate_account_intelligence(sample_df)
        assert 'predicted_clv'   in results.columns
        assert 'revenue_at_risk' in results.columns
        assert 'health_score'    in results.columns
        assert 'priority_tier'   in results.columns

    def test_priority_tiers_valid(self, check_revenue_models_exist, sample_df):
        from src.pipelines.revenue_pipeline import load_revenue_pipeline
        pipeline     = load_revenue_pipeline()
        results      = pipeline.generate_account_intelligence(sample_df)
        valid_tiers  = {'P1 - Critical', 'P2 - High', 'P3 - Medium', 'P4 - Low'}
        assert set(results['priority_tier'].unique()).issubset(valid_tiers)

    def test_pipeline_info(self, check_revenue_models_exist):
        from src.pipelines.revenue_pipeline import load_revenue_pipeline
        pipeline = load_revenue_pipeline()
        info     = pipeline.get_pipeline_info()
        assert 'mrr_model_type'  in info
        assert info['clv_model_type'] == 'deterministic_formula'
