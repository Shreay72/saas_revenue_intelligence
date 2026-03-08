import pandas as pd
from typing import Dict, Any, Optional

from src.utils.logger import get_logger
from src.pipelines.churn_pipeline import ChurnPredictionPipeline
from src.inference.schema import PredictionRequest, PredictionResponse
# Will add Revenue pipeline later when it's fully built in Phase 4

logger = get_logger(__name__)

class PredictionService:
    """Core Service wrapping ML Pipelines for real-time JSON inference."""
    
    def __init__(self):
        logger.info("Initializing Prediction Service...")
        self.churn_pipeline = ChurnPredictionPipeline()
        self.churn_pipeline.load_models()
        self.revenue_pipeline = None # Will load lazily if built
        
    def predict_account(self, request: PredictionRequest) -> PredictionResponse:
        """Executes full prediction logic for a single account."""
        
        # Convert single dict feature set to DataFrame (1 row)
        df = pd.DataFrame([request.features])
        
        # Must have account_id for pipeline logic
        df['account_id'] = request.account_id
        
        # Execute Pipeline
        churn_results = self.churn_pipeline.predict(df)
        
        # Handle empty/error outputs
        if churn_results.empty:
            raise ValueError(f"Failed to generate prediction for account {request.account_id}")
            
        row = churn_results.iloc[0]
        
        # Map back to unified schema
        return PredictionResponse(
            account_id=request.account_id,
            churn_probability=float(row['churn_probability']),
            risk_level=str(row['risk_level']),
            recommendation=str(row['recommendation'])
        )
