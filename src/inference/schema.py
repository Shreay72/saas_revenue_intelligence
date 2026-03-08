from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class PredictionRequest(BaseModel):
    """Schema for a single account prediction request."""
    account_id: str
    features: Dict[str, Any] = Field(..., description="Raw account features required for model input")

class BatchPredictionRequest(BaseModel):
    """Schema for uploading a batch of accounts for prediction."""
    accounts: List[PredictionRequest]
    
class PredictionResponse(BaseModel):
    """Schema for returning risk and revenue forecasts."""
    account_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    predicted_mrr: Optional[float] = None
    clv: Optional[float] = None
    recommendation: Optional[str] = None
    
class BatchPredictionResponse(BaseModel):
    """Schema for returning a batch of predictions."""
    results: List[PredictionResponse]
    summary_stats: Dict[str, Any]
