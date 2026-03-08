import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.utils.logger import get_logger
from src.pipelines.churn_pipeline import ChurnPredictionPipeline

logger = get_logger(__name__)

class BatchPredictor:
    """Service to process large datasets (CSVs) for daily/weekly scoring syncs."""
    
    def __init__(self):
        self.churn_pipeline = ChurnPredictionPipeline()
        self.churn_pipeline.load_models()
        
    def score_csv(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Loads CSV, runs predictions, saves results, and returns summary stats.
        """
        in_path = Path(input_path)
        out_path = Path(output_path)
        
        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")
            
        logger.info(f"Loading batch data from {in_path}")
        df = pd.read_csv(in_path)
        
        logger.info(f"Scoring {len(df)} accounts via Batch Pipeline...")
        scored_df = self.churn_pipeline.predict(df)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(out_path, index=False)
        logger.info(f"Batch results saved to {out_path}")
        
        # Basic stats
        high_risk_count = len(scored_df[scored_df['risk_level'] == 'High'])
        
        stats = {
            "total_accounts": len(scored_df),
            "high_risk_count": high_risk_count,
            "average_churn_prob": float(scored_df['churn_probability'].mean())
        }
        
        return stats
