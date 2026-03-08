import os
import glob
import json
import shutil
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelRegistry:
    """
    Local filesystem-based Model Registry to manage ML artifacts.
    Provides MLFlow-lite capabilities.
    """
    
    def __init__(self, root_dir: str = "models"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
    def get_latest_model(self, model_type: str = "churn") -> Optional[Path]:
        """Get the path to the most recently trained model of a given type."""
        model_dir = self.root_dir / model_type
        if not model_dir.exists():
            return None
            
        models = glob.glob(str(model_dir / "*_model_*.pkl"))
        if not models:
            return None
            
        # Basic sorting by name semantics or creation time
        latest = max(models, key=os.path.getctime)
        return Path(latest)
        
    def load_model(self, model_type: str = "churn", version: str = None) -> Any:
        """Load a specified model version or the latest if None."""
        if version:
            model_path = self.root_dir / model_type / f"{model_type}_model_{version}.pkl"
        else:
            model_path = self.get_latest_model(model_type)
            
        if not model_path or not model_path.exists():
            logger.error(f"No model found for type '{model_type}' and version '{version}'")
            raise FileNotFoundError(f"Model artifact not found.")
            
        logger.info(f"Loading {model_type} model from {model_path}")
        return joblib.load(model_path)
        
    def get_metadata(self, model_type: str = "churn") -> Dict[str, Any]:
        """Load metadata json for a specific model family."""
        meta_path = self.root_dir / model_type / f"{model_type}_metadata.json"
        
        # Mapping for the existing churn structure which just uses "model_metadata.json"
        if not meta_path.exists() and model_type == 'churn':
            meta_path = self.root_dir / model_type / "model_metadata.json"
            
        if not meta_path.exists():
            logger.warning(f"No metadata found at {meta_path}")
            return {}
            
        with open(meta_path, 'r') as f:
            return json.load(f)
            
    def promote_to_production(self, model_type: str, candidate_path: Path):
        """Copies a candidate model to production alias."""
        target = self.root_dir / model_type / f"{model_type}_production.pkl"
        logger.info(f"Promoting {candidate_path.name} to Production at {target}")
        shutil.copy2(candidate_path, target)
