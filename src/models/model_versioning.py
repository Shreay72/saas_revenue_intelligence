import json
from pathlib import Path
from typing import Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelVersioner:
    """Manages semantic versioning for models in the registry."""
    
    def __init__(self, metadata_path: str = "models/version_history.json"):
        self.metadata_path = Path(metadata_path)
        self._ensure_history_exists()
        
    def _ensure_history_exists(self):
        if not self.metadata_path.exists():
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, 'w') as f:
                json.dump({"history": []}, f)
                
    def get_history(self) -> list:
        with open(self.metadata_path, 'r') as f:
            return json.load(f).get("history", [])
            
    def get_next_version(self, model_type: str, is_major: bool = False) -> str:
        """Determines the next vX.Y string."""
        history = self.get_history()
        type_history = [h for h in history if h.get('model_type') == model_type]
        
        if not type_history:
            return "v1.0"
            
        latest = type_history[-1]['version']
        v = latest.lstrip('v').split('.')
        major, minor = int(v[0]), int(v[1])
        
        if is_major:
            major += 1
            minor = 0
        else:
            minor += 1
            
        return f"v{major}.{minor}"
        
    def record_version(self, model_type: str, version: str, metrics: Dict[str, float], author: str = "system"):
        """Logs a newly trained version to history."""
        history = self.get_history()
        
        entry = {
            "model_type": model_type,
            "version": version,
            "metrics": metrics,
            "author": author,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        history.append(entry)
        
        with open(self.metadata_path, 'w') as f:
            json.dump({"history": history}, f, indent=4)
            
        logger.info(f"Recorded new model version: {model_type} {version}")
