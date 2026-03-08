"""
Config Loader Utility
SaaS Revenue Intelligence System
"""

import yaml
from pathlib import Path
from typing import Any, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = 'config/model_config.yaml') -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Usage:
        from src.utils.config_loader import load_config
        config = load_config()
    """
    path = Path(config_path)
    
    if not path.exists():
        logger.warning(f"Config file not found: {path}. Using defaults.")
        return {}
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✅ Config loaded from: {path}")
    return config


def get_business_costs(config: Dict = None) -> Dict[str, float]:
    """Get business cost assumptions."""
    if config is None:
        config = load_config()
    
    costs = config.get('business_costs', {})
    return {
        'cost_fp': costs.get('false_positive_cost', 100.0),
        'cost_fn': costs.get('false_negative_cost', 5000.0)
    }


def get_risk_thresholds(config: Dict = None) -> Dict[str, float]:
    """Get risk level thresholds."""
    if config is None:
        config = load_config()
    
    thresholds = config.get('risk_thresholds', {})
    return {
        'low': thresholds.get('low', 0.3),
        'high': thresholds.get('high', 0.6)
    }


if __name__ == "__main__":
    config = load_config()
    print("Config loaded successfully.")
    print(f"Business costs: {get_business_costs(config)}")
    print(f"Risk thresholds: {get_risk_thresholds(config)}")
