"""
SaaS Revenue Intelligence System
Setup configuration for package installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8") \
    if (Path(__file__).parent / "README.md").exists() else ""

setup(
    name="saas_revenue_intelligence",
    version="1.0.0",
    author="SaaS Intelligence Team",
    description="ML-powered SaaS revenue intelligence: churn prediction, CLV, risk scoring, and account recommendations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests*", "venv*", "*.egg-info"]),
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "streamlit>=1.31.0",
        "plotly>=5.19.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.0.0",
            "httpx>=0.26.0",
        ],
        "tune": [
            "optuna>=3.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "saas-train=scripts.train_all:main",
            "saas-api=scripts.run_api:main",
            "saas-monitor=scripts.monitor_model:main",
            "saas-report=scripts.generate_report:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
