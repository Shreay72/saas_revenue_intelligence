# 🔧 Troubleshooting Guide

## Common Errors

---

### ❌ `KeyError: 'revenue_at_risk'`
**Cause**: `generate_intelligence()` called before revenue columns computed.

**Fix**: Ensure these lines run BEFORE calling `generate_intelligence()`:
```python
df["revenue_at_risk"] = rev_pipeline.calculate_revenue_at_risk(df)
df["health_score"]    = rev_pipeline.calculate_health_scores(df)
df["clv"]             = rev_pipeline.predict_clv(df)
```

---

### ❌ `No module named 'X'`
**Fix**:
```bash
pip install -r requirements.txt
```

---

### ❌ `FileNotFoundError: churn_model_v1.pkl`
**Fix**: Models not trained yet.
```bash
python scripts/train_all.py --skip-week1 --no-tune
```

---

### ❌ `API not reachable` in dashboard
**Fix**: Start API first.
```bash
# Terminal 1
python scripts/run_api.py

# Terminal 2 (wait until API shows "Application startup complete")
python scripts/run_streamlit.py
```

---

### ❌ ROC-AUC = 1.0000 (suspiciously perfect)
**Cause**: `churn_probability` column in training features → data leakage.

**Fix**: Remove `churn_probability` from `account_level_features.csv` before training:
```python
df = pd.read_csv("data/processed/account_level_features.csv")
df = df.drop(columns=["churn_probability"], errors="ignore")
df.to_csv("data/processed/account_level_features.csv", index=False)
python scripts/train_all.py --skip-week1 --no-tune
```

---

### ❌ `Baseline not found: monitoring/baseline_snapshot.csv`
**Cause**: First time running monitoring — no baseline yet.

**Fix**: This is expected. The detector auto-creates the baseline on first run.
```bash
python scripts/generate_report.py --run-monitoring
```

---

### ❌ `use_container_width` Streamlit warnings
**Cause**: Streamlit deprecated this parameter.

**Fix (PowerShell)**:
```powershell
Get-ChildItem -Path dashboard -Recurse -Filter "*.py" | ForEach-Object {
    (Get-Content $_.FullName) `
      -replace "use_container_width=True",  "width='stretch'" `
      -replace "use_container_width=False", "width='content'" |
    Set-Content $_.FullName
}
```

---

### ❌ `Early stopping not available`
**Cause**: XGBoost version doesn't support sklearn API early stopping.

**Impact**: None — model trains correctly without early stopping.

---

### ❌ Docker: port already in use
**Fix**:
```bash
# Kill process on port 8000
netstat -ano | findstr :8000       # Windows
lsof -ti:8000 | xargs kill -9     # Linux/Mac
```

---

### ❌ CLV values seem very high ($500K+)
**Cause**: Low churn probability + high MRR → large CLV by formula.

**Check**: CLV = (MRR × 0.70) / (churn/12 + 0.10/12)
At churn_prob=0.01, MRR=$10K: CLV = $7,000 / 0.0092 = $761K — expected.

---

## Useful Debug Commands

```bash
# Check all artifacts exist
python scripts/train_all.py --verify-only

# Test API health
curl http://localhost:8000/health

# Check account count
python -c "import pandas as pd; df=pd.read_csv('data/processed/account_intelligence.csv'); print(df.shape)"

# Check risk distribution
python -c "import pandas as pd; df=pd.read_csv('data/processed/account_intelligence.csv'); print(df['risk_tier'].value_counts())"

# Run all 182 tests
pytest tests/ -v

# Re-run full pipeline
python scripts/train_all.py --skip-week1 --no-tune
```
