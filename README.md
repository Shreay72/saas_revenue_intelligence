# 📊 SaaS Revenue Risk & Retention Intelligence System

> An end-to-end ML platform that predicts customer churn, models revenue risk,
> and recommends prioritized retention actions for SaaS Customer Success teams.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-150%2B%20passing-brightgreen.svg)]()

---

## 🎯 What It Does

| Problem | Solution |
|---------|----------|
| Which customers will churn? | ML churn model → `churn_probability` per account |
| How much revenue is at risk? | `Revenue at Risk = MRR × churn_probability` |
| What should CSM do? | Rule-based engine → action, owner, urgency, expected recovery |
| Who to call first? | Priority score = risk × revenue × urgency |

---

## 🏗️ Architecture

