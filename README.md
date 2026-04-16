# 🚗 Drivr Intelligence — End-to-End ML System

> A **production-grade Data Science & ML portfolio project** built on a real startup use case: Drivr, a driver-for-hire platform connecting car owners with verified drivers in Miami.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.ai)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green.svg)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io)

---

## 📌 What is Drivr?

Drivr is a **"Uber for your own car"** concept — customers hire verified drivers to operate their personal vehicle, instead of booking a taxi. Target market: Miami nightlife, tourism, weddings, events.

This repository contains the **entire ML/DS backend** that would power the platform — built end-to-end as a senior-level portfolio project.

---

## 🧠 System Components

| # | Component | Tech | Key Result |
|---|-----------|------|------------|
| 1 | [Dynamic Pricing Model](./01_pricing_model) | LightGBM, XGBoost, FastAPI | R² = 0.977, MAE = $1.61 |
| 2 | [Driver Demand Heatmap](./02_demand_heatmap) | Gradient Boosting, Folium, Geospatial | R² = 0.815, 3 interactive maps |
| 3 | [Driver-Customer Matching](./03_driver_matching) | Two-stage RecSys, SVD, Cosine, FastAPI | Content + Collaborative + Hybrid ranker |
| 4 | [Analytics Dashboard](./04_analytics_dashboard) | Streamlit, Plotly, Pandas | 4-page business intelligence dashboard |
| 5 | [Fraud Detection](./05_fraud_detection) | Isolation Forest, Autoencoder, XGBoost | ROC-AUC = 1.00, 5 fraud types detected |

---

## 🏗️ Full Architecture

```
                        DRIVR INTELLIGENCE
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Customer Request                                          │
│         │                                                   │
│         ├──► [01] Pricing Model ──► Real-time fare ($)     │
│         │         LightGBM                                  │
│         │         FastAPI /predict                          │
│         │                                                   │
│         ├──► [02] Demand Heatmap ──► Driver positioning    │
│         │         Gradient Boosting                         │
│         │         Folium maps                               │
│         │                                                   │
│         ├──► [03] Matching System ──► Top-K drivers        │
│         │         Content-based + SVD + Hybrid              │
│         │         FastAPI /match                            │
│         │                                                   │
│         └──► [05] Fraud Detection ──► Risk score (0-1)     │
│                   Isolation Forest + Autoencoder + XGBoost  │
│                                                             │
│   Business Layer                                            │
│         └──► [04] Analytics Dashboard                       │
│                   Streamlit + Plotly                        │
│                   Revenue, trips, drivers, trends           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/drivr-intelligence.git
cd drivr-intelligence
pip install -r requirements.txt
```

### Run any component
```bash
# Pricing Model API
cd 01_pricing_model/api && uvicorn main:app --reload

# Demand maps
cd 02_demand_heatmap/app && python generate_maps.py

# Analytics Dashboard
cd 04_analytics_dashboard/app && streamlit run dashboard.py

# Fraud Detection
cd 05_fraud_detection/models && python train_fraud.py
```

---

## 📦 Requirements

```bash
pip install xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn \
            fastapi uvicorn joblib scipy folium streamlit plotly
```

---

## 📊 Key Results Summary

| Model | Metric | Value |
|-------|--------|-------|
| Pricing — LightGBM | R² | **0.977** |
| Pricing — LightGBM | MAE | **$1.61** |
| Demand — Gradient Boosting | R² | **0.815** |
| Fraud — Ensemble | ROC-AUC | **1.000** |
| Fraud — Recall (fraud class) | Recall | **100%** |
| Matching — Hybrid Ranker | Retrieval pool | **40 candidates → Top-K** |

---

## 💡 Why This Project Stands Out

- **Real business context** — every model solves a real problem, not a toy dataset
- **End-to-end** — data generation → feature engineering → training → serving → monitoring
- **Multiple ML paradigms** — supervised, unsupervised, recommender systems, geospatial, time-series
- **Production patterns** — two-stage retrieval, model ensembles, FastAPI endpoints, Streamlit dashboards
- **Live demo-able** — run the dashboard or hit the API in 60 seconds during an interview

---

## 👤 Author

Built as a senior-level DS/ML portfolio project demonstrating end-to-end machine learning engineering.

> *"The best portfolio projects solve real problems, not toy datasets."*
