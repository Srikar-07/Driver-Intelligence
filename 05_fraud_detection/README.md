# 🚨 Drivr Intelligence — Fraud & Anomaly Detection

> **Three-model ensemble fraud detector** combining unsupervised anomaly detection (Isolation Forest + Autoencoder) with supervised classification (XGBoost) to flag suspicious trips in real time.

---

## 📌 Project Summary

Fraud is an existential threat for any ride-hailing startup — fake completions, fare manipulation, and GPS spoofing directly destroy unit economics. This system detects 5 fraud types with a **weighted ensemble achieving ROC-AUC = 1.00** on the test set.

---

## 🏆 Model Results

| Model | Type | ROC-AUC | Ensemble Weight |
|-------|------|---------|-----------------|
| Isolation Forest | Unsupervised | 0.9856 | 0.327 |
| Autoencoder | Unsupervised (deep) | 0.9936 | 0.330 |
| **XGBoost** | Supervised | **1.0000** | **0.333** |
| **Ensemble** | Weighted avg | **1.0000** | — |

**Classification report (test set):**
```
              precision    recall  f1-score
Normal         1.00        1.00      1.00
Fraud          0.91        1.00      0.95
Accuracy                             1.00
```

---

## 🔍 Fraud Types Detected

| Type | Signal | Key Feature |
|------|--------|-------------|
| GPS spoofing | Impossible movement speed | `speed_kmh` > 90, `location_jump_km` > 15 |
| Fare manipulation | Distance/fare mismatch | `fare_distance_ratio` outlier |
| Fake completion | Trip too fast to be real | `duration_min` < 3 for `distance_km` > 5 |
| Account takeover | New device + new payment + location jump | `new_device` + `location_jump_km` |
| Promo abuse | Excessive trips/cancellations | `trip_count_today` > 15, `cancellations_today` |

---

## 🏗️ Architecture

```
Trip data
    │
    ▼
┌─────────────────────────────────┐
│ Feature Engineering (16 feats)  │
└──────────┬──────────────────────┘
           │
    ┌──────┼──────────────┐
    ▼      ▼              ▼
Isolation  Autoencoder   XGBoost
Forest     (recon error) (supervised)
    │          │              │
    └──────────┴──────────────┘
               │
         Weighted Ensemble
         (AUC-weighted avg)
               │
          Fraud Score (0–1)
```

**Why three models?**
- **Isolation Forest** — catches outliers with no labels (unsupervised). Works on day 1 with no fraud history.
- **Autoencoder** — learns what "normal" looks like. Anomalies have high reconstruction error.
- **XGBoost** — uses fraud labels for precise classification once labeled data is available.
- **Ensemble** — weighted by each model's AUC — more accurate than any single model.

---

## 🗂️ Structure

```
05_fraud_detection/
├── data/
│   ├── generate_fraud_data.py    # Injects 5 realistic fraud patterns
│   └── fraud_trips.csv           # 5,000 trips (4% fraud rate)
├── models/
│   ├── train_fraud.py            # Full training pipeline
│   ├── isolation_forest.pkl
│   ├── autoencoder.pkl
│   ├── xgb_fraud.pkl
│   └── fraud_scaler.pkl
└── notebooks/
    ├── confusion_matrix.png
    ├── score_distributions.png
    ├── feature_importance.png
    └── fraud_by_type.png
```

---

## 🚀 Quickstart

```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn joblib

cd data/ && python generate_fraud_data.py
cd models/ && python train_fraud.py
```

---

## 🧠 Key Interview Talking Points

- **"I used two unsupervised models for cold-start"** — at launch you have no fraud labels, so Isolation Forest + Autoencoder detect anomalies without them. As labeled data accumulates, XGBoost takes over with higher precision.
- **"AUC-weighted ensemble"** — each model's weight is proportional to its ROC-AUC, so better models automatically contribute more.
- **"Autoencoder trained only on normal data"** — this is the standard approach: learn the distribution of legitimate trips; fraud shows high reconstruction error because it's out of distribution.

---

## 🔭 Next Steps

- [ ] Real-time scoring API with <50ms latency SLA
- [ ] SHAP explanations for each flagged trip ("flagged because speed_kmh = 187")
- [ ] Active learning loop — human review of borderline cases feeds new labels back into XGBoost
- [ ] Graph-based fraud detection — detect coordinated fraud rings using driver-customer networks

---

## 👤 Author

Part of the **Drivr Intelligence** portfolio — a full DS/ML system built on a real startup use case.
