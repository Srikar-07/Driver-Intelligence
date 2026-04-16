# 🚗 Drivr Intelligence — Dynamic Pricing Model

> **End-to-end ML system for real-time ride pricing** | Built as the core ML engine for Drivr, a driver-for-hire startup targeting Miami's nightlife and tourism market.

---

## 📌 Project Summary

Drivr connects car owners with verified drivers — think *"Uber for your own car."* This repository contains the **dynamic pricing ML system** that powers real-time fare estimation based on demand, supply, weather, time, and local events.

This is a **production-grade ML project** showcasing:
- End-to-end pipeline from data generation → feature engineering → training → serving
- Model comparison (XGBoost vs LightGBM) with rigorous evaluation
- REST API deployment with FastAPI
- Business-framed thinking throughout (not just model accuracy)

---

## 🏆 Model Results

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|------|
| XGBoost | $1.64 | $3.07 | 0.977 | 3.77% |
| **LightGBM** ✅ | **$1.61** | **$3.06** | **0.977** | **3.68%** |

**LightGBM selected** as production model — better MAE and RMSE with similar MAPE.

**Business impact framing:** A $1.61 average pricing error on a $40 average fare = **4.0% error rate**, well within acceptable range for dynamic pricing. Comparable to Uber's surge pricing accuracy benchmarks.

---

## 🔑 Key Features Driving Price

Based on feature importance analysis:

1. **`event_demand_multiplier`** — Local events (Art Basel, Ultra) are the strongest price driver
2. **`distance_km`** — Core fare component
3. **`driver_supply_ratio`** — Supply-demand dynamics
4. **`hour`** — Time-of-day surge (late night = 1.8x multiplier)
5. **`weather_encoded`** — Stormy weather → 1.6x multiplier
6. **`demand_supply_ratio`** — Computed feature: pending requests / active drivers

---

## 🗂️ Project Structure

```
01_pricing_model/
├── data/
│   ├── generate_data.py        # Synthetic Miami ride data generator
│   └── rides.csv               # 5,000 trip records
├── models/
│   ├── train.py                # Full training pipeline
│   ├── pricing_model.pkl       # Saved LightGBM model
│   ├── label_encoder.pkl       # Neighborhood encoder
│   ├── features.json           # Feature list
│   └── results.json            # Model metrics
├── api/
│   └── main.py                 # FastAPI prediction endpoint
└── notebooks/
    ├── actual_vs_predicted.png
    ├── feature_importance_xgboost.png
    └── feature_importance_lightgbm.png
```

---

## 🚀 Quickstart

### 1. Install dependencies
```bash
pip install xgboost lightgbm scikit-learn pandas numpy fastapi uvicorn joblib
```

### 2. Generate data
```bash
cd data/
python generate_data.py
```

### 3. Train models
```bash
cd models/
python train.py
```

### 4. Run the API
```bash
cd api/
uvicorn main:app --reload
```

API docs at: `http://localhost:8000/docs`

---

## 🔌 API Usage

### Predict price for a trip

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 23,
    "day_of_week": 4,
    "month": 12,
    "distance_km": 8.5,
    "weather": "rainy",
    "pickup_neighborhood": "South Beach",
    "dropoff_neighborhood": "Brickell",
    "driver_supply_ratio": 0.3,
    "event_demand_multiplier": 2.5,
    "active_drivers": 18,
    "pending_requests": 95
  }'
```

**Response:**
```json
{
  "estimated_price_usd": 208.26,
  "surge_active": true,
  "surge_reason": "late night, rainy weather, local event",
  "confidence": "high",
  "breakdown": {
    "base_fare": 15.0,
    "distance_component": 18.7,
    "surge_multiplier": 6.18,
    "weather": "rainy",
    "is_weekend": false,
    "is_night": true
  }
}
```

> **Art Basel night, rainy, South Beach → $208 surge** 🌧️

---

## 🧠 Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Model | LightGBM | Faster training, better on tabular data vs XGBoost here |
| API framework | FastAPI | Async, auto-docs, Pydantic validation |
| Feature encoding | LabelEncoder for neighborhoods | Low cardinality, interpretable |
| Pricing floor | $8.00 | Business constraint — minimum viable fare |
| Train/test split | 80/20 | Standard; no time-series leakage (random trips) |

---

## 📊 Data Features

| Feature | Type | Description |
|---------|------|-------------|
| `hour` | int | Hour of trip (0–23) |
| `day_of_week` | int | 0=Monday, 6=Sunday |
| `is_weekend` | bool | Fri/Sat/Sun |
| `is_night` | bool | 10pm–6am |
| `is_rush_hour` | bool | 7–10am, 5–8pm |
| `distance_km` | float | Trip distance |
| `weather_encoded` | int | 0=sunny → 3=stormy |
| `driver_supply_ratio` | float | Available drivers ratio (0–1) |
| `event_demand_multiplier` | float | 1.0 = no event, up to 2.8 |
| `demand_supply_ratio` | float | Pending requests / active drivers |
| `pickup_neighborhood` | str | One of 10 Miami neighborhoods |

---

## 🔭 Next Steps / Roadmap

- [ ] Add real weather data via OpenWeatherMap API
- [ ] Integrate Miami event calendar API
- [ ] Retrain pipeline with SHAP explainability layer
- [ ] Add price elasticity analysis (what discount % increases bookings)
- [ ] Deploy to Hugging Face Spaces with Streamlit demo UI
- [ ] A/B test simulation notebook comparing static vs dynamic pricing revenue

---

## 👤 Author

Built as part of the **Drivr Intelligence** portfolio project — a full DS/ML system demonstrating end-to-end machine learning engineering on a real startup use case.

> *"The best portfolio projects solve real problems, not toy datasets."*
