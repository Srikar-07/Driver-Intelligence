# 📍 Drivr Intelligence — Driver Demand Heatmap

> **Geospatial ML system for predicting ride demand across Miami** | Identifies when and where drivers should position themselves to maximize earnings and minimize customer wait times.

---

## 📌 Project Summary

This component predicts **hyperlocal ride demand** across Miami's 10 key neighborhoods using temporal features, cyclical time encoding, and gradient boosting. The output powers three interactive Folium maps that can be opened directly in a browser — no server needed.

**Core DS skills demonstrated:**
- Geospatial data generation with realistic lat/lon coordinates
- Cyclical feature engineering (sin/cos encoding for time)
- Demand forecasting with Random Forest vs Gradient Boosting
- Interactive map visualization with Folium & HeatMapWithTime
- Business framing: driver positioning strategy

---

## 🏆 Model Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Random Forest | 0.0433 | 0.0556 | 0.811 |
| **Gradient Boosting** ✅ | **0.0433** | **0.0551** | **0.815** |

**Business interpretation:** The model predicts demand scores (0–1) with an average error of 0.043 — meaning if we predict "high demand" (0.8), the true value is within ±0.04. That's sufficient precision for driver positioning decisions.

---

## 🗺️ Maps Generated

| Map | Description | Key Insight |
|-----|-------------|-------------|
| `map_01_pickup_density.html` | Static heatmap of all 8,000 pickups | South Beach & Wynwood are hotspots |
| `map_02_hourly_animation.html` | Time-animated demand by hour (press play) | Demand shifts from Downtown (daytime) → South Beach (night) |
| `map_03_neighborhood_bubbles.html` | Bubble map: volume + avg demand | Doral has lowest demand — avoid positioning drivers there |

---

## 🔑 Key Features

| Feature | Engineering | Why |
|---------|-------------|-----|
| `hour_sin / hour_cos` | Cyclical encoding | Avoids 23→0 discontinuity |
| `dow_sin / dow_cos` | Cyclical encoding | Day-of-week patterns |
| `month_sin / month_cos` | Cyclical encoding | Seasonal tourism peaks |
| `is_weekend` | Binary | Friday/Saturday surge |
| `neighborhood_enc` | Label encoded | Location identity |

**Why cyclical encoding?** A naive `hour=23` and `hour=0` are 23 apart numerically but only 1 hour apart in reality. Sin/cos encoding wraps the time axis into a circle — a standard technique in time-aware ML systems.

---

## 🗂️ Structure

```
02_demand_heatmap/
├── data/
│   ├── generate_geo_data.py     # GPS + demand data generator
│   ├── trips_geo.csv            # 8,000 trips with lat/lon
│   └── demand_grid.csv          # Grid demand for animation
├── models/
│   ├── train_demand.py          # Training pipeline
│   ├── demand_model.pkl         # Gradient Boosting model
│   └── demand_results.json      # Model metrics
├── app/
│   └── generate_maps.py         # Folium map generator
└── notebooks/
    ├── map_01_pickup_density.html
    ├── map_02_hourly_animation.html
    ├── map_03_neighborhood_bubbles.html
    ├── hourly_demand_forecast.png
    ├── neighborhood_hour_heatmap.png
    └── demand_feature_importance.png
```

---

## 🚀 Quickstart

```bash
pip install folium scikit-learn pandas numpy matplotlib seaborn joblib

# Generate data
cd data/ && python generate_geo_data.py

# Train model
cd models/ && python train_demand.py

# Generate maps
cd app/ && python generate_maps.py

# Open maps in browser
open notebooks/map_01_pickup_density.html
open notebooks/map_02_hourly_animation.html
```

---

## 📊 Key Findings

**When demand peaks:**
- South Beach & Wynwood: 10pm–2am (nightlife)
- Brickell & Downtown: 7–9am, 5–7pm (business commute)
- All neighborhoods: Saturday > Friday > weekdays

**Event impact (December Art Basel):**
- Wynwood demand: +180% vs baseline
- South Beach demand: +160% vs baseline
- Doral: virtually unaffected (residential/industrial)

**Driver positioning strategy:**
- *Before 6pm:* Stage in Brickell/Downtown for commuter demand
- *After 10pm weekends:* Shift to South Beach/Wynwood
- *Event nights:* Pre-position 30min before event end

---

## 🔭 Next Steps

- [ ] Integrate real OpenStreetMap road network for routing
- [ ] Add live weather API to adjust demand predictions in real-time
- [ ] Build Streamlit dashboard combining pricing + demand maps
- [ ] Cluster neighborhoods using DBSCAN for micro-zone discovery
- [ ] Add Miami event calendar API (Ticketmaster, SeatGeek)

---

## 👤 Author

Part of the **Drivr Intelligence** portfolio — a full DS/ML system built on a real startup use case.
