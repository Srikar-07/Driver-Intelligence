"""
Drivr Dynamic Pricing API
FastAPI endpoint to serve real-time price predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import json
import numpy as np
import os

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "../models/pricing_model.pkl")
LE_PATH    = os.path.join(BASE, "../models/label_encoder.pkl")
FEAT_PATH  = os.path.join(BASE, "../models/features.json")

model = joblib.load(MODEL_PATH)
le    = joblib.load(LE_PATH)
with open(FEAT_PATH) as f:
    FEATURES = json.load(f)

NEIGHBORHOODS = [
    "South Beach", "Brickell", "Wynwood", "Coral Gables",
    "Coconut Grove", "Little Havana", "Midtown", "Downtown",
    "Aventura", "Doral"
]

app = FastAPI(
    title="Drivr Pricing API",
    description="Real-time dynamic pricing for driver-for-hire trips in Miami.",
    version="1.0.0"
)


class PriceRequest(BaseModel):
    hour: int                     = Field(..., ge=0, le=23,  example=22)
    day_of_week: int              = Field(..., ge=0, le=6,   example=5)
    month: int                    = Field(..., ge=1, le=12,  example=3)
    distance_km: float            = Field(..., gt=0,         example=12.5)
    weather: Literal["sunny","cloudy","rainy","stormy"] = Field(..., example="rainy")
    pickup_neighborhood: str      = Field(..., example="South Beach")
    dropoff_neighborhood: str     = Field(..., example="Brickell")
    driver_supply_ratio: float    = Field(..., ge=0, le=1,   example=0.4)
    event_demand_multiplier: float= Field(..., ge=1,         example=1.0)
    active_drivers: int           = Field(..., ge=0,         example=25)
    pending_requests: int         = Field(..., ge=0,         example=90)


class PriceResponse(BaseModel):
    estimated_price_usd: float
    surge_active: bool
    surge_reason: str
    confidence: str
    breakdown: dict


@app.get("/")
def root():
    return {"service": "Drivr Pricing API", "status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "LightGBM", "features": len(FEATURES)}


@app.post("/predict", response_model=PriceResponse)
def predict_price(req: PriceRequest):
    if req.pickup_neighborhood not in NEIGHBORHOODS:
        raise HTTPException(400, f"Unknown neighborhood: {req.pickup_neighborhood}")
    if req.dropoff_neighborhood not in NEIGHBORHOODS:
        raise HTTPException(400, f"Unknown neighborhood: {req.dropoff_neighborhood}")

    weather_map = {"sunny": 0, "cloudy": 1, "rainy": 2, "stormy": 3}
    is_weekend  = int(req.day_of_week >= 5)
    is_night    = int(req.hour >= 22 or req.hour <= 5)
    is_rush     = int(req.hour in list(range(7,10)) + list(range(17,20)))
    pickup_enc  = int(le.transform([req.pickup_neighborhood])[0])
    dropoff_enc = int(le.transform([req.dropoff_neighborhood])[0])
    dsratio     = round(req.pending_requests / (req.active_drivers + 1), 2)

    row = [
        req.hour, req.day_of_week, req.month,
        is_weekend, is_night, is_rush,
        req.distance_km, weather_map[req.weather],
        req.driver_supply_ratio, req.event_demand_multiplier,
        req.active_drivers, req.pending_requests,
        dsratio, pickup_enc, dropoff_enc
    ]

    price = float(model.predict([row])[0])
    price = round(max(price, 8.0), 2)

    # Surge logic
    base_est = 15 + req.distance_km * 2.2
    surge = price > base_est * 1.3
    reasons = []
    if is_night:                         reasons.append("late night")
    if req.weather in ("rainy","stormy"): reasons.append(f"{req.weather} weather")
    if req.event_demand_multiplier > 1.3: reasons.append("local event")
    if req.driver_supply_ratio < 0.3:    reasons.append("low driver supply")
    if is_weekend:                       reasons.append("weekend")

    return PriceResponse(
        estimated_price_usd=price,
        surge_active=surge,
        surge_reason=", ".join(reasons) if reasons else "none",
        confidence="high" if req.distance_km < 40 else "medium",
        breakdown={
            "base_fare": 15.0,
            "distance_component": round(req.distance_km * 2.2, 2),
            "surge_multiplier": round(price / base_est, 2),
            "weather": req.weather,
            "is_weekend": bool(is_weekend),
            "is_night": bool(is_night),
        }
    )


@app.get("/neighborhoods")
def neighborhoods():
    return {"neighborhoods": NEIGHBORHOODS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
