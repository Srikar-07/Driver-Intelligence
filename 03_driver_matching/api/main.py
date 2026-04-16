"""
Drivr — Driver Matching API
Returns ranked list of best-matched drivers for a given customer request.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
import joblib
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "../models")
DATA   = os.path.join(BASE, "../data")

cb_matcher    = joblib.load(os.path.join(MODELS, "cb_matcher.pkl"))
hybrid_ranker = joblib.load(os.path.join(MODELS, "hybrid_ranker.pkl"))
drivers       = pd.read_csv(os.path.join(DATA, "drivers.csv"))
driver_map    = drivers.set_index("driver_id")

CAR_TYPES = ["Sedan", "SUV", "Luxury", "Minivan", "Convertible"]

app = FastAPI(
    title="Drivr Matching API",
    description="Ranks and returns best-matched drivers for a customer trip request.",
    version="1.0.0"
)


class MatchRequest(BaseModel):
    preferred_car_type: Literal["Sedan","SUV","Luxury","Minivan","Convertible"] = Field(..., example="Luxury")
    is_nightlife_user:  int = Field(default=0, ge=0, le=1, example=1)
    is_airport_user:    int = Field(default=0, ge=0, le=1)
    is_event_user:      int = Field(default=0, ge=0, le=1, example=1)
    prefers_spanish:    int = Field(default=0, ge=0, le=1, example=1)
    has_pet:            int = Field(default=0, ge=0, le=1)
    prefers_quiet:      int = Field(default=0, ge=0, le=1)
    price_sensitivity:  float = Field(default=0.5, ge=0.0, le=1.0, example=0.2)
    top_k:              int = Field(default=5, ge=1, le=20)


class DriverMatch(BaseModel):
    rank:           int
    driver_id:      str
    match_score:    float
    rating:         float
    car_type:       str
    experience_yrs: float
    on_time_rate:   float
    speaks_spanish: bool
    allows_pets:    bool
    specialties:    str


class MatchResponse(BaseModel):
    top_matches:    List[DriverMatch]
    total_drivers:  int
    match_strategy: str


@app.get("/")
def root():
    return {"service": "Drivr Matching API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "drivers_available": len(drivers)}

@app.post("/match", response_model=MatchResponse)
def match_drivers(req: MatchRequest):
    customer = {
        "preferred_car_enc": CAR_TYPES.index(req.preferred_car_type),
        "is_nightlife_user": req.is_nightlife_user,
        "is_airport_user":   req.is_airport_user,
        "is_event_user":     req.is_event_user,
        "prefers_spanish":   req.prefers_spanish,
        "has_pet":           req.has_pet,
        "prefers_quiet":     req.prefers_quiet,
        "price_sensitivity": req.price_sensitivity,
    }

    # Step 1: content-based candidates (top 40 pool)
    cb_ranks = cb_matcher.rank_drivers(customer, top_k=40)

    # Step 2: hybrid re-rank
    rows = []
    for _, r in cb_ranks.iterrows():
        did = r["driver_id"]
        if did not in driver_map.index:
            continue
        d = driver_map.loc[did]
        rows.append({
            "driver_id":         did,
            "cb_score":          r["cb_score"],
            "cf_score":          0.5,
            "rating":            d["rating"],
            "on_time_rate":      d["on_time_rate"],
            "cancellation_rate": d["cancellation_rate"],
            "acceptance_rate":   d["acceptance_rate"],
            "experience_years":  d["experience_years"],
        })

    df_pool = pd.DataFrame(rows)
    df_pool["hybrid_score"] = hybrid_ranker.predict(df_pool)
    df_pool = df_pool.sort_values("hybrid_score", ascending=False).head(req.top_k).reset_index(drop=True)

    matches = []
    for rank, (_, row) in enumerate(df_pool.iterrows(), 1):
        did = row["driver_id"]
        d   = driver_map.loc[did]
        matches.append(DriverMatch(
            rank=rank,
            driver_id=did,
            match_score=round(float(row["hybrid_score"]), 3),
            rating=float(d["rating"]),
            car_type=d["car_type"],
            experience_yrs=float(d["experience_years"]),
            on_time_rate=float(d["on_time_rate"]),
            speaks_spanish=bool(d["speaks_spanish"]),
            allows_pets=bool(d["allows_pets"]),
            specialties=str(d["specialties"]),
        ))

    return MatchResponse(
        top_matches=matches,
        total_drivers=len(drivers),
        match_strategy="content-based (cosine) → hybrid gradient boosting re-rank",
    )

@app.get("/drivers/{driver_id}")
def get_driver(driver_id: str):
    if driver_id not in driver_map.index:
        raise HTTPException(404, f"Driver {driver_id} not found")
    d = driver_map.loc[driver_id].to_dict()
    return {"driver_id": driver_id, **d}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
