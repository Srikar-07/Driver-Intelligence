"""
Drivr — Synthetic Ride Data Generator
Generates realistic Miami ride data for pricing model training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

MIAMI_NEIGHBORHOODS = [
    "South Beach", "Brickell", "Wynwood", "Coral Gables",
    "Coconut Grove", "Little Havana", "Midtown", "Downtown",
    "Aventura", "Doral"
]

MIAMI_EVENTS = {
    "Ultra Music Festival": {"months": [3], "demand_boost": 2.8},
    "Art Basel": {"months": [12], "demand_boost": 2.5},
    "Miami Grand Prix": {"months": [5], "demand_boost": 2.2},
    "Heat Game": {"months": [1,2,3,4,10,11,12], "demand_boost": 1.6},
    "Dolphins Game": {"months": [9,10,11,12], "demand_boost": 1.5},
}


def generate_weather():
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    weights = [0.55, 0.25, 0.15, 0.05]
    return np.random.choice(conditions, p=weights)


def compute_price(row):
    base = 14.0
    # Distance — lower per-km rate
    price = base + row["distance_km"] * 1.5
    # Time of day surge
    hour = row["hour"]
    if hour in range(22, 24) or hour in range(0, 3):   # late night
        price *= 1.5
    elif hour in range(17, 20):                          # evening rush
        price *= 1.2
    elif hour in range(7, 10):                           # morning rush
        price *= 1.15
    elif hour in range(3, 6):                            # dead hours
        price *= 0.85
    # Weekend
    if row["day_of_week"] >= 5:
        price *= 1.1
    # Weather
    if row["weather"] == "rainy":
        price *= 1.2
    elif row["weather"] == "stormy":
        price *= 1.4
    # Event boost — dampened so rare spikes don't skew mean
    event_mult = 1 + (row["event_demand_multiplier"] - 1) * 0.4
    price *= event_mult
    # Driver supply (lower supply → higher price)
    price *= (1 + (1 - row["driver_supply_ratio"]) * 0.25)
    # Neighborhood premium
    premium_zones = ["South Beach", "Brickell", "Coral Gables"]
    if row["pickup_neighborhood"] in premium_zones:
        price *= 1.08
    # Add noise
    price += np.random.normal(0, 1.0)
    return round(max(price, 6.0), 2)


def generate_dataset(n=5000):
    start_date = datetime(2023, 1, 1)
    records = []

    for _ in range(n):
        dt = start_date + timedelta(
            days=random.randint(0, 364),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        hour = dt.hour
        dow = dt.weekday()
        month = dt.month

        # Event multiplier
        event_mult = 1.0
        for event, meta in MIAMI_EVENTS.items():
            if month in meta["months"] and random.random() < 0.15:
                event_mult = max(event_mult, meta["demand_boost"])
                break

        distance = round(np.random.exponential(scale=8) + 1.5, 2)
        distance = min(distance, 60)

        weather = generate_weather()
        driver_supply = round(np.random.beta(5, 2), 2)  # skewed towards available

        record = {
            "trip_id": f"TR{_:05d}",
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": hour,
            "day_of_week": dow,
            "month": month,
            "is_weekend": int(dow >= 5),
            "is_night": int(hour >= 22 or hour <= 5),
            "is_rush_hour": int(hour in list(range(7,10)) + list(range(17,20))),
            "distance_km": distance,
            "pickup_neighborhood": random.choice(MIAMI_NEIGHBORHOODS),
            "dropoff_neighborhood": random.choice(MIAMI_NEIGHBORHOODS),
            "weather": weather,
            "weather_encoded": {"sunny": 0, "cloudy": 1, "rainy": 2, "stormy": 3}[weather],
            "driver_supply_ratio": driver_supply,
            "event_demand_multiplier": round(event_mult, 2),
            "active_drivers": int(driver_supply * 80),
            "pending_requests": random.randint(5, 120),
        }

        record["price_usd"] = compute_price(record)
        records.append(record)

    df = pd.DataFrame(records)
    # Demand ratio feature
    df["demand_supply_ratio"] = round(df["pending_requests"] / (df["active_drivers"] + 1), 2)
    return df


if __name__ == "__main__":
    print("Generating Miami ride dataset...")
    df = generate_dataset(5000)
    df.to_csv("rides.csv", index=False)
    print(f"Generated {len(df)} trips")
    print(f"Price range: ${df['price_usd'].min():.2f} — ${df['price_usd'].max():.2f}")
    print(f"Average price: ${df['price_usd'].mean():.2f}")
    print(df.head(3).to_string())
