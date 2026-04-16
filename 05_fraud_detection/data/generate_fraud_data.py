"""
Drivr — Fraud / Anomaly Data Generator
Generates normal trip data + injects realistic fraud patterns.
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

FRAUD_TYPES = [
    "gps_spoofing",        # fake pickup/dropoff location
    "fare_manipulation",   # inflated distance/fare
    "fake_completion",     # trip marked complete but never happened
    "account_takeover",    # login from new device/location
    "promo_abuse",         # rapid bookings to exploit discounts
]

NEIGHBORHOODS = [
    "South Beach","Brickell","Wynwood","Coral Gables","Coconut Grove",
    "Little Havana","Midtown","Downtown","Aventura","Doral"
]


def generate_normal_trip():
    hour      = random.randint(0, 23)
    distance  = round(np.random.exponential(8) + 1.5, 2)
    fare      = round(max(8, 14 + distance * 1.5 + np.random.normal(0, 3)), 2)
    speed     = round(np.random.normal(35, 8), 1)   # km/h, realistic city speed
    duration  = round(distance / max(speed, 5) * 60, 1)  # minutes
    return {
        "distance_km":         min(distance, 55),
        "fare_usd":            min(fare, 200),
        "speed_kmh":           np.clip(speed, 5, 80),
        "duration_min":        np.clip(duration, 1, 120),
        "hour":                hour,
        "is_night":            int(hour >= 22 or hour <= 5),
        "driver_rating":       round(np.clip(np.random.normal(4.5, 0.3), 3.0, 5.0), 2),
        "customer_rating":     round(np.clip(np.random.normal(4.4, 0.4), 3.0, 5.0), 2),
        "trip_count_today":    random.randint(1, 8),
        "new_device":          int(random.random() < 0.05),
        "new_payment_method":  int(random.random() < 0.08),
        "location_jump_km":    round(max(0, np.random.normal(0.5, 0.8)), 2),
        "surge_multiplier":    round(np.random.choice([1.0,1.2,1.5,2.0], p=[0.6,0.2,0.15,0.05]), 1),
        "promo_applied":       int(random.random() < 0.12),
        "cancellations_today": random.randint(0, 2),
        "fare_distance_ratio": 0.0,   # filled below
        "is_fraud":            0,
        "fraud_type":          "none",
    }


def inject_fraud(record, fraud_type):
    r = record.copy()
    r["is_fraud"]   = 1
    r["fraud_type"] = fraud_type

    if fraud_type == "gps_spoofing":
        r["location_jump_km"] = round(np.random.uniform(15, 80), 2)
        r["speed_kmh"]        = round(np.random.uniform(90, 200), 1)  # impossible speed

    elif fraud_type == "fare_manipulation":
        r["distance_km"] = round(np.random.uniform(40, 100), 2)
        r["fare_usd"]    = round(np.random.uniform(150, 500), 2)
        r["duration_min"]= round(np.random.uniform(5, 20), 1)   # too fast for distance

    elif fraud_type == "fake_completion":
        r["duration_min"] = round(np.random.uniform(0.5, 3), 1)  # suspiciously fast
        r["distance_km"]  = round(np.random.uniform(0.1, 1.0), 2)
        r["speed_kmh"]    = round(np.random.uniform(150, 300), 1)

    elif fraud_type == "account_takeover":
        r["new_device"]          = 1
        r["new_payment_method"]  = 1
        r["location_jump_km"]    = round(np.random.uniform(50, 500), 2)  # diff city
        r["hour"]                = random.randint(1, 4)   # odd hours

    elif fraud_type == "promo_abuse":
        r["promo_applied"]       = 1
        r["trip_count_today"]    = random.randint(15, 40)  # too many trips
        r["cancellations_today"] = random.randint(8, 20)
        r["new_payment_method"]  = 1

    return r


def generate_dataset(n_normal=4800, n_fraud=200):
    records = []

    for _ in range(n_normal):
        r = generate_normal_trip()
        r["fare_distance_ratio"] = round(r["fare_usd"] / max(r["distance_km"], 0.1), 2)
        records.append(r)

    for _ in range(n_fraud):
        base  = generate_normal_trip()
        ftype = random.choice(FRAUD_TYPES)
        r     = inject_fraud(base, ftype)
        r["fare_distance_ratio"] = round(r["fare_usd"] / max(r["distance_km"], 0.1), 2)
        records.append(r)

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    df["trip_id"] = [f"FR{i:05d}" for i in range(len(df))]
    return df


if __name__ == "__main__":
    print("Generating fraud dataset...")
    df = generate_dataset()
    df.to_csv("fraud_trips.csv", index=False)
    print(f"  {len(df)} trips | Fraud rate: {df['is_fraud'].mean():.1%}")
    print("\nFraud type breakdown:")
    print(df[df["is_fraud"]==1]["fraud_type"].value_counts().to_string())
    print("\nKey stats comparison:")
    print(df.groupby("is_fraud")[["fare_usd","distance_km","speed_kmh","location_jump_km"]].mean().round(2).to_string())
