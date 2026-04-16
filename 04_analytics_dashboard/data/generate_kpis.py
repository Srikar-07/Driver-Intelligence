"""
Drivr — Business KPI Data Generator
Generates daily/weekly metrics for the analytics dashboard.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

NEIGHBORHOODS = [
    "South Beach","Brickell","Wynwood","Coral Gables","Coconut Grove",
    "Little Havana","Midtown","Downtown","Aventura","Doral"
]
CAR_TYPES  = ["Sedan","SUV","Luxury","Minivan","Convertible"]
TRIP_TYPES = ["nightlife","airport","errand","event","long_distance","wedding"]


def generate_daily_kpis(days=180):
    """180 days of daily business metrics — Jan to Jun 2024."""
    start = datetime(2024, 1, 1)
    records = []
    base_trips  = 60   # starting daily trips
    base_rev    = 40   # avg fare $40

    for d in range(days):
        dt  = start + timedelta(days=d)
        dow = dt.weekday()
        mon = dt.month
        is_weekend = dow >= 5

        # Organic growth trend
        growth = 1 + (d / days) * 0.8   # 80% growth over 6 months

        # Seasonality — Miami peaks Dec-Mar, slight dip Apr-Jun
        season = 1.0
        if mon in [1, 2, 3]:  season = 1.25
        if mon in [4, 5, 6]:  season = 0.95

        weekend_boost = 1.4 if is_weekend else 1.0

        trips     = int(base_trips * growth * season * weekend_boost * np.random.uniform(0.88, 1.12))
        avg_fare  = base_rev * (1 + d * 0.0005) * np.random.uniform(0.93, 1.07)
        revenue   = round(trips * avg_fare, 2)
        drivers   = max(10, int(trips * 0.18 * np.random.uniform(0.85, 1.15)))
        util_rate = round(min(0.97, trips / (drivers * 6 + 1) * np.random.uniform(0.9, 1.1)), 3)
        cancel    = round(np.random.beta(2, 25), 3)
        rating    = round(np.random.normal(4.55, 0.08), 2)
        new_cust  = int(np.random.poisson(5 + d * 0.06))

        records.append({
            "date":              dt.strftime("%Y-%m-%d"),
            "day_of_week":       dow,
            "is_weekend":        int(is_weekend),
            "month":             mon,
            "total_trips":       trips,
            "total_revenue":     revenue,
            "avg_fare":          round(avg_fare, 2),
            "active_drivers":    drivers,
            "driver_utilization":util_rate,
            "cancellation_rate": cancel,
            "avg_rating":        rating,
            "new_customers":     new_cust,
        })

    df = pd.DataFrame(records)
    df["cumulative_revenue"] = df["total_revenue"].cumsum().round(2)
    df["cumulative_trips"]   = df["total_trips"].cumsum()
    return df


def generate_trip_log(n=5000):
    """Individual trip log for granular analysis."""
    start = datetime(2024, 1, 1)
    records = []
    for i in range(n):
        dt  = start + timedelta(days=random.randint(0, 179),
                                hours=random.randint(0, 23),
                                minutes=random.randint(0, 59))
        hood      = random.choice(NEIGHBORHOODS)
        car       = np.random.choice(CAR_TYPES, p=[0.40,0.30,0.15,0.10,0.05])
        trip_type = random.choice(TRIP_TYPES)
        dist      = round(np.random.exponential(8) + 1.5, 1)
        fare      = round(max(8, np.random.normal(40, 12)), 2)
        rating    = min(5, max(1, round(np.random.normal(4.5, 0.4))))

        records.append({
            "trip_id":       f"T{i:05d}",
            "datetime":      dt.strftime("%Y-%m-%d %H:%M"),
            "date":          dt.strftime("%Y-%m-%d"),
            "month":         dt.month,
            "hour":          dt.hour,
            "neighborhood":  hood,
            "car_type":      car,
            "trip_type":     trip_type,
            "distance_km":   dist,
            "fare_usd":      fare,
            "driver_rating": rating,
            "completed":     int(random.random() < 0.93),
        })

    return pd.DataFrame(records)


def generate_driver_performance(n=80):
    """Weekly driver performance snapshots."""
    records = []
    for i in range(n):
        trips  = random.randint(20, 120)
        rating = round(np.clip(np.random.normal(4.5, 0.35), 3.0, 5.0), 2)
        records.append({
            "driver_id":       f"DRV{i:04d}",
            "weekly_trips":    trips,
            "weekly_revenue":  round(trips * np.random.normal(40, 5), 2),
            "avg_rating":      rating,
            "on_time_rate":    round(np.random.beta(9, 1.5), 2),
            "cancellation_rate": round(np.random.beta(1, 20), 3),
            "car_type":        np.random.choice(CAR_TYPES, p=[0.40,0.30,0.15,0.10,0.05]),
            "active_days":     random.randint(3, 7),
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Generating KPI data...")
    daily = generate_daily_kpis(180)
    daily.to_csv("daily_kpis.csv", index=False)
    print(f"  {len(daily)} days | Total revenue: ${daily['total_revenue'].sum():,.0f}")

    print("Generating trip log...")
    trips = generate_trip_log(5000)
    trips.to_csv("trip_log.csv", index=False)
    print(f"  {len(trips)} trips | Avg fare: ${trips['fare_usd'].mean():.2f}")

    print("Generating driver performance...")
    drivers = generate_driver_performance(80)
    drivers.to_csv("driver_performance.csv", index=False)
    print(f"  {len(drivers)} drivers | Avg rating: {drivers['avg_rating'].mean():.2f}")
