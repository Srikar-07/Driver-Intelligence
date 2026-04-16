"""
Drivr — Geospatial Demand Data Generator
Generates realistic Miami trip data with GPS coordinates per neighborhood.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json

np.random.seed(42)
random.seed(42)

# Real Miami neighborhood centroids with spread radius (degrees)
NEIGHBORHOODS = {
    "South Beach":    {"lat": 25.7907, "lon": -80.1300, "radius": 0.018, "nightlife": 0.95, "tourism": 0.95},
    "Brickell":       {"lat": 25.7589, "lon": -80.1918, "radius": 0.015, "nightlife": 0.75, "tourism": 0.50},
    "Wynwood":        {"lat": 25.8008, "lon": -80.1994, "radius": 0.012, "nightlife": 0.85, "tourism": 0.80},
    "Coral Gables":   {"lat": 25.7215, "lon": -80.2684, "radius": 0.020, "nightlife": 0.30, "tourism": 0.40},
    "Coconut Grove":  {"lat": 25.7310, "lon": -80.2380, "radius": 0.015, "nightlife": 0.60, "tourism": 0.55},
    "Little Havana":  {"lat": 25.7665, "lon": -80.2294, "radius": 0.013, "nightlife": 0.50, "tourism": 0.60},
    "Midtown":        {"lat": 25.8100, "lon": -80.1950, "radius": 0.010, "nightlife": 0.65, "tourism": 0.40},
    "Downtown":       {"lat": 25.7748, "lon": -80.1977, "radius": 0.014, "nightlife": 0.55, "tourism": 0.65},
    "Aventura":       {"lat": 25.9564, "lon": -80.1394, "radius": 0.018, "nightlife": 0.40, "tourism": 0.50},
    "Doral":          {"lat": 25.8195, "lon": -80.3556, "radius": 0.020, "nightlife": 0.20, "tourism": 0.25},
}

EVENTS = {
    "Ultra Music Festival": {"months": [3],       "neighborhoods": ["Brickell", "Downtown"],       "boost": 3.0},
    "Art Basel":            {"months": [12],      "neighborhoods": ["Wynwood", "South Beach"],     "boost": 2.8},
    "Miami Grand Prix":     {"months": [5],       "neighborhoods": ["Brickell", "Downtown"],       "boost": 2.5},
    "Heat Game":            {"months": list(range(1,13)), "neighborhoods": ["Brickell", "Downtown"], "boost": 1.7},
    "Dolphins Game":        {"months": [9,10,11,12],  "neighborhoods": ["Downtown"],               "boost": 1.6},
}

def jitter_coords(lat, lon, radius):
    """Add realistic Gaussian scatter around neighborhood centroid."""
    dlat = np.random.normal(0, radius * 0.4)
    dlon = np.random.normal(0, radius * 0.4)
    return round(lat + dlat, 6), round(lon + dlon, 6)

def demand_score(hour, dow, month, neighborhood):
    """Compute expected demand score 0–1 for a given context."""
    n = NEIGHBORHOODS[neighborhood]
    base = 0.3

    # Time of day
    if hour in range(22, 24) or hour in range(0, 3):
        base += 0.4 * n["nightlife"]
    elif hour in range(17, 20):
        base += 0.25
    elif hour in range(7, 10):
        base += 0.20
    elif hour in range(10, 16):
        base += 0.10 * n["tourism"]
    else:
        base += 0.05

    # Weekend boost
    if dow >= 5:
        base += 0.15 * n["nightlife"]

    # Seasonal — Dec/Mar peak for Miami tourism
    if month in [12, 1, 2, 3]:
        base += 0.10 * n["tourism"]

    # Event boost
    for event, meta in EVENTS.items():
        if month in meta["months"] and neighborhood in meta["neighborhoods"]:
            if random.random() < 0.12:
                base *= meta["boost"] * 0.5

    return min(round(base + np.random.normal(0, 0.05), 3), 1.0)

def generate_trips(n=8000):
    start = datetime(2023, 1, 1)
    records = []

    for i in range(n):
        dt = start + timedelta(
            days=random.randint(0, 364),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        hour, dow, month = dt.hour, dt.weekday(), dt.month

        # Weight pickup by demand score
        hoods = list(NEIGHBORHOODS.keys())
        weights = [max(demand_score(hour, dow, month, h), 0.05) for h in hoods]
        total = sum(weights)
        weights = [w / total for w in weights]

        pickup = np.random.choice(hoods, p=weights)
        dropoff = np.random.choice(hoods)

        plat, plon = jitter_coords(*[NEIGHBORHOODS[pickup]["lat"], NEIGHBORHOODS[pickup]["lon"]],
                                    NEIGHBORHOODS[pickup]["radius"])
        dlat, dlon = jitter_coords(*[NEIGHBORHOODS[dropoff]["lat"], NEIGHBORHOODS[dropoff]["lon"]],
                                    NEIGHBORHOODS[dropoff]["radius"])

        ds = demand_score(hour, dow, month, pickup)

        records.append({
            "trip_id":            f"GEO{i:05d}",
            "datetime":           dt.strftime("%Y-%m-%d %H:%M:%S"),
            "hour":               hour,
            "day_of_week":        dow,
            "month":              month,
            "is_weekend":         int(dow >= 5),
            "pickup_neighborhood": pickup,
            "dropoff_neighborhood": dropoff,
            "pickup_lat":         plat,
            "pickup_lon":         plon,
            "dropoff_lat":        dlat,
            "dropoff_lon":        dlon,
            "demand_score":       ds,
            "trip_count":         1,
        })

    return pd.DataFrame(records)

def generate_grid_demand(resolution=0.005):
    """Generate demand predictions over a lat/lon grid for heatmap."""
    lat_range = np.arange(25.70, 25.97, resolution)
    lon_range = np.arange(-80.37, -80.11, resolution)

    grid = []
    hours_to_sample = [2, 8, 12, 18, 22]  # night, morning, midday, evening, late night

    for hour in hours_to_sample:
        for lat in lat_range:
            for lon in lon_range:
                # Find nearest neighborhood
                min_dist = float("inf")
                nearest = None
                for name, meta in NEIGHBORHOODS.items():
                    dist = ((lat - meta["lat"])**2 + (lon - meta["lon"])**2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest = name

                if min_dist > 0.06:  # Too far from any neighborhood
                    continue

                ds = demand_score(hour, 5, 12, nearest)  # Saturday in December
                decay = max(0, 1 - (min_dist / 0.04))   # falloff from centroid
                grid.append({
                    "lat": round(lat, 4),
                    "lon": round(lon, 4),
                    "hour": hour,
                    "demand": round(ds * decay, 3),
                    "neighborhood": nearest,
                })

    return pd.DataFrame(grid)


if __name__ == "__main__":
    print("Generating trip data...")
    trips = generate_trips(8000)
    trips.to_csv("trips_geo.csv", index=False)
    print(f"  {len(trips)} trips saved to trips_geo.csv")

    print("Generating demand grid...")
    grid = generate_grid_demand()
    grid.to_csv("demand_grid.csv", index=False)
    print(f"  {len(grid)} grid points saved to demand_grid.csv")

    print("\nDemand by neighborhood (avg):")
    print(trips.groupby("pickup_neighborhood")["demand_score"].mean().sort_values(ascending=False).round(3).to_string())
