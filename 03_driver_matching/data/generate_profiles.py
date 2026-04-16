"""
Drivr — Driver & Customer Profile Generator
Generates realistic driver profiles, customer preferences, and trip interaction history.
"""

import numpy as np
import pandas as pd
import random
import json

np.random.seed(42)
random.seed(42)

CAR_TYPES    = ["Sedan", "SUV", "Luxury", "Minivan", "Convertible"]
LANGUAGES    = ["English", "Spanish", "Portuguese", "French", "Creole"]
NEIGHBORHOODS = [
    "South Beach", "Brickell", "Wynwood", "Coral Gables", "Coconut Grove",
    "Little Havana", "Midtown", "Downtown", "Aventura", "Doral"
]
TRIP_TYPES   = ["nightlife", "airport", "errand", "event", "long_distance", "wedding"]
MUSIC_PREFS  = ["none", "pop", "latin", "hip_hop", "classical", "jazz"]


def generate_drivers(n=200):
    drivers = []
    for i in range(n):
        exp_years = round(np.random.exponential(3) + 0.5, 1)
        base_rating = np.clip(np.random.normal(4.5, 0.3), 3.0, 5.0)
        # More experienced → slightly better rating
        rating = round(np.clip(base_rating + exp_years * 0.02, 3.0, 5.0), 2)

        car_type = np.random.choice(CAR_TYPES, p=[0.40, 0.30, 0.15, 0.10, 0.05])
        home_zone = random.choice(NEIGHBORHOODS)

        specialties = random.sample(TRIP_TYPES, k=random.randint(1, 3))
        languages   = [random.choice(LANGUAGES)] + (
            [random.choice(LANGUAGES)] if random.random() < 0.4 else []
        )
        languages = list(set(languages))

        drivers.append({
            "driver_id":          f"DRV{i:04d}",
            "rating":             rating,
            "experience_years":   exp_years,
            "total_trips":        int(exp_years * random.randint(80, 200)),
            "car_type":           car_type,
            "car_type_enc":       CAR_TYPES.index(car_type),
            "home_zone":          home_zone,
            "home_zone_enc":      NEIGHBORHOODS.index(home_zone),
            "specialties":        ",".join(specialties),
            "is_nightlife_spec":  int("nightlife" in specialties),
            "is_airport_spec":    int("airport" in specialties),
            "is_event_spec":      int("event" in specialties),
            "is_wedding_spec":    int("wedding" in specialties),
            "languages":          ",".join(languages),
            "speaks_spanish":     int("Spanish" in languages),
            "speaks_english":     int("English" in languages),
            "acceptance_rate":    round(np.random.beta(8, 2), 2),
            "on_time_rate":       round(np.random.beta(9, 1.5), 2),
            "cancellation_rate":  round(np.random.beta(1, 20), 3),
            "avg_response_sec":   int(np.random.exponential(45) + 10),
            "music_pref":         random.choice(MUSIC_PREFS),
            "allows_pets":        int(random.random() < 0.3),
            "prefers_quiet":      int(random.random() < 0.4),
        })

    return pd.DataFrame(drivers)


def generate_customers(n=500):
    customers = []
    for i in range(n):
        preferred_car = np.random.choice(CAR_TYPES, p=[0.35, 0.30, 0.20, 0.10, 0.05])
        trip_types    = random.sample(TRIP_TYPES, k=random.randint(1, 3))

        customers.append({
            "customer_id":          f"CST{i:04d}",
            "preferred_car_type":   preferred_car,
            "preferred_car_enc":    CAR_TYPES.index(preferred_car),
            "preferred_trip_types": ",".join(trip_types),
            "is_nightlife_user":    int("nightlife" in trip_types),
            "is_airport_user":      int("airport" in trip_types),
            "is_event_user":        int("event" in trip_types),
            "prefers_spanish":      int(random.random() < 0.35),  # Miami is 70% Hispanic
            "has_pet":              int(random.random() < 0.25),
            "prefers_quiet":        int(random.random() < 0.35),
            "price_sensitivity":    round(np.random.beta(3, 3), 2),  # 0=price insensitive
            "avg_trip_distance":    round(np.random.exponential(8) + 2, 1),
            "total_bookings":       random.randint(1, 150),
            "preferred_music":      random.choice(MUSIC_PREFS),
        })

    return pd.DataFrame(customers)


def generate_interactions(drivers, customers, n=5000):
    """Simulate trip interactions with implicit ratings based on compatibility."""
    driver_ids   = drivers["driver_id"].tolist()
    customer_ids = customers["customer_id"].tolist()
    driver_map   = drivers.set_index("driver_id")
    customer_map = customers.set_index("customer_id")

    records = []
    for i in range(n):
        did = random.choice(driver_ids)
        cid = random.choice(customer_ids)
        d   = driver_map.loc[did]
        c   = customer_map.loc[cid]

        # Compute compatibility score (ground truth for training)
        compat = 0.5

        # Car type match
        if d["car_type"] == c["preferred_car_type"]:
            compat += 0.20

        # Language match
        if c["prefers_spanish"] and d["speaks_spanish"]:
            compat += 0.12

        # Pet compatibility
        if c["has_pet"] and d["allows_pets"]:
            compat += 0.10
        elif c["has_pet"] and not d["allows_pets"]:
            compat -= 0.15

        # Quiet preference match
        if c["prefers_quiet"] and d["prefers_quiet"]:
            compat += 0.08

        # Music preference match
        if d["music_pref"] == c["preferred_music"] or d["music_pref"] == "none":
            compat += 0.05

        # Trip type specialization
        if c["is_nightlife_user"] and d["is_nightlife_spec"]:
            compat += 0.10
        if c["is_airport_user"] and d["is_airport_spec"]:
            compat += 0.08
        if c["is_event_user"] and d["is_event_spec"]:
            compat += 0.08

        # Driver quality
        compat += (d["rating"] - 4.0) * 0.10
        compat += d["on_time_rate"] * 0.08
        compat -= d["cancellation_rate"] * 0.20

        # Add noise
        compat = float(np.clip(compat + np.random.normal(0, 0.05), 0.0, 1.0))

        # Simulate explicit rating (1–5) from compatibility
        explicit_rating = min(5, max(1, round(1 + compat * 4 + np.random.normal(0, 0.3))))

        records.append({
            "driver_id":         did,
            "customer_id":       cid,
            "compatibility":     round(compat, 3),
            "explicit_rating":   int(explicit_rating),
            "trip_type":         random.choice(TRIP_TYPES),
            "completed":         int(random.random() < 0.93),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Generating driver profiles...")
    drivers = generate_drivers(200)
    drivers.to_csv("drivers.csv", index=False)
    print(f"  {len(drivers)} drivers | avg rating: {drivers['rating'].mean():.2f}")

    print("Generating customer profiles...")
    customers = generate_customers(500)
    customers.to_csv("customers.csv", index=False)
    print(f"  {len(customers)} customers")

    print("Generating interaction history...")
    interactions = generate_interactions(drivers, customers, 5000)
    interactions.to_csv("interactions.csv", index=False)
    print(f"  {len(interactions)} interactions | avg compatibility: {interactions['compatibility'].mean():.3f}")

    print("\nCar type distribution (drivers):")
    print(drivers["car_type"].value_counts().to_string())
