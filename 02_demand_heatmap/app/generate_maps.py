"""
Drivr — Interactive Miami Demand Heatmap
Generates Folium HTML maps showing ride demand across Miami neighborhoods.
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, HeatMapWithTime
import json
import os

BASE     = os.path.dirname(__file__)
TRIPS    = os.path.join(BASE, "../data/trips_geo.csv")
GRID     = os.path.join(BASE, "../data/demand_grid.csv")
OUT      = os.path.join(BASE, "../notebooks")
os.makedirs(OUT, exist_ok=True)

MIAMI_CENTER = [25.805, -80.220]

NEIGHBORHOOD_COORDS = {
    "South Beach":   [25.7907, -80.1300],
    "Brickell":      [25.7589, -80.1918],
    "Wynwood":       [25.8008, -80.1994],
    "Coral Gables":  [25.7215, -80.2684],
    "Coconut Grove": [25.7310, -80.2380],
    "Little Havana": [25.7665, -80.2294],
    "Midtown":       [25.8100, -80.1950],
    "Downtown":      [25.7748, -80.1977],
    "Aventura":      [25.9564, -80.1394],
    "Doral":         [25.8195, -80.3556],
}

COLORS = {
    "South Beach":   "#378ADD",
    "Brickell":      "#1D9E75",
    "Wynwood":       "#D85A30",
    "Coral Gables":  "#7F77DD",
    "Coconut Grove": "#BA7517",
    "Little Havana": "#D4537E",
    "Midtown":       "#639922",
    "Downtown":      "#E24B4A",
    "Aventura":      "#5DCAA5",
    "Doral":         "#888780",
}


def make_base_map():
    return folium.Map(
        location=MIAMI_CENTER,
        zoom_start=12,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )


def map_1_trip_density(trips):
    """Static heatmap of all pickup locations."""
    m = make_base_map()
    heat_data = [[r.pickup_lat, r.pickup_lon, r.demand_score]
                 for _, r in trips.iterrows()]
    HeatMap(
        heat_data,
        radius=14, blur=18, max_zoom=14,
        gradient={0.2: "#0C447C", 0.5: "#EF9F27", 0.8: "#E24B4A", 1.0: "#FCEBEB"},
        min_opacity=0.4,
    ).add_to(m)

    # Neighborhood markers
    for name, coords in NEIGHBORHOOD_COORDS.items():
        avg = trips[trips["pickup_neighborhood"] == name]["demand_score"].mean()
        folium.CircleMarker(
            location=coords,
            radius=10,
            color=COLORS[name],
            fill=True, fill_color=COLORS[name], fill_opacity=0.85,
            tooltip=folium.Tooltip(f"<b>{name}</b><br>Avg demand: {avg:.2f}"),
        ).add_to(m)
        folium.Marker(
            location=[coords[0] + 0.005, coords[1]],
            icon=folium.DivIcon(
                html=f'<div style="color:white;font-size:11px;font-weight:600;'
                     f'text-shadow:1px 1px 2px #000;white-space:nowrap">{name}</div>',
                icon_size=(120, 20), icon_anchor=(60, 10)
            )
        ).add_to(m)

    title = """<div style="position:fixed;top:16px;left:50%;transform:translateX(-50%);
        background:rgba(0,0,0,0.75);color:white;padding:10px 20px;
        border-radius:8px;font-size:14px;font-weight:600;z-index:9999">
        🚗 Drivr — Miami Trip Pickup Density (8,000 trips)</div>"""
    m.get_root().html.add_child(folium.Element(title))
    return m


def map_2_hourly_heatmap(grid):
    """Time-animated heatmap showing demand by hour of day."""
    m = make_base_map()
    hours = sorted(grid["hour"].unique())
    heat_frames = []
    for h in hours:
        frame = grid[grid["hour"] == h]
        heat_frames.append(
            [[r.lat, r.lon, r.demand] for _, r in frame.iterrows() if r.demand > 0.05]
        )

    HeatMapWithTime(
        heat_frames,
        index=[f"{h:02d}:00" for h in hours],
        radius=18, blur=20,
        gradient={0.2: "#0C447C", 0.5: "#EF9F27", 1.0: "#E24B4A"},
        min_opacity=0.3,
        max_opacity=0.9,
        auto_play=True,
        display_index=True,
    ).add_to(m)

    title = """<div style="position:fixed;top:16px;left:50%;transform:translateX(-50%);
        background:rgba(0,0,0,0.75);color:white;padding:10px 20px;
        border-radius:8px;font-size:14px;font-weight:600;z-index:9999">
        🕐 Drivr — Demand by Hour (Saturday in December · press play)</div>"""
    m.get_root().html.add_child(folium.Element(title))
    return m


def map_3_neighborhood_bubbles(trips):
    """Bubble map: size = trip volume, color = avg demand."""
    m = make_base_map()
    summary = trips.groupby("pickup_neighborhood").agg(
        trips=("trip_id","count"),
        avg_demand=("demand_score","mean")
    ).reset_index()

    max_trips = summary["trips"].max()
    for _, row in summary.iterrows():
        coords = NEIGHBORHOOD_COORDS[row["pickup_neighborhood"]]
        radius = 10 + 35 * (row["trips"] / max_trips)
        color  = COLORS[row["pickup_neighborhood"]]

        folium.CircleMarker(
            location=coords,
            radius=radius,
            color=color, fill=True,
            fill_color=color, fill_opacity=0.6,
            weight=2,
            tooltip=folium.Tooltip(
                f"<b>{row['pickup_neighborhood']}</b><br>"
                f"Trips: {row['trips']:,}<br>"
                f"Avg demand: {row['avg_demand']:.2f}"
            )
        ).add_to(m)
        folium.Marker(
            location=coords,
            icon=folium.DivIcon(
                html=f'<div style="color:white;font-size:10px;font-weight:700;'
                     f'text-align:center;text-shadow:1px 1px 2px #000">'
                     f'{row["pickup_neighborhood"]}<br>{row["trips"]:,}</div>',
                icon_size=(110, 30), icon_anchor=(55, 15)
            )
        ).add_to(m)

    title = """<div style="position:fixed;top:16px;left:50%;transform:translateX(-50%);
        background:rgba(0,0,0,0.75);color:white;padding:10px 20px;
        border-radius:8px;font-size:14px;font-weight:600;z-index:9999">
        📍 Drivr — Trip Volume by Neighborhood (bubble size = volume)</div>"""
    m.get_root().html.add_child(folium.Element(title))
    return m


def main():
    print("Loading data...")
    trips = pd.read_csv(TRIPS)
    grid  = pd.read_csv(GRID)

    print("Generating Map 1: Trip pickup density heatmap...")
    m1 = map_1_trip_density(trips)
    p1 = os.path.join(OUT, "map_01_pickup_density.html")
    m1.save(p1)
    print(f"  Saved: {p1}")

    print("Generating Map 2: Hourly demand animation...")
    m2 = map_2_hourly_heatmap(grid)
    p2 = os.path.join(OUT, "map_02_hourly_animation.html")
    m2.save(p2)
    print(f"  Saved: {p2}")

    print("Generating Map 3: Neighborhood bubble map...")
    m3 = map_3_neighborhood_bubbles(trips)
    p3 = os.path.join(OUT, "map_03_neighborhood_bubbles.html")
    m3.save(p3)
    print(f"  Saved: {p3}")

    print("\nAll 3 maps generated. Open the HTML files in your browser.")


if __name__ == "__main__":
    main()
