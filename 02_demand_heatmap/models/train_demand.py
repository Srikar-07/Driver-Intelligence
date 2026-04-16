"""
Drivr — Demand Forecasting Model
Predicts trip demand per neighborhood per hour using Random Forest.
Outputs feature importance, model metrics, and hourly forecast curves.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

BASE     = os.path.dirname(__file__)
DATA     = os.path.join(BASE, "../data/trips_geo.csv")
OUT_DIR  = os.path.join(BASE, "../notebooks")
os.makedirs(OUT_DIR, exist_ok=True)

NEIGHBORHOODS = [
    "South Beach","Brickell","Wynwood","Coral Gables","Coconut Grove",
    "Little Havana","Midtown","Downtown","Aventura","Doral"
]

def load_features(df):
    le = LabelEncoder()
    df["neighborhood_enc"] = le.fit_transform(df["pickup_neighborhood"])

    # Cyclical time encoding — avoids discontinuity at hour 23→0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    features = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_weekend", "neighborhood_enc"
    ]
    return df, features, le

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 1)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"\n{name}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    return {"model": name, "MAE": round(mae,4), "RMSE": round(rmse,4), "R2": round(r2,4)}, preds

def plot_hourly_demand(model, le, features_list, outdir):
    """Plot predicted demand curves for top neighborhoods across 24 hours."""
    hours = list(range(24))
    top_hoods = ["South Beach", "Brickell", "Wynwood", "Downtown", "Doral"]
    colors = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD", "#888780"]

    fig, ax = plt.subplots(figsize=(11, 5))

    for hood, color in zip(top_hoods, colors):
        enc = le.transform([hood])[0]
        preds = []
        for h in hours:
            row = {
                "hour_sin":       np.sin(2 * np.pi * h / 24),
                "hour_cos":       np.cos(2 * np.pi * h / 24),
                "dow_sin":        np.sin(2 * np.pi * 5 / 7),   # Saturday
                "dow_cos":        np.cos(2 * np.pi * 5 / 7),
                "month_sin":      np.sin(2 * np.pi * 12 / 12), # December
                "month_cos":      np.cos(2 * np.pi * 12 / 12),
                "is_weekend":     1,
                "neighborhood_enc": enc,
            }
            X = pd.DataFrame([[row[f] for f in features_list]], columns=features_list)
            preds.append(float(model.predict(X)[0]))

        ax.plot(hours, preds, label=hood, color=color, lw=2.2, marker="o", markersize=3)

    ax.set_xlabel("Hour of day", fontsize=12)
    ax.set_ylabel("Predicted demand (0–1)", fontsize=12)
    ax.set_title("Predicted demand by neighborhood — Saturday in December", fontsize=13)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=30, ha="right")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, 1)
    ax.axvspan(22, 24, alpha=0.07, color="#378ADD", label="_Late night")
    ax.axvspan(0,  3,  alpha=0.07, color="#378ADD")
    plt.tight_layout()
    path = os.path.join(outdir, "hourly_demand_forecast.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def plot_neighborhood_heatmap(df, outdir):
    """Pivot: neighborhood × hour → avg demand score."""
    pivot = df.groupby(["pickup_neighborhood","hour"])["demand_score"].mean().unstack()

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3,
                cbar_kws={"label": "Avg demand score"}, vmin=0, vmax=1)
    ax.set_title("Demand heatmap — Neighborhood × Hour of day", fontsize=13)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    plt.tight_layout()
    path = os.path.join(outdir, "neighborhood_hour_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def plot_feature_importance(model, features, outdir):
    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    fi.plot(kind="barh", ax=ax, color="#7F77DD")
    ax.set_title("Feature importance — Demand model", fontsize=13)
    ax.set_xlabel("Importance")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(outdir, "demand_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def main():
    print("=" * 52)
    print("  Drivr Demand Forecasting — Training Pipeline")
    print("=" * 52)

    df = pd.read_csv(DATA)
    df, features, le = load_features(df)

    X = df[features]
    y = df["demand_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nDataset: {len(df)} trips | Features: {features}")

    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_metrics, _ = evaluate("Random Forest", rf, X_test, y_test)

    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    gb_metrics, _ = evaluate("Gradient Boosting", gb, X_test, y_test)

    best = rf if rf_metrics["R2"] >= gb_metrics["R2"] else gb
    best_name = "Random Forest" if rf_metrics["R2"] >= gb_metrics["R2"] else "Gradient Boosting"
    print(f"\nBest model: {best_name}")

    print("\nGenerating plots...")
    plot_hourly_demand(best, le, features, OUT_DIR)
    plot_neighborhood_heatmap(df, OUT_DIR)
    plot_feature_importance(best, features, OUT_DIR)

    # Save
    model_dir = BASE
    joblib.dump(best, os.path.join(model_dir, "demand_model.pkl"))
    joblib.dump(le,   os.path.join(model_dir, "neighborhood_encoder.pkl"))
    with open(os.path.join(model_dir, "demand_features.json"), "w") as f:
        json.dump(features, f)
    results = {"random_forest": rf_metrics, "gradient_boosting": gb_metrics, "best": best_name}
    with open(os.path.join(model_dir, "demand_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved: demand_model.pkl, neighborhood_encoder.pkl, demand_features.json")
    print("\nDone!")
    return results

if __name__ == "__main__":
    main()
