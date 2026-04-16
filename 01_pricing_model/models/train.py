"""
Drivr Dynamic Pricing Model — Training Pipeline
Trains XGBoost and LightGBM models, compares performance, saves best model.
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

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/rides.csv")
MODEL_DIR = os.path.dirname(__file__)


def load_and_prepare(path):
    df = pd.read_csv(path)

    # Encode neighborhood
    le = LabelEncoder()
    df["pickup_enc"] = le.fit_transform(df["pickup_neighborhood"])
    df["dropoff_enc"] = le.fit_transform(df["dropoff_neighborhood"])

    features = [
        "hour", "day_of_week", "month", "is_weekend", "is_night",
        "is_rush_hour", "distance_km", "weather_encoded",
        "driver_supply_ratio", "event_demand_multiplier",
        "active_drivers", "pending_requests", "demand_supply_ratio",
        "pickup_enc", "dropoff_enc"
    ]

    X = df[features]
    y = df["price_usd"]
    return X, y, le, features


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    print(f"\n{name}")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    return {"model": name, "MAE": round(mae,2), "RMSE": round(rmse,2),
            "R2": round(r2,4), "MAPE": round(mape,2)}, preds


def plot_feature_importance(model, features, name, outdir):
    imp = model.feature_importances_
    fi = pd.Series(imp, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="barh", ax=ax, color="#1D9E75")
    ax.set_title(f"Feature Importance — {name}", fontsize=13)
    ax.set_xlabel("Importance score")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(outdir, f"feature_importance_{name.lower().replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_predictions(y_test, preds_xgb, preds_lgbm, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, preds, name, color in zip(
        axes,
        [preds_xgb, preds_lgbm],
        ["XGBoost", "LightGBM"],
        ["#378ADD", "#1D9E75"]
    ):
        ax.scatter(y_test, preds, alpha=0.3, s=10, color=color)
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual Price ($)")
        ax.set_ylabel("Predicted Price ($)")
        ax.set_title(f"{name} — Actual vs Predicted")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(outdir, "actual_vs_predicted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 50)
    print("  Drivr Dynamic Pricing — Training Pipeline")
    print("=" * 50)

    X, y, le, features = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nDataset: {len(X)} trips | Train: {len(X_train)} | Test: {len(X_test)}")

    # XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_metrics, preds_xgb = evaluate(xgb, X_test, y_test, "XGBoost")

    # LightGBM
    print("\nTraining LightGBM...")
    lgbm = LGBMRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    lgbm_metrics, preds_lgbm = evaluate(lgbm, X_test, y_test, "LightGBM")

    # Plots
    print("\nGenerating plots...")
    outdir = os.path.join(MODEL_DIR, "../notebooks")
    os.makedirs(outdir, exist_ok=True)
    plot_feature_importance(xgb,  features, "XGBoost",  outdir)
    plot_feature_importance(lgbm, features, "LightGBM", outdir)
    plot_predictions(y_test, preds_xgb, preds_lgbm, outdir)

    # Pick best by MAE
    best_name = "XGBoost" if xgb_metrics["MAE"] <= lgbm_metrics["MAE"] else "LightGBM"
    best_model = xgb if best_name == "XGBoost" else lgbm
    print(f"\nBest model: {best_name}")

    # Save
    joblib.dump(best_model, os.path.join(MODEL_DIR, "pricing_model.pkl"))
    joblib.dump(le,         os.path.join(MODEL_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODEL_DIR, "features.json"), "w") as f:
        json.dump(features, f)

    results = {"xgboost": xgb_metrics, "lightgbm": lgbm_metrics, "best": best_name}
    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved: pricing_model.pkl, label_encoder.pkl, features.json, results.json")
    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
