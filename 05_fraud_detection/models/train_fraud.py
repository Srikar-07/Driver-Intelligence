"""
Drivr — Fraud Detection Pipeline
Three-model ensemble:
  1. Isolation Forest (unsupervised anomaly detection)
  2. Autoencoder (deep anomaly detection via reconstruction error)
  3. XGBoost (supervised, uses fraud labels for training)
  → Ensemble: weighted average of all three scores
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

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, RocCurveDisplay, precision_recall_curve)
from xgboost import XGBClassifier

BASE    = os.path.dirname(__file__)
DATA    = os.path.join(BASE, "../data/fraud_trips.csv")
OUT     = os.path.join(BASE, "../notebooks")
os.makedirs(OUT, exist_ok=True)

FEATURES = [
    "distance_km", "fare_usd", "speed_kmh", "duration_min",
    "hour", "is_night", "driver_rating", "customer_rating",
    "trip_count_today", "new_device", "new_payment_method",
    "location_jump_km", "surge_multiplier", "promo_applied",
    "cancellations_today", "fare_distance_ratio",
]


# ─────────────────────────────────────────────
# Simple Autoencoder (NumPy — no torch/keras dependency)
# ─────────────────────────────────────────────

class SimpleAutoencoder:
    """Shallow autoencoder using NumPy for anomaly detection via reconstruction error."""

    def __init__(self, input_dim, hidden_dim=8, latent_dim=4, lr=0.01, epochs=60):
        self.lr, self.epochs = lr, epochs
        # Encoder weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b2 = np.zeros(latent_dim)
        # Decoder weights
        self.W3 = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b3 = np.zeros(hidden_dim)
        self.W4 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b4 = np.zeros(input_dim)
        self.threshold = None

    @staticmethod
    def relu(x):       return np.maximum(0, x)
    @staticmethod
    def relu_grad(x):  return (x > 0).astype(float)

    def forward(self, X):
        self.h1 = self.relu(X @ self.W1 + self.b1)
        self.z  = self.relu(self.h1 @ self.W2 + self.b2)
        self.h2 = self.relu(self.z @ self.W3 + self.b3)
        self.Xh = self.h2 @ self.W4 + self.b4
        return self.Xh

    def fit(self, X):
        for ep in range(self.epochs):
            Xh   = self.forward(X)
            loss = ((X - Xh) ** 2).mean()
            # Backprop
            dL   = -2 * (X - Xh) / X.shape[0]
            dW4  = self.h2.T @ dL;         db4 = dL.sum(0)
            dh2  = dL @ self.W4.T * self.relu_grad(self.h2)
            dW3  = self.z.T @ dh2;         db3 = dh2.sum(0)
            dz   = dh2 @ self.W3.T * self.relu_grad(self.z)
            dW2  = self.h1.T @ dz;         db2 = dz.sum(0)
            dh1  = dz @ self.W2.T * self.relu_grad(self.h1)
            dW1  = X.T @ dh1;              db1 = dh1.sum(0)
            for W, dW in [(self.W1,dW1),(self.W2,dW2),(self.W3,dW3),(self.W4,dW4)]:
                W -= self.lr * dW
            for b, db in [(self.b1,db1),(self.b2,db2),(self.b3,db3),(self.b4,db4)]:
                b -= self.lr * db
            if (ep + 1) % 20 == 0:
                print(f"    Epoch {ep+1}/{self.epochs} — loss: {loss:.6f}")
        # Set threshold at 95th percentile of training errors
        errors = ((X - self.forward(X)) ** 2).mean(axis=1)
        self.threshold = float(np.percentile(errors, 95))
        return self

    def reconstruction_error(self, X):
        return ((X - self.forward(X)) ** 2).mean(axis=1)

    def predict_proba(self, X):
        errors = self.reconstruction_error(X)
        return np.clip(errors / (self.threshold * 2), 0, 1)


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def plot_confusion_matrix(cm, out):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal","Fraud"], yticklabels=["Normal","Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix — Ensemble model")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  Saved: confusion_matrix.png")

def plot_score_distributions(scores_normal, scores_fraud, out):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores_normal, bins=50, alpha=0.65, color="#378ADD", label="Normal", density=True)
    ax.hist(scores_fraud,  bins=50, alpha=0.65, color="#E24B4A", label="Fraud",  density=True)
    ax.axvline(0.5, color="#888780", lw=1.5, linestyle="--", label="Threshold 0.5")
    ax.set_xlabel("Ensemble fraud score (0–1)"); ax.set_ylabel("Density")
    ax.set_title("Fraud score distribution — Normal vs Fraud trips")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "score_distributions.png"), dpi=150)
    plt.close()
    print("  Saved: score_distributions.png")

def plot_feature_importance(xgb_model, features, out):
    fi = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    fi.plot(kind="barh", ax=ax, color="#E24B4A")
    ax.set_title("Feature importance — XGBoost fraud detector")
    ax.set_xlabel("Importance"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "feature_importance.png"), dpi=150)
    plt.close()
    print("  Saved: feature_importance.png")

def plot_fraud_by_type(df_test, ensemble_scores, out):
    df_test = df_test.copy()
    df_test["ensemble_score"] = ensemble_scores
    fraud_df = df_test[df_test["is_fraud"] == 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    order = fraud_df.groupby("fraud_type")["ensemble_score"].mean().sort_values(ascending=False).index
    sns.boxplot(data=fraud_df, x="fraud_type", y="ensemble_score",
                order=order, palette="Reds", ax=ax)
    ax.set_title("Ensemble fraud score by fraud type")
    ax.set_xlabel(""); ax.set_ylabel("Fraud score"); ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fraud_by_type.png"), dpi=150)
    plt.close()
    print("  Saved: fraud_by_type.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 54)
    print("  Drivr Fraud Detection Pipeline")
    print("=" * 54)

    df = pd.read_csv(DATA)
    X  = df[FEATURES].fillna(0)
    y  = df["is_fraud"]

    print(f"\nDataset: {len(df)} trips | Fraud rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    df_test = df.iloc[X_test.index].reset_index(drop=True)

    # Scale
    scaler  = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    # ── 1. Isolation Forest ───────────────────────────────
    print("\n[1] Training Isolation Forest...")
    iso = IsolationForest(n_estimators=300, contamination=0.04, random_state=42, n_jobs=-1)
    iso.fit(Xs_train)
    iso_scores_test = -iso.score_samples(Xs_test)
    iso_scores_test = (iso_scores_test - iso_scores_test.min()) / (iso_scores_test.max() - iso_scores_test.min())
    auc_iso = roc_auc_score(y_test, iso_scores_test)
    print(f"  ROC-AUC: {auc_iso:.4f}")

    # ── 2. Autoencoder ────────────────────────────────────
    print("\n[2] Training Autoencoder...")
    ae = SimpleAutoencoder(input_dim=Xs_train.shape[1], hidden_dim=10, latent_dim=5, lr=0.005, epochs=60)
    X_normal_train = Xs_train[y_train == 0]
    ae.fit(X_normal_train)
    ae_scores_test = ae.predict_proba(Xs_test)
    auc_ae = roc_auc_score(y_test, ae_scores_test)
    print(f"  ROC-AUC: {auc_ae:.4f}")

    # ── 3. XGBoost (supervised) ────────────────────────────
    print("\n[3] Training XGBoost classifier...")
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos, eval_metric="auc",
        random_state=42, verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_scores_test = xgb.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, xgb_scores_test)
    print(f"  ROC-AUC: {auc_xgb:.4f}")

    # ── Ensemble ──────────────────────────────────────────
    print("\n[4] Ensemble (weighted average)...")
    # Weight by individual AUC performance
    total = auc_iso + auc_ae + auc_xgb
    w_iso, w_ae, w_xgb = auc_iso/total, auc_ae/total, auc_xgb/total
    ensemble = w_iso * iso_scores_test + w_ae * ae_scores_test + w_xgb * xgb_scores_test
    ensemble = (ensemble - ensemble.min()) / (ensemble.max() - ensemble.min())

    auc_ens  = roc_auc_score(y_test, ensemble)
    preds    = (ensemble >= 0.5).astype(int)
    cm       = confusion_matrix(y_test, preds)

    print(f"\n  Ensemble ROC-AUC: {auc_ens:.4f}")
    print(f"\n{classification_report(y_test, preds, target_names=['Normal','Fraud'])}")

    # ── Plots ─────────────────────────────────────────────
    print("Generating plots...")
    plot_confusion_matrix(cm, OUT)
    plot_score_distributions(ensemble[y_test==0], ensemble[y_test==1], OUT)
    plot_feature_importance(xgb, FEATURES, OUT)
    plot_fraud_by_type(df_test, ensemble, OUT)

    # ── Save ─────────────────────────────────────────────
    joblib.dump(iso,    os.path.join(BASE, "isolation_forest.pkl"))
    joblib.dump(ae,     os.path.join(BASE, "autoencoder.pkl"))
    joblib.dump(xgb,    os.path.join(BASE, "xgb_fraud.pkl"))
    joblib.dump(scaler, os.path.join(BASE, "fraud_scaler.pkl"))

    results = {
        "isolation_forest": {"ROC_AUC": round(auc_iso, 4), "weight": round(w_iso, 3)},
        "autoencoder":      {"ROC_AUC": round(auc_ae,  4), "weight": round(w_ae,  3)},
        "xgboost":          {"ROC_AUC": round(auc_xgb, 4), "weight": round(w_xgb, 3)},
        "ensemble":         {"ROC_AUC": round(auc_ens, 4)},
        "fraud_rate":       round(float(y.mean()), 4),
        "features":         FEATURES,
    }
    with open(os.path.join(BASE, "fraud_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved: isolation_forest.pkl, autoencoder.pkl, xgb_fraud.pkl, fraud_scaler.pkl")
    print("Done!")
    return results


if __name__ == "__main__":
    main()
