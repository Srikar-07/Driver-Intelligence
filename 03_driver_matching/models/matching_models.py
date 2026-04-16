"""
Drivr — Matching Model Classes
Importable module so joblib pkl files deserialize correctly across scripts.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from scipy.sparse import csr_matrix

DRIVER_FEATURES = [
    "rating", "experience_years", "car_type_enc",
    "is_nightlife_spec", "is_airport_spec", "is_event_spec", "is_wedding_spec",
    "speaks_spanish", "speaks_english",
    "acceptance_rate", "on_time_rate", "cancellation_rate",
    "allows_pets", "prefers_quiet",
]

CUSTOMER_FEATURES = [
    "preferred_car_enc", "is_nightlife_user", "is_airport_user", "is_event_user",
    "prefers_spanish", "has_pet", "prefers_quiet", "price_sensitivity",
]


class ContentBasedMatcher:
    def __init__(self):
        self.driver_scaler  = MinMaxScaler()
        self.driver_vectors = None
        self.driver_ids     = None

    def fit(self, drivers: pd.DataFrame):
        X = drivers[DRIVER_FEATURES].fillna(0).values
        self.driver_vectors = self.driver_scaler.fit_transform(X)
        self.driver_ids     = drivers["driver_id"].tolist()
        return self

    def _customer_vector(self, customer: dict) -> np.ndarray:
        vec = np.zeros(len(DRIVER_FEATURES))
        feat_map = {
            "rating":            ("price_sensitivity", lambda x: 1 - x),
            "car_type_enc":      ("preferred_car_enc", lambda x: x),
            "is_nightlife_spec": ("is_nightlife_user", lambda x: x),
            "is_airport_spec":   ("is_airport_user",   lambda x: x),
            "is_event_spec":     ("is_event_user",     lambda x: x),
            "speaks_spanish":    ("prefers_spanish",   lambda x: x),
            "allows_pets":       ("has_pet",           lambda x: x),
            "prefers_quiet":     ("prefers_quiet",     lambda x: x),
        }
        for i, feat in enumerate(DRIVER_FEATURES):
            if feat in feat_map:
                cust_key, fn = feat_map[feat]
                vec[i] = fn(customer.get(cust_key, 0))
        vec = self.driver_scaler.transform(vec.reshape(1, -1))[0]
        return vec

    def rank_drivers(self, customer: dict, top_k=10) -> pd.DataFrame:
        cvec = self._customer_vector(customer)
        sims = cosine_similarity(cvec.reshape(1, -1), self.driver_vectors)[0]
        idx  = np.argsort(sims)[::-1][:top_k]
        return pd.DataFrame({
            "driver_id": [self.driver_ids[i] for i in idx],
            "cb_score":  [round(float(sims[i]), 4) for i in idx],
        })


class CollaborativeFilter:
    def __init__(self, n_components=30):
        self.svd          = TruncatedSVD(n_components=n_components, random_state=42)
        self.customer_idx = {}
        self.driver_idx   = {}
        self.driver_ids   = []
        self.U = self.Vt  = None

    def fit(self, interactions: pd.DataFrame):
        customers = interactions["customer_id"].unique()
        drivers   = interactions["driver_id"].unique()
        self.customer_idx = {c: i for i, c in enumerate(customers)}
        self.driver_idx   = {d: i for i, d in enumerate(drivers)}
        self.driver_ids   = list(drivers)
        rows = interactions["customer_id"].map(self.customer_idx)
        cols = interactions["driver_id"].map(self.driver_idx)
        vals = interactions["explicit_rating"].values
        mat  = csr_matrix((vals, (rows, cols)), shape=(len(customers), len(drivers)))
        self.U  = self.svd.fit_transform(mat)
        self.Vt = self.svd.components_
        return self

    def rank_drivers(self, customer_id: str, top_k=10) -> pd.DataFrame:
        if customer_id not in self.customer_idx:
            return pd.DataFrame({"driver_id": [], "cf_score": []})
        cidx   = self.customer_idx[customer_id]
        scores = self.U[cidx] @ self.Vt
        idx    = np.argsort(scores)[::-1][:top_k]
        return pd.DataFrame({
            "driver_id": [self.driver_ids[i] for i in idx],
            "cf_score":  [round(float(scores[i]), 4) for i in idx],
        })


class HybridRanker:
    def __init__(self):
        self.model    = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
        self.features = ["cb_score", "cf_score", "rating", "on_time_rate",
                         "cancellation_rate", "acceptance_rate", "experience_years"]

    def fit(self, df_train: pd.DataFrame):
        self.model.fit(df_train[self.features], df_train["compatibility"])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features])
