# ⭐ Drivr Intelligence — Driver-Customer Matching System

> **Three-stage recommender system** that ranks the best available drivers for each customer using content-based filtering, collaborative filtering (SVD), and a hybrid gradient boosting re-ranker.

---

## 📌 Project Summary

Matching the right driver to the right customer is critical for Drivr — a bad match means low ratings, cancellations, and churn. This system implements a **two-stage retrieval + re-ranking pipeline**, the same architecture used by Uber, Lyft, and DoorDash for driver-customer matching.

---

## 🏗️ Architecture

```
Customer Request
      │
      ▼
┌─────────────────────────────┐
│  Stage 1: Retrieval         │  ← Fast, recall-focused
│  Content-Based (Cosine)     │  14 driver + customer features
│  Candidates: Top 40 drivers │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 2: Re-ranking        │  ← Precise, precision-focused
│  Hybrid Gradient Boosting   │  CB score + CF score + quality signals
│  Output: Top K ranked list  │
└─────────────────────────────┘
```

**Why two stages?** Scoring all 200 drivers with an expensive model is slow. The retrieval stage narrows to 40 candidates in milliseconds using fast cosine similarity. The re-ranker then applies a more nuanced model on just those 40.

---

## 🔑 Matching Signals

| Signal | Type | Weight |
|--------|------|--------|
| Car type preference match | Content-based | High |
| Language match (Spanish) | Content-based | High |
| Trip type specialization | Content-based | Medium |
| Pet compatibility | Content-based | Medium |
| Quiet preference match | Content-based | Low |
| Past rating patterns | Collaborative (SVD) | Medium |
| Driver rating | Quality signal | High |
| On-time rate | Quality signal | High |
| Cancellation rate | Quality signal | Medium |
| Acceptance rate | Quality signal | Low |

---

## 📊 Dataset

| Entity | Count |
|--------|-------|
| Drivers | 200 |
| Customers | 500 |
| Interactions | 5,000 |
| Avg compatibility score | 0.715 |

---

## 🗂️ Structure

```
03_driver_matching/
├── data/
│   ├── generate_profiles.py   # Driver, customer, interaction generator
│   ├── drivers.csv            # 200 driver profiles
│   ├── customers.csv          # 500 customer profiles
│   └── interactions.csv       # 5,000 trip interactions
├── models/
│   ├── train_matching.py      # Full training pipeline
│   ├── cb_matcher.pkl         # Content-based model
│   ├── cf_filter.pkl          # SVD collaborative filter
│   └── hybrid_ranker.pkl      # Gradient boosting re-ranker
├── api/
│   └── main.py                # FastAPI matching endpoint
└── notebooks/
    ├── score_distributions.png
    ├── matching_feature_importance.png
    └── top10_driver_matches.png
```

---

## 🚀 Quickstart

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib scipy fastapi uvicorn

# Generate data
cd data/ && python generate_profiles.py

# Train models
cd models/ && python train_matching.py

# Run API
cd api/ && uvicorn main:app --reload --port 8001
```

---

## 🔌 API Usage

```bash
curl -X POST http://localhost:8001/match \
  -H "Content-Type: application/json" \
  -d '{
    "preferred_car_type": "Luxury",
    "is_nightlife_user": 1,
    "is_event_user": 1,
    "prefers_spanish": 1,
    "price_sensitivity": 0.2,
    "top_k": 5
  }'
```

**Response:**
```json
{
  "top_matches": [
    {
      "rank": 1,
      "driver_id": "DRV0153",
      "match_score": 0.924,
      "rating": 4.5,
      "car_type": "SUV",
      "speaks_spanish": true,
      "specialties": "nightlife,event"
    }
  ],
  "total_drivers": 200,
  "match_strategy": "content-based (cosine) → hybrid gradient boosting re-rank"
}
```

---

## 🧠 Key Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Retrieval | Cosine similarity | O(n) fast scan on 200 drivers |
| CF method | TruncatedSVD | Handles sparse matrix, interpretable latent factors |
| Re-ranker | Gradient Boosting | Handles mixed feature types, robust to noise |
| Compatibility label | Computed from feature rules | No ground-truth labels needed at launch |
| Two-stage pipeline | Retrieve 40 → re-rank to top K | Industry standard: same pattern as Uber/DoorDash |

---

## 🔭 Next Steps

- [ ] Replace approximate CB score in hybrid with real per-request score
- [ ] Add NDCG@10 as primary ranking evaluation metric
- [ ] Implement online learning — update CF model with new trip ratings daily
- [ ] Add real-time driver availability filter (only rank online drivers)
- [ ] Explore LightFM for joint content + collaborative learning

---

## 👤 Author

Part of the **Drivr Intelligence** portfolio — a full DS/ML system built on a real startup use case.
