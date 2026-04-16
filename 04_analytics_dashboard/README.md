# 📊 Drivr Intelligence — Analytics Dashboard

> **4-page interactive business intelligence dashboard** built with Streamlit and Plotly. Tracks revenue, trip patterns, driver performance, and pricing trends across 180 days of Miami operations.

---

## 📌 Project Summary

Every DS/ML system needs a business-facing layer — this dashboard is the window into Drivr's operations. It combines 3 datasets (daily KPIs, 5k trip logs, 80 driver profiles) into a live, filterable BI tool that a business stakeholder or investor could use on day one.

**Skills demonstrated:**
- ETL pipeline: raw CSVs → aggregated KPIs → interactive visuals
- Streamlit multi-page architecture with sidebar navigation
- Plotly: area charts, box plots, violin plots, heatmaps, scatter plots, pie charts
- Business thinking: what metrics actually matter for a ride-hailing startup

---

## 📊 Dashboard Pages

### Page 1 — Business Overview
- 5 headline KPI cards (revenue, trips, avg fare, rating, utilization)
- Weekly revenue area chart with growth trend
- Weekday vs weekend trip distribution (box plot)
- Cumulative revenue + trips dual-axis chart

### Page 2 — Trip Analytics
- Trips by neighborhood (horizontal bar)
- Revenue by trip type (horizontal bar — wedding/event > errand)
- Demand heatmap: hour × day of week
- Fare distribution by car type (violin plot)

### Page 3 — Driver Performance
- Rating histogram across 80 drivers
- Weekly trips by car type (box plot)
- Driver quality quadrant: trips vs rating, bubble = revenue
- Top 10 drivers table with formatted metrics

### Page 4 — Revenue Deep Dive
- Monthly revenue bar chart with labels
- Weekday vs weekend revenue split (donut chart)
- 7-day rolling avg fare trend
- Cancellation rate vs revenue scatter with OLS trendline

---

## 🔑 Key Business Findings

| Insight | Finding |
|---------|---------|
| Revenue growth | **+80%** over 6 months (Jan → Jun) |
| Peak demand | **Saturday 10pm–2am** (South Beach / Wynwood) |
| Best trip type | **Weddings** — highest avg fare |
| Best car type | **Luxury** — highest fare, lower volume |
| Driver sweet spot | **60–90 trips/week**, rating **4.5+** |
| Cancellations | Negatively correlated with revenue (r = -0.31) |
| Total 6-month revenue | **$764,988** |

---

## 🗂️ Structure

```
04_analytics_dashboard/
├── data/
│   ├── generate_kpis.py          # Generates all 3 datasets
│   ├── daily_kpis.csv            # 180 days of daily business metrics
│   ├── trip_log.csv              # 5,000 individual trips
│   └── driver_performance.csv   # 80 driver weekly stats
├── app/
│   └── dashboard.py             # Streamlit app (4 pages)
└── notebooks/
    ├── revenue_trend.png
    ├── demand_heatmap.png
    ├── driver_quadrant.png
    └── fare_by_trip_type.png
```

---

## 🚀 Quickstart

```bash
pip install streamlit plotly pandas numpy seaborn matplotlib

# Generate data
cd data/ && python generate_kpis.py

# Run dashboard
cd app/ && streamlit run dashboard.py
```

Dashboard opens at `http://localhost:8501`

---

## 🧠 Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Framework | Streamlit | Fastest path to a shareable DS dashboard |
| Charts | Plotly | Interactive hover, zoom — better than matplotlib for BI |
| Layout | Multi-page sidebar | Mirrors real BI tools (Tableau, Looker) |
| Caching | `@st.cache_data` | Prevents reloading CSVs on every interaction |
| Filters | Month multiselect | Most useful filter for time-series business data |

---

## 🔭 Next Steps

- [ ] Connect to a live database (PostgreSQL) instead of CSVs
- [ ] Add real-time refresh (WebSocket or Streamlit `st.experimental_rerun`)
- [ ] Deploy to Streamlit Community Cloud (free, public URL)
- [ ] Add driver churn prediction widget (30-day churn risk score per driver)
- [ ] Export to PDF report button for stakeholder sharing

---

## 👤 Author

Part of the **Drivr Intelligence** portfolio — a full DS/ML system built on a real startup use case.
