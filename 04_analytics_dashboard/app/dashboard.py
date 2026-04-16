"""
Drivr Intelligence — Analytics Dashboard
Interactive Streamlit dashboard showing business KPIs, trip analytics,
driver performance, and revenue trends.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drivr Intelligence Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 16px 20px; border-left: 4px solid #1D9E75;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #1a1a1a; }
    .metric-label { font-size: 13px; color: #666; margin-bottom: 4px; }
    .metric-delta { font-size: 12px; color: #1D9E75; }
    .section-header { font-size: 18px; font-weight: 600; margin: 8px 0 16px; }
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "primary":  "#1D9E75",
    "blue":     "#378ADD",
    "orange":   "#D85A30",
    "purple":   "#7F77DD",
    "amber":    "#EF9F27",
    "red":      "#E24B4A",
    "gray":     "#888780",
}

# ── Data loading ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "../data")

@st.cache_data
def load_data():
    daily   = pd.read_csv(os.path.join(DATA, "daily_kpis.csv"), parse_dates=["date"])
    trips   = pd.read_csv(os.path.join(DATA, "trip_log.csv"),   parse_dates=["date"])
    drivers = pd.read_csv(os.path.join(DATA, "driver_performance.csv"))
    return daily, trips, drivers

daily, trips, drivers = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=60)
    st.title("Drivr Intelligence")
    st.caption("Business Analytics Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🗺️ Trip Analytics", "🚗 Driver Performance", "📈 Revenue Deep Dive"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("**Date range**")
    months = sorted(daily["month"].unique())
    month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun"}
    selected_months = st.multiselect(
        "Months", months,
        default=months,
        format_func=lambda x: month_labels[x]
    )

    st.markdown("---")
    st.caption("v1.0 · Drivr Intelligence")

# ── Filter data ───────────────────────────────────────────────────────────────
daily_f = daily[daily["month"].isin(selected_months)]
trips_f = trips[trips["month"].isin(selected_months)]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Business Overview")
    st.caption(f"Jan–Jun 2024 · Miami, FL · {len(daily_f)} days selected")

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    total_rev    = daily_f["total_revenue"].sum()
    total_trips  = daily_f["total_trips"].sum()
    avg_fare     = daily_f["avg_fare"].mean()
    avg_rating   = daily_f["avg_rating"].mean()
    avg_util     = daily_f["driver_utilization"].mean()

    col1.metric("Total Revenue",      f"${total_rev:,.0f}",   "+80% vs prior period")
    col2.metric("Total Trips",        f"{total_trips:,}",     "+65% vs prior period")
    col3.metric("Avg Fare",           f"${avg_fare:.2f}",     "+8%")
    col4.metric("Avg Driver Rating",  f"{avg_rating:.2f} ★",  "")
    col5.metric("Driver Utilization", f"{avg_util:.1%}",      "")

    st.markdown("---")

    # Revenue + Trips trend
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Revenue trend (daily)</div>', unsafe_allow_html=True)
        weekly = daily_f.set_index("date").resample("W")["total_revenue"].sum().reset_index()
        fig = px.area(weekly, x="date", y="total_revenue",
                      color_discrete_sequence=[COLORS["primary"]])
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=260,
                          xaxis_title="", yaxis_title="Revenue ($)",
                          showlegend=False, plot_bgcolor="white",
                          yaxis=dict(gridcolor="#f0f0f0"))
        fig.update_traces(fillcolor="rgba(29,158,117,0.15)", line_width=2)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Daily trips (weekday vs weekend)</div>', unsafe_allow_html=True)
        daily_f["day_type"] = daily_f["is_weekend"].map({0:"Weekday", 1:"Weekend"})
        fig = px.box(daily_f, x="day_type", y="total_trips",
                     color="day_type",
                     color_discrete_map={"Weekday": COLORS["blue"], "Weekend": COLORS["orange"]})
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=260,
                          showlegend=False, xaxis_title="", yaxis_title="Trips/day",
                          plot_bgcolor="white", yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig, use_container_width=True)

    # Cumulative growth
    st.markdown('<div class="section-header">Cumulative growth</div>', unsafe_allow_html=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=daily_f["date"], y=daily_f["cumulative_revenue"],
                             name="Revenue ($)", line=dict(color=COLORS["primary"], width=2.5)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_f["date"], y=daily_f["cumulative_trips"],
                             name="Trips", line=dict(color=COLORS["blue"], width=2.5, dash="dash")),
                  secondary_y=True)
    fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0),
                      plot_bgcolor="white", legend=dict(orientation="h", y=1.1),
                      yaxis=dict(gridcolor="#f0f0f0", title="Revenue ($)"),
                      yaxis2=dict(title="Trips"))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: TRIP ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Trip Analytics":
    st.title("🗺️ Trip Analytics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Fare",         f"${trips_f['fare_usd'].mean():.2f}")
    col2.metric("Completion Rate",  f"{trips_f['completed'].mean():.1%}")
    col3.metric("Avg Distance",     f"{trips_f['distance_km'].mean():.1f} km")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Trips by neighborhood</div>', unsafe_allow_html=True)
        hood_counts = trips_f["neighborhood"].value_counts().reset_index()
        hood_counts.columns = ["neighborhood","trips"]
        fig = px.bar(hood_counts, x="trips", y="neighborhood",
                     orientation="h",
                     color="trips", color_continuous_scale=["#B5D4F4","#1D9E75"])
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=320,
                          showlegend=False, coloraxis_showscale=False,
                          xaxis_title="Trip count", yaxis_title="",
                          plot_bgcolor="white", xaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Revenue by trip type</div>', unsafe_allow_html=True)
        tt = trips_f.groupby("trip_type")["fare_usd"].agg(["mean","count"]).reset_index()
        tt.columns = ["trip_type","avg_fare","count"]
        fig = px.bar(tt.sort_values("avg_fare"), x="avg_fare", y="trip_type",
                     orientation="h",
                     color="avg_fare", color_continuous_scale=["#B5D4F4","#D85A30"])
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=320,
                          showlegend=False, coloraxis_showscale=False,
                          xaxis_title="Avg fare ($)", yaxis_title="",
                          plot_bgcolor="white", xaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig, use_container_width=True)

    # Demand heatmap by hour
    st.markdown('<div class="section-header">Demand by hour × day of week</div>', unsafe_allow_html=True)
    trips_f2 = trips_f.copy()
    trips_f2["dow_name"] = pd.to_datetime(trips_f2["date"]).dt.day_name()
    pivot = trips_f2.groupby(["hour","dow_name"])["trip_id"].count().unstack()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex(columns=[d for d in dow_order if d in pivot.columns])
    fig = px.imshow(pivot, color_continuous_scale=["#E6F1FB","#1D9E75"],
                    labels=dict(x="Day", y="Hour", color="Trips"), aspect="auto")
    fig.update_layout(height=340, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Fare distribution
    st.markdown('<div class="section-header">Fare distribution by car type</div>', unsafe_allow_html=True)
    fig = px.violin(trips_f, x="car_type", y="fare_usd", box=True,
                    color="car_type",
                    color_discrete_sequence=list(COLORS.values()))
    fig.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0),
                      showlegend=False, plot_bgcolor="white",
                      xaxis_title="Car type", yaxis_title="Fare ($)",
                      yaxis=dict(gridcolor="#f0f0f0"))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: DRIVER PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚗 Driver Performance":
    st.title("🚗 Driver Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Drivers",   f"{len(drivers)}")
    col2.metric("Avg Weekly Trips", f"{drivers['weekly_trips'].mean():.0f}")
    col3.metric("Avg Rating",       f"{drivers['avg_rating'].mean():.2f} ★")
    col4.metric("Avg On-Time Rate", f"{drivers['on_time_rate'].mean():.1%}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Rating distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(drivers, x="avg_rating", nbins=20,
                           color_discrete_sequence=[COLORS["primary"]])
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0),
                          plot_bgcolor="white", xaxis_title="Rating",
                          yaxis_title="# Drivers", yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Weekly trips by car type</div>', unsafe_allow_html=True)
        fig = px.box(drivers, x="car_type", y="weekly_trips",
                     color="car_type",
                     color_discrete_sequence=list(COLORS.values()))
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0),
                          showlegend=False, plot_bgcolor="white",
                          xaxis_title="", yaxis_title="Trips/week",
                          yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: trips vs rating
    st.markdown('<div class="section-header">Driver quality quadrant — trips vs rating</div>', unsafe_allow_html=True)
    drivers["revenue_size"] = drivers["weekly_revenue"] / 10
    fig = px.scatter(drivers, x="weekly_trips", y="avg_rating",
                     size="revenue_size", color="car_type",
                     color_discrete_sequence=list(COLORS.values()),
                     hover_data=["driver_id","weekly_revenue","on_time_rate"],
                     labels={"weekly_trips":"Weekly trips","avg_rating":"Avg rating"})
    med_trips  = drivers["weekly_trips"].median()
    med_rating = drivers["avg_rating"].median()
    fig.add_vline(x=med_trips,  line_dash="dash", line_color=COLORS["gray"], opacity=0.5)
    fig.add_hline(y=med_rating, line_dash="dash", line_color=COLORS["gray"], opacity=0.5)
    fig.update_layout(height=380, margin=dict(l=0,r=0,t=0,b=0),
                      plot_bgcolor="white",
                      xaxis=dict(gridcolor="#f0f0f0"),
                      yaxis=dict(gridcolor="#f0f0f0"))
    st.plotly_chart(fig, use_container_width=True)

    # Top 10 drivers table
    st.markdown('<div class="section-header">Top 10 drivers by revenue</div>', unsafe_allow_html=True)
    top10 = drivers.nlargest(10, "weekly_revenue")[
        ["driver_id","car_type","weekly_trips","weekly_revenue","avg_rating","on_time_rate","cancellation_rate"]
    ].reset_index(drop=True)
    top10.index += 1
    top10.columns = ["Driver","Car","Trips","Revenue ($)","Rating","On-time","Cancel rate"]
    st.dataframe(top10.style.format({"Revenue ($)":"${:,.0f}","Rating":"{:.2f}","On-time":"{:.1%}","Cancel rate":"{:.3f}"}),
                 use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: REVENUE DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Revenue Deep Dive":
    st.title("📈 Revenue Deep Dive")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue",   f"${daily_f['total_revenue'].sum():,.0f}")
    col2.metric("Best Day",        f"${daily_f['total_revenue'].max():,.0f}")
    col3.metric("Avg Daily Rev",   f"${daily_f['total_revenue'].mean():,.0f}")

    st.markdown("---")

    # Revenue by month
    st.markdown('<div class="section-header">Revenue by month</div>', unsafe_allow_html=True)
    month_rev = daily_f.groupby("month")["total_revenue"].sum().reset_index()
    month_rev["month_name"] = month_rev["month"].map(
        {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun"})
    fig = px.bar(month_rev, x="month_name", y="total_revenue",
                 color="total_revenue", color_continuous_scale=["#B5D4F4","#1D9E75"],
                 text="total_revenue")
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0),
                      showlegend=False, coloraxis_showscale=False,
                      plot_bgcolor="white", xaxis_title="",
                      yaxis_title="Revenue ($)", yaxis=dict(gridcolor="#f0f0f0"))
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Weekday vs weekend revenue</div>', unsafe_allow_html=True)
        wk = daily_f.groupby("is_weekend")["total_revenue"].sum().reset_index()
        wk["label"] = wk["is_weekend"].map({0:"Weekday",1:"Weekend"})
        fig = px.pie(wk, values="total_revenue", names="label",
                     color_discrete_sequence=[COLORS["blue"], COLORS["orange"]],
                     hole=0.45)
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Avg fare trend (7-day rolling)</div>', unsafe_allow_html=True)
        daily_f2 = daily_f.copy().sort_values("date")
        daily_f2["rolling_fare"] = daily_f2["avg_fare"].rolling(7).mean()
        fig = px.line(daily_f2, x="date", y="rolling_fare",
                      color_discrete_sequence=[COLORS["amber"]])
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0),
                          plot_bgcolor="white", xaxis_title="",
                          yaxis_title="Avg fare ($)", yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig, use_container_width=True)

    # Cancellation vs revenue
    st.markdown('<div class="section-header">Cancellation rate vs daily revenue</div>', unsafe_allow_html=True)
    fig = px.scatter(daily_f, x="cancellation_rate", y="total_revenue",
                     color="is_weekend",
                     color_discrete_map={0: COLORS["blue"], 1: COLORS["orange"]},
                     labels={"cancellation_rate":"Cancellation rate","total_revenue":"Revenue ($)",
                             "is_weekend":"Weekend"},
                     trendline="ols")
    fig.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0),
                      plot_bgcolor="white",
                      xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"))
    st.plotly_chart(fig, use_container_width=True)
