import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

st.set_page_config(
    page_title="Metaculus Calibration Engine",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1d27; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    
    /* All text white */
    .stApp p, .stApp span, .stApp label, .stApp div { color: #e2e8f0; }
    
    /* Headers */
    .stApp h1 { color: #7c9ef5 !important; }
    .stApp h2 { color: #a0b4f7 !important; }
    .stApp h3 { color: #c3d0fa !important; }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1a1d27;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    [data-testid="stMetricLabel"] { color: #a0aec0 !important; }
    
    /* Radio buttons and selectbox */
    .stRadio label { color: #e2e8f0 !important; }
    .stSelectbox label { color: #e2e8f0 !important; }
    
    /* Input fields */
    .stTextInput input { 
        background-color: #1a1d27 !important; 
        color: #ffffff !important;
        border-color: #2d3748 !important;
    }

    /* Divider */
    hr { border-color: #2d3748; }
</style>
""", unsafe_allow_html=True)
def load_data():
    df = pd.read_csv("data/processed/questions_clean.csv")
    bins = pd.read_csv("data/processed/bins.csv")
    df["resolve_time"] = pd.to_datetime(df["resolve_time"], errors="coerce")
    df["brier"] = (df["community_prediction"] - df["resolution"]) ** 2
    return df, bins

def compute_ece(df):
    bin_edges = np.linspace(0, 1, 11)
    df = df.copy()
    df["bin"] = pd.cut(df["community_prediction"], bins=bin_edges, labels=False, include_lowest=True)
    bins = df.groupby("bin").agg(
        mean_prediction=("community_prediction", "mean"),
        actual_rate=("resolution", "mean"),
        count=("resolution", "count")
    ).dropna()
    if bins["count"].sum() == 0:
        return 0
    return np.sum(bins["count"] * np.abs(bins["mean_prediction"] - bins["actual_rate"])) / bins["count"].sum()

def reliability_diagram(bins):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(dash="dash", color="gray"),
        name="Perfect Calibration"
    ))
    fig.add_trace(go.Scatter(
        x=bins["mean_prediction"], y=bins["actual_rate"],
        mode="markers+lines",
        marker=dict(size=bins["count"] / bins["count"].max() * 30 + 5, color="royalblue"),
        name="Metaculus",
        text=[f"n={c}" for c in bins["count"]],
        hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<br>%{text}"
    ))
    fig.update_layout(
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Actual Resolution Rate",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        template="plotly_white", height=450
    )
    return fig

def overconfidence_map(bins):
    gap = bins["mean_prediction"] - bins["actual_rate"]
    colors = ["crimson" if g > 0 else "steelblue" for g in gap]
    fig = go.Figure(go.Bar(
        x=bins["mean_prediction"], y=gap,
        marker_color=colors,
        text=[f"{g:+.3f}" for g in gap],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="Predicted Probability Bin",
        yaxis_title="Gap (+ = Overconfident)",
        template="plotly_white", height=400
    )
    return fig

def brier_histogram(df):
    fig = go.Figure(go.Histogram(
        x=df["brier"], nbinsx=40,
        marker_color="royalblue", opacity=0.8
    ))
    fig.add_vline(x=df["brier"].mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {df['brier'].mean():.3f}")
    fig.update_layout(
        xaxis_title="Brier Score", yaxis_title="Count",
        template="plotly_white", height=400
    )
    return fig

# ── Load data ──────────────────────────────────────────────
df, bins = load_data()

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("📊 Calibration Engine")
page = st.sidebar.radio("Navigate", ["Overview", "Calibration Explorer", "Forecaster Analysis", "Question Browser", "Findings"])

# ══════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Metaculus Prediction Market Calibration Engine")
    st.markdown("*How well does the crowd know what it doesn't know?*")

    bs = np.mean(df["brier"])
    base_rate = df["resolution"].mean()
    bs_baseline = np.mean((base_rate - df["resolution"]) ** 2)
    bss = 1 - (bs / bs_baseline)
    ece = compute_ece(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Questions Analyzed", f"{len(df):,}")
    col2.metric("Brier Score", f"{bs:.4f}", delta="vs random: 0.25", delta_color="inverse")
    col3.metric("Skill Score", f"{bss:.4f}", delta="vs baseline: 0.00")
    col4.metric("ECE", f"{ece:.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Reliability Diagram")
        st.plotly_chart(reliability_diagram(bins), use_container_width=True)
    with col2:
        st.subheader("Overconfidence Map")
        st.plotly_chart(overconfidence_map(bins), use_container_width=True)

# ══════════════════════════════════════════════════════════
elif page == "Calibration Explorer":
    st.title("Calibration Explorer")

    st.subheader("Brier Score Distribution")
    st.plotly_chart(brier_histogram(df), use_container_width=True)

    st.subheader("Worst Calibrated Questions")
    worst = df.nlargest(20, "brier")[["title", "community_prediction", "resolution", "brier"]]
    worst.columns = ["Question", "Predicted", "Resolved", "Brier Score"]
    st.dataframe(worst, use_container_width=True)

# ══════════════════════════════════════════════════════════
elif page == "Forecaster Analysis":
    st.title("Does Crowd Size Matter?")
    st.markdown("Splitting questions by number of forecasters reveals a striking pattern.")

    df_f = df.dropna(subset=["number_of_forecasters"])
    median = df_f["number_of_forecasters"].median()
    low = df_f[df_f["number_of_forecasters"] <= median]
    high = df_f[df_f["number_of_forecasters"] > median]

    col1, col2 = st.columns(2)
    col1.metric(f"Low Forecasters (≤{median:.0f})", f"ECE: {compute_ece(low):.4f}", delta=f"Brier: {low['brier'].mean():.4f}")
    col2.metric(f"High Forecasters (>{median:.0f})", f"ECE: {compute_ece(high):.4f}", delta=f"Brier: {high['brier'].mean():.4f}")

    st.markdown("""
    **Key Finding:** Questions with more forecasters are both *more accurate* and *better 
    calibrated* across every metric. Crucially, this effect holds within every question 
    age group — controlling for how old a question is doesn't explain it away. 
    More participation genuinely improves forecast quality (p < 0.0001, n=4,851).
    
    **Correlation between forecaster count and Brier Score: -0.146**  
    More forecasters → lower error, consistently.
    """)

    # Scatter: forecaster count vs brier score
    fig = px.scatter(
        df_f, x="number_of_forecasters", y="brier",
        opacity=0.4, trendline="lowess",
        labels={"number_of_forecasters": "Number of Forecasters", "brier": "Brier Score"},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
elif page == "Question Browser":
    st.title("Question Browser")

    search = st.text_input("Search questions", "")
    sort_by = st.selectbox("Sort by", ["brier", "community_prediction", "resolution"])

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["title"].str.contains(search, case=False, na=False)]

    filtered = filtered.sort_values(sort_by, ascending=False)
    display = filtered[["title", "community_prediction", "resolution", "number_of_forecasters", "brier"]].head(100)
    display.columns = ["Question", "Predicted Prob", "Resolved (1=Yes)", "Forecasters", "Brier Score"]
    st.dataframe(display, use_container_width=True)

# ══════════════════════════════════════════════════════════
elif page == "Findings":
    st.title("Key Findings")

    st.markdown("""
    ## 1. Metaculus is exceptionally well-calibrated
    With a Brier Score of **0.1178** (vs 0.25 for random) and a Skill Score of **0.4759**, 
    the Metaculus community is nearly twice as good as a naive baseline. ECE of 0.0489 
    means predictions are on average less than 5 percentage points off from true frequencies.

    ## 2. More forecasters improves both accuracy AND calibration
    Questions with above-median forecaster counts have a **31% lower Brier Score** 
    (0.0961 vs 0.1394) and **lower ECE** (0.0400 vs 0.0577). Wisdom of crowds works — 
    more participation genuinely improves forecast quality across every dimension.
    This effect is highly statistically significant (p < 0.0001, n=4,851).

    ## 3. Geopolitics is best calibrated, Natural Sciences worst
    Geopolitics questions have an ECE of just 0.022 — nearly perfect calibration.
    Natural Sciences questions have an ECE of 0.160 — the worst of any category.
    This suggests the crowd has well-developed intuitions for political outcomes 
    but struggles with scientific predictions.

    ## 4. Extreme predictions are remarkably accurate
    Among 1,302 questions predicted below 10%: only 1.15% resolved Yes.
    Among 435 questions predicted above 90%: 98.39% resolved Yes.
    The crowd avoids false certainty at the tails with striking consistency.

    ## 5. Calibration has improved over time
    ECE peaked around 0.144 in 2019 and has trended down to 0.034 by 2023-2025,
    suggesting the Metaculus community has matured as a forecasting platform.
    """)