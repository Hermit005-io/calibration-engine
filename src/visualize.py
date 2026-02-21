import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

os.makedirs("data/processed", exist_ok=True)

def plot_reliability_diagram(bins, title="Reliability Diagram"):
    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect Calibration"
    ))

    # Actual calibration points
    fig.add_trace(go.Scatter(
        x=bins["mean_prediction"],
        y=bins["actual_rate"],
        mode="markers+lines",
        marker=dict(size=bins["count"] / bins["count"].max() * 30 + 5, color="royalblue"),
        name="Metaculus Community",
        text=[f"n={c}" for c in bins["count"]],
        hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<br>%{text}"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Actual Resolution Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=700, height=600,
        template="plotly_white"
    )
    return fig

def plot_calibration_by_category(df):
    categories = df["category"].value_counts()
    # Only keep categories with enough questions
    valid_cats = categories[categories >= 30].index.tolist()
    df_filtered = df[df["category"].isin(valid_cats)]

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect Calibration"
    ))

    colors = px.colors.qualitative.Set2
    bin_edges = np.linspace(0, 1, 11)

    for i, cat in enumerate(valid_cats):
        cat_df = df_filtered[df_filtered["category"] == cat].copy()
        cat_df["bin"] = pd.cut(cat_df["community_prediction"], bins=bin_edges, labels=False, include_lowest=True)
        cat_bins = cat_df.groupby("bin").agg(
            mean_prediction=("community_prediction", "mean"),
            actual_rate=("resolution", "mean"),
            count=("resolution", "count")
        ).reset_index().dropna()

        fig.add_trace(go.Scatter(
            x=cat_bins["mean_prediction"],
            y=cat_bins["actual_rate"],
            mode="markers+lines",
            name=cat,
            marker=dict(color=colors[i % len(colors)], size=8),
        ))

    fig.update_layout(
        title="Calibration by Category",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Actual Resolution Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=800, height=600,
        template="plotly_white"
    )
    return fig

def plot_overconfidence_map(bins):
    gap = bins["mean_prediction"] - bins["actual_rate"]
    colors = ["crimson" if g > 0 else "steelblue" for g in gap]

    fig = go.Figure(go.Bar(
        x=bins["mean_prediction"],
        y=gap,
        marker_color=colors,
        text=[f"{g:+.3f}" for g in gap],
        textposition="outside",
        hovertemplate="Predicted: %{x:.2f}<br>Gap: %{y:.3f}"
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Overconfidence Map (Predicted - Actual)",
        xaxis_title="Predicted Probability Bin",
        yaxis_title="Gap (+ = Overconfident, - = Underconfident)",
        width=750, height=500,
        template="plotly_white"
    )
    return fig

def plot_brier_distribution(df):
    df["brier"] = (df["community_prediction"] - df["resolution"]) ** 2

    fig = go.Figure(go.Histogram(
        x=df["brier"],
        nbinsx=40,
        marker_color="royalblue",
        opacity=0.8
    ))

    fig.add_vline(x=df["brier"].mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {df['brier'].mean():.3f}")

    fig.update_layout(
        title="Distribution of Per-Question Brier Scores",
        xaxis_title="Brier Score",
        yaxis_title="Count",
        width=750, height=500,
        template="plotly_white"
    )
    return fig

if __name__ == "__main__":
    df = pd.read_csv("data/processed/questions_clean.csv")
    bins = pd.read_csv("data/processed/bins.csv")

    fig1 = plot_reliability_diagram(bins)
    fig1.show()

    fig2 = plot_calibration_by_category(df)
    fig2.show()

    fig3 = plot_overconfidence_map(bins)
    fig3.show()

    fig4 = plot_brier_distribution(df)
    fig4.show()