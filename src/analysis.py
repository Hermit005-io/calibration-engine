import pandas as pd
import numpy as np
from scipy import stats
import os

def ece(df):
    bin_edges = np.linspace(0, 1, 11)
    df = df.copy()
    df["bin"] = pd.cut(df["community_prediction"], bins=bin_edges, labels=False, include_lowest=True)
    bins = df.groupby("bin").agg(
        mean_prediction=("community_prediction", "mean"),
        actual_rate=("resolution", "mean"),
        count=("resolution", "count")
    ).dropna()
    return np.sum(bins["count"] * np.abs(bins["mean_prediction"] - bins["actual_rate"])) / bins["count"].sum()

def analysis_by_category(df):
    """Q1: Does category affect calibration quality?"""
    print("\n" + "=" * 50)
    print("Q1: CALIBRATION BY CATEGORY")
    print("=" * 50)

    categories = df["category"].value_counts()
    valid_cats = categories[categories >= 30].index.tolist()

    results = []
    for cat in valid_cats:
        cat_df = df[df["category"] == cat]
        cat_ece = ece(cat_df)
        cat_bs = np.mean((cat_df["community_prediction"] - cat_df["resolution"]) ** 2)
        results.append({
            "category": cat,
            "n": len(cat_df),
            "ece": cat_ece,
            "brier_score": cat_bs,
            "resolution_rate": cat_df["resolution"].mean()
        })

    results_df = pd.DataFrame(results).sort_values("ece")
    print(results_df.to_string(index=False))
    return results_df

def analysis_forecaster_count(df):
    """Q2: Do questions with more forecasters calibrate better?"""
    print("\n" + "=" * 50)
    print("Q2: DOES FORECASTER COUNT MATTER?")
    print("=" * 50)

    df = df.dropna(subset=["number_of_forecasters"])
    median = df["number_of_forecasters"].median()

    low = df[df["number_of_forecasters"] <= median]
    high = df[df["number_of_forecasters"] > median]

    low_ece = ece(low)
    high_ece = ece(high)
    low_bs = np.mean((low["community_prediction"] - low["resolution"]) ** 2)
    high_bs = np.mean((high["community_prediction"] - high["resolution"]) ** 2)

    print(f"Median forecaster count: {median:.0f}")
    print(f"\nLow forecaster group  (n={len(low)}): ECE={low_ece:.4f}, Brier={low_bs:.4f}")
    print(f"High forecaster group (n={len(high)}): ECE={high_ece:.4f}, Brier={high_bs:.4f}")

    # Statistical significance
    low_briers = (low["community_prediction"] - low["resolution"]) ** 2
    high_briers = (high["community_prediction"] - high["resolution"]) ** 2
    t_stat, p_value = stats.ttest_ind(low_briers, high_briers)
    print(f"\nt-test p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'} at 0.05)")

    return {"low_ece": low_ece, "high_ece": high_ece, "p_value": p_value}

def analysis_extreme_probabilities(df):
    """Q3: Are extreme predictions (tails) well calibrated?"""
    print("\n" + "=" * 50)
    print("Q3: CALIBRATION AT EXTREME PROBABILITIES")
    print("=" * 50)

    tails = df[
        (df["community_prediction"] < 0.1) |
        (df["community_prediction"] > 0.9)
    ]
    middle = df[
        (df["community_prediction"] >= 0.1) &
        (df["community_prediction"] <= 0.9)
    ]

    tail_ece = ece(tails)
    middle_ece = ece(middle)

    # Check if high confidence predictions are correct more often
    high_conf = df[df["community_prediction"] > 0.9]
    low_conf = df[df["community_prediction"] < 0.1]

    print(f"Tail questions (<10% or >90%): n={len(tails)}, ECE={tail_ece:.4f}")
    print(f"Middle questions (10%-90%):     n={len(middle)}, ECE={middle_ece:.4f}")
    print(f"\nHigh confidence (>90%) questions: n={len(high_conf)}")
    if len(high_conf) > 0:
        print(f"  Actual resolution rate: {high_conf['resolution'].mean():.2%} (should be ~90%+)")
    print(f"\nLow confidence (<10%) questions: n={len(low_conf)}")
    if len(low_conf) > 0:
        print(f"  Actual resolution rate: {low_conf['resolution'].mean():.2%} (should be ~10%-)")

    return {"tail_ece": tail_ece, "middle_ece": middle_ece}

def analysis_resolution_rate_over_time(df):
    """Q4: Has calibration improved over the years?"""
    print("\n" + "=" * 50)
    print("Q4: CALIBRATION OVER THE YEARS")
    print("=" * 50)

    df = df.dropna(subset=["resolve_time"])
    df["resolve_time"] = pd.to_datetime(df["resolve_time"], errors="coerce")
    df["year"] = df["resolve_time"].dt.year

    year_counts = df["year"].value_counts()
    valid_years = year_counts[year_counts >= 30].index.tolist()
    valid_years.sort()

    results = []
    for year in valid_years:
        year_df = df[df["year"] == year]
        year_ece = ece(year_df)
        year_bs = np.mean((year_df["community_prediction"] - year_df["resolution"]) ** 2)
        results.append({
            "year": year,
            "n": len(year_df),
            "ece": year_ece,
            "brier_score": year_bs
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/questions_clean.csv")

    cat_results = analysis_by_category(df)
    forecaster_results = analysis_forecaster_count(df)
    extreme_results = analysis_extreme_probabilities(df)
    time_results = analysis_resolution_rate_over_time(df)