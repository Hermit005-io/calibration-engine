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
def analysis_forecaster_confounds(df):
    """
    Q5: Is the forecaster count effect real, or a confound?
    We control for question age and resolution rate to isolate the effect.
    """
    print("\n" + "=" * 50)
    print("Q5: CONTROLLING FOR CONFOUNDS IN FORECASTER EFFECT")
    print("=" * 50)

    df = df.dropna(subset=["number_of_forecasters", "resolve_time"]).copy()
    df["resolve_time"] = pd.to_datetime(df["resolve_time"], errors="coerce")
    df["question_age_days"] = (df["resolve_time"] - df["created_time"].apply(pd.Timestamp)).dt.days.abs()
    df["brier"] = (df["community_prediction"] - df["resolution"]) ** 2

    # Bin forecaster count into quartiles
    df["forecaster_quartile"] = pd.qcut(df["number_of_forecasters"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    # Bin question age into quartiles
    df["age_quartile"] = pd.qcut(df["question_age_days"], q=4, labels=["young", "mid-young", "mid-old", "old"])

    # For each age group, check if forecaster count still predicts calibration
    print("\nBrier Score by forecaster quartile, controlling for question age:")
    print(f"{'Age Group':<12} {'Q1 (few)':<12} {'Q2':<12} {'Q3':<12} {'Q4 (many)':<12}")
    print("-" * 52)

    for age in ["young", "mid-young", "mid-old", "old"]:
        age_df = df[df["age_quartile"] == age]
        row = f"{age:<12}"
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            subset = age_df[age_df["forecaster_quartile"] == q]
            if len(subset) > 5:
                row += f"{subset['brier'].mean():<12.4f}"
            else:
                row += f"{'N/A':<12}"
        print(row)

    # Overall correlation
    corr = df["number_of_forecasters"].corr(df["brier"])
    print(f"\nOverall correlation (forecasters vs Brier): {corr:.4f}")
    print("(Negative = more forecasters → lower Brier Score → more accurate)")

    return df
def analysis_time_horizon(df):
    """
    Q6: Does calibration improve as resolution date approaches?
    We use the aggregation history to measure calibration at different time horizons.
    """
    import requests
    import time as time_module

    print("\n" + "=" * 50)
    print("Q6: CALIBRATION OVER TIME HORIZON")
    print("=" * 50)

    # Sample 200 questions to keep API calls manageable
    sample = df.dropna(subset=["resolve_time"]).sample(200, random_state=42)
    sample["resolve_time"] = pd.to_datetime(sample["resolve_time"], errors="coerce")

    horizons = {30: [], 7: [], 1: []}

    print("Fetching prediction histories (this may take a few minutes)...")

    for i, (_, row) in enumerate(sample.iterrows()):
        try:
            r = requests.get(
                f"https://www.metaculus.com/api2/questions/{row['id']}/",
                timeout=10
            )
            if r.status_code != 200:
                continue

            data = r.json()
            question = data.get("question", {})
            resolve_time = pd.to_datetime(row["resolve_time"], utc=True)
            resolution = row["resolution"]

            # Get prediction history
            aggs = question.get("aggregations", {})
            history = None
            for key in ["recency_weighted", "unweighted"]:
                h = aggs.get(key, {}).get("history", [])
                if h:
                    history = h
                    break

            if not history:
                continue

            # For each horizon, find the prediction closest to N days before resolution
            for days in [30, 7, 1]:
                target_time = resolve_time - pd.Timedelta(days=days)
                best_pred = None
                best_diff = float("inf")

                for entry in history:
                    try:
                        t = pd.Timestamp(entry["start_time"], unit="s", tz="UTC")
                        centers = entry.get("centers", [])
                        if not centers:
                            continue
                        diff = abs((t - target_time).total_seconds())
                        if diff < best_diff:
                            best_diff = diff
                            best_pred = centers[0]
                    except Exception:
                        continue

                if best_pred is not None:
                    brier = (best_pred - resolution) ** 2
                    horizons[days].append(brier)

            time_module.sleep(0.3)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/200 questions...")

        except Exception:
            continue

    print("\nResults:")
    print(f"{'Horizon':<15} {'N':<8} {'Mean Brier':<15} {'ECE proxy'}")
    print("-" * 45)
    for days in [30, 7, 1]:
        scores = horizons[days]
        if scores:
            print(f"{days} days before  {len(scores):<8} {np.mean(scores):<15.4f}")

    return horizons
if __name__ == "__main__":
    df = pd.read_csv("data/processed/questions_clean.csv")
    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")

    cat_results = analysis_by_category(df)
    forecaster_results = analysis_forecaster_count(df)
    extreme_results = analysis_extreme_probabilities(df)
    time_results = analysis_resolution_rate_over_time(df)
    confound_results = analysis_forecaster_confounds(df)
    horizon_results = analysis_time_horizon(df)