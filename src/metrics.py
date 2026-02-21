import pandas as pd
import numpy as np

def brier_score(df):
    """Mean squared error between predicted probability and actual outcome."""
    return np.mean((df["community_prediction"] - df["resolution"]) ** 2)

def brier_score_decomposition(df, bins):
    """
    Decompose Brier Score into:
    - Reliability: how far predictions are from actual rates (calibration error)
    - Resolution: how spread out the bin outcomes are from the base rate
    - Uncertainty: inherent difficulty of the questions
    """
    base_rate = df["resolution"].mean()
    n = len(df)

    reliability = np.sum(bins["count"] * (bins["mean_prediction"] - bins["actual_rate"]) ** 2) / n
    resolution = np.sum(bins["count"] * (bins["actual_rate"] - base_rate) ** 2) / n
    uncertainty = base_rate * (1 - base_rate)

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty
    }

def expected_calibration_error(bins):
    """Weighted average gap between predicted probability and actual frequency."""
    total = bins["count"].sum()
    ece = np.sum(bins["count"] * np.abs(bins["mean_prediction"] - bins["actual_rate"])) / total
    return ece

def maximum_calibration_error(bins):
    """Worst-case gap across any single bin."""
    return np.max(np.abs(bins["mean_prediction"] - bins["actual_rate"]))

def log_score(df):
    """Log loss - stricter than Brier, penalizes overconfident wrong predictions."""
    p = df["community_prediction"]
    o = df["resolution"]
    return -np.mean(o * np.log(p) + (1 - o) * np.log(1 - p))

def brier_skill_score(df):
    """How much better is the market vs always predicting the base rate?"""
    base_rate = df["resolution"].mean()
    bs = brier_score(df)
    bs_baseline = np.mean((base_rate - df["resolution"]) ** 2)
    return 1 - (bs / bs_baseline)

def run_all_metrics(df, bins):
    bs = brier_score(df)
    decomp = brier_score_decomposition(df, bins)
    ece = expected_calibration_error(bins)
    mce = maximum_calibration_error(bins)
    ls = log_score(df)
    bss = brier_skill_score(df)

    print("=" * 40)
    print("CALIBRATION METRICS")
    print("=" * 40)
    print(f"Brier Score:        {bs:.4f}  (lower is better, random=0.25)")
    print(f"Brier Skill Score:  {bss:.4f}  (higher is better, 0=baseline)")
    print(f"Log Score:          {ls:.4f}  (lower is better)")
    print(f"ECE:                {ece:.4f}  (lower is better)")
    print(f"MCE:                {mce:.4f}  (lower is better)")
    print("-" * 40)
    print("Brier Decomposition:")
    print(f"  Reliability:      {decomp['reliability']:.4f}  (calibration error)")
    print(f"  Resolution:       {decomp['resolution']:.4f}  (spread of outcomes)")
    print(f"  Uncertainty:      {decomp['uncertainty']:.4f}  (inherent difficulty)")
    print("=" * 40)

    return {
        "brier_score": bs,
        "brier_skill_score": bss,
        "log_score": ls,
        "ece": ece,
        "mce": mce,
        **decomp
    }

if __name__ == "__main__":
    df = pd.read_csv("data/processed/questions_clean.csv")
    bins = pd.read_csv("data/processed/bins.csv")
    metrics = run_all_metrics(df, bins)