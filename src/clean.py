import pandas as pd
import numpy as np
import os

def load_raw_data():
    df = pd.read_csv("data/raw/questions.csv")
    print(f"Loaded {len(df)} raw questions")
    return df

def clean_data(df):
    # Drop rows missing critical fields
    df = df.dropna(subset=["community_prediction", "resolution"])
    print(f"After dropping nulls: {len(df)} questions")

    # Ensure resolution is integer (0 or 1)
    df["resolution"] = df["resolution"].astype(int)

    # Clip predictions to avoid exactly 0 or 1 (causes issues in log score)
    df["community_prediction"] = df["community_prediction"].clip(0.001, 0.999)

    # Convert times to datetime
    df["resolve_time"] = pd.to_datetime(df["resolve_time"], errors="coerce")
    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")

    # Fill missing categories
    df["category"] = df["category"].fillna("Uncategorized")

    print(f"Final clean dataset: {len(df)} questions")
    print(f"Resolution rate: {df['resolution'].mean():.2%}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    
    return df

def bin_predictions(df, n_bins=10):
    """Assign each question to a probability bin for calibration analysis."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["community_prediction"], bins=bin_edges, labels=False, include_lowest=True)
    
    # Compute per-bin stats
    bins = df.groupby("bin").agg(
        mean_prediction=("community_prediction", "mean"),
        actual_rate=("resolution", "mean"),
        count=("resolution", "count")
    ).reset_index()
    
    return df, bins

def save_clean_data(df, bins):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/questions_clean.csv", index=False)
    bins.to_csv("data/processed/bins.csv", index=False)
    print("Saved cleaned data to data/processed/")

if __name__ == "__main__":
    df = load_raw_data()
    df = clean_data(df)
    df, bins = bin_predictions(df)
    save_clean_data(df, bins)
    print("\nBin summary:")
    print(bins.to_string())