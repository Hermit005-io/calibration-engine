# Metaculus Prediction Market Calibration Engine

A tool for measuring and analyzing the calibration quality of the Metaculus prediction market community.

**Live Dashboard:** https://calibration-engine-ylueapd62m3kddnqquqhdh.streamlit.app/

## What is Calibration?

A forecaster is well-calibrated if, among all predictions made with X% confidence, roughly X% actually come true. This project measures how well the Metaculus crowd achieves this.

## Key Findings

**1. Metaculus is exceptionally well-calibrated.** With a Brier Score of 0.1178 (vs 0.25 for random) and a Skill Score of 0.4759, the crowd is nearly twice as good as a naive baseline across 4,851 resolved questions.

**2. More forecasters improves both accuracy and calibration.** Questions with above-median forecaster counts have a 31% lower Brier Score and lower ECE. Wisdom of crowds works — this effect is statistically significant (p < 0.0001).

**3. Geopolitics is best calibrated, Natural Sciences worst.** ECE ranges from 0.022 (Geopolitics) to 0.160 (Natural Sciences) across 15 categories, suggesting the crowd has stronger intuitions for political outcomes than scientific ones.

**4. Extreme predictions are remarkably accurate.** Among 1,302 questions predicted below 10%, only 1.15% resolved Yes. Among 435 questions above 90%, 98.39% resolved Yes.

**5. Calibration has improved over time.** ECE peaked around 0.144 in 2019 and trended down to 0.034 by 2023-2025.

## Metrics Implemented

- **Brier Score** – Mean squared error between predicted probability and outcome
- **Brier Score Decomposition** – Reliability, Resolution, and Uncertainty components
- **Expected Calibration Error (ECE)** – Weighted average gap between predicted and actual frequencies
- **Maximum Calibration Error (MCE)** – Worst-case calibration gap across any bin
- **Log Score** – Stricter metric penalizing overconfident wrong predictions
- **Brier Skill Score** – Performance relative to a naive baseline

## Project Structure
```
calibration-engine/
├── src/
│   ├── fetch.py        # Metaculus API data collection
│   ├── clean.py        # Data cleaning and binning
│   ├── metrics.py      # Calibration metrics implementation
│   ├── visualize.py    # Plotly visualizations
│   ├── analysis.py     # Deep analysis and hypothesis testing
│   └── dashboard.py    # Streamlit interactive dashboard
├── data/
│   ├── raw/            # Raw API responses
│   └── processed/      # Cleaned data and bin summaries
└── findings.md         # Detailed analytical findings
```

## Running Locally
```bash
git clone https://github.com/Hermit005-io/calibration-engine
cd calibration-engine
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python src/fetch.py
python src/clean.py
streamlit run src/dashboard.py
```

## Tech Stack

Python, Pandas, NumPy, SciPy, Plotly, Streamlit, Metaculus API