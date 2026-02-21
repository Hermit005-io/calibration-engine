# Metaculus Prediction Market Calibration Engine

A tool for measuring and analyzing the calibration quality of the Metaculus prediction market community.

**Live Dashboard:** https://calibration-engine-ylueapd62m3kddnqquqhdh.streamlit.app/

## What is Calibration?

A forecaster is well-calibrated if, among all predictions made with X% confidence, roughly X% actually come true. This project measures how well the Metaculus crowd achieves this.

## Key Findings

**1. Metaculus is genuinely skilled.** With a Brier Score of 0.1717 vs 0.25 for a random forecaster, and a Skill Score of 0.2433, the crowd meaningfully beats naive baselines.

**2. More forecasters = more accurate but less calibrated.** Questions with above-median forecaster counts have a 37% lower Brier Score but 2x higher ECE. Larger crowds converge on correct answers but become systematically overconfident. This effect is statistically significant (p < 0.0001).

**3. Extreme predictions are well-grounded.** Questions predicted below 10% resolved No 100% of the time. Questions above 90% resolved Yes 100% of the time.

**4. Recent questions show improving accuracy.** 2026 questions have a Brier Score of 0.119 vs 0.215 for 2025, mirroring the forecaster count effect.

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