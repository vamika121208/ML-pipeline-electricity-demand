# ML-pipeline-electricity-demand
# Electricity Demand Forecasting

Predicts next-hour grid electricity demand (`demand_mw`) using classical ML.  
**Primary metric:** MAPE | **Models:** LightGBM (primary), XGBoost, Random Forest

---

## Repository Structure

```
electricity-demand-forecast/
│
├── data/                        # place raw datasets here
│   ├── PGCB_date_power_demand.xlsx
│   ├── weather_data.xlsx
│   └── economic_full_1.csv
│
├── src/
│   ├── data_preprocessing.py    # load, clean, anomaly removal, econ merge
│   ├── feature_engineering.py   # lags, rolling stats, calendar, weather features
│   ├── train_model.py           # LightGBM, XGBoost, RF + TimeSeriesSplit CV
│   └── evaluate.py              # metrics, plots, sanity checks
│
├── notebooks/
│   └── final_pipeline.ipynb               
│
├── outputs/                     
│   ├── predictions.csv
│   ├── predictions_vs_actual.png
│   └── feature_importance.png
│
├── main.py                      # end-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place the three raw data files in the `data/` folder, then run:

```bash
python main.py
```

---

## Pipeline Overview

| Step | File | What it does |
|---|---|---|
| Load | `data_preprocessing.py` | Reads all three datasets |
| Clean | `data_preprocessing.py` | Dedup, resample to hourly, tiered missing fill |
| Anomaly removal | `data_preprocessing.py` | Rolling 7-day median ± 3σ clipping |
| Econ merge | `data_preprocessing.py` | Yearly macro indicators joined by calendar year |
| Feature engineering | `feature_engineering.py` | Lags, rolling stats, cyclic time features, weather interactions |
| Train/test split | `main.py` | Chronological — 2023 as hold-out test set |
| CV | `train_model.py` | TimeSeriesSplit (5 folds) with early stopping |
| Training | `train_model.py` | LightGBM (primary), XGBoost, Random Forest |
| Evaluation | `evaluate.py` | MAPE (primary), RMSE, sanity checks |
| Outputs | `evaluate.py` | Predictions CSV, actual vs predicted plot, feature importance plot |

---

## Key Design Decisions

- **No future leakage:** all rolling features use `.shift(1)` before `.rolling()`. Lag features use only past timestamps.
- **Tiered missing fill:** short gaps (≤ 3h) → linear interpolation; long gaps → same hour previous day.
- **Anomaly clipping, not deletion:** preserves the continuous hourly index required for lag features.
- **Cyclic encoding:** hour, month, and day-of-week encoded as sin/cos pairs to prevent boundary discontinuities.
- **Baseline check:** naive lag-1 prediction computed first — any model failing to beat it signals a pipeline bug.
