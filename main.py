"""
electricity_demand_forecast/main.py
────────────────────────────────────
End-to-end pipeline runner.
Run from the project root:  python main.py
"""

import matplotlib
matplotlib.use("Agg")   # headless — remove this line if running interactively

from src.data_preprocessing import (
    load_data, clean_and_align, handle_missing,
    remove_anomalies, merge_econ,
)
from src.feature_engineering import build_features, get_feature_cols
from src.train_model import cross_validate_lgb, train_all, mape
from src.evaluate import (
    evaluate, sanity_check,
    plot_predictions, plot_feature_importance, save_predictions,
)

# ── Config ────────────────────────────────────────────────────────────────────
DEMAND_PATH  = "data/PGCB_date_power_demand.xlsx"
WEATHER_PATH = "data/weather_data.xlsx"
ECON_PATH    = "data/economic_full_1.csv"
SPLIT_DATE   = "2023-01-01"
TARGET_COL   = "demand_mw"


def main():
    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n── Step 1: Load Data ─────────────────────────────────────────────")
    demand, weather, econ = load_data(DEMAND_PATH, WEATHER_PATH, ECON_PATH)

    # ── 2. Clean & align ──────────────────────────────────────────────────────
    print("\n── Step 2: Clean & Align ─────────────────────────────────────────")
    df = clean_and_align(demand, weather)
    df = handle_missing(df, TARGET_COL)

    # ── 3. Anomaly handling ───────────────────────────────────────────────────
    print("\n── Step 3: Anomaly Handling ──────────────────────────────────────")
    df = remove_anomalies(df, TARGET_COL)

    # ── 4. Economic merge ─────────────────────────────────────────────────────
    print("\n── Step 4: Economic Merge ────────────────────────────────────────")
    df = merge_econ(df, econ)

    # ── 5. Feature engineering ────────────────────────────────────────────────
    print("\n── Step 5: Feature Engineering ──────────────────────────────────")
    df = build_features(df)
    feature_cols = get_feature_cols(df)

    # ── 6. Train / test split ─────────────────────────────────────────────────
    print("\n── Step 6: Train/Test Split ──────────────────────────────────────")
    train = df[df.index < SPLIT_DATE]
    test  = df[df.index >= SPLIT_DATE]

    X_train, y_train = train[feature_cols], train["target"]
    X_test,  y_test  = test[feature_cols],  test["target"]

    print(f"Train: {len(train)} rows  |  {train.index.min().date()} → {train.index.max().date()}")
    print(f"Test : {len(test)}  rows  |  {test.index.min().date()}  → {test.index.max().date()}")
    print(f"Features: {len(feature_cols)}")

    # ── 7. Baseline ───────────────────────────────────────────────────────────
    print("\n── Step 7: Naive Baseline ────────────────────────────────────────")
    baseline_pred = X_test["lag_1"].values
    baseline_mape, _ = evaluate("Baseline (lag_1)", y_test, baseline_pred)
    print("→ All models must beat this.")

    # ── 8. Cross-validation ───────────────────────────────────────────────────
    print("\n── Step 8: Cross-Validation ──────────────────────────────────────")
    cross_validate_lgb(X_train, y_train, n_splits=5)

    # ── 9. Train all models ───────────────────────────────────────────────────
    print("\n── Step 9: Model Training ────────────────────────────────────────")
    lgb_model, xgb_model, rf_model = train_all(X_train, y_train)

    # ── 10. Evaluate ──────────────────────────────────────────────────────────
    print("\n── Step 10: Evaluation ───────────────────────────────────────────")
    lgb_pred = lgb_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    rf_pred  = rf_model.predict(X_test)

    print("\n=== FINAL TEST SET RESULTS ===")
    evaluate("Baseline (lag_1)", y_test, baseline_pred)
    evaluate("LightGBM",         y_test, lgb_pred)
    evaluate("XGBoost",          y_test, xgb_pred)
    evaluate("Random Forest",    y_test, rf_pred)

    sanity_check(lgb_pred, df[TARGET_COL].max())

    # ── 11. Plots & outputs ───────────────────────────────────────────────────
    print("\n── Step 11: Outputs ──────────────────────────────────────────────")
    plot_predictions(y_test, lgb_pred)
    plot_feature_importance(lgb_model, feature_cols)
    save_predictions(test.index, y_test, lgb_pred, xgb_pred, rf_pred, baseline_pred)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
