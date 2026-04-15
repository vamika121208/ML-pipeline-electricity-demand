import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# ── Metrics ───────────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 1
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(name, y_true, y_pred):
    m = mape(y_true, y_pred)
    r = rmse(y_true, y_pred)
    print(f"{name:30s}  MAPE={m:.3f}%   RMSE={r:.2f} MW")
    return m, r


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(y_pred, hist_max, label="LightGBM"):
    assert (y_pred > 0).all(),                 f"ERROR [{label}]: negative predictions found"
    assert (y_pred < hist_max * 1.5).all(),    f"ERROR [{label}]: predictions exceed 1.5× historical max"
    print(f"✓ Sanity check passed  ({label} range: {y_pred.min():.0f}–{y_pred.max():.0f} MW)")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_predictions(y_test, lgb_pred, n_days=14, save_path="outputs/predictions_vs_actual.png"):
    n = 24 * n_days
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_test.values[:n], label="Actual",   linewidth=1.2)
    ax.plot(lgb_pred[:n],      label="LightGBM", linewidth=1.0, alpha=0.85)
    ax.set_title(f"Actual vs Predicted — First {n_days} Days of Test Set")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved → {save_path}")


def plot_feature_importance(model, feature_cols, top_n=20,
                            save_path="outputs/feature_importance.png"):
    feat_imp = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=True)
        .tail(top_n)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    feat_imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"LightGBM — Top {top_n} Feature Importances", fontsize=13)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Feature importance plot saved → {save_path}")
    print("\nTop 10 features:")
    print(feat_imp.tail(10).sort_values(ascending=False).to_string())


# ── Save predictions ──────────────────────────────────────────────────────────

def save_predictions(test_index, y_test, lgb_pred, xgb_pred, rf_pred,
                     baseline_pred, save_path="outputs/predictions.csv"):
    df = pd.DataFrame({
        "datetime":   test_index,
        "actual":     y_test.values,
        "lgb_pred":   lgb_pred,
        "xgb_pred":   xgb_pred,
        "rf_pred":    rf_pred,
        "baseline":   baseline_pred,
    })
    df.to_csv(save_path, index=False)
    print(f"Predictions saved → {save_path}")
