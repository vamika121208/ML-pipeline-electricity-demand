import numpy as np
import lightgbm as lgbm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

RANDOM_STATE = 42


# ── Metric ────────────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 1   # guard against near-zero division
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_lgb(X_train, y_train, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    lgb_cv = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    print(f"TimeSeriesSplit CV ({n_splits} folds):")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr,  X_val = X_train.iloc[tr_idx],  X_train.iloc[val_idx]
        y_tr,  y_val = y_train.iloc[tr_idx],  y_train.iloc[val_idx]

        lgb_cv.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgbm.early_stopping(stopping_rounds=50, verbose=False),
                lgbm.log_evaluation(period=-1),
            ],
        )
        score = mape(y_val, lgb_cv.predict(X_val))
        cv_scores.append(score)
        print(f"  Fold {fold}: MAPE = {score:.3f}%")

    print(f"\nMean CV MAPE: {np.mean(cv_scores):.3f}% ± {np.std(cv_scores):.3f}%")
    return cv_scores


# ── Final model training ──────────────────────────────────────────────────────

def train_lightgbm(X_train, y_train):
    """
    Uses last 10% of training data as internal validation for early stopping.
    n_estimators=2000 with early stopping finds the true optimal count.
    """
    val_cutoff = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:val_cutoff], X_train.iloc[val_cutoff:]
    y_tr, y_val = y_train.iloc[:val_cutoff], y_train.iloc[val_cutoff:]

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgbm.early_stopping(stopping_rounds=50, verbose=True),
            lgbm.log_evaluation(period=100),
        ],
    )
    print(f"LightGBM best iteration: {model.best_iteration_}")
    return model


def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_all(X_train, y_train):
    print("\nTraining LightGBM...")
    lgb = train_lightgbm(X_train, y_train)
    print("\nTraining XGBoost...")
    xgb = train_xgboost(X_train, y_train)
    print("\nTraining Random Forest...")
    rf  = train_random_forest(X_train, y_train)
    print("\nAll models trained.")
    return lgb, xgb, rf
