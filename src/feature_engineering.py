import numpy as np


LAGS = [1, 2, 3, 6, 12, 24, 48, 168]

TARGET_COL = "demand_mw"
DROP_COLS  = [TARGET_COL, "target", "year"]


def add_calendar_features(df):
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    df["quarter"]    = df.index.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # cyclic encoding — prevents discontinuity at period boundaries (23→0, Dec→Jan etc.)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]       / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]       / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]      / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]      / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"]  / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"]  / 7)
    return df


def add_lag_features(df, target_col=TARGET_COL, lags=LAGS):
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col=TARGET_COL):
    # .shift(1) BEFORE .rolling() → strictly no future leakage
    base = df[target_col].shift(1)
    df["rolling_mean_3h"]   = base.rolling(3).mean()
    df["rolling_mean_24h"]  = base.rolling(24).mean()
    df["rolling_std_24h"]   = base.rolling(24).std()
    df["rolling_mean_168h"] = base.rolling(168).mean()
    df["rolling_max_24h"]   = base.rolling(24).max()
    df["rolling_min_24h"]   = base.rolling(24).min()
    return df


def add_weather_features(df):
    if "temperature" in df.columns:
        df["cooling_demand"] = np.maximum(0, df["temperature"] - 24)  # AC load above comfort
        df["heating_demand"] = np.maximum(0, 18 - df["temperature"])  # heating below comfort
        df["temp_change"]    = df["temperature"].diff()
        if "humidity" in df.columns:
            df["heat_index"] = df["temperature"] * df["humidity"]
    return df


def add_target(df, target_col=TARGET_COL):
    df["target"] = df[target_col].shift(-1)
    return df


def drop_nulls_and_finalize(df, lags=LAGS):
    critical = [f"lag_{l}" for l in lags] + ["rolling_mean_24h", "target"]
    before = len(df)
    df = df.dropna(subset=critical)
    print(f"Rows dropped (NaN cleanup): {before - len(df)}")
    print(f"Final dataset shape: {df.shape}")
    return df


def build_features(df):
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_weather_features(df)
    df = add_target(df)
    df = drop_nulls_and_finalize(df)
    return df


def get_feature_cols(df):
    return [c for c in df.columns if c not in DROP_COLS]
