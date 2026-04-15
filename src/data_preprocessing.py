import pandas as pd
import numpy as np


def load_data(demand_path, weather_path, econ_path):
    demand = pd.read_excel(demand_path)
    weather = pd.read_excel(weather_path)
    econ = pd.read_csv(econ_path)
    print(f"Demand shape : {demand.shape}")
    print(f"Weather shape: {weather.shape}")
    print(f"Econ shape   : {econ.shape}")
    return demand, weather, econ


def clean_and_align(demand, weather):
    # parse datetimes
    demand["datetime"]  = pd.to_datetime(demand["datetime"],  errors="coerce")
    weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
    demand  = demand.dropna(subset=["datetime"])
    weather = weather.dropna(subset=["datetime"])

    # remove duplicates
    before = len(demand)
    demand  = demand.drop_duplicates(subset="datetime")
    weather = weather.drop_duplicates(subset="datetime")
    print(f"Demand duplicates removed: {before - len(demand)}")

    # merge weather onto demand
    df = demand.merge(weather, on="datetime", how="left")

    # strict hourly index
    df = df.set_index("datetime").sort_index()
    df = df.resample("H").mean()

    print(f"Date range : {df.index.min()} → {df.index.max()}")
    print(f"Total hours: {len(df)}")
    return df


def tiered_fill(series, short_gap=3):
    """
    Gaps <= short_gap hours → linear interpolation (sensor dropout).
    Gaps >  short_gap hours → same hour previous day (structural outage).
    """
    s = series.copy()
    mask   = s.isna()
    run_id  = (mask != mask.shift()).cumsum()
    run_len = mask.groupby(run_id).transform("sum")

    short_mask = mask & (run_len <= short_gap)
    s[short_mask] = s.interpolate(method="time")[short_mask]

    still_nan = s.isna()
    s[still_nan] = s.shift(24)[still_nan]

    s = s.ffill().bfill()
    return s


def handle_missing(df, target_col="demand_mw"):
    print(f"Missing {target_col} before fill: {df[target_col].isna().sum()}")
    df[target_col] = tiered_fill(df[target_col])

    weather_cols = [c for c in df.columns if c != target_col]
    df[weather_cols] = df[weather_cols].interpolate(method="time").ffill().bfill()

    print(f"Missing {target_col} after fill : {df[target_col].isna().sum()}")
    return df


def remove_anomalies(df, target_col="demand_mw", window=24 * 7, n_sigma=3):
    """
    Rolling 7-day median ± n_sigma * std clipping.
    Uses center=True for symmetric window — only applied at prep time, not live.
    """
    roll_med = df[target_col].rolling(window=window, center=True, min_periods=12).median()
    roll_std = df[target_col].rolling(window=window, center=True, min_periods=12).std()

    upper = roll_med + n_sigma * roll_std
    lower = roll_med - n_sigma * roll_std

    df[target_col] = df[target_col].clip(lower=lower, upper=upper)
    df[target_col] = df[target_col].interpolate(method="time").ffill().bfill()
    print("Anomaly clipping done.")
    return df


def merge_econ(df, econ):
    """
    Join annual macro indicators onto hourly data by calendar year.
    Only keeps demand-relevant columns.
    """
    ECON_COLS = ["year", "gdp_growth", "industrial_output", "population"]
    available = [c for c in ECON_COLS if c in econ.columns]
    econ = econ[available].copy()
    econ["year"] = econ["year"].astype(int)

    df["year"] = df.index.year
    df = df.merge(econ, on="year", how="left")
    print(f"Econ columns added: {[c for c in available if c != 'year']}")
    return df
