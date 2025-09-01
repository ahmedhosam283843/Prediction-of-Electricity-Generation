# src/data_loader.py
from pathlib import Path
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import requests
import warnings
from .config import (
    SELECTED_WEATHER_VARS, TIME_FEATURES,
    ARCHIVE_URL, FORECAST_URL, MAX_PAST_DAYS_FORECAST, CHUNK_LENGTH_FORECAST,
    COUNTRY, COUNTRY_COORDS, LAT, LON
)

try:
    from codegreen_core.data import energy
    CODEGREEN_AVAILABLE = True
except ImportError:
    print("Warning: codegreen_core package is not available. Using simulated data instead.")
    CODEGREEN_AVAILABLE = False

# ────────────────────────────────────────────────────────────────
# disk-cache helper (key per-country)
# ────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _combined_cache_path(country_code: str, start_date: datetime, end_date: datetime) -> Path:
    key = f"{country_code}_elec_combined_{start_date:%Y%m%d}_{end_date:%Y%m%d}.pkl"
    return CACHE_DIR / key


def load_weather_data(
    lat: float | None = None,
    lon: float | None = None,
    start_date: datetime | None = None,
    end_date:   datetime | None = None,
    selected_params=None,
):
    """Fetch hourly weather data from Open-Meteo."""
    if start_date is None:
        start_date = datetime.utcnow() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.utcnow()

    # default to config country coords
    if lat is None or lon is None:
        lat = LAT
        lon = LON

    today_utc = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    oldest_forecast_ok = today_utc - timedelta(days=MAX_PAST_DAYS_FORECAST)

    chunks = []
    if start_date < oldest_forecast_ok:
        hist_end = min(end_date, oldest_forecast_ok - timedelta(days=1))
        chunks.append(("archive", start_date, hist_end))

    recent_start = max(start_date, oldest_forecast_ok)
    if recent_start <= end_date:
        cur = recent_start
        while cur <= end_date:
            cur_end = min(cur + timedelta(days=CHUNK_LENGTH_FORECAST - 1), end_date)
            chunks.append(("forecast", cur, cur_end))
            cur = cur_end + timedelta(days=1)

    # Map your internal names → Open-Meteo field names
    PARAM_MAPPING = {
        "temperature":     "temperature_2m",
        "wind_speed":      "wind_speed_10m",
        "wind_direction":  "wind_direction_10m",
        "pressure":        "pressure_msl",
        "global_radiation": "shortwave_radiation",
        "diffuse_solar_radiation": "diffuse_radiation",
        "sunshine_duration":      "sunshine_duration",
        "humidity":        "relative_humidity_2m",
        "cloudiness":      "cloud_cover",
    }
    open_meteo_vars = list({PARAM_MAPPING.get(p, p) for p in selected_params})

    frames = []
    session = requests.Session()

    for kind, s, e in chunks:
        base_url = ARCHIVE_URL if kind == "archive" else FORECAST_URL
        params = {
            "latitude":  lat,
            "longitude": lon,
            "start_date": s.strftime("%Y-%m-%d"),
            "end_date":   e.strftime("%Y-%m-%d"),
            "hourly":     ",".join(open_meteo_vars),
            "timezone":   "UTC",
        }
        resp = session.get(base_url, params=params, timeout=30)
        try:
            resp.raise_for_status()
        except Exception as ex:
            raise RuntimeError(f"[Open-Meteo] {kind} {s.date()}–{e.date()} failed: {ex}") from ex

        data = resp.json()
        if "hourly" not in data:
            raise ValueError("No hourly block in API response")

        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        frames.append(df)

    df_all = pd.concat(frames).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]
    if "wind_speed_10m" in df_all.columns:
        df_all["wind_speed_10m"] /= 3.6  # km/h → m/s
    if "shortwave_radiation" in df_all.columns:
        df_all["shortwave_radiation"] /= 1000  # W/m² → kWh/m² (hourly)

    weather = pd.DataFrame(index=df_all.index)
    for p in selected_params:
        api_name = PARAM_MAPPING.get(p, p)
        if api_name in df_all.columns:
            weather[p] = df_all[api_name]
        else:
            warnings.warn(f"{p} not returned by API – filled with NaN")
            weather[p] = np.nan

    weather = weather.interpolate(method="time")
    print(f"[Open-Meteo] got {weather.shape} rows ({weather.index.min()} → {weather.index.max()})")
    return weather


def load_electricity_data_from_api(energy_type, start_date=None, end_date=None, country_code: str | None = None):
    if not CODEGREEN_AVAILABLE:
        raise ImportError("codegreen_core package is not available")

    if start_date is None:
        start_date = datetime.now() - timedelta(days=60)
    if end_date is None:
        end_date = datetime.now()
    if country_code is None:
        country_code = COUNTRY

    # serve cached combined data per-country
    if energy_type == "combined":
        cache_file = _combined_cache_path(country_code, start_date, end_date)
        if cache_file.exists():
            print(f"[cache] {country_code} electricity {start_date.date()}–{end_date.date()} loaded")
            return pd.read_pickle(cache_file), {}

    data_type = "generation"
    result = energy(country_code, start_date, end_date, data_type)
    if not result.get("data_available", False):
        err = result.get("error", "Unknown error")
        raise ValueError(f"Data not available: {err}")

    df = result["data"]
    columns = result.get("columns", {})

    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = next((c for c in ["startTime", "startTimeUTC", "time", "timestamp", "datetime"] if c in df.columns), None)
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col])
            df = df.set_index(ts_col)
        else:
            raise ValueError("Cannot locate timestamp column in API response")

    electricity_data = pd.DataFrame(index=df.index)

    if energy_type == "combined":
        required_cols = {"percentRenewable", "total"}
        if required_cols.issubset(df.columns):
            electricity_data["percentRenewable"] = df["percentRenewable"] / 100.0
            cache_file = _combined_cache_path(country_code, start_date, end_date)
            electricity_data.to_pickle(cache_file)
            print(f"[cache] saved → {cache_file.name}")
        else:
            raise ValueError(f"Columns {required_cols} missing in API response")

    elif energy_type == "pv":
        solar_col = next((c for c in ["Solar_per"] if c in df.columns), None)
        if not solar_col:
            raise ValueError("Solar generation data not found")
        electricity_data["generation"] = df[solar_col]

    elif energy_type == "wind":
        wind_cols = [c for c in ["Wind_per"] if c in df.columns]
        if not wind_cols:
            raise ValueError("Wind generation data not found")
        electricity_data["generation"] = df[wind_cols].sum(axis=1)

    return electricity_data, columns


def load_electricity_data(data_path, energy_type, start_date=None, end_date=None, country: str | None = None):
    if CODEGREEN_AVAILABLE:
        try:
            electricity_data, _ = load_electricity_data_from_api(
                energy_type, start_date, end_date, country_code=(country or COUNTRY)
            )
            if electricity_data.index.tz is None:
                electricity_data.index = electricity_data.index.tz_localize('Europe/Berlin')
                electricity_data.index = electricity_data.index.tz_convert('UTC').tz_localize(None)
            print(f"Successfully loaded {energy_type} data from CodeGreen API ({country or COUNTRY})")
            return electricity_data
        except Exception as e:
            print(f"Failed to load data from CodeGreen API: {str(e)}")
            raise e
    else:
        raise ImportError("codegreen_core package is not available.")


def preprocess_data(weather_data, electricity_data, selected_params=None):
    if selected_params:
        weather_data = weather_data[selected_params]
    if len(weather_data) == 0:
        raise ValueError("Weather data is empty")
    if len(electricity_data) == 0:
        raise ValueError("Electricity data is empty")

    if not isinstance(weather_data.index, pd.DatetimeIndex):
        raise ValueError("Weather data index must be a DatetimeIndex")
    if not isinstance(electricity_data.index, pd.DatetimeIndex):
        raise ValueError("Electricity data index must be a DatetimeIndex")

    print(f"Weather data date range: {weather_data.index.min()} to {weather_data.index.max()}")
    print(f"Electricity data date range: {electricity_data.index.min()} to {electricity_data.index.max()}")

    if weather_data.index.tz is not None:
        print("Converting weather data timezone to UTC")
        weather_data.index = weather_data.index.tz_convert('UTC').tz_localize(None)
    else:
        print("Weather data index is naive; no timezone conversion applied")

    if electricity_data.index.tz is not None:
        print("Converting electricity data timezone to UTC")
        electricity_data.index = electricity_data.index.tz_convert('UTC').tz_localize(None)
    else:
        print("Electricity data index is naive; no timezone conversion applied")

    assert weather_data.index.tz is None
    assert electricity_data.index.tz is None

    print("After timezone conversion:")
    print(f"Weather data date range: {weather_data.index.min()} to {weather_data.index.max()}")
    print(f"Electricity data date range: {electricity_data.index.min()} to {electricity_data.index.max()}")

    start_date = max(weather_data.index.min(), electricity_data.index.min())
    end_date = min(weather_data.index.max(), electricity_data.index.max())
    print(f"Overlapping date range: {start_date} to {end_date}")
    if start_date >= end_date:
        raise ValueError("No overlapping data between weather and electricity datasets")

    weather_data = weather_data.loc[start_date:end_date]
    electricity_data = electricity_data.loc[start_date:end_date]
    merged_data = pd.merge(weather_data, electricity_data, left_index=True, right_index=True, how='inner')
    if len(merged_data) == 0:
        raise ValueError("No overlapping data after merging")
    if merged_data.isnull().sum().sum() > 0:
        print(f"Found {merged_data.isnull().sum().sum()} missing values. Interpolating...")
        merged_data = merged_data.interpolate(method='time')

    return merged_data, {}


def fetch_and_prepare(start: datetime, end: datetime, country: str | None = None):
    """
    Returns merged & cleaned dataframe and the list of feature columns.
    Country and location are sourced from config by default.
    """
    country = (country or COUNTRY).upper()
    lat, lon = COUNTRY_COORDS.get(country, (LAT, LON))

    weather = load_weather_data(selected_params=SELECTED_WEATHER_VARS,
                                start_date=start, end_date=end,
                                lat=lat, lon=lon)

    elec_raw = load_electricity_data(None, "combined", start, end, country=country)

    if {"renewableTotalWS", "total"}.issubset(elec_raw.columns):
        elec_raw["y"] = elec_raw["renewableTotalWS"]
    elif "percentRenewable" in elec_raw.columns:
        elec_raw = elec_raw.rename(columns={"percentRenewable": "y"})
    else:
        raise RuntimeError("Could not build renewable-percentage target")

    df, _ = preprocess_data(weather, elec_raw, selected_params=SELECTED_WEATHER_VARS)

    hr, wkd, doy = df.index.hour, df.index.weekday, df.index.dayofyear
    df["hr_sin"]  = np.sin(2*np.pi*hr / 24)
    df["hr_cos"]  = np.cos(2*np.pi*hr / 24)
    df["wkd_sin"] = np.sin(2*np.pi*wkd/ 7)
    df["wkd_cos"] = np.cos(2*np.pi*wkd/ 7)
    df["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    df["doy_cos"] = np.cos(2*np.pi*doy/365.25)

    feat_cols = SELECTED_WEATHER_VARS + TIME_FEATURES
    return df[feat_cols + ["y"]], feat_cols


def build_sequences(df, lookback, horizon, include_y_hist: bool = False, y_hist_k: int = 72):
    """
    Build rolling windows:
      - X: (N, lookback, F) features (optionally + past target channel)
      - y: (N, horizon)      future path
      - t: (N, horizon)      timestamps for the future window
    """
    vals  = df.values.astype(np.float32)
    n     = len(vals)
    feats = df.shape[1] - 1
    X, y, t = [], [], []
    for i in range(lookback, n - horizon + 1):
        X_feats = vals[i-lookback:i, :feats]
        if include_y_hist:
            y_hist_full = vals[i-lookback:i, -1]  # (lookback,)
            k = min(y_hist_k, lookback)
            y_ch = np.empty(lookback, dtype=np.float32)
            y_ch[:lookback-k] = y_hist_full[lookback-k]
            y_ch[lookback-k:] = y_hist_full[lookback-k:]
            X_win = np.concatenate([X_feats, y_ch.reshape(-1, 1)], axis=1)
        else:
            X_win = X_feats
        X.append(X_win)
        y.append(vals[i:i+horizon, -1])
        t.append(df.index.values[i:i+horizon])
    return np.asarray(X), np.asarray(y), np.asarray(t, dtype='datetime64[ns]')