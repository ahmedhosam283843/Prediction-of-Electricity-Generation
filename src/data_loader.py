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

try:
    from codegreen_core.data import energy
    CODEGREEN_AVAILABLE = True
except ImportError:
    print("Warning: codegreen_core package not available. Using simulated data instead.")
    CODEGREEN_AVAILABLE = False

class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback_window, forecast_horizon, target_column, predict_mode='sequence'):
        self.data = data
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.target_column = target_column
        self.predict_mode = predict_mode
        self.features = data.drop(columns=[target_column]).values
        self.target = data[target_column].values
        self.valid_indices = len(data) - lookback_window - forecast_horizon + 1
    
    def __len__(self):
        return self.valid_indices
    
    def __getitem__(self, idx):
        x_start = idx
        x_end = idx + self.lookback_window
        x = self.features[x_start:x_end]
        if self.predict_mode == 'single':
            y_idx = idx + self.lookback_window + self.forecast_horizon - 1
            y = self.target[y_idx]
            y = torch.FloatTensor([y])  # Shape (1,)
        else:
            y_start = x_end
            y_end = y_start + self.forecast_horizon
            y = self.target[y_start:y_end]
            y = torch.FloatTensor(y)  # Shape (forecast_horizon,)
        return torch.FloatTensor(x), y

def load_weather_data_real(lat=52.52, lon=13.405, start_date=None, end_date=None, selected_params=None):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    print(f"Loading real weather data from Open-Meteo for coordinates ({lat}, {lon})")
    print(f"Date range: {start_date} to {end_date}")
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    PARAM_MAPPING = {
        'temperature': 'temperature_2m',
        'wind_speed': 'wind_speed_10m',
        'wind_direction': 'wind_direction_10m',
        'pressure': 'pressure_msl',
        'global_radiation': 'shortwave_radiation',
        'diffuse_solar_radiation': 'diffuse_radiation',
        'sunshine_duration': 'sunshine_duration',
        'humidity': 'relative_humidity_2m',
        'cloudiness': 'cloud_cover',
    }
    open_meteo_vars = list(set(PARAM_MAPPING.get(param, param) for param in selected_params))
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_str,
        'end_date': end_str,
        'hourly': ','.join(open_meteo_vars),
        'timezone': 'auto',
    }
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'hourly' not in data:
            raise ValueError("No hourly data in response")
        hourly = data['hourly']
        df = pd.DataFrame(hourly)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        weather_data = pd.DataFrame(index=df.index)
        for param in selected_params:
            om_var = PARAM_MAPPING.get(param, param)
            if om_var in df.columns:
                weather_data[param] = df[om_var]
            else:
                print(f"Warning: Parameter {param} not available in data")
                weather_data[param] = np.nan
        weather_data = weather_data.interpolate(method='time')
        print(f"Successfully loaded real weather data from Open-Meteo with shape: {weather_data.shape}")
        print(f"Available parameters: {weather_data.columns.tolist()}")
        print(f"Date range: {weather_data.index.min()} to {weather_data.index.max()}")
        return weather_data
    except Exception as e:
        print(f"Error loading real weather data from Open-Meteo: {e}")
        print("Falling back to simulated data")
        return load_weather_data_simulated(start_date, end_date, selected_params)

def load_weather_data_simulated(start_date, end_date, selected_params=None):
    print("Using simulated weather data as fallback")
    raise NotImplementedError("Simulated weather data generation is not implemented yet")

def load_weather_data(data_path=None, selected_params=None, use_real_data=True, 
                     lat=52.52, lon=13.405, start_date=None, end_date=None):
    if use_real_data:
        try:
            return load_weather_data_real(lat, lon, start_date, end_date, selected_params)
        except Exception as e:
            print(f"Error loading real weather data: {e}")
            print("Falling back to simulated data")
    if start_date is None:
        start_date = datetime(2015, 1, 1)
    if end_date is None:
        end_date = datetime(2021, 12, 31)
    return load_weather_data_simulated(start_date, end_date, selected_params)

def load_electricity_data_from_api(energy_type, start_date=None, end_date=None):
    if not CODEGREEN_AVAILABLE:
        raise ImportError("codegreen_core package is not available")
    if start_date is None:
        start_date = datetime.now() - timedelta(days=60)
    if end_date is None:
        end_date = datetime.now()
    country_code = 'DE'
    data_type = 'generation'
    try:
        result = energy(country_code, start_date, end_date, data_type)
        if not result.get('data_available', False):
            raise ValueError(f"Data not available: {result.get('error', 'Unknown error')}")
        df = result['data']
        columns = result.get('columns', {})
        if not isinstance(df.index, pd.DatetimeIndex):
            timestamp_col = next((col for col in ['startTime', 'startTimeUTC', 'time', 'timestamp', 'datetime'] if col in df.columns), None)
            if timestamp_col:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df.set_index(timestamp_col, inplace=True)
            else:
                raise ValueError(f"No timestamp column found in API response. Columns: {df.columns.tolist()}")
        electricity_data = pd.DataFrame(index=df.index)
        if energy_type == 'pv':
            solar_col = next((col for col in ['Solar_per'] if col in df.columns), None)
            if solar_col:
                electricity_data['generation'] = df[solar_col]
            else:
                raise ValueError("Solar generation data not found")
        elif energy_type == 'wind':
            wind_cols = [col for col in ['Wind_per'] if col in df.columns]
            if wind_cols:
                electricity_data['generation'] = df[wind_cols].sum(axis=1)
            else:
                raise ValueError("Wind generation data not found")
        elif energy_type == 'combined':
            if 'renewableTotalWS' in df.columns:
                electricity_data['generation'] = df['renewableTotalWS']
                print("Using 'renewableTotalWS' for combined energy generation")
            else:
                raise ValueError("'renewableTotalWS' column not found in API response")
        return electricity_data, columns
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        raise

def load_electricity_data(data_path, energy_type, start_date=None, end_date=None):
    if CODEGREEN_AVAILABLE:
        try:
            electricity_data, _ = load_electricity_data_from_api(energy_type, start_date, end_date)
            print(f"Successfully loaded {energy_type} data from CodeGreen API")
            return electricity_data
        except Exception as e:
            print(f"Failed to load data from CodeGreen API: {str(e)}")
            print("Falling back to simulated data")
            raise e
    print(f"Using simulated {energy_type} data")
    if start_date is None:
        start_date = datetime(2015, 1, 1)
    if end_date is None:
        end_date = datetime(2021, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    electricity_data = pd.DataFrame(index=dates)
    if energy_type == 'pv':
        values = np.random.uniform(0, 20000, size=len(dates))
        values[electricity_data.index.hour < 6] = 0
        values[electricity_data.index.hour > 20] = 0
        season = np.sin(2 * np.pi * (electricity_data.index.dayofyear / 365))
        values *= (0.5 + 0.5 * season)
        daily = np.sin(np.pi * (electricity_data.index.hour / 12))
        values *= np.maximum(0, daily)
    elif energy_type == 'wind':
        values = np.random.uniform(0, 30000, size=len(dates))
        season = np.cos(2 * np.pi * (electricity_data.index.dayofyear / 365))
        values *= (0.5 + 0.5 * season)
    electricity_data['generation'] = values
    print(f"Generated simulated {energy_type} data with shape: {electricity_data.shape}")
    print(f"Date range: {electricity_data.index.min()} to {electricity_data.index.max()}")
    return electricity_data

def create_time_features(dates):
    time_features = pd.DataFrame(index=dates)
    time_features['hour_sin'] = np.sin(2 * np.pi * dates.hour / 24)
    time_features['hour_cos'] = np.cos(2 * np.pi * dates.hour / 24)
    time_features['day_sin'] = np.sin(2 * np.pi * dates.day / dates.days_in_month)
    time_features['day_cos'] = np.cos(2 * np.pi * dates.day / dates.days_in_month)
    time_features['month_sin'] = np.sin(2 * np.pi * dates.month / 12)
    time_features['month_cos'] = np.cos(2 * np.pi * dates.month / 12)
    return time_features

def preprocess_data(weather_data, electricity_data, selected_params=None):
    if selected_params:
        weather_data = weather_data[selected_params]
    if len(weather_data) == 0:
        raise ValueError("Weather data is empty")
    if len(electricity_data) == 0:
        raise ValueError("Electricity data is empty")
    print(f"Weather data date range: {weather_data.index.min()} to {weather_data.index.max()}")
    print(f"Electricity data date range: {electricity_data.index.min()} to {electricity_data.index.max()}")
    if weather_data.index.tz is not None:
        print("Converting weather data timezone to UTC")
        weather_data.index = weather_data.index.tz_convert('UTC').tz_localize(None)
    if electricity_data.index.tz is not None:
        print("Converting electricity data timezone to UTC")
        electricity_data.index = electricity_data.index.tz_convert('UTC').tz_localize(None)
    print(f"After timezone conversion:")
    print(f"Weather data date range: {weather_data.index.min()} to {weather_data.index.max()}")
    print(f"Electricity data date range: {electricity_data.index.min()} to {electricity_data.index.max()}")
    start_date = max(weather_data.index.min(), electricity_data.index.min())
    end_date = min(weather_data.index.max(), electricity_data.index.max())
    print(f"Overlapping date range: {start_date} to {end_date}")
    if start_date >= end_date:
        print("ERROR: No overlapping time period between weather and electricity data!")
        print(f"Weather data: {weather_data.index.min()} to {weather_data.index.max()}")
        print(f"Electricity data: {electricity_data.index.min()} to {electricity_data.index.max()}")
        raise ValueError("No overlapping data between weather and electricity datasets")
    weather_data = weather_data.loc[start_date:end_date]
    electricity_data = electricity_data.loc[start_date:end_date]
    merged_data = pd.merge(weather_data, electricity_data, left_index=True, right_index=True, how='inner')
    if len(merged_data) == 0:
        raise ValueError("No overlapping data between weather and electricity datasets after merging")
    print(f"Merged data shape: {merged_data.shape}")
    print(f"Merged data columns: {merged_data.columns.tolist()}")
    if merged_data.isnull().sum().sum() > 0:
        print(f"Found {merged_data.isnull().sum().sum()} missing values. Interpolating...")
        merged_data = merged_data.interpolate(method='time')
    
    # Check for empty columns and remove them
    empty_cols = [col for col in merged_data.columns if merged_data[col].isna().all()]
    if empty_cols:
        print(f"Removing empty columns: {empty_cols}")
        merged_data = merged_data.drop(columns=empty_cols)
    for lag in range(1, 25):
        merged_data[f'generation_lag_{lag}'] = merged_data['generation'].shift(lag)
    merged_data['generation_roll_mean_24'] = merged_data['generation'].rolling(window=24).mean()
    merged_data['generation_roll_std_24'] = merged_data['generation'].rolling(window=24).std()
    merged_data.dropna(inplace=True)
    time_features = create_time_features(merged_data.index)
    merged_data = pd.concat([merged_data, time_features], axis=1)
    return merged_data, {}

def create_data_loaders(data, lookback_window, forecast_horizon, target_column, 
                        batch_size, validation_split=0.2, test_split=0.1, predict_mode='sequence'):
    n = len(data)
    train_end = int(n * (1 - validation_split - test_split))
    val_end = int(n * (1 - test_split))
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    scalers = {}
    for column in train_data.columns:
        scaler = MinMaxScaler()
        train_data[column] = scaler.fit_transform(train_data[column].values.reshape(-1, 1)).flatten()
        val_data[column] = scaler.transform(val_data[column].values.reshape(-1, 1)).flatten()
        test_data[column] = scaler.transform(test_data[column].values.reshape(-1, 1)).flatten()
        scalers[column] = scaler
    train_dataset = TimeSeriesDataset(train_data, lookback_window, forecast_horizon, target_column, predict_mode)
    val_dataset = TimeSeriesDataset(val_data, lookback_window, forecast_horizon, target_column, predict_mode)
    test_dataset = TimeSeriesDataset(test_data, lookback_window, forecast_horizon, target_column, predict_mode)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_index = test_data.index
    return train_loader, val_loader, test_loader, scalers, test_index

def prepare_data(config, energy_type):
    selected_params = config.WEATHER_PARAMS_ALL if energy_type == 'combined' else (
        config.WEATHER_PARAMS_PV if energy_type == 'pv' else config.WEATHER_PARAMS_WIND
    )
    start_date = pd.to_datetime(config.DATA_START_DATE) if getattr(config, 'DATA_START_DATE', None) else datetime.now() - timedelta(days=60)
    end_date = pd.to_datetime(config.DATA_END_DATE) if getattr(config, 'DATA_END_DATE', None) else datetime.now()
    cache_dir = os.path.join(config.DATA_PATH, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"processed_{energy_type}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
    )
    if os.path.exists(cache_file):
        print(f"Loading processed data from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            processed_data, scalers = pickle.load(f)
    else:
        electricity_data = load_electricity_data(config.DATA_PATH, energy_type, start_date, end_date)
        weather_data = load_weather_data(config.DATA_PATH, selected_params, True, config.LOCATION_LAT, config.LOCATION_LON, start_date, end_date)
        processed_data, scalers = preprocess_data(weather_data, electricity_data, selected_params)
        with open(cache_file, "wb") as f:
            pickle.dump((processed_data, scalers), f)
        print(f"Processed data saved to cache: {cache_file}")
    train_loader, val_loader, test_loader, scalers, test_index = create_data_loaders(
        processed_data, config.LOOKBACK_WINDOW, config.FORECAST_HORIZON, 'generation',
        config.BATCH_SIZE, config.VALIDATION_SPLIT, config.TEST_SPLIT, config.PREDICT_MODE
    )
    return train_loader, val_loader, test_loader, scalers, test_index