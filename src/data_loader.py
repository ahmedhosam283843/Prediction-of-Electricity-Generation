"""
Data loading and preprocessing utilities for wind and solar energy prediction.

This module handles:
1. Loading weather data from German Weather Service (DWD)
2. Loading electricity generation data from SMARD via CodeGreen API
3. Data preprocessing and feature engineering
4. Dataset creation for model training
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta

# Import CodeGreen API for real data
try:
    from codegreen_core.data import energy
    CODEGREEN_AVAILABLE = True
except ImportError:
    print("Warning: codegreen_core package not available. Using simulated data instead.")
    CODEGREEN_AVAILABLE = False

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting with sliding window approach.
    """
    def __init__(self, data, lookback_window, forecast_horizon, target_column):
        """
        Initialize the dataset.
        
        Args:
            data (pd.DataFrame): DataFrame containing features and target
            lookback_window (int): Number of time steps to look back
            forecast_horizon (int): Number of time steps to predict ahead
            target_column (str): Name of the target column to predict
        """
        self.data = data
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.target_column = target_column
        
        # Extract features and target
        self.features = data.drop(columns=[target_column]).values
        self.target = data[target_column].values
        
        # Calculate valid indices
        self.valid_indices = len(data) - lookback_window - forecast_horizon + 1
        
    def __len__(self):
        return self.valid_indices
    
    def __getitem__(self, idx):
        # Get input sequence
        x_start = idx
        x_end = idx + self.lookback_window
        x = self.features[x_start:x_end]
        
        # Get target sequence
        y_start = x_end
        y_end = y_start + self.forecast_horizon
        y = self.target[y_start:y_end]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

def load_weather_data(data_path, selected_params=None):
    """
    Load weather data from DWD.
    
    Args:
        data_path (str): Path to the weather data
        selected_params (list): List of weather parameters to select
        
    Returns:
        pd.DataFrame: Processed weather data
    """
    # In a real implementation, this would load data from DWD
    # For this implementation, we'll simulate the data loading
    
    # Create a date range for demonstration
    dates = pd.date_range(start='2015-01-01', end='2021-12-31', freq='h')
    
    # Create a DataFrame with all weather parameters
    weather_params = [
        'sunshine_duration', 'global_radiation', 'diffuse_solar_radiation',
        'wind_speed', 'wind_direction', 'temperature', 'pressure',
        'humidity', 'cloudiness'
    ]
    
    # Initialize DataFrame with dates
    weather_data = pd.DataFrame(index=dates)
    
    # Add weather parameters with random values
    for param in weather_params:
        # Generate random values appropriate for each parameter
        if param == 'sunshine_duration':
            # Sunshine duration in minutes (0-60)
            values = np.random.randint(0, 61, size=len(dates))
            # Set to 0 during night hours
            values[weather_data.index.hour < 6] = 0
            values[weather_data.index.hour > 20] = 0
        elif param in ['global_radiation', 'diffuse_solar_radiation']:
            # Radiation in W/m² (0-1000)
            values = np.random.randint(0, 1001, size=len(dates))
            # Set to 0 during night hours
            values[weather_data.index.hour < 6] = 0
            values[weather_data.index.hour > 20] = 0
        elif param == 'wind_speed':
            # Wind speed in m/s (0-30)
            values = np.random.uniform(0, 30, size=len(dates))
        elif param == 'wind_direction':
            # Wind direction in degrees (0-360)
            values = np.random.uniform(0, 360, size=len(dates))
        elif param == 'temperature':
            # Temperature in °C (-10 to 35)
            values = np.random.uniform(-10, 35, size=len(dates))
            # Add seasonal pattern
            season = np.sin(2 * np.pi * (weather_data.index.dayofyear / 365))
            values += 15 * season
        elif param == 'pressure':
            # Pressure in hPa (980-1030)
            values = np.random.uniform(980, 1030, size=len(dates))
        elif param == 'humidity':
            # Humidity in % (20-100)
            values = np.random.uniform(20, 100, size=len(dates))
        elif param == 'cloudiness':
            # Cloudiness in % (0-100)
            values = np.random.uniform(0, 100, size=len(dates))
        
        weather_data[param] = values
    
    # Filter selected parameters if specified
    if selected_params:
        weather_data = weather_data[selected_params]
    
    return weather_data

def load_electricity_data_from_api(energy_type, start_date=None, end_date=None):
    """
    Load electricity generation data from CodeGreen API.
    
    Args:
        energy_type (str): Type of energy ('pv' or 'wind')
        start_date (datetime): Start date for data retrieval
        end_date (datetime): End date for data retrieval
        
    Returns:
        pd.DataFrame: Processed electricity generation data
        dict: Column categories from the API
    """
    if not CODEGREEN_AVAILABLE:
        raise ImportError("codegreen_core package is not available")
    
    # Default to recent data if dates not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=60)
    if end_date is None:
        end_date = datetime.now()
    
    # Get data from CodeGreen API
    country_code = 'DE'  # Germany
    data_type = 'generation'
    
    try:
        result = energy(country_code, start_date, end_date, data_type)
        
        if not result.get('data_available', False):
            error_msg = result.get('error', 'Unknown error')
            raise ValueError(f"Data not available from CodeGreen API: {error_msg}")
        
        # Extract dataframe and column categories
        df = result['data']
        columns = result.get('columns', {})
        
        # Process the dataframe for the specific energy type
        if energy_type == 'pv':
            # For PV, use Solar column
            if 'Solar' in df.columns:
                electricity_data = pd.DataFrame(index=df.index)
                electricity_data['generation'] = df['Solar']
            else:
                # Try to find solar in renewable columns
                solar_cols = [col for col in df.columns if 'solar' in col.lower()]
                if solar_cols:
                    electricity_data = pd.DataFrame(index=df.index)
                    electricity_data['generation'] = df[solar_cols[0]]
                else:
                    raise ValueError("Solar generation data not found in API response")
        
        elif energy_type == 'wind':
            # For wind, combine onshore and offshore if available
            wind_cols = [col for col in df.columns if 'wind' in col.lower()]
            if wind_cols:
                electricity_data = pd.DataFrame(index=df.index)
                electricity_data['generation'] = df[wind_cols].sum(axis=1)
            else:
                raise ValueError("Wind generation data not found in API response")
        
        return electricity_data, columns
    
    except Exception as e:
        print(f"Error fetching data from CodeGreen API: {str(e)}")
        raise

def load_electricity_data(data_path, energy_type, start_date=None, end_date=None):
    """
    Load electricity generation data from SMARD via CodeGreen API or fallback to simulation.
    
    Args:
        data_path (str): Path to the electricity data (used only for simulated data)
        energy_type (str): Type of energy ('pv' or 'wind')
        start_date (datetime): Start date for data retrieval (used only for API)
        end_date (datetime): End date for data retrieval (used only for API)
        
    Returns:
        pd.DataFrame: Processed electricity generation data
    """
    # Try to load data from CodeGreen API if available
    if CODEGREEN_AVAILABLE:
        try:
            electricity_data, _ = load_electricity_data_from_api(energy_type, start_date, end_date)
            print(f"Successfully loaded {energy_type} data from CodeGreen API")
            return electricity_data
        except Exception as e:
            print(f"Failed to load data from CodeGreen API: {str(e)}")
            print("Falling back to simulated data")
    
    # Fallback to simulated data
    print(f"Using simulated {energy_type} data")
    
    # Create a date range for demonstration
    dates = pd.date_range(start='2015-01-01', end='2021-12-31', freq='h')
    
    # Create a DataFrame with electricity generation
    electricity_data = pd.DataFrame(index=dates)
    
    # Add electricity generation with random values and patterns
    if energy_type == 'pv':
        # PV generation in MWh (0-20000)
        values = np.random.uniform(0, 20000, size=len(dates))
        # Set to 0 during night hours
        values[electricity_data.index.hour < 6] = 0
        values[electricity_data.index.hour > 20] = 0
        # Add seasonal pattern
        season = np.sin(2 * np.pi * (electricity_data.index.dayofyear / 365))
        values *= (0.5 + 0.5 * season)
        # Add daily pattern
        daily = np.sin(np.pi * (electricity_data.index.hour / 12))
        values *= np.maximum(0, daily)
    elif energy_type == 'wind':
        # Wind generation in MWh (0-30000)
        values = np.random.uniform(0, 30000, size=len(dates))
        # Add seasonal pattern (more wind in winter)
        season = np.cos(2 * np.pi * (electricity_data.index.dayofyear / 365))
        values *= (0.5 + 0.5 * season)
    
    electricity_data['generation'] = values
    
    return electricity_data

def preprocess_data(weather_data, electricity_data, selected_params=None):
    """
    Preprocess and merge weather and electricity data.
    
    Args:
        weather_data (pd.DataFrame): Weather data
        electricity_data (pd.DataFrame): Electricity generation data
        selected_params (list): List of weather parameters to select
        
    Returns:
        pd.DataFrame: Merged and preprocessed data
        dict: Scalers used for normalization
    """
    # Filter selected parameters if specified
    if selected_params:
        weather_data = weather_data[selected_params]
    
    # Ensure both dataframes have data
    if len(weather_data) == 0:
        raise ValueError("Weather data is empty")
    if len(electricity_data) == 0:
        raise ValueError("Electricity data is empty")
    
    # Align the date ranges - use the overlapping period
    start_date = max(weather_data.index.min(), electricity_data.index.min())
    end_date = min(weather_data.index.max(), electricity_data.index.max())
    
    # Filter data to the common date range
    weather_data = weather_data.loc[start_date:end_date]
    electricity_data = electricity_data.loc[start_date:end_date]
    
    # Merge data on datetime index
    merged_data = pd.merge(weather_data, electricity_data, left_index=True, right_index=True, how='inner')
    
    # Check if merged data is empty
    if len(merged_data) == 0:
        raise ValueError("No overlapping data between weather and electricity datasets")
    
    # Check for missing values
    if merged_data.isnull().sum().sum() > 0:
        print(f"Found {merged_data.isnull().sum().sum()} missing values. Interpolating...")
        merged_data = merged_data.interpolate(method='time')
    
    # Check for empty columns and remove them
    empty_cols = [col for col in merged_data.columns if merged_data[col].isna().all()]
    if empty_cols:
        print(f"Removing empty columns: {empty_cols}")
        merged_data = merged_data.drop(columns=empty_cols)
    
    # Normalize data
    scalers = {}
    for column in merged_data.columns:
        # Check if column has data
        if len(merged_data[column]) > 0 and not merged_data[column].isna().all():
            scaler = StandardScaler()
            merged_data[column] = scaler.fit_transform(merged_data[column].values.reshape(-1, 1))
            scalers[column] = scaler
        else:
            print(f"Warning: Column '{column}' has no valid data for normalization")
            # Fill with zeros as a fallback
            merged_data[column] = 0
    
    return merged_data, scalers

def create_data_loaders(data, lookback_window, forecast_horizon, target_column, 
                        batch_size, validation_split=0.2, test_split=0.1):
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        data (pd.DataFrame): Preprocessed data
        lookback_window (int): Number of time steps to look back
        forecast_horizon (int): Number of time steps to predict ahead
        target_column (str): Name of the target column to predict
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation
        test_split (float): Fraction of data to use for testing
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Check if data is sufficient for the requested windows
    min_required_length = lookback_window + forecast_horizon + 1
    if len(data) < min_required_length:
        raise ValueError(f"Data length ({len(data)}) is insufficient for the requested lookback window ({lookback_window}) and forecast horizon ({forecast_horizon})")
    
    # Split data into train, validation, and test sets
    n = len(data)
    train_end = int(n * (1 - validation_split - test_split))
    val_end = int(n * (1 - test_split))
    
    # Ensure each split has at least the minimum required length
    if train_end < min_required_length:
        train_end = min_required_length
        print(f"Warning: Training set size adjusted to minimum required length ({min_required_length})")
    
    if val_end - train_end < min_required_length:
        val_end = train_end + min_required_length
        print(f"Warning: Validation set size adjusted to minimum required length ({min_required_length})")
    
    if n - val_end < min_required_length:
        val_end = n - min_required_length
        print(f"Warning: Test set size adjusted to minimum required length ({min_required_length})")
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, lookback_window, forecast_horizon, target_column)
    val_dataset = TimeSeriesDataset(val_data, lookback_window, forecast_horizon, target_column)
    test_dataset = TimeSeriesDataset(test_data, lookback_window, forecast_horizon, target_column)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def prepare_data(config, energy_type):
    """
    Prepare data for model training.
    
    Args:
        config: Configuration object with data settings
        energy_type (str): Type of energy ('pv' or 'wind')
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, scalers)
    """
    # Select appropriate weather parameters based on energy type
    if energy_type == 'pv':
        selected_params = config.WEATHER_PARAMS_PV
    elif energy_type == 'wind':
        selected_params = config.WEATHER_PARAMS_WIND
    else:
        selected_params = config.WEATHER_PARAMS_ALL
    
    # Load data
    weather_data = load_weather_data(config.DATA_PATH, selected_params)
    
    # Try to use real data from CodeGreen API with a date range
    # Default to recent data (last 2 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    # Load electricity data (will use API if available, otherwise simulated)
    electricity_data = load_electricity_data(
        config.DATA_PATH, 
        energy_type,
        start_date=start_date,
        end_date=end_date
    )
    
    # Print data info for debugging
    print(f"Weather data shape: {weather_data.shape}, date range: {weather_data.index.min()} to {weather_data.index.max()}")
    print(f"Electricity data shape: {electricity_data.shape}, date range: {electricity_data.index.min()} to {electricity_data.index.max()}")
    
    # Preprocess data
    processed_data, scalers = preprocess_data(weather_data, electricity_data, selected_params)
    
    # Print processed data info
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Columns after preprocessing: {processed_data.columns.tolist()}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        processed_data, 
        config.LOOKBACK_WINDOW, 
        config.FORECAST_HORIZON, 
        'generation', 
        config.BATCH_SIZE,
        config.VALIDATION_SPLIT,
        config.TEST_SPLIT
    )
    
    return train_loader, val_loader, test_loader, scalers
