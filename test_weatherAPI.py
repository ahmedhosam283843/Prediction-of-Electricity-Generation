"""
Test script for validating the weather data loading functionality.

This script tests the data loader with real DWD weather data and simulated data
to ensure it can successfully retrieve and process weather data.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_weather_data

def test_weather_data_loading():
    """Test the weather data loading functionality."""
    print("Testing weather data loading...")
    
    # Set parameters
    lat = 52.52  # Berlin
    lon = 13.405
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    selected_params = ['temperature', 'wind_speed', 'global_radiation', 'sunshine_duration']

    # Test real data loading
    print("\nTesting real DWD data loading...")
    try:
        weather_data_real = load_weather_data(
            data_path="./data",
            selected_params=selected_params,
            use_real_data=True,
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date
        )
        print(f"Real weather data shape: {weather_data_real.shape}")
        print(f"Date range: {weather_data_real.index.min()} to {weather_data_real.index.max()}")
        print(f"Available parameters: {weather_data_real.columns.tolist()}")
        weather_data_real.to_csv('weather_data_real.csv')
        # Plot temperature if available
        if 'temperature' in weather_data_real.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(weather_data_real.index, weather_data_real['temperature'])
            plt.title('Temperature')
            plt.xlabel('Date')
            plt.ylabel('Temperature (°C)')
            plt.grid(True)
            plt.savefig('temperature_real.png')
            plt.close()
    except Exception as e:
        print(f"Error loading real weather data: {str(e)}")
    
    # Test simulated data loading
    print("\nTesting simulated data loading...")
    try:
        weather_data_sim = load_weather_data(
            data_path="./data",
            selected_params=selected_params,
            use_real_data=False,
            start_date=start_date,
            end_date=end_date
        )
        print(f"Simulated weather data shape: {weather_data_sim.shape}")
        print(f"Date range: {weather_data_sim.index.min()} to {weather_data_sim.index.max()}")
        print(f"Available parameters: {weather_data_sim.columns.tolist()}")
        
        # Plot temperature if available
        if 'temperature' in weather_data_sim.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(weather_data_sim.index, weather_data_sim['temperature'])
            plt.title('Simulated Temperature')
            plt.xlabel('Date')
            plt.ylabel('Temperature (°C)')
            plt.grid(True)
            plt.savefig('temperature_sim.png')
            plt.close()
    except Exception as e:
        print(f"Error loading simulated weather data: {str(e)}")

if __name__ == "__main__":
    test_weather_data_loading()