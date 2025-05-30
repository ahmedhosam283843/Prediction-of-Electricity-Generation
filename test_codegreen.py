"""
Test script for validating the CodeGreen API integration.

This script tests the data loader with the CodeGreen API to ensure
it can successfully retrieve and process real energy data.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data_loader import load_electricity_data, CODEGREEN_AVAILABLE

def test_codegreen_integration():
    """Test the CodeGreen API integration for both PV and wind data."""
    print("Testing CodeGreen API integration...")
    
    # Set date range for testing
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Get last 30 days of data
    
    results = {}
    
    # Test PV data
    print("\nTesting PV data retrieval...")
    try:
        pv_data = load_electricity_data(
            data_path="./data",
            energy_type="pv",
            start_date=start_date,
            end_date=end_date
        )
        print(f"PV data shape: {pv_data.shape}")
        print(f"PV data range: {pv_data.index.min()} to {pv_data.index.max()}")
        print(f"PV data sample:\n{pv_data.head()}")
        
        # Plot PV data
        plt.figure(figsize=(12, 6))
        plt.plot(pv_data.index, pv_data['generation'])
        plt.title('PV Generation')
        plt.xlabel('Date')
        plt.ylabel('Generation (MWh)')
        plt.grid(True)
        plt.savefig('pv_data_test.png')
        plt.close()
        
        results['pv'] = {
            'success': True,
            'shape': pv_data.shape,
            'date_range': (pv_data.index.min(), pv_data.index.max())
        }
    except Exception as e:
        print(f"Error retrieving PV data: {str(e)}")
        results['pv'] = {
            'success': False,
            'error': str(e)
        }
    
    # Test wind data
    print("\nTesting wind data retrieval...")
    try:
        wind_data = load_electricity_data(
            data_path="./data",
            energy_type="wind",
            start_date=start_date,
            end_date=end_date
        )
        print(f"Wind data shape: {wind_data.shape}")
        print(f"Wind data range: {wind_data.index.min()} to {wind_data.index.max()}")
        print(f"Wind data sample:\n{wind_data.head()}")
        
        # Plot wind data
        plt.figure(figsize=(12, 6))
        plt.plot(wind_data.index, wind_data['generation'])
        plt.title('Wind Generation')
        plt.xlabel('Date')
        plt.ylabel('Generation (MWh)')
        plt.grid(True)
        plt.savefig('wind_data_test.png')
        plt.close()
        
        results['wind'] = {
            'success': True,
            'shape': wind_data.shape,
            'date_range': (wind_data.index.min(), wind_data.index.max())
        }
    except Exception as e:
        print(f"Error retrieving wind data: {str(e)}")
        results['wind'] = {
            'success': False,
            'error': str(e)
        }
    
    # Print summary
    print("\nTest Summary:")
    print(f"CodeGreen API available: {CODEGREEN_AVAILABLE}")
    print(f"PV data retrieval: {'Success' if results['pv']['success'] else 'Failed'}")
    print(f"Wind data retrieval: {'Success' if results['wind']['success'] else 'Failed'}")
    
    return results

if __name__ == "__main__":
    test_codegreen_integration()
