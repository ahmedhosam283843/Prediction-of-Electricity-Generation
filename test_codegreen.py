"""
Test script for validating the CodeGreen API integration.

This script tests the data loader with the CodeGreen API to ensure
it can successfully retrieve and process real energy data.
"""

from src.data_loader import load_electricity_data, CODEGREEN_AVAILABLE
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_codegreen_integration():
    """Test the CodeGreen API integration for electricity data."""
    print("Testing CodeGreen API integration...")

    # Set date range for testing
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Get last 30 days of data
    country_code = "DE"

    print(f"\nTesting electricity data retrieval for country: {country_code}")
    try:
        electricity_data = load_electricity_data(
            start_date=start_date,
            end_date=end_date,
            country=country_code
        )
        print(f"Data shape: {electricity_data.shape}")
        print(
            f"Date range: {electricity_data.index.min()} to {electricity_data.index.max()}")
        print(f"Data sample:\n{electricity_data.head()}")

        # Plot electricity data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(electricity_data.index,
                electricity_data['percentRenewable'] * 100)
        ax.set_title(f'Renewable Energy Percentage in {country_code}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Renewable Percentage (%)')
        ax.grid(True)
        fig.autofmt_xdate()  # Rotate date labels for readability
        plt.tight_layout()
        fig.savefig('electricity_data_test.png')
        plt.close(fig)

        success = True
        error_msg = None
        print("test plot saved as 'electricity_data_test.png'")
    except Exception as e:
        print(f"Error retrieving electricity data: {str(e)}")
        success = False
        error_msg = str(e)

    # Print summary
    print("\nTest Summary:")
    print(f"CodeGreen API available: {CODEGREEN_AVAILABLE}")
    print(f"Electricity data retrieval: {'Success' if success else 'Failed'}")
    if not success:
        print(f"Error: {error_msg}")


if __name__ == "__main__":
    test_codegreen_integration()
