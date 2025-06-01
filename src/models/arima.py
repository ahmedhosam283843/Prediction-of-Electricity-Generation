"""
ARIMA model implementation for wind and solar energy prediction.

This module implements the Autoregressive Integrated Moving Average (ARIMA) model
as described in the paper "Prediction of Electricity Generation Using Onshore Wind and Solar Energy in Germany".
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ARIMAModel:
    """
    ARIMA model for time series forecasting.
    
    As described in the paper, this model serves as a baseline for comparison with
    deep learning approaches.
    """
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize the ARIMA model.
        
        Args:
            order (tuple): ARIMA order parameters (p, d, q)
                p: The number of lag observations included in the model (AR)
                d: The number of times the raw observations are differenced (I)
                q: The size of the moving average window (MA)
        """
        self.order = order
        self.model = None
        
    def fit(self, train_data):
        """
        Fit the ARIMA model to the training data.
        
        Args:
            train_data (np.array): Training data time series
            
        Returns:
            self: The fitted model
        """
        # Convert to pandas Series for statsmodels
        if not isinstance(train_data, pd.Series):
            train_data = pd.Series(train_data)
        
        # Fit ARIMA model
        self.model = ARIMA(train_data, order=self.order)
        self.model_fit = self.model.fit()
        
        return self
    
    def predict(self, n_steps):
        """
        Generate forecasts for future time steps.
        
        Args:
            n_steps (int): Number of steps to forecast
            
        Returns:
            np.array: Forecasted values
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Generate forecast
        forecast = self.model_fit.forecast(steps=n_steps)
        
        return forecast.values
    

