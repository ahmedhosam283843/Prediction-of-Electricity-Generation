import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
from pathlib import Path

class XGBoostForecaster(BaseEstimator, RegressorMixin):
    """XGBoost model for multi-horizon forecasting"""
    
    def __init__(self, **kwargs):
        # Set default parameters
        self.params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Update with user parameters
        self.params.update(kwargs)
        
        # Handle device parameter for CPU-only systems
        if 'device' in self.params and self.params['device'] == 'cuda':
            # Remove cuda device if not available, let XGBoost use CPU
            self.params.pop('device')
            print("CUDA not available, using CPU training")
            
        self.model = None
        self.is_fitted = False
        
    def _flatten_sequences(self, X):
        """Flatten sequence data for XGBoost input"""
        if X.ndim == 3:  # (samples, timesteps, features)
            return X.reshape(X.shape[0], -1)
        return X
    
    def fit(self, X, y):
        """Train the XGBoost model"""
        X_flat = self._flatten_sequences(X)
        
        # Create base XGBoost model with all parameters
        base_model = xgb.XGBRegressor(**self.params)
        
        # Use MultiOutputRegressor for multi-horizon prediction
        if y.ndim > 1 and y.shape[1] > 1:
            self.model = MultiOutputRegressor(base_model, n_jobs=-1)
        else:
            self.model = base_model
            
        self.model.fit(X_flat, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_flat = self._flatten_sequences(X)
        predictions = self.model.predict(X_flat)
        
        # Ensure predictions have correct shape
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
            
        return predictions
    
    def save(self, filepath):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        """Load a trained model"""
        self.model = joblib.load(filepath)
        self.is_fitted = True