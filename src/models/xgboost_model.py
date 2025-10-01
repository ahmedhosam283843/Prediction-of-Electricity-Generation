import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
from pathlib import Path
from typing import Any


class XGBoostForecaster(BaseEstimator, RegressorMixin):
    """A wrapper for the XGBoost model for multi-horizon time series forecasting.

    This class flattens the input sequence and uses scikit-learn's
    MultiOutputRegressor to enable XGBoost to predict a multi-step forecast horizon.
    It adheres to the scikit-learn estimator API.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the XGBoostForecaster.

        Args:
            **kwargs: Hyperparameters for the underlying `xgboost.XGBRegressor`.
                      These will override the defaults.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1,
            'verbosity': 0,
        }
        self.params = {**default_params, **kwargs}

        # This logic handles cases where 'device' might be passed but XGBoost
        # is not built with GPU support. It prevents crashes by falling back to CPU.
        if self.params.get('device') == 'cuda':
            try:
                # A simple check to see if GPU is usable by XGBoost
                xgb.XGBRegressor(device='cuda', n_estimators=1).fit(
                    np.zeros((1, 1)), np.zeros(1))
            except xgb.core.XGBoostError:
                print("XGBoost CUDA support not available, falling back to CPU.")
                self.params.pop('device')

        self.model = None

    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """Flattens 3D sequence data (samples, timesteps, features) to 2D."""
        if X.ndim == 3:
            n_samples = X.shape[0]
            return X.reshape(n_samples, -1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostForecaster":
        """Fits the forecasting model.

        Args:
            X (np.ndarray): Input data of shape (n_samples, lookback, n_features).
            y (np.ndarray): Target data of shape (n_samples, horizon).

        Returns:
            XGBoostForecaster: The fitted forecaster instance.
        """
        X_flat = self._flatten_sequences(X)

        base_model = xgb.XGBRegressor(**self.params)

        # Use MultiOutputRegressor for multi-step forecasting
        if y.ndim > 1 and y.shape[1] > 1:
            self.model = MultiOutputRegressor(base_model, n_jobs=-1)
        else:
            self.model = base_model

        self.model.fit(X_flat, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates forecasts for the given input data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, lookback, n_features).

        Returns:
            np.ndarray: The forecasted values, shape (n_samples, horizon).
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        X_flat = self._flatten_sequences(X)
        predictions = self.model.predict(X_flat)

        # Ensure output is always 2D
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return predictions

    def save(self, filepath: str | Path) -> None:
        """Saves the trained model to a file.

        Args:
            filepath (str | Path): The path to save the model file.
        """
        if self.model is None:
            raise ValueError("Cannot save unfitted model")
        joblib.dump(self.model, filepath)

    def load(self, filepath: str | Path) -> None:
        """Loads a trained model from a file.

        Args:
            filepath (str | Path): The path to the model file.
        """
        self.model = joblib.load(filepath)
