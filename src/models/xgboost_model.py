from __future__ import annotations
from typing import Any, Dict, Union
import joblib
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor

class XGBoostForecaster(BaseEstimator, RegressorMixin):
    """
    XGBoost model for multi-horizon forecasting.

    - Accepts inputs of shape (B, L, F) and flattens them to (B, L*F)
    - Uses MultiOutputRegressor when y has multiple columns (multi-step)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Args:
            **kwargs: Any parameters supported by xgboost.XGBRegressor, e.g.:
                      n_estimators, max_depth, learning_rate, random_state, tree_method, n_jobs, verbosity, etc.
        """
        # Default parameters (safe, CPU-friendly)
        self.params: Dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
        }
        self.params.update(kwargs)

        # Remove 'device=cuda' if present; keep it CPU-only unless the environment guarantees CUDA
        if self.params.get("device", None) == "cuda":
            self.params.pop("device")
            print("CUDA not available, using CPU training")

        self.model: Union[xgb.XGBRegressor, MultiOutputRegressor, None] = None
        self.is_fitted: bool = False

    @staticmethod
    def _flatten_sequences(X: np.ndarray) -> np.ndarray:
        """
        Flatten sequence inputs to 2D for XGBoost.

        Args:
            X: Input array of shape (B, L, F) or (B, F).

        Returns:
            2D array of shape (B, L*F) or (B, F).
        """
        X = np.asarray(X)
        if X.ndim == 3:  # (samples, timesteps, features)
            return X.reshape(X.shape[0], -1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostForecaster":
        """
        Train the XGBoost model.

        Args:
            X: Features, shape (B, L, F) or (B, F).
            y: Targets, shape (B,) or (B, H).

        Returns:
            self
        """
        X_flat = self._flatten_sequences(X)
        base_model = xgb.XGBRegressor(**self.params)

        # Multi-output wrapper for multi-horizon predictions
        if y.ndim > 1 and y.shape[1] > 1:
            self.model = MultiOutputRegressor(base_model, n_jobs=-1)
        else:
            self.model = base_model

        self.model.fit(X_flat, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict targets for the given inputs.

        Args:
            X: Features, shape (B, L, F) or (B, F).

        Returns:
            Predictions of shape (B, H) or (B, 1).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_flat = self._flatten_sequences(X)
        preds = self.model.predict(X_flat)

        # Ensure 2D shape for consistency
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Destination path for the serialized model.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        joblib.dump(self.model, filepath)

    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the serialized model file.
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True