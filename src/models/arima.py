from __future__ import annotations
from typing import Optional, Dict, Union, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .. import config as CFG

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute MAPE robustly, avoiding division by zero.

    Args:
        y_true: Ground truth values (shape: [n]).
        y_pred: Predicted values (shape: [n]).
        eps: Small epsilon to avoid division by zero.

    Returns:
        MAPE percentage as a float.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


class ARIMAModel:
    """
    ARIMA model for univariate time series forecasting.

    This class wraps statsmodels' ARIMA to provide a consistent interface
    with other forecasters used in the pipeline.
    """

    def __init__(self, order: tuple[int, int, int] = CFG.MODEL_DEFAULTS["ARIMA"]["order"]) -> None:
        """
        Initialize the ARIMA model.

        Args:
            order: ARIMA order parameters (p, d, q), where:
                p = autoregressive lags,
                d = differences,
                q = moving average terms.
        """
        self.order = tuple(order)
        self.model: Optional[ARIMA] = None
        self.model_fit: Optional[Any] = None  # statsmodels ARIMAResults

    def fit(self, train_data: Union[np.ndarray, pd.Series]) -> "ARIMAModel":
        """
        Fit the ARIMA model to the training data.

        Args:
            train_data: 1D training time series (np.ndarray or pd.Series).

        Returns:
            self
        """
        # Ensure pandas Series for statsmodels
        if not isinstance(train_data, pd.Series):
            train_data = pd.Series(np.asarray(train_data).ravel())

        self.model = ARIMA(train_data, order=self.order)
        self.model_fit = self.model.fit()
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        """
        Generate forecasts for future time steps.

        Args:
            n_steps: Number of steps to forecast into the future.

        Returns:
            Forecasted values as a numpy array (shape: [n_steps]).
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before making predictions.")
        forecast = self.model_fit.forecast(steps=n_steps)
        return np.asarray(forecast)

    def evaluate(self, test_data: Union[np.ndarray, pd.Series], n_steps: int) -> Dict[str, float]:
        """
        Evaluate the model on test data using standard metrics.

        Args:
            test_data: 1D test time series (np.ndarray or pd.Series).
            n_steps: Number of steps to forecast and compare.

        Returns:
            Dictionary of evaluation metrics: mse, rmse, mae, mape.
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before evaluation.")

        y_true = np.asarray(test_data).ravel()[:n_steps]
        y_pred = self.predict(n_steps)

        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, y_pred)
        mape = _safe_mape(y_true, y_pred)

        return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}

    def fit_predict(self, train_data: Union[np.ndarray, pd.Series], n_steps: int) -> np.ndarray:
        """
        Convenience method to fit and forecast in one step.

        Args:
            train_data: 1D training time series.
            n_steps: Forecast horizon.

        Returns:
            Forecasted values (shape: [n_steps]).
        """
        self.fit(train_data)
        return self.predict(n_steps)
