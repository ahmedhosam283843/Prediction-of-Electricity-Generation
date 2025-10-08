import os
import torch
from datetime import datetime

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data date range
START_DATE = datetime(2023, 8, 13)
END_DATE = datetime(2025, 8, 12)

LOOKBACK_HOURS = 72
HORIZON_HOURS = 72
VAL_FRAC = 0.15
TEST_FRAC = 0.15
BATCH_SIZE = 256
EPOCHS = 50
LR = 3e-4
EARLY_STOPPING_PATIENCE = 8

SELECTED_WEATHER_VARS = [
    "temperature", "wind_speed", "pressure",
    "global_radiation", "humidity", "cloudiness"
]
TIME_FEATURES = ["hr_sin", "hr_cos",
                 "wkd_sin", "wkd_cos", "doy_sin", "doy_cos"]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

MAX_PAST_DAYS_FORECAST = 92    # API limit
CHUNK_LENGTH_FORECAST = 16    # ≤16 days per request

# ---------------- Country configuration ----------------
COUNTRY = "DE"

# ---------------- General execution settings ----------------
TRACK_EMISSIONS = True
SAVE_PLOTS = True

# List of models to run in the benchmark.
MODEL_RUN_LIST = [
    "LSTM", "GRU", "TCN", "CNN-LSTM", "Seq2Seq", "Informer", "Transformer",
    "CycleLSTM", "XGBoost", "ARIMA(1,1,1)"
]

# ---------------- Carbon emissions & Scheduling ----------------
AVG_CI_G_PER_KWH = 400.0      # fallback proxy if CI data unavailable
USE_GLOBAL_CI_CURVE = True   # prefetch full CI span and slice in-memory

SCHEDULER_RUNTIME_H = 8
SCHEDULER_THRESHOLD = 0.75
SCHEDULER_THRESHOLDS = (0.7, 0.75, 0.9)


# ---------------- Carbon emissions configuration ----------------
DEFAULT_SERVER = dict(
    country=COUNTRY,
    number_core=8,
    memory_gb=32,
    power_draw_core=15.8,  # W / core  (Green-Algorithms defaults)
    usage_factor_core=1.0,
    power_draw_mem=0.3725,  # W / GiB
    power_usage_efficiency=1.6,
)

# ---------------- Model defaults ----------------
MODEL_DEFAULTS = {
    "ARIMA":       {"order": (1, 1, 1), "refit_every": 24},
    "CNN-LSTM":    {"conv_channels": [64, 64, 64], "lstm_hidden": 256, "dropout": 0.2},
    "CycleLSTM":   {"hidden_size": 256, "num_layers": 1, "cycle_len": 24, "dropout": 0.2},
    "GRU":         {"hidden_size1": 352, "hidden_size2": 128, "dropout": 0.3},
    "Informer":    {"d_model": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256, "dropout": 0.25, "distill": True},
    "LSTM":        {"hidden_size1": 256, "hidden_size2": 128, "dropout1": 0.3, "dropout2": 0.1},
    "Seq2Seq":     {"hidden_size": 128, "num_layers": 2, "dropout": 0.2, "tf_ratio": 0.7},
    "TCN":         {"channels": 128, "levels": 7, "kernel_size": 3, "dropout": 0.3},
    "Transformer": {"d_model": 128, "nhead": 4, "num_layers": 2, "dim_feedforward": 256, "dropout": 0.1},
}

# XGBoost defaults
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "learning_rate": 0.01,
    "n_estimators": 200,
    "max_depth": 6,
    "random_state": 42,
    "eval_metric": "rmse",
    "tree_method": "hist",
    "lambda": 1.2,
    "alpha": 0.2,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
}
