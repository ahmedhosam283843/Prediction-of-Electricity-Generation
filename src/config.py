import torch

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOOKBACK_HOURS = 300
HORIZON_HOURS = 72
VAL_FRAC = 0.15
TEST_FRAC = 0.15
BATCH_SIZE = 256
EPOCHS = 50
LR = 3e-4

SELECTED_WEATHER_VARS = [
    "temperature", "wind_speed", "pressure",
    "global_radiation", "humidity", "cloudiness"
]
TIME_FEATURES = ["hr_sin","hr_cos","wkd_sin","wkd_cos","doy_sin","doy_cos"]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

MAX_PAST_DAYS_FORECAST = 92     # hard API limit
CHUNK_LENGTH_FORECAST = 16     # ≤16-day slices keep us safe