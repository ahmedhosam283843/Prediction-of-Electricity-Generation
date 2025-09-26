import os
import torch
from datetime import datetime

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data date range
START_DATE = datetime(2021, 8, 13)
END_DATE = datetime(2025, 8, 12)

LOOKBACK_HOURS = 72
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
TIME_FEATURES = ["hr_sin", "hr_cos",
                 "wkd_sin", "wkd_cos", "doy_sin", "doy_cos"]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

MAX_PAST_DAYS_FORECAST = 92    # API limit
CHUNK_LENGTH_FORECAST = 16    # ≤16 days per request

# ---------------- Country configuration ----------------
COUNTRY = "DE"

# Representative coordinates for Open-Meteo (capital or grid hub per country)
COUNTRY_COORDS = {
    "DE": (52.5200, 13.4050),  # Berlin
    "DK": (55.6760, 12.5680),  # Copenhagen
    "NL": (52.3676, 4.9041),   # Amsterdam
    "BE": (50.8503, 4.3517),   # Brussels
    "ES": (40.4168, -3.7038),  # Madrid
    "FR": (48.8566, 2.3522),   # Paris
    "IT": (41.9028, 12.4964),  # Rome
    "PL": (52.2297, 21.0122),  # Warsaw
    "CZ": (50.0755, 14.4378),  # Prague
    "AT": (48.2082, 16.3738),  # Vienna
    "SE": (59.3293, 18.0686),  # Stockholm
    "NO": (59.9139, 10.7522),  # Oslo
    "PT": (38.7223, -9.1393),  # Lisbon
    "IE": (53.3498, -6.2603),  # Dublin
    "UK": (51.5074, -0.1278),  # London
    "FI": (60.1699, 24.9384),  # Helsinki
    "CH": (46.9480, 7.4474),   # Bern
}

LAT, LON = COUNTRY_COORDS.get(COUNTRY, COUNTRY_COORDS["DE"])

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
