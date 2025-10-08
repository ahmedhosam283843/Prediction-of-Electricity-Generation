from __future__ import annotations
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from .config import BATCH_SIZE
import os
from datetime import datetime
import json
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

def get_country_coords(country_code: str) -> tuple[float, float]:
    """
    Returns (lat, lon) for a country's capital using geopy.
    Falls back to a hardcoded default for DE on failure.
    """
    try:
        geolocator = Nominatim(user_agent="wind_solar_forecasting_app")
        location = geolocator.geocode(f"Capital of {country_code}", timeout=5)
        if location:
            return (location.latitude, location.longitude)
    except (GeocoderTimedOut, GeocoderUnavailable):
        print(f"[geopy] Service unavailable, falling back for {country_code}")
    print(f"[geopy] Could not find coordinates for {country_code}, falling back to DE.")
    return (52.52, 13.40) # Fallback for Berlin, DE

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_env(seed: int = 42) -> torch.device:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| Device:", device)
    return device

def make_out_dir(country: str) -> str:
    out_dir = os.path.join("results", f"benchmarks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{country}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def print_markdown_table(rows, headers):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            elif v is None:
                cells.append("")
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")

def make_loaders(Xtr: np.ndarray, ytr: np.ndarray,
                 Xva: np.ndarray, yva: np.ndarray,
                 Xte: np.ndarray, yte: np.ndarray) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr)),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(Xva), torch.FloatTensor(yva)),
                            batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(Xte), torch.FloatTensor(yte)),
                             batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

def compute_cycle_hour_index(Tsplit: np.ndarray, lookback: int, cycle_len: int = 24) -> np.ndarray:
    anchor = pd.to_datetime(Tsplit[:, 0])
    start_hist = anchor - pd.to_timedelta(lookback, unit="h")
    return start_hist.hour.values.astype(np.int64) % cycle_len

def make_loaders_cycle(Xtr, ytr, idx_tr, Xva, yva, idx_va, Xte, yte, idx_te):
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr),
                                            torch.LongTensor(idx_tr).view(-1, 1)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(Xva), torch.FloatTensor(yva),
                                          torch.LongTensor(idx_va).view(-1, 1)),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(Xte), torch.FloatTensor(yte),
                                           torch.LongTensor(idx_te).view(-1, 1)),
                             batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader

def save_json(results: list[dict], out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), "w") as f:
        json.dump(results, f, indent=2)

def append_results(results, name, n_params, metrics):
    """Append model results to the results list."""
    results.append(dict(model=name, params=n_params,
                            mae_t72=metrics["mae_t72"], rmse_t72=metrics["rmse_t72"],
                            mae_path=metrics["mae_path"], rmse_path=metrics["rmse_path"],
                            soft_dtw=metrics["soft_dtw"],
                            train_time_s=metrics["train_time_s"], infer_ms_per_sample=metrics["infer_ms_per_sample"],
                            carbon_kg=metrics["carbon_kg"]))

def track_training_emissions(project_name: str, out_dir: str, fn):
    """
    Run fn() while tracking carbon emissions. Returns (result, carbon_kg).
    """
    tracker = EmissionsTracker(project_name=project_name, output_dir=out_dir, log_level="error")
    tracker.start()
    result = fn()
    carbon_kg = tracker.stop()
    return result, carbon_kg

def save_forecast_plots(predictions: np.ndarray, targets: np.ndarray, times: np.ndarray,
                        model_name: str, out_dir: str, horizon: int):
    """
    Save forecast plots (h-step ahead points and 72h path for last sample).
    """
    plot_72h_path_last_sample(predictions, targets, times, model_name, out_dir) 
    plot_predictions(predictions, targets, times, horizon, model_name, out_dir)

def plot_predictions(predictions: np.ndarray, targets: np.ndarray, times: np.ndarray,
                     horizon: int, model_name: str,
                     save_dir: str | None = None,
                     num_points: int = 300):
    """
    Plot h-step ahead points (uses last step when horizon>1) and save PNG.
    """
    # Normalize dims to 2-D
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if times.ndim == 1:
        times = times.reshape(-1, 1)

    forecast_horizon = predictions.shape[1]
    h = 0 if forecast_horizon == 1 else horizon - 1
    if h >= forecast_horizon:
        raise ValueError(f"Horizon {horizon} exceeds forecast_horizon {forecast_horizon}")

    times_h = times[:, h]
    pred_h = predictions[:, h]
    target_h = targets[:, h]

    idx = np.argsort(times_h)
    times_h = times_h[idx][:num_points]
    pred_h = pred_h[idx][:num_points]
    target_h = target_h[idx][:num_points]

    fig, ax = plt.subplots(figsize=(17, 6))
    ax.plot(times_h, target_h, label='Actual',  lw=1.7, alpha=.7)
    ax.plot(times_h, pred_h, label=f'{horizon}-h Forecast', lw=1.7, ls='--', alpha=.7)
    ax.set_xlabel('Time'), ax.set_ylabel('% Renewable')
    ax.set_title(f'{model_name}: {horizon}-h Ahead Forecast (first {len(times_h)} points)')
    ax.grid(alpha=.3)
    ax.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(Path(save_dir, f'{model_name}_{horizon}h_forecast.png'), dpi=300)
    return fig

def plot_72h_path_last_sample(y_pred: np.ndarray, y_true: np.ndarray,
                              Tte: np.ndarray, model_name: str, out_dir: str):
    """
    Plot full 72h path for the last test sample and save PNG.
    """
    t = pd.to_datetime(Tte[-1, :])
    yp = y_pred[-1, :]
    yt = y_true[-1, :]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, yt * 100, label="Actual", lw=2, alpha=.8, color="tab:green")
    ax.plot(t, yp * 100, label="Predicted", lw=2, alpha=.8, ls="--", color="tab:blue")
    ax.set_title(f"{model_name} — 72-hour forecast (last test sample)")
    ax.set_ylabel("% renewable")
    ax.grid(alpha=.3)
    ax.legend()
    fig.autofmt_xdate()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{model_name}_72h_path_last_sample.png"), dpi=300)
    plt.close(fig)