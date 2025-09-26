from __future__ import annotations
import warnings
from src.carbon_emissions import (
    set_global_ci,
    carbon_plot_single,
    scheduler_metrics,
    scheduling_stats,
    precompute_emissions,
    CG_AVAILABLE,
)
from codegreen_core.tools.carbon_intensity import compute_ci
from src.models.informer import InformerForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.cycle_lstm import CycleLSTMModel
from src.models.seq2seq import Seq2SeqForecaster
from src.models.cnn_lstm import CNNLSTMModel
from src.models.tcn import TCNForecaster
from src.models.gru import GRUForecaster
from src.models.lstm import LSTMForecaster
from src.train_eval import train_model, evaluate_model, plot_predictions
from src.data_loader import fetch_and_prepare, build_sequences

import os
import json
import random
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
from tqdm import tqdm

# Central config (robust import)
from src import config as CFG

LOOKBACK_HOURS = CFG.LOOKBACK_HOURS
HORIZON_HOURS = CFG.HORIZON_HOURS
VAL_FRAC = CFG.VAL_FRAC
TEST_FRAC = CFG.TEST_FRAC
BATCH_SIZE = CFG.BATCH_SIZE
EPOCHS = CFG.EPOCHS
LR = CFG.LR
START_DATE = CFG.START_DATE
END_DATE = CFG.END_DATE
COUNTRY = getattr(CFG, "COUNTRY", "DE")

# Data + train/eval utils
try:
    from tslearn.metrics import SoftDTWLossPyTorch
    HAS_SDTW = True
except Exception:
    HAS_SDTW = False

# Optional ARIMA baseline
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except Exception:
    HAS_ARIMA = False

# Optional XGBoost (backend lib)
try:
    import xgboost  # noqa: F401  # ensure lib is present
    HAS_XGB = True
except Exception:
    HAS_XGB = False

SCHEDULE_SAMPLE_STRIDE = getattr(CFG, "SCHEDULE_SAMPLE_STRIDE", 1)
EMISSIONS_PRECOMPUTE_SCOPE = getattr(
    CFG, "EMISSIONS_PRECOMPUTE_SCOPE", "start_now")


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    



# benchmark.py
def prepare_data(start, end):
    # df has columns: feat_cols + ["y"] with DateTimeIndex
    df, feat_cols = fetch_and_prepare(start, end)

    # Include past targets channel in X
    X, y, T = build_sequences(
        df, LOOKBACK_HOURS, HORIZON_HOURS, include_y_hist=True, y_hist_k=LOOKBACK_HOURS
    )

    # chronological split with target-overlap purge
    n = len(X)
    test_cut = int(n * (1 - TEST_FRAC))
    val_cut = int(test_cut * (1 - VAL_FRAC))
    purge = max(HORIZON_HOURS - 1, 0)

    # Purge only on the earlier side of each boundary to avoid overlap but keep data:
    #  - drop last `purge` train samples
    #  - drop last `purge` val samples
    tr_end = max(val_cut - purge, 0)
    va_start = val_cut
    va_end = max(test_cut - purge, va_start)
    te_start = test_cut

    if tr_end == 0 or va_end <= va_start or te_start >= n:
        warnings.warn(
            f"[split] Small/empty set after purge (n={n}, purge={purge}). "
            f"Consider adjusting VAL_FRAC/TEST_FRAC or date range."
        )

    Xtr, ytr, Ttr = X[:tr_end], y[:tr_end], T[:tr_end]
    Xva, yva, Tva = X[va_start:va_end], y[va_start:va_end], T[va_start:va_end]
    Xte, yte, Tte = X[te_start:], y[te_start:], T[te_start:]

    # Print test data date range
    if len(Tte) > 0:
        test_start_date = pd.to_datetime(Tte[0, 0])
        test_end_date = pd.to_datetime(Tte[-1, -1])
        print(
            f"Test data from {test_start_date.date()} to {test_end_date.date()} (purge={purge})")
    else:
        print("[warn] No test samples after purge")

    # scale X (fit on train only)
    scaler_X = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))
    Xtr_sc = scaler_X.transform(
        Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape)
    Xva_sc = scaler_X.transform(
        Xva.reshape(-1, Xva.shape[-1])).reshape(Xva.shape)
    Xte_sc = scaler_X.transform(
        Xte.reshape(-1, Xte.shape[-1])).reshape(Xte.shape)

    data = dict(
        df=df, feat_cols=feat_cols,
        Xtr=Xtr_sc, Xva=Xva_sc, Xte=Xte_sc,
        ytr=ytr, yva=yva, yte=yte,
        Ttr=Ttr, Tva=Tva, Tte=Tte, scaler_X=scaler_X, scaler_y=None
    )
    return data


def make_loaders(Xtr, ytr, Xva, yva, Xte, yte):
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xva), torch.FloatTensor(yva)),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xte), torch.FloatTensor(yte)),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    return train_loader, val_loader, test_loader


def compute_cycle_hour_index(Tsplit: np.ndarray, lookback: int, cycle_len: int = 24) -> np.ndarray:
    # index = hour-of-day of the first hour in the input window (forecast_start - lookback)
    anchor = pd.to_datetime(Tsplit[:, 0])
    start_hist = anchor - pd.to_timedelta(lookback, unit="h")
    return start_hist.hour.values.astype(np.int64) % cycle_len


def make_loaders_cycle(Xtr, ytr, idx_tr, Xva, yva, idx_va, Xte, yte, idx_te):
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(
            ytr), torch.LongTensor(idx_tr).view(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xva), torch.FloatTensor(
            yva), torch.LongTensor(idx_va).view(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xte), torch.FloatTensor(
            yte), torch.LongTensor(idx_te).view(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    return train_loader, val_loader, test_loader


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_metrics_original(y_pred, y_true):
    # y_pred, y_true: (N, H)
    mae_path = float(np.mean(np.abs(y_pred - y_true)))
    rmse_path = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    mae_t72 = float(np.mean(np.abs(y_pred[:, -1] - y_true[:, -1])))
    rmse_t72 = float(np.sqrt(np.mean((y_pred[:, -1] - y_true[:, -1]) ** 2)))

    soft_dtw = None
    if HAS_SDTW:
        with torch.no_grad():
            sdtw = SoftDTWLossPyTorch(gamma=0.1)
            a = torch.tensor(y_pred, dtype=torch.float32).unsqueeze(-1)
            b = torch.tensor(y_true, dtype=torch.float32).unsqueeze(-1)
            soft_dtw = float(sdtw(a, b).mean().item())

    return dict(mae_path=mae_path, rmse_path=rmse_path,
                mae_t72=mae_t72, rmse_t72=rmse_t72, soft_dtw=soft_dtw)


def train_and_eval_torch_model(name, model, loaders, device):
    train_loader, val_loader, test_loader = loaders

    t0 = time.time()
    _ = train_model(model.to(device), train_loader, val_loader, EPOCHS, device)
    train_time = time.time() - t0

    # evaluate
    test_mse, test_mae, test_sdtw, y_pred, y_true = evaluate_model(
        model, test_loader, device)

    # inference latency (ms/sample)
    t1 = time.time()
    with torch.no_grad():
        for bx, _ in test_loader:
            _ = model(bx.to(device))
    t2 = time.time()
    n_samples = len(test_loader.dataset)
    infer_ms_per_sample = 1000.0 * (t2 - t1) / max(n_samples, 1)

    metrics = compute_metrics_original(y_pred, y_true)
    metrics.update(dict(train_time_s=float(train_time),
                        infer_ms_per_sample=float(infer_ms_per_sample),
                        scaled_mse=None, scaled_mae=None, scaled_soft_dtw=None))  # No longer scaled
    return metrics, y_pred, y_true


def train_and_eval_cycle_lstm(model, loaders, device):
    import torch.nn as nn
    import torch.optim as optim
    train_loader, val_loader, test_loader = loaders
    crit = nn.L1Loss()
    opt = optim.Adam(model.parameters(), lr=LR)
    best, patience, wait = float("inf"), 8, 0

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for bx, by, bidx in train_loader:
            bx, by, bidx = bx.to(device), by.to(device), bidx.to(device)
            opt.zero_grad()
            out = model(bx, bidx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
        # val
        model.eval()
        vloss, n = 0.0, 0
        with torch.no_grad():
            for bx, by, bidx in val_loader:
                bx, by, bidx = bx.to(device), by.to(device), bidx.to(device)
                vloss += crit(model(bx, bidx), by).item() * bx.size(0)
                n += bx.size(0)
        vloss /= max(n, 1)
        if vloss < best:
            best, wait = vloss, 0
            torch.save(model.state_dict(), "_best_cycle_lstm.pt")
        else:
            wait += 1
            if wait >= patience:
                break
    train_time = time.time() - t0
    model.load_state_dict(torch.load("_best_cycle_lstm.pt"))

    # test
    model.eval()
    preds, trues = [], []
    import torch.nn as nn
    mse_loss, l1_loss = nn.MSELoss(), nn.L1Loss()
    tot_mse, tot_mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for bx, by, bidx in test_loader:
            bx, by, bidx = bx.to(device), by.to(device), bidx.to(device)
            out = model(bx, bidx)
            # These are now unscaled losses
            tot_mse += mse_loss(out, by).item() * bx.size(0)
            # These are now unscaled losses
            tot_mae += l1_loss(out, by).item() * bx.size(0)
            n += bx.size(0)
            preds.append(out.cpu().numpy())
            trues.append(by.cpu().numpy())
    y_pred = np.vstack(preds)
    y_true = np.vstack(trues)

    # latency
    t1 = time.time()
    with torch.no_grad():
        for bx, _, bidx in test_loader:
            _ = model(bx.to(device), bidx.to(device))
    t2 = time.time()
    infer_ms_per_sample = 1000.0 * (t2 - t1) / max(len(test_loader.dataset), 1)

    metrics = compute_metrics_original(y_pred, y_true)
    metrics.update(dict(train_time_s=float(train_time),
                        infer_ms_per_sample=float(infer_ms_per_sample),
                        scaled_mse=None, scaled_mae=None, scaled_soft_dtw=None))  # No longer scaled
    return metrics, y_pred, y_true


# ---------------- TransformerEncoder forecaster (lightweight, no external deps) ----------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4000):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(
            0, max_len, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: (L, B, D)
        L = x.size(0)
        x = x + self.pe[:L].unsqueeze(1)
        return self.drop(x)


class TransformerEncoderForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int,
                 d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False, activation="relu"
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )

    def forward(self, x):  # x: (B, L, F)
        x = self.proj(x)     # (B, L, D)
        x = x.transpose(0, 1)  # (L, B, D)
        x = self.pe(x)
        z = self.encoder(x)  # (L, B, D)
        z_last = z[-1]       # (B, D)
        out = self.head(z_last)  # (B, H)
        return out


# ---------------- XGBoost full 72-h path ----------------

def train_and_eval_xgboost_full(Xtr, ytr, Xva, yva, Xte, yte):
    """
    Train multi-output XGBoost (full 72-step path) using the provided wrapper.
    """
    assert HAS_XGB, "xgboost is not installed"

    # XGBoostForecaster flattens 3D sequences internally
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.01,
        'max_depth': 6,
        'random_state': 42,
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'lambda': 1.2,
        'alpha': 0.2,
        'colsample_bytree': 0.8,
        'subsample': 0.8
    }
    model = XGBoostForecaster(**params)

    t0 = time.time()
    model.fit(Xtr, ytr)  # full path (N, H)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(Xte)  # (N, H)
    t2 = time.time()
    infer_ms_per_sample = 1000.0 * (t2 - t1) / max(len(Xte), 1)

    metrics = compute_metrics_original(y_pred, yte)
    metrics.update(dict(train_time_s=float(train_time),
                        infer_ms_per_sample=float(infer_ms_per_sample)))
    return metrics, y_pred, yte


# ---------------- ARIMA baseline (optional) ----------------

def train_and_eval_arima(df: pd.DataFrame, Tte: np.ndarray, yte: np.ndarray,
                         order=(1, 1, 1), refit_every=24):
    assert HAS_ARIMA, "statsmodels not installed"
    idx_map = {ts: i for i, ts in enumerate(df.index)}
    y_full = df["y"].values.astype(float)
    N, H = yte.shape
    preds = np.zeros((N, H), dtype=float)

    last_fit_pos, model_fit = None, None
    t0 = time.time()
    for i in range(N):
        anchor_ts = pd.to_datetime(Tte[i, 0])
        train_end = idx_map[anchor_ts]
        if (last_fit_pos is None) or (train_end - last_fit_pos >= refit_every) or (model_fit is None):
            model_fit = ARIMA(y_full[:train_end], order=order).fit()
            last_fit_pos = train_end
        preds[i, :] = model_fit.forecast(steps=H)
    train_time = time.time() - t0

    metrics = compute_metrics_original(preds, yte)
    metrics.update(dict(train_time_s=float(train_time),
                        infer_ms_per_sample=0.0))
    return metrics, preds, yte


# ---------------- plotting ----------------

def plot_72h_path_last_sample(y_pred: np.ndarray, y_true: np.ndarray,
                              Tte: np.ndarray, model_name: str, out_dir: str):
    """Plot full 72-hour trajectory for the last test sample."""
    t = pd.to_datetime(Tte[-1, :])
    yp = y_pred[-1, :]
    yt = y_true[-1, :]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, yt * 100, label="Actual", lw=2, alpha=.8, color="tab:green")
    ax.plot(t, yp * 100, label="Predicted", lw=2,
            alpha=.8, ls="--", color="tab:blue")
    ax.set_title(f"{model_name} — 72-hour forecast (last test sample)")
    ax.set_ylabel("% renewable")
    ax.grid(alpha=.3)
    ax.legend()
    fig.autofmt_xdate()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(
        out_dir, f"{model_name}_72h_path_last_sample.png"), dpi=300)
    plt.close(fig)


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


# ---------------- ensemble ----------------

def build_mean_ensemble(selected_preds: list[np.ndarray]) -> np.ndarray:
    """
    Simple unweighted mean ensemble across models with identical shape (N, H).
    """
    stack = np.stack(selected_preds, axis=0)  # (K, N, H)
    return np.mean(stack, axis=0)             # (N, H)


# ---------------- scheduling metrics helpers (added) ----------------

def summarize_scheduler(y_pred: np.ndarray,
                        y_true: np.ndarray,
                        Tte: np.ndarray,
                        model_name: str,
                        out_dir: str,
                        runtime_h: int = 8,
                        threshold: float = 0.75,
                        country: str = COUNTRY):
    """
    Computes per-sample metrics + writes the aggregate KPIs requested for R2:
        avg_saved_kg, avg_saved_pct, avg_attainment_pct,
        threshold_hit_rate_pred, avg_overlap_h, exact_match_rate_pct,
        avg_delay_h, avg_regret_kg
    """
    rows = []
    for i in tqdm(range(len(y_pred)),
                  desc=f"[{model_name}] scheduler",
                  unit="sample"):
        start_ts = pd.to_datetime(Tte[i, 0])
        m = scheduler_metrics(
            pred_renew=y_pred[i],
            true_renew=y_true[i],
            start_ts=start_ts,
            runtime_h=runtime_h,
            threshold=threshold,
            country=country,
        )
        rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{model_name}_scheduler_samples.csv"),
              index=False)

    # ---------- aggregate KPIs ----------
    exact_match = float(
        (df["pred_start_idx"] == df["oracle_start_idx"]).mean() * 100.0)

    summary = dict(
        model=model_name,
        N=len(df),
        runtime_h=runtime_h,
        threshold=threshold,
        avg_saved_kg=float(df["saved_kg_pred"].mean()),
        avg_saved_pct=float(df["saved_pct_pred"].mean()),
        avg_attainment_pct=float(df["attainment_pct"].mean()),
        threshold_hit_rate_pred=float(df["thr_met_pred"].mean() * 100.0),
        avg_overlap_h=float(df["overlap_h"].mean()),
        exact_match_rate_pct=exact_match,
        avg_delay_h=float(df["pred_start_idx"].mean()),
        avg_regret_kg=float(df["regret_kg"].mean()),
    )
    with open(os.path.join(out_dir, f"{model_name}_scheduler_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[scheduler] {model_name:>15} | "
          f"saved {summary['avg_saved_kg']:.2f} kg "
          f"({summary['avg_saved_pct']:.1f} %), "
          f"attain {summary['avg_attainment_pct']:.1f} %, "
          f"hit-rate {summary['threshold_hit_rate_pred']:.1f} %")

    return df, summary


def table_R2_last_sample(model_name: str,
                         model_preds: dict[str, np.ndarray],
                         y_true_ref: np.ndarray,
                         Tte: np.ndarray,
                         out_dir: str,
                         runtime_h: int = 8,
                         threshold: float = 0.75):
    """
    Build R2 table (CSV) for last test sample:
      - Start now
      - Energy-max (no target)
      - Energy-max (avg ≥ threshold)
      - Energy-max (oracle)
    """
    i = len(y_true_ref) - 1
    start_ts = pd.to_datetime(Tte[i, 0])
    pred = model_preds[model_name][i]
    true = y_true_ref[i]

    # Start-now + no-target + threshold rows (forecast)
    df_sched = scheduling_stats(
        pred_renew=pred, start_ts=start_ts, runtime_h=runtime_h,
        thresholds=(threshold,), server=None  # uses defaults
    )

    # Oracle row
    m = scheduler_metrics(pred, true, start_ts, runtime_h,
                          threshold, country=COUNTRY)
    oracle_row = dict(
        criterion="Energy-max (oracle)",
        suggested_start=m["oracle_start_ts"],
        avg_renew_pct=(m["avg_ren_true_win"] *
                       100.0 if m["avg_ren_true_win"] is not None else None),
        emissions_kg=m["kg_oracle"],
        saved_kg=(m["kg_now"] - m["kg_oracle"]
                  ) if m["kg_oracle"] is not None else None,
        saved_pct=(100.0 * (m["kg_now"] - m["kg_oracle"]) /
                   m["kg_now"]) if m["kg_oracle"] is not None else None
    )
    df_sched = pd.concat(
        [df_sched, pd.DataFrame([oracle_row])], ignore_index=True)
    df_sched_path = os.path.join(out_dir, f"{model_name}_R2_last_sample.csv")
    df_sched.to_csv(df_sched_path, index=False)
    print(
        f"[R2] Saved last-sample scheduling table for {model_name} → {df_sched_path}")


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch:", torch.__version__, "| CUDA:",
          torch.cuda.is_available(), "| Device:", device)
    if not HAS_SDTW:
        print("[warn] SoftDTW not available; soft_dtw will be None")
    if HAS_ARIMA:
        import statsmodels
        print("statsmodels:", statsmodels.__version__)
    if HAS_XGB:
        import xgboost as xgb
        print("xgboost:", xgb.__version__)

    # Prepare data using dates from config
    data = prepare_data(START_DATE, END_DATE)

    loaders = make_loaders(data["Xtr"], data["ytr"],
                           data["Xva"], data["yva"],
                           data["Xte"], data["yte"])
    # CycleLSTM loaders
    idx_tr = compute_cycle_hour_index(data["Ttr"], LOOKBACK_HOURS, 24)
    idx_va = compute_cycle_hour_index(data["Tva"], LOOKBACK_HOURS, 24)
    idx_te = compute_cycle_hour_index(data["Tte"], LOOKBACK_HOURS, 24)
    cycle_loaders = make_loaders_cycle(
        data["Xtr"], data["ytr"], idx_tr,
        data["Xva"], data["yva"], idx_va,
        data["Xte"], data["yte"], idx_te
    )

    # Use actual input_dim from the array (includes +1 past-y channel)
    F = data["Xtr"].shape[-1]
    H = HORIZON_HOURS

    # Torch models (full 72h path)
    torch_models = {
        "LSTM": lambda: LSTMForecaster(input_dim=F, lookback=LOOKBACK_HOURS, horizon=H),
        "GRU": lambda: GRUForecaster(input_dim=F, horizon=H),
        "TCN": lambda: TCNForecaster(input_dim=F, horizon=H),
        "CNN-LSTM": lambda: CNNLSTMModel(input_dim=F, output_dim=H, dropout=0.2),
        "Seq2Seq": lambda: Seq2SeqForecaster(in_dim=F, horizon=H, hid=128, n_layers=2, drop=0.2, tf_ratio=0.7),
        "Informer": lambda: InformerForecaster(input_dim=F, horizon=H, d_model=128, nhead=4, num_layers=3, dropout=0.25, distill=True),
        "Transformer": lambda: TransformerEncoderForecaster(
            input_dim=F, horizon=H, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.2
        ),
    }

    results = []
    model_preds: dict[str, np.ndarray] = {}
    y_true_ref: np.ndarray | None = None

    out_dir = os.path.join(
        "results", f"benchmarks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{COUNTRY}"
    )
    os.makedirs(out_dir, exist_ok=True)

    runtime_h = 8
    if CG_AVAILABLE and EMISSIONS_PRECOMPUTE_SCOPE != "none":
        print(
            f"[precompute] CodeGreen available → scope={EMISSIONS_PRECOMPUTE_SCOPE}")
        if EMISSIONS_PRECOMPUTE_SCOPE == "start_now":
            all_start_ts = pd.to_datetime(data["Tte"][:, 0])
            n = precompute_emissions(
                all_start_ts, runtime_h=runtime_h, country=COUNTRY)
            print(f"[precompute] warmed {n} start-now hours")
        elif EMISSIONS_PRECOMPUTE_SCOPE == "all_windows":
            # Pre-warm all candidate window starts across (optionally subsampled) test set
            ts_all = []
            N = len(data["Tte"])
            for i in range(0, N, max(1, SCHEDULE_SAMPLE_STRIDE)):
                anchor = pd.to_datetime(data["Tte"][i, 0])
                # all window starts over the horizon (every hour)
                ts_all.append(pd.date_range(anchor, periods=H, freq="H"))
            ts_all = pd.DatetimeIndex(
                np.unique(np.concatenate([np.asarray(t) for t in ts_all])))
            n = precompute_emissions(
                ts_all, runtime_h=runtime_h, country=COUNTRY)
            print(f"[precompute] warmed {n} candidate window hours")
    else:
        print("[precompute] CodeGreen not available or disabled; will use on-demand calls/linear proxy")

    # Train/eval Torch models + 72h path plot
    for name, ctor in torch_models.items():
        print(f"\n=== Training {name} ===")
        model = ctor()
        n_params = param_count(model)

        # Start CodeCarbon tracker
        tracker = EmissionsTracker(
            project_name=f"carbon_{name}", output_dir=out_dir, log_level="error")
        tracker.start()

        metrics, y_pred, y_true = train_and_eval_torch_model(
            name, model, loaders, device)

        # Stop tracker and get emissions
        emissions = tracker.stop()
        metrics["carbon_kg"] = emissions

        y_true_ref = y_true  # same for all models
        results.append(dict(model=name, params=n_params,
                            mae_t72=metrics["mae_t72"], rmse_t72=metrics["rmse_t72"],
                            mae_path=metrics["mae_path"], rmse_path=metrics["rmse_path"],
                            soft_dtw=metrics["soft_dtw"],
                            train_time_s=metrics["train_time_s"], infer_ms_per_sample=metrics["infer_ms_per_sample"],
                            carbon_kg=metrics["carbon_kg"]))
        with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        plot_72h_path_last_sample(y_pred, y_true, data["Tte"], name, out_dir)
        plot_predictions(
            y_pred, y_true, data["Tte"], horizon=H, model_name=name, save_dir=out_dir)
        model_preds[name] = y_pred

    # CycleLSTM
    print(f"\n=== Training CycleLSTM ===")
    cycle_model = CycleLSTMModel(
        input_size=F, hidden_size=256, num_layers=1, output_size=H, cycle_len=24, seq_len=LOOKBACK_HOURS, dropout=0.2
    ).to(device)
    n_params = param_count(cycle_model)

    tracker = EmissionsTracker(
        project_name="carbon_CycleLSTM", output_dir=out_dir, log_level="error")
    tracker.start()
    metrics_cyc, y_pred_cyc, y_true_cyc = train_and_eval_cycle_lstm(
        cycle_model, cycle_loaders, device)
    emissions = tracker.stop()
    metrics_cyc["carbon_kg"] = emissions

    y_true_ref = y_true_cyc
    results.append(dict(model="CycleLSTM", params=n_params,
                        mae_t72=metrics_cyc["mae_t72"], rmse_t72=metrics_cyc["rmse_t72"],
                        mae_path=metrics_cyc["mae_path"], rmse_path=metrics_cyc["rmse_path"],
                        soft_dtw=metrics_cyc["soft_dtw"],
                        train_time_s=metrics_cyc["train_time_s"], infer_ms_per_sample=metrics_cyc["infer_ms_per_sample"],
                        carbon_kg=metrics_cyc["carbon_kg"]))
    with open(os.path.join(out_dir, "CycleLSTM_metrics.json"), "w") as f:
        json.dump(metrics_cyc, f, indent=2)
    plot_72h_path_last_sample(y_pred_cyc, y_true_cyc,
                              data["Tte"], "CycleLSTM", out_dir)
    plot_predictions(y_pred_cyc, y_true_cyc,
                     data["Tte"], horizon=H, model_name="CycleLSTM", save_dir=out_dir)
    model_preds["CycleLSTM"] = y_pred_cyc

    # XGBoost (full 72-step path)
    if HAS_XGB:
        print(f"\n=== Training XGBoost (full 72-step path) ===")
        tracker = EmissionsTracker(
            project_name="carbon_XGBoost", output_dir=out_dir, log_level="error")
        tracker.start()
        xgb_metrics, y_pred_xgb, y_true_xgb = train_and_eval_xgboost_full(
            data["Xtr"], data["ytr"], data["Xva"], data["yva"], data["Xte"], data["yte"]
        )
        emissions = tracker.stop()
        xgb_metrics["carbon_kg"] = emissions

        y_true_ref = y_true_xgb
        results.append(dict(model="XGBoost", params="trees",
                            mae_t72=xgb_metrics["mae_t72"], rmse_t72=xgb_metrics["rmse_t72"],
                            mae_path=xgb_metrics["mae_path"], rmse_path=xgb_metrics["rmse_path"],
                            soft_dtw=xgb_metrics["soft_dtw"],
                            train_time_s=xgb_metrics["train_time_s"], infer_ms_per_sample=xgb_metrics["infer_ms_per_sample"],
                            carbon_kg=xgb_metrics["carbon_kg"]))
        with open(os.path.join(out_dir, "XGBoost_metrics.json"), "w") as f:
            json.dump(xgb_metrics, f, indent=2)
        plot_72h_path_last_sample(
            y_pred_xgb, y_true_xgb, data["Tte"], "XGBoost", out_dir)
        plot_predictions(y_pred_xgb, y_true_xgb,
                         data["Tte"], horizon=H, model_name="XGBoost", save_dir=out_dir)
        model_preds["XGBoost"] = y_pred_xgb
    else:
        print("[skip] xgboost not available")

    # ARIMA baseline (optional, full path)
    if HAS_ARIMA:
        print(f"\n=== ARIMA baseline (order=(1,1,1)) ===")
        tracker = EmissionsTracker(
            project_name="carbon_ARIMA", output_dir=out_dir, log_level="error")
        tracker.start()
        arima_metrics, arima_pred, arima_true = train_and_eval_arima(
            data["df"], data["Tte"], data["yte"], order=(1, 1, 1), refit_every=24
        )
        emissions = tracker.stop()
        arima_metrics["carbon_kg"] = emissions

        y_true_ref = arima_true
        results.append(dict(model="ARIMA(1,1,1)", params="statsmodels",
                            mae_t72=arima_metrics["mae_t72"], rmse_t72=arima_metrics["rmse_t72"],
                            mae_path=arima_metrics["mae_path"], rmse_path=arima_metrics["rmse_path"],
                            soft_dtw=arima_metrics["soft_dtw"],
                            train_time_s=arima_metrics["train_time_s"], infer_ms_per_sample=arima_metrics["infer_ms_per_sample"],
                            carbon_kg=arima_metrics["carbon_kg"]))
        with open(os.path.join(out_dir, "ARIMA_metrics.json"), "w") as f:
            json.dump(arima_metrics, f, indent=2)
        plot_72h_path_last_sample(
            arima_pred, arima_true, data["Tte"], "ARIMA(1,1,1)", out_dir)
        plot_predictions(arima_pred, arima_true,
                         data["Tte"], horizon=H, model_name="ARIMA(1,1,1)", save_dir=out_dir)
        model_preds["ARIMA(1,1,1)"] = arima_pred
    else:
        print("[skip] statsmodels (ARIMA) not available")

    # Save base-model results
    with open(os.path.join(out_dir, "summary_base_models.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Build ensembles of top-K models by mae_path
    print("\n=== Ensembles (mean of top-K by mae_path) ===")
    eligible = [r for r in results if (
        r.get("mae_path") is not None) and (r["model"] in model_preds)]
    eligible_sorted = sorted(eligible, key=lambda r: r["mae_path"])
    topks = [2, 3, 4, 5]
    for k in topks:
        if len(eligible_sorted) < k:
            print(
                f"[skip] Not enough eligible models for top-{k} ensemble (have {len(eligible_sorted)})")
            continue
        top_models = [r["model"] for r in eligible_sorted[:k]]
        preds_list = [model_preds[name] for name in top_models]
        ens_pred = build_mean_ensemble(preds_list)
        ens_metrics = compute_metrics_original(ens_pred, y_true_ref)
        row = dict(model=f"Ensemble_top{k}", params=f"{k} models",
                   mae_t72=ens_metrics["mae_t72"], rmse_t72=ens_metrics["rmse_t72"],
                   mae_path=ens_metrics["mae_path"], rmse_path=ens_metrics["rmse_path"],
                   soft_dtw=ens_metrics["soft_dtw"],
                   train_time_s=0.0, infer_ms_per_sample=0.0)
        results.append(row)
        with open(os.path.join(out_dir, f"Ensemble_top{k}_metrics.json"), "w") as f:
            json.dump(ens_metrics, f, indent=2)
        plot_72h_path_last_sample(
            ens_pred, y_true_ref, data["Tte"], f"Ensemble_top{k}", out_dir)
        plot_predictions(
            ens_pred, y_true_ref, data["Tte"], horizon=H, model_name=f"Ensemble_top{k}", save_dir=out_dir)
        model_preds[f"Ensemble_top{k}"] = ens_pred  # store for window plots

    # Save combined results and print
    with open(os.path.join(out_dir, "summary_all.json"), "w") as f:
        json.dump(results, f, indent=2)

    headers = ["model", "mae_t72", "rmse_t72", "mae_path", "rmse_path",
               "soft_dtw", "params", "train_time_s", "infer_ms_per_sample", "carbon_kg"]
    print("\nComparison (original units):")
    print_markdown_table(results, headers)

    # === Scheduling metrics across test set (added) ===
    span_start = pd.to_datetime(data["Tte"][0, 0])
    span_end = pd.to_datetime(data["Tte"][-1, 0]) + \
        timedelta(hours=HORIZON_HOURS)
    ci_global = compute_ci(COUNTRY, span_start, span_end)
    set_global_ci(ci_global)
    print("\n=== Scheduling metrics across test set (R=8h, thr=0.75) ===")
    # Best base (exclude ensembles)
    base_sorted_all = sorted(
        [r for r in results if r.get(
            "mae_path") is not None and not r["model"].startswith("Ensemble_")],
        key=lambda r: r["mae_path"]
    )
    best_base = base_sorted_all[0]["model"] if base_sorted_all else None

    # Best ensemble (include only ensembles)
    ens_sorted_all = sorted(
        [r for r in results if r.get(
            "mae_path") is not None and r["model"].startswith("Ensemble_")],
        key=lambda r: r["mae_path"]
    )
    best_ens = ens_sorted_all[0]["model"] if ens_sorted_all else None
    scheduler_summaries: list[dict] = []

    for mname in model_preds:
        if mname in model_preds:
            df_sched, summary = summarize_scheduler(
                y_pred=model_preds[mname],
                y_true=y_true_ref,
                Tte=data["Tte"],
                model_name=mname,
                out_dir=out_dir,
                runtime_h=8,
                threshold=0.7,
                country=COUNTRY
            )
            scheduler_summaries.append(summary)
            table_R2_last_sample(
                model_name=mname,
                model_preds=model_preds,
                y_true_ref=y_true_ref,
                Tte=data["Tte"],
                out_dir=out_dir,
                runtime_h=8,
                threshold=0.7
            )
    if scheduler_summaries:
        with open(os.path.join(out_dir, "all_scheduler_summaries.json"), "w") as f:
            json.dump(scheduler_summaries, f, indent=2)       # NEW
        print(
            f"[scheduler] wrote combined KPI file → {os.path.join(out_dir,'all_scheduler_summaries.json')}")
    # === Energy-based optimal-window plots for best-2 base and best-2 ensembles ===
    # Rank by mae_path (lower better)
    base_sorted = sorted(
        [r for r in results if r.get(
            "mae_path") is not None and not r["model"].startswith("Ensemble_")],
        key=lambda r: r["mae_path"]
    )[:2]
    ens_sorted = sorted(
        [r for r in results if r.get(
            "mae_path") is not None and r["model"].startswith("Ensemble_")],
        key=lambda r: r["mae_path"]
    )[:2]

    start_ts = pd.to_datetime(data["Tte"][-1, 0])
    start_ts_earlier = start_ts - timedelta(days=7)
    true_last = y_true_ref[-1]  # ground truth path for last test sample

    def _save_window_plot(model_name: str):
        pred_last = model_preds[model_name][-1]
        # Energy-based optimal window (clarified savings vs optimal inside the function)
        fig, _, _ = carbon_plot_single(
            pred_renew=pred_last,
            true_renew=true_last,
            start_ts=start_ts,
            runtime_h=8,
            threshold=0.7,
            country=COUNTRY
        )
        fig.savefig(os.path.join(
            out_dir, f"{model_name}_optimal_window.png"), dpi=300)

        fig2, _, _ = carbon_plot_single(
            pred_renew=pred_last,
            true_renew=true_last,
            start_ts=start_ts_earlier,
            runtime_h=8,
            threshold=0.7,
            country=COUNTRY
        )
        fig2.savefig(os.path.join(
            out_dir, f"{model_name}_optimal_window_week_earlier.png"), dpi=300)

    for r in base_sorted:
        if r["model"] in model_preds:
            _save_window_plot(r["model"])

    for r in ens_sorted:
        if r["model"] in model_preds:
            _save_window_plot(r["model"])

    print(
        f'\nSaved metrics, plots, scheduling summaries, and window plots to: {out_dir}')


if __name__ == "__main__":
    main()
