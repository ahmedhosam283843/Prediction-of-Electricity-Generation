import time
import os
from typing import Callable, Optional, Tuple
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tslearn.metrics import SoftDTWLossPyTorch
from . import config as CFG
from statsmodels.tsa.arima.model import ARIMA


def _unpack_batch(batch):
    """
    Supports 2-tensor batches (x, y) and 3-tensor batches (x, y, extra index).
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            x, y = batch
            extra = None
        elif len(batch) == 3:
            x, y, extra = batch
        else:
            raise ValueError(f"Unexpected batch format of length {len(batch)}")
    else:
        raise ValueError("Batch must be a tuple/list")
    return x, y, extra


def _train_model_core(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    device: torch.device,
    forward_fn: Callable[[torch.nn.Module, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    loss_fn: nn.Module = nn.L1Loss(),
    patience: int = CFG.EARLY_STOPPING_PATIENCE,
    best_path: str = "best_model.pt",
) -> Tuple[list[float], list[float]]:
    """
    Shared training loop with early stopping.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=CFG.LR)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    wait = 0
    print(
        f"\nTraining {model.__class__.__name__} for {epochs} epochs on {device}")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_batches = 0.0, 0
        for batch in train_loader:
            x, y, extra = _unpack_batch(batch)
            x, y = x.to(device), y.to(device)
            extra = extra.to(device) if extra is not None else None

            optimizer.zero_grad()
            out = forward_fn(model, x, extra)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train = train_loss / max(train_batches, 1)
        train_losses.append(avg_train)

        # Validate
        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, extra = _unpack_batch(batch)
                x, y = x.to(device), y.to(device)
                extra = extra.to(device) if extra is not None else None
                out = forward_fn(model, x, extra)
                val_loss += loss_fn(out, y).item() * x.size(0)
                n += x.size(0)
        avg_val = val_loss / max(n, 1)
        val_losses.append(avg_val)

        print(
            f"Epoch [{epoch+1}/{epochs}] - Train {avg_train:.4f} | Val {avg_val:.4f}")

        # Early stopping + checkpoint
        if avg_val < best_val:
            best_val, wait = avg_val, 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best weights
    model.load_state_dict(torch.load(best_path))
    model.eval()
    return train_losses, val_losses


def _predict_core(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    forward_fn: Callable[[torch.nn.Module, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict full test set using a shared forward function.
    """
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y, extra = _unpack_batch(batch)
            x = x.to(device)
            extra = extra.to(device) if extra is not None else None
            out = forward_fn(model, x, extra)
            preds.append(out.cpu().numpy())
            trues.append(y.numpy() if not isinstance(
                y, torch.Tensor) else y.cpu().numpy())
    return np.vstack(preds), np.vstack(trues)


def _time_infer_core(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    forward_fn: Callable[[torch.nn.Module, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
) -> float:
    """
    Measure average inference latency per sample in milliseconds.
    """
    t1 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            x, _, extra = _unpack_batch(batch)
            x = x.to(device)
            extra = extra.to(device) if extra is not None else None
            _ = forward_fn(model, x, extra)
    t2 = time.time()
    try:
        n_samples = len(test_loader.dataset)
    except Exception:
        n_samples = sum(len(b[0]) for b in test_loader)
    return 1000.0 * (t2 - t1) / max(n_samples, 1)


def compute_metrics_original(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Original metrics for consistency with existing outputs.
    """
    mae_path = float(np.mean(np.abs(y_pred - y_true)))
    rmse_path = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae_t72 = float(np.mean(np.abs(y_pred[:, -1] - y_true[:, -1])))
    rmse_t72 = float(np.sqrt(np.mean((y_pred[:, -1] - y_true[:, -1]) ** 2)))
    soft_dtw = None
    try:
        with torch.no_grad():
            sdtw = SoftDTWLossPyTorch(gamma=0.1)
            a = torch.tensor(y_pred, dtype=torch.float32).unsqueeze(-1)
            b = torch.tensor(y_true, dtype=torch.float32).unsqueeze(-1)
            soft_dtw = float(sdtw(a, b).mean().item())
    except Exception:
        soft_dtw = None
    return dict(mae_path=mae_path, rmse_path=rmse_path,
                mae_t72=mae_t72, rmse_t72=rmse_t72, soft_dtw=soft_dtw)


def train_and_eval_generic(
    model: torch.nn.Module,
    loaders,
    device: torch.device,
    epochs: int,
    forward_fn: Callable[[torch.nn.Module, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    best_path: str,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Generic runner for torch models (vanilla and models needing an extra index).
    """
    train_loader, val_loader, test_loader = loaders

    # Train
    t0 = time.time()
    _ = _train_model_core(
        model=model.to(device),
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device,
        forward_fn=forward_fn,
        best_path=best_path,
    )
    train_time = time.time() - t0

    # Predict + metrics
    y_pred, y_true = _predict_core(
        model, test_loader, device, forward_fn=forward_fn)

    # Inference timing
    infer_ms_per_sample = _time_infer_core(
        model, test_loader, device, forward_fn=forward_fn)

    metrics = compute_metrics_original(y_pred, y_true)
    metrics.update(dict(train_time_s=float(train_time),
                        infer_ms_per_sample=float(infer_ms_per_sample)))
    return metrics, y_pred, y_true


def train_and_eval_torch_model(name, model, loaders, device, epochs):
    """
    Thin wrapper for vanilla torch models (forward(x)).
    """
    return train_and_eval_generic(
        model=model,
        loaders=loaders,
        device=device,
        epochs=epochs,
        forward_fn=lambda m, x, extra: m(x),
        best_path="best_model.pt",
    )


def train_and_eval_xgboost_full(Xtr, ytr, Xva, yva, Xte, yte, xgb_forecaster_ctor, best_path: str):
    """
    XGBoost runner: fit on train, predict on test, measure train time and latency.
    """
    print("\nTraining XGBoost model")
    # Use centrally defined XGBoost parameters
    model = xgb_forecaster_ctor(**CFG.XGB_PARAMS)

    # Train time
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0
    model.save(best_path)
    # Predict + inference timing
    t1 = time.time()
    y_pred = model.predict(Xte)
    infer_ms_per_sample = 1000.0 * (time.time() - t1) / max(len(Xte), 1)

    metrics = compute_metrics_original(y_pred, yte)
    metrics.update(dict(train_time_s=float(train_time),
                        infer_ms_per_sample=float(infer_ms_per_sample)))
    return metrics, y_pred, yte


def train_and_eval_arima(df: pd.DataFrame, Tte: np.ndarray, yte: np.ndarray,
                         order=(1, 1, 1), refit_every=24):
    """
    ARIMA runner: refit every refit_every hours, forecast per test anchor.
    """
    print(f"\nTraining ARIMA{order} with refit every {refit_every} steps")
    idx_map = {ts: i for i, ts in enumerate(df.index)}
    y_full = df["y"].values.astype(float)
    N, H = yte.shape
    preds = np.zeros((N, H), dtype=float)

    t0 = time.time()
    last_fit_pos, model_fit = None, None
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


def forward_auto(model: torch.nn.Module, x: torch.Tensor, extra: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Call model(x[, extra]) depending on whether `extra` is provided and supported.
    """
    # Special handling for CycleLSTM which needs seq_len in forward
    if model.__class__.__name__ == "CycleLSTM":
        return model(x, extra, seq_len=CFG.LOOKBACK_HOURS)
    if extra is None:
        return model(x)
    try:
        return model(x, extra)
    except TypeError:  # pragma: no cover
        # Fallback if model does not accept extra
        return model(x)


def run_model_any(
    spec: dict,
    data: dict,
    loaders,
    cycle_loaders,
    device: torch.device,
    epochs: int,
    out_dir: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Universal runner covering torch (with/without extra), XGBoost, and ARIMA.
    Expects spec keys:
      - name (str)
      - type ('torch'|'xgb'|'arima')
      - build (callable)  for 'torch' or 'xgb' (returns model or XGB class)
      - needs_cycle_idx (bool, optional) for torch models that use extra index
      - order (tuple), refit_every (int) for ARIMA
    """
    name = spec["name"]
    mtype = spec["type"]

    tracker = None
    if CFG.TRACK_EMISSIONS:
        tracker = EmissionsTracker(
            project_name=f"carbon_{name}", output_dir=out_dir, log_level="error")
        tracker.start()

    if mtype == "torch":
        # Build model and select loaders
        model = spec["build"]().to(device)
        use_cycle = spec.get("needs_cycle_idx", False)
        chosen_loaders = cycle_loaders if use_cycle else loaders
        checkpoint_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(
            checkpoint_dir, f"{name.replace(' ', '_').replace('/', '_')}.pt")

        metrics, y_pred, y_true = train_and_eval_generic(
            model=model,
            loaders=chosen_loaders,
            device=device,
            epochs=epochs,
            forward_fn=forward_auto,
            best_path=best_path,
        )
        # Parameter count
        metrics["param_count"] = int(sum(p.numel()
                                     for p in model.parameters()))

    elif mtype == "xgb":
        # build returns the constructor/class
        checkpoint_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(
            checkpoint_dir, f"{name.replace(' ', '_').replace('/', '_')}.joblib")

        xgb_ctor = spec["build"]()
        metrics, y_pred, y_true = train_and_eval_xgboost_full(
            data["Xtr"], data["ytr"], data["Xva"], data["yva"], data["Xte"], data["yte"], xgb_ctor, best_path
        )
        metrics["param_count"] = "N/A"

    elif mtype == "arima":
        order = spec.get("order", (1, 1, 1))
        refit_every = spec.get("refit_every", 24)
        metrics, y_pred, y_true = train_and_eval_arima(
            df=data["df"], Tte=data["Tte"], yte=data["yte"], order=order, refit_every=refit_every
        )
        metrics["param_count"] = "N/A"

    else:
        if tracker:
            tracker.stop()
        raise ValueError(f"Unknown model type: {mtype}")

    carbon_kg = 0.0
    if tracker:
        carbon_kg = tracker.stop() or 0.0
    metrics["carbon_kg"] = float(carbon_kg or 0.0)
    return metrics, y_pred, y_true
