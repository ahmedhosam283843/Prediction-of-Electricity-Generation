"""
Training and evaluation utilities for wind and solar energy prediction models.

This module provides functions for training, evaluating, and comparing different models
for wind and solar energy prediction as described in the paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import os
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models.cycle_lstm import CycleLSTMModel
from src.config import LR
from tslearn.metrics import SoftDTWLossPyTorch


def train_model(model, train_loader, val_loader, epochs, device):
    """Train the PyTorch model"""
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8

    model = model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        print(
            f'Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))

    return train_losses, val_losses


def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    sdtw_criterion = SoftDTWLossPyTorch(gamma=0.1).to(device)

    total_mse = 0.0
    total_mae = 0.0
    total_sdtw = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            mse = criterion(outputs, batch_y)
            mae = mae_criterion(outputs, batch_y)
            # SDTW expects 3D tensors: [batch, length, 1]
            sdtw = sdtw_criterion(outputs.unsqueeze(-1),
                                  batch_y.unsqueeze(-1)).mean()

            total_mse += mse.item() * batch_x.size(0)
            total_mae += mae.item() * batch_x.size(0)
            total_sdtw += sdtw.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    avg_sdtw = total_sdtw / total_samples

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    return avg_mse, avg_mae, avg_sdtw, predictions, targets


def plot_predictions(predictions: np.ndarray, targets: np.ndarray, times: np.ndarray,
                     horizon: int, model_name: str,
                     save_dir: str | None = None,
                     num_points: int = 300):

    # ── always reshape 1-D inputs to 2-D ──────────────────────────
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if times.ndim == 1:
        times = times.reshape(-1, 1)

    forecast_horizon = predictions.shape[1]
    h = 0 if forecast_horizon == 1 else horizon - 1
    if h >= forecast_horizon:
        raise ValueError(f"Horizon {horizon} exceeds forecast_horizon "
                         f"{forecast_horizon}")

    times_h = times[:, h]
    pred_h = predictions[:, h]
    target_h = targets[:, h]

    # sort chronologically and trim
    idx = np.argsort(times_h)
    times_h = times_h[idx][:num_points]
    pred_h = pred_h[idx][:num_points]
    target_h = target_h[idx][:num_points]

    fig, ax = plt.subplots(figsize=(17, 6))
    ax.plot(times_h, target_h, label='Actual',  lw=1.7, alpha=.7)
    ax.plot(times_h, pred_h,
            label=f'{horizon}-h Forecast', lw=1.7, ls='--', alpha=.7)
    ax.set_xlabel('Time'), ax.set_ylabel('% Renewable')
    ax.set_title(
        f'{model_name}: {horizon}-h Ahead Forecast (first {len(times_h)} points)')
    ax.grid(alpha=.3)
    ax.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            Path(save_dir, f'{model_name}_{horizon}h_forecast.png'), dpi=300)
    return fig


def plot_learning_curve(train_losses, val_losses, model_name: str, save_dir: str | None = None):
    """
    Plot training and validation losses over epochs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss')
    ax.plot(epochs, val_losses, label='Val Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_name} Learning Curve')
    ax.grid(alpha=.3)
    ax.legend()
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(Path(save_dir, f'{model_name}_learning.png'), dpi=300)
    return fig


def train_xgboost_model(model, X_train, y_train, X_val, y_val):
    """Train XGBoost model with validation monitoring"""
    print("Training XGBoost model...")

    import time
    start_time = time.time()

    # Fit the model
    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    print(f"Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")

    return model, [val_mse], [val_mse]


def evaluate_xgboost_model(model, X_test, y_test):
    """Evaluate XGBoost model"""
    print("Making predictions on test set...")
    predictions = model.predict(X_test)

    print("Calculating metrics...")
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Skip Soft-DTW for faster evaluation
    avg_sdtw = 0.0
    print("Skipping Soft-DTW calculation for faster evaluation")

    return mse, mae, avg_sdtw, predictions, y_test
