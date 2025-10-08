from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding, added to sequence embeddings.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4000) -> None:
        """
        Args:
            d_model: Embedding dimension.
            dropout: Dropout probability applied after adding positional encoding.
            max_len: Maximum supported sequence length.
        """
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(
            0, max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Args:
            x: Input of shape (L, B, D).

        Returns:
            Tensor of shape (L, B, D).
        """
        L = x.size(0)
        x = x + self.pe[:L].unsqueeze(1)
        return self.drop(x)


class InformerForecaster(nn.Module):
    """
    Lightweight Informer-style forecaster for time series.
    - Projects inputs to d_model
    - Optionally performs Conv1d-based temporal distillation (downsampling)
    - Uses a standard TransformerEncoder stack
    - Pools the last token and predicts the horizon via an MLP head

    Input/Output:
      - forward expects x of shape (B, L, F)
      - returns predictions of shape (B, H)
    """

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        distill: bool = False,
    ) -> None:
        """
        Args:
            input_dim: Number of input features F.
            horizon: Forecast horizon H (number of steps to predict).
            d_model: Transformer embedding dimension.
            nhead: Number of attention heads.
            num_layers: Number of encoder layers.
            dim_feedforward: Hidden dimension of the feedforward network.
            dropout: Dropout probability.
            distill: If True, apply a Conv1d-based downsampling in time (factor 2).
        """
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.distill = distill
        self.nhead = nhead

        # Input projection to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Optional distillation conv (downsample along time by stride=2)
        self.distill_conv = (
            nn.Conv1d(d_model, d_model, kernel_size=3,
                      stride=2, padding=1) if distill else None
        )

        # Positional encoding
        self.pe = PositionalEncoding(d_model, dropout=dropout, max_len=4096)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # MLP head mapping last token to horizon outputs
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, L, F).

        Returns:
            Predictions of shape (B, H).
        """
        # Project features to model dimension: (B, L, D)
        z = self.input_proj(x)

        # Optional downsampling in time via Conv1d
        if self.distill_conv is not None:
            # Conv1d expects (B, C, L) where C == D
            z = z.transpose(1, 2).contiguous()  # (B, D, L)
            z = self.distill_conv(z)            # (B, D, L//2)
            z = z.transpose(1, 2).contiguous()  # (B, L2, D)

        # Transformer expects (L, B, D)
        z = z.transpose(0, 1)            # (L, B, D)
        z = self.pe(z)                   # add positional encoding + dropout
        z_enc = self.encoder(z)          # (L, B, D)

        # Use last time-step representation
        z_last = z_enc[-1]               # (B, D)
        out = self.head(z_last)          # (B, H)
        return out
