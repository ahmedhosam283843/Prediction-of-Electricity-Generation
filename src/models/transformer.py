"""
Transformer Encoder-based forecaster for time series.

Projects input features to an embedding, adds positional encoding,
passes through a TransformerEncoder stack, and predicts the horizon
from the last token via a small MLP head.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


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

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Args:
            x: Input tensor of shape (L, B, D).

        Returns:
            Tensor of shape (L, B, D).
        """
        L = x.size(0)
        x = x + self.pe[:L].unsqueeze(1)
        return self.drop(x)


class TransformerForecaster(nn.Module):
    """
    Transformer Encoder-based model for time series forecasting.

    Input/Output:
      - forward expects x of shape (B, L, F)
      - returns predictions of shape (B, H)
    """

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            input_dim: Number of input features F.
            horizon: Forecast horizon H (number of steps to predict).
            d_model: Transformer embedding dimension.
            nhead: Number of attention heads.
            num_layers: Number of encoder layers.
            dim_feedforward: Hidden dimension in the feedforward network.
            dropout: Dropout probability.
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.pe = PositionalEncoding(d_model, dropout=dropout)

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
        # Project input features to model dimension
        x = self.proj(x)       # (B, L, D)

        # Transformer expects (L, B, D)
        x = x.transpose(0, 1)  # (L, B, D)

        # Add positional encoding and run the encoder
        x = self.pe(x)
        z = self.encoder(x)    # (L, B, D)

        # Use the last time step for prediction
        z_last = z[-1]         # (B, D)
        out = self.head(z_last)  # (B, H)
        return out
