import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Injects positional information into the input tensor.

    This module generates fixed sinusoidal positional encodings and adds them
    to the input tensor. It's a standard component in Transformer architectures.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4000):
        """
        Args:
            d_model (int): The dimensionality of the input embeddings.
            dropout (float): The dropout probability.
            max_len (int): The maximum possible sequence length.
        """
        super().__init__()
        self.drop = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (L, B, D), where L is the
                              sequence length, B is the batch size, and D is
                              the embedding dimension.

        Returns:
            torch.Tensor: The input tensor with positional encodings added.
        """
        L = x.size(0)
        x = x + self.pe[:L].unsqueeze(1)
        return self.drop(x)


class TransformerEncoderForecaster(nn.Module):
    """A Transformer Encoder-based model for time series forecasting."""

    def __init__(self, input_dim: int, horizon: int,
                 d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.1):
        """
        Args:
            input_dim (int): The number of features in the input sequence.
            horizon (int): The forecast horizon (number of steps to predict).
            d_model (int): The dimensionality of the Transformer model.
            nhead (int): The number of attention heads in the Transformer.
            num_layers (int): The number of Transformer encoder layers.
            dim_feedforward (int): The dimension of the feedforward network.
            dropout (float): The dropout probability.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TransformerEncoderForecaster.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, F).

        Returns:
            torch.Tensor: Output tensor of shape (B, H).
        """
        # Project input features to the model dimension
        x = self.proj(x)  # Shape: (B, L, D)
        # Transformer expects (L, B, D), so we transpose
        x = x.transpose(0, 1)
        # Add positional encoding
        x = self.pe(x)
        # Pass through the Transformer encoder
        z = self.encoder(x)  # Shape: (L, B, D)
        # Use the output of the last time step for prediction
        z_last = z[-1]  # Shape: (B, D)
        out = self.head(z_last)  # Shape: (B, H)
        return out
