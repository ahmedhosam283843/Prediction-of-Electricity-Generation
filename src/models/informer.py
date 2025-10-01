import numpy as np
from torch import nn
import torch


class PositionalEncoding(nn.Module):
    """Injects positional information into the input tensor."""

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
            x (torch.Tensor): Input tensor of shape (L, B, D).

        Returns:
            torch.Tensor: The input tensor with positional encodings added.
        """
        L = x.size(0)
        x = x + self.pe[:L].unsqueeze(1)
        return self.drop(x)


class InformerForecaster(nn.Module):
    def __init__(self,
                 input_dim: int,
                 horizon: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 distill: bool = True):
        """A lightweight Informer-style forecaster for benchmarking.

        This is not a full ProbSparse-Informer implementation. Instead, it:
          - projects inputs to d_model.
          - optionally performs Conv1d-based distillation (downsampling).
          - uses a standard TransformerEncoder to produce sequence features.
          - pools the final encoder output and uses an MLP head for prediction.

        Args:
            input_dim (int): Number of input features.
            horizon (int): Forecast horizon length.
            d_model (int): Dimensionality of the Transformer model.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            distill (bool): Whether to use a distillation convolution layer.
        """
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.distill = distill

        # input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Optional distillation conv (downsamples sequence length by a factor of 2)
        if distill:
            self.distill_conv = nn.Conv1d(
                d_model, d_model, kernel_size=3, stride=2, padding=1)
        else:
            self.distill_conv = None

        # Positional encoding
        self.pe = PositionalEncoding(d_model, dropout=dropout, max_len=4096)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False, activation="relu"
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # head: MLP that maps pooled encoder features to horizon outputs
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the InformerForecaster.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, F).

        Returns:
            torch.Tensor: Output tensor of shape (B, H).
        """
        # Project input features to the model dimension
        z = self.input_proj(x)

        # Optionally distill/downsample the sequence
        if self.distill_conv is not None:
            z = z.transpose(1, 2)  # To (B, D, L) for Conv1d
            z = self.distill_conv(z)
            z = z.transpose(1, 2)  # Back to (B, L_new, D)

        # Transformer expects (L, B, D), so we transpose and add positional encoding
        z = z.transpose(0, 1)
        z = self.pe(z)
        z_enc = self.encoder(z)

        # Use the output of the last time step for prediction
        z_last = z_enc[-1]
        out = self.head(z_last)
        return out
