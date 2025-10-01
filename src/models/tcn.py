# src/models/tcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    """A residual block for a Temporal Convolutional Network (TCN).

    It consists of two causal convolutional layers with weight normalization,
    ReLU activation, and dropout. A residual connection is also used.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.3):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): The size of the convolutional kernel.
            dilation (int): The dilation factor for the convolutions.
            dropout (float): The dropout probability.
        """
        super().__init__()
        # Causal padding: (kernel_size - 1) * dilation on the left.
        self.causal_padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=0, dilation=dilation
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=0, dilation=dilation
        ))
        self.dropout = nn.Dropout(dropout)

        # 1x1 convolution for the residual connection if channel dimensions differ.
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the temporal block."""
        residual = x if self.downsample is None else self.downsample(x)

        # First convolutional layer
        y = F.pad(x, (self.causal_padding, 0))
        y = self.conv1(y)
        y = F.relu(y)
        y = self.dropout(y)

        # Second convolutional layer
        y = F.pad(y, (self.causal_padding, 0))
        y = self.conv2(y)
        y = F.relu(y)
        y = self.dropout(y)

        return F.relu(y + residual)


class TCNForecaster(nn.Module):
    """A Temporal Convolutional Network (TCN) for time series forecasting."""

    def __init__(self, input_dim: int, horizon: int, channels: int = 128,
                 levels: int = 7, kernel_size: int = 3, dropout: float = 0.3):
        """
        Args:
            input_dim (int): The number of features in the input sequence.
            horizon (int): The forecast horizon (number of steps to predict).
            channels (int): The number of channels in the TCN hidden layers.
            levels (int): The number of residual blocks in the TCN.
            kernel_size (int): The kernel size for the convolutions.
            dropout (float): The dropout probability.
        """
        super().__init__()
        layers = []
        num_channels = [input_dim] + [channels] * levels
        for i in range(levels):
            dilation_size = 2**i
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size=kernel_size,
                dilation=dilation_size, dropout=dropout
            ))
        self.net = nn.Sequential(*layers)
        self.fc_out = nn.Linear(channels, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, L, F)
        x = x.transpose(1, 2)  # To (B, F, L) for Conv1d
        features = self.net(x)
        # Use the output of the last time step for prediction
        last_timestep_features = features[:, :, -1]
        # The sigmoid ensures the output is in [0, 1], suitable for scaled targets.
        return torch.sigmoid(self.fc_out(last_timestep_features))
