import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class TemporalBlock(nn.Module):
    """
    Dilated causal convolutional block with residual connection.

    Structure:
      Pad → Conv1d → ReLU → Dropout → Pad → Conv1d → ReLU → Dropout → Residual Add → ReLU
    """

    def __init__(self, in_ch, out_ch, k=3, dilation=1, dropout=0.3):
        """
        Args:
            in_ch (int): Input channels.
            out_ch (int): Output channels.
            k (int): Kernel size.
            dilation (int): Dilation factor.
            dropout (float): Dropout after activations.
        """
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=0, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=0, dilation=dilation))
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C_in, L)

        Returns:
            Tensor: (B, C_out, L)
        """
        y = F.pad(x, (self.pad, 0)); y = self.drop(F.relu(self.conv1(y)))
        y = F.pad(y, (self.pad, 0)); y = self.drop(F.relu(self.conv2(y)))
        res = x if self.down is None else self.down(x)
        return F.relu(y + res)

class TCNForecaster(nn.Module):
    """
    Temporal Convolutional Network (TCN) forecaster.

    Applies a stack of dilated causal conv blocks and predicts the horizon
    from the final time step's features via a linear layer (optionally squashed by sigmoid).
    """

    def __init__(self, input_dim, horizon, channels=128, levels=7, kernel_size=3, dropout=0.3):
        """
        Args:
            input_dim (int): Number of input features per time step.
            horizon (int): Forecast horizon length.
            channels (int): Number of channels per TCN layer.
            levels (int): Number of temporal blocks (exponential dilations).
            kernel_size (int): Convolution kernel size.
            dropout (float): Dropout rate inside temporal blocks.
        """
        super().__init__()
        layers, in_ch = [], input_dim
        for i in range(levels):
            layers.append(TemporalBlock(in_ch, channels, k=kernel_size, dilation=2**i, dropout=dropout))
            in_ch = channels
        self.net = nn.Sequential(*layers)
        self.fc_out = nn.Linear(channels, horizon)

    def forward(self, x):     # x: (B, L, F)
        """
        Args:
            x (Tensor): (batch, seq_len, input_dim)

        Returns:
            Tensor: (batch, horizon)
        """
        x = x.transpose(1, 2) # (B, F, L)
        x = self.net(x)
        x = x[:, :, -1]       # last time step features
        return torch.sigmoid(self.fc_out(x))