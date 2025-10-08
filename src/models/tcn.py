# src/models/tcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=1, dropout=0.3):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=0, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=0, dilation=dilation))
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = F.pad(x, (self.pad, 0)); y = self.drop(F.relu(self.conv1(y)))
        y = F.pad(y, (self.pad, 0)); y = self.drop(F.relu(self.conv2(y)))
        res = x if self.down is None else self.down(x)
        return F.relu(y + res)

class TCNForecaster(nn.Module):
    def __init__(self, input_dim, horizon, channels=128, levels=7, kernel_size=3, dropout=0.3):
        super().__init__()
        layers, in_ch = [], input_dim
        for i in range(levels):
            layers.append(TemporalBlock(in_ch, channels, k=kernel_size, dilation=2**i, dropout=dropout))
            in_ch = channels
        self.net = nn.Sequential(*layers)
        self.fc_out = nn.Linear(channels, horizon)

    def forward(self, x):     # x: (B, L, F)
        x = x.transpose(1, 2) # (B, F, L)
        x = self.net(x)
        x = x[:, :, -1]
        return torch.sigmoid(self.fc_out(x))