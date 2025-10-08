# src/models/gru.py
import torch
import torch.nn as nn

class GRUForecaster(nn.Module):
    """
    2-layer GRU → FC → horizon
    """
    def __init__(self, input_dim: int, horizon: int,
                 hid1: int =352, hid2: int = 128, p: float = 0.3):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hid1, batch_first=True)
        self.gru2 = nn.GRU(hid1,    hid2, batch_first=True)
        self.fc1  = nn.Linear(hid2,  hid2)
        self.drop = nn.Dropout(p)
        self.fc_out = nn.Linear(hid2, horizon)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.drop(x)
        x, _ = self.gru2(x)
        x = self.drop(x)
        x = x[:, -1, :]          # last time step
        x = torch.relu(self.fc1(x))
        
        return self.fc_out(x)