import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """
    Two-layer LSTM forecaster with dropout and a small MLP head.

    Input:
      x: (batch, seq_len, input_dim)

    Output:
      (batch, horizon)
    """

    def __init__(self, input_dim, lookback, horizon):
        """
        Args:
            input_dim (int): Number of input features per time step.
            lookback (int): Lookback length (unused here but kept for interface parity).
            horizon (int): Forecast horizon length.
        """
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.1)
        self.fc_out = nn.Linear(128, horizon)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch, seq_len, input_dim)

        Returns:
            Tensor: (batch, horizon)
        """
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x[:, -1, :]               # last time step features
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc_out(x)