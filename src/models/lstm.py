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

    def __init__(self, input_dim, lookback, horizon, hidden_size1, hidden_size2, dropout1, dropout2):
        """
        Args:
            input_dim (int): Number of input features per time step.
            lookback (int): Lookback length (unused here but kept for interface parity).
            horizon (int): Forecast horizon length.
            hidden_size1 (int): Hidden size of the first LSTM layer.
            hidden_size2 (int): Hidden size of the second LSTM layer.
            dropout1 (float): Dropout rate after the first and second LSTM layers.
            dropout2 (float): Dropout rate after the ReLU activation in the head.
        """
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim,
                             hidden_size=hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1,
                             hidden_size=hidden_size2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size2, hidden_size2)
        self.dropout = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc_out = nn.Linear(hidden_size2, horizon)

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
