import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    """
    Two-layer GRU forecaster with a small MLP head.

    Architecture:
      GRU(input_dim -> hid1) → Dropout → GRU(hid1 -> hid2) → Dropout
      → ReLU(Linear(hid2 -> hid2)) → Linear(hid2 -> horizon)
    """

    def __init__(self, input_dim: int, horizon: int,
                 hidden_size1: int = 352, hidden_size2: int = 128, dropout: float = 0.3):
        """
        Args:
            input_dim (int): Number of input features per time step.
            horizon (int): Forecast horizon length.
            hidden_size1 (int): Hidden size of first GRU.
            hidden_size2 (int): Hidden size of second GRU and MLP.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hidden_size1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size1,    hidden_size2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size2,  hidden_size2)
        self.drop = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size2, horizon)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch, seq_len, input_dim)

        Returns:
            Tensor: (batch, horizon)
        """
        x, _ = self.gru1(x)
        x = self.drop(x)
        x, _ = self.gru2(x)
        x = self.drop(x)
        x = x[:, -1, :]          # Take last time step
        x = torch.relu(self.fc1(x))
        return self.fc_out(x)
