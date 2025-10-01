import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    """A two-layer GRU model for time series forecasting.

    This model uses a stack of two GRU layers to process the input sequence,
    followed by a fully connected head to produce the forecast.
    """

    def __init__(self, input_dim: int, horizon: int,
                 hidden_size_1: int = 352, hidden_size_2: int = 128,
                 dropout: float = 0.3):
        """
        Args:
            input_dim (int): The number of features in the input sequence.
            horizon (int): The forecast horizon (number of steps to predict).
            hidden_size_1 (int): The number of hidden units in the first GRU layer.
            hidden_size_2 (int): The number of hidden units in the second GRU layer.
            dropout (float): The dropout probability.
        """
        super().__init__()

        # GRU layers
        self.gru_layers = nn.Sequential(
            nn.GRU(input_dim, hidden_size_1, batch_first=True),
            # The first element of the tuple from GRU is the output tensor
            nn.Lambda(lambda x: x[0]),
            nn.Dropout(dropout),
            nn.GRU(hidden_size_1, hidden_size_2, batch_first=True),
            nn.Lambda(lambda x: x[0]),
            nn.Dropout(dropout)
        )

        # Fully connected head for prediction
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_size_2, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GRUForecaster.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, F), where B is the
                              batch size, L is the sequence length, and F is
                              the number of input features.

        Returns:
            torch.Tensor: Output tensor of shape (B, H), where H is the
                          forecast horizon.
        """
        # x shape: (B, L, F) -> (B, L, hidden_size_2)
        gru_out = self.gru_layers(x)

        # Use the output of the last time step for prediction
        last_timestep_out = gru_out[:, -1, :]  # Shape: (B, hidden_size_2)

        # Pass through the fully connected head
        output = self.fc_head(last_timestep_out)  # Shape: (B, H)
        return output
