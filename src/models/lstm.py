import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """A two-layer LSTM model for time series forecasting.

    This model uses a stack of two LSTM layers to process the input sequence,
    followed by a fully connected head to produce the forecast.
    """

    def __init__(self, input_dim: int, horizon: int, lookback: int | None = None):
        """
        Args:
            input_dim (int): The number of features in the input sequence.
            horizon (int): The forecast horizon (number of steps to predict).
            lookback (int | None): The length of the input sequence.
                                   Note: This parameter is not used by the model
                                   but is kept for API consistency.
        """
        super().__init__()

        lstm1_hidden_size = 256
        lstm2_hidden_size = 128
        fc_hidden_size = 128

        # LSTM layers
        self.lstm_layers = nn.Sequential(
            nn.LSTM(input_size=input_dim,
                    hidden_size=lstm1_hidden_size, batch_first=True),
            # The first element of the tuple from LSTM is the output tensor
            nn.Lambda(lambda x: x[0]),
            nn.Dropout(0.3),
            nn.LSTM(input_size=lstm1_hidden_size,
                    hidden_size=lstm2_hidden_size, batch_first=True),
            nn.Lambda(lambda x: x[0]),
            nn.Dropout(0.3)
        )

        # Fully connected head for prediction
        self.fc_head = nn.Sequential(
            nn.Linear(lstm2_hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fc_hidden_size, horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LSTMForecaster.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, F).

        Returns:
            torch.Tensor: Output tensor of shape (B, H).
        """
        # x shape: (B, L, F) -> (B, L, lstm2_hidden_size)
        lstm_out = self.lstm_layers(x)
        # Use the output of the last time step for prediction
        last_timestep_out = lstm_out[:, -1, :]  # Shape: (B, lstm2_hidden_size)
        # Pass through the fully connected head
        output = self.fc_head(last_timestep_out)  # Shape: (B, H)
        return output
