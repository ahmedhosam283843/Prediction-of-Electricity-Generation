import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """A CNN-LSTM model for time series forecasting.

    This model first applies a series of 1D convolutional layers to extract
    features from the input sequence, then feeds the result into an LSTM layer
    to capture temporal dependencies. Finally, a fully connected head produces
    the forecast.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        """
        Args:
            input_dim (int): The number of features in the input sequence.
            output_dim (int): The forecast horizon (number of steps to predict).
            dropout (float): The dropout probability to be used in the network.
        """
        super().__init__()

        cnn_out_channels = 64
        lstm_hidden_size = 256
        fc_hidden_size = 256

        # 1D Convolutional feature extractor
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_out_channels, cnn_out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_out_channels, cnn_out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM layer to process the extracted features
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # Fully connected head for prediction
        self.fc_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc_hidden_size),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN-LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, F), where B is the
                              batch size, L is the sequence length, and F is
                              the number of input features.

        Returns:
            torch.Tensor: Output tensor of shape (B, H), where H is the
                          forecast horizon.
        """
        # x shape: (B, L, F)
        x = x.permute(0, 2, 1)  # To (B, F, L) for Conv1d
        x = self.cnn_feature_extractor(x)
        x = x.permute(0, 2, 1)  # Back to (B, L, C) for LSTM

        # LSTM processes the sequence of features
        out, _ = self.lstm(x)  # out shape: (B, L, lstm_hidden_size)

        # Use the output of the last time step for prediction
        last_timestep_out = out[:, -1, :]  # Shape: (B, lstm_hidden_size)

        # Pass through the fully connected head
        out = self.fc_head(last_timestep_out)  # Shape: (B, output_dim)
        return out
