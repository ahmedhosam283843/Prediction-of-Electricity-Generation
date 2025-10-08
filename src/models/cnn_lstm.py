import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN + LSTM forecaster for multi-horizon prediction.

    This model:
      - Applies a stack of 1D convolutions over the time axis to extract local patterns.
      - Feeds the convolved features into an LSTM to capture longer temporal dependencies.
      - Uses a small fully-connected head to output the full forecast horizon.
    """

    def __init__(self, input_dim, output_dim, conv_channels, lstm_hidden, dropout=0.0):
        """
        Args:
            input_dim (int): Number of input features per time step.
            output_dim (int): Forecast horizon length (number of future steps).
            conv_channels (list[int]): List of output channels for each conv layer.
            lstm_hidden (int): Hidden size of the LSTM layer.
            dropout (float): Dropout rate applied in conv and FC blocks.
        """
        super(CNNLSTMModel, self).__init__()
        if len(conv_channels) != 3:
            raise ValueError("conv_channels must be a list of 3 integers.")

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            input_dim, conv_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        self.conv2 = nn.Conv1d(
            conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        self.conv3 = nn.Conv1d(
            conv_channels[1], conv_channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(conv_channels[2])
        self.dropout = nn.Dropout(dropout)

        # LSTM layer (unidirectional)
        self.lstm = nn.LSTM(conv_channels[2], lstm_hidden, num_layers=1, batch_first=True,
                            dropout=dropout, bidirectional=False)

        # Fully connected head
        self.fc1 = nn.Linear(lstm_hidden, lstm_hidden)
        self.bn_fc = nn.BatchNorm1d(lstm_hidden)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(lstm_hidden, output_dim)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): (batch, seq_len, input_dim)

        Returns:
            Tensor: (batch, output_dim)
        """
        # To (batch, channels=input_dim, seq_len)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        # Back to (batch, seq_len, channels=conv_channels[2])
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)      # (batch, seq_len, lstm_hidden)
        # last time step features (batch, lstm_hidden)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.bn_fc(out)
        out = self.fc_dropout(out)
        out = self.fc2(out)        # (batch, output_dim)
        return out
