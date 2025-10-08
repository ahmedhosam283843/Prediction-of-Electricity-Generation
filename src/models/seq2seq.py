import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    LSTM encoder that maps an input sequence to final hidden/cell states.

    Input:
      x: (B, L, F)

    Output:
      (h, c): each shaped (n_layers, B, hid)
    """

    def __init__(self, in_dim: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Args:
            in_dim (int): Number of features per time step.
            hidden_size (int): Hidden size of the LSTM.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout between LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)

    def forward(self, x):                             # x: B × L × F
        _, (h, c) = self.lstm(x)
        return h, c                                   # n_layers × B × hid


class Decoder(nn.Module):
    """
    Autoregressive LSTM decoder for sequence generation with optional teacher forcing.

    At each step:
      - inputs previous target (predicted or ground truth),
      - updates hidden state,
      - emits next scalar prediction.

    Produces horizon-length univariate output.
    """

    def __init__(self, hidden_size: int, horizon: int,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Args:
            hidden_size (int): Hidden size of the LSTM (matches encoder).
            horizon (int): Number of future steps to generate.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout between LSTM layers.
        """
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(1, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, y0, h, c,
                teacher=None, tf_ratio: float = 0.5):
        """
        Args:
            y0 (Tensor): Last observed target at time t (B,).
            h (Tensor): Encoder hidden state (n_layers, B, hid).
            c (Tensor): Encoder cell state (n_layers, B, hid).
            teacher (Tensor|None): Ground-truth sequence (B, horizon) used for teacher forcing.
            tf_ratio (float): Probability of using teacher at each step during training.

        Returns:
            Tensor: (B, horizon) predicted sequence.
        """
        B = y0.size(0)
        y_prev = y0.view(B, 1, 1)       # seed as (B, 1, 1)

        outs = []
        for t in range(self.horizon):
            out, (h, c) = self.lstm(y_prev, (h, c))
            y_hat = self.fc(out[:, -1])  # (B, 1)
            outs.append(y_hat)

            use_teacher = (teacher is not None) and \
                          (torch.rand(1).item() < tf_ratio)
            next_in = teacher[:, t:t+1] if use_teacher else y_hat
            y_prev = next_in.unsqueeze(1)  # keep shape (B, 1, 1)

        return torch.cat(outs, dim=1)      # B × horizon


class Seq2SeqForecaster(nn.Module):
    """
    Sequence-to-sequence forecaster (Encoder-Decoder LSTM).

    Assumes the last channel of the input sequence contains the past target
    (scaled), whose last value is used as the decoder seed (y0).
    """

    def __init__(self, in_dim: int, horizon: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, tf_ratio: float = 0.7):
        """
        Args:
            in_dim (int): Number of input features per time step.
            horizon (int): Number of future steps to predict.
            hidden_size (int): Hidden size for both encoder and decoder LSTM.
            num_layers (int): Number of layers in both encoder and decoder.
            dropout (float): Dropout in LSTMs.
            tf_ratio (float): Teacher forcing ratio used during training.
        """
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, horizon, num_layers, dropout)
        self.tf_ratio = tf_ratio

    def forward(self, x, y_future=None):
        """
        Args:
            x (Tensor): (B, L, F) where the last feature is past target.
            y_future (Tensor|None): (B, H) ground truth for teacher forcing (train only).

        Returns:
            Tensor: (B, H) forecast.
        """
        self.decoder.horizon = self.decoder.horizon  # no-op; keeps interface consistent
        h, c = self.encoder(x)
        # Use the LAST time step's target-like channel as decoder seed
        y0 = x[:, -1, -1]      # shape (B)
        tf_r = self.tf_ratio if self.training else 0.0
        return self.decoder(y0, h, c,
                            teacher=y_future, tf_ratio=tf_r)
