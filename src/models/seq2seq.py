import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encodes the input sequence into a context vector (hidden and cell states)."""

    def __init__(self, in_dim: int, hid: int = 128,
                 n_layers: int = 2, drop: float = 0.1):
        """
        Args:
            in_dim (int): Number of input features.
            hid (int): Number of hidden units in the LSTM.
            n_layers (int): Number of LSTM layers.
            drop (float): Dropout probability.
        """
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, n_layers,
                            dropout=drop, batch_first=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, F).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Hidden and cell states of shape (n_layers, B, hid).
        """
        _, (h, c) = self.lstm(x)
        return h, c


class Decoder(nn.Module):
    """Decodes the context vector to generate the forecast sequence."""

    def __init__(self, hid: int, horizon: int,
                 n_layers: int = 2, drop: float = 0.1):
        """
        Args:
            hid (int): Number of hidden units in the LSTM.
            horizon (int): The forecast horizon length.
            n_layers (int): Number of LSTM layers.
            drop (float): Dropout probability.
        """
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(1, hid, n_layers,
                            dropout=drop, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, y0: torch.Tensor, h: torch.Tensor, c: torch.Tensor,
                teacher: torch.Tensor | None = None, tf_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass for the decoder, generating the sequence autoregressively.

        Args:
            y0 (torch.Tensor): The last known target value, shape (B,).
            h (torch.Tensor): Initial hidden state from the encoder.
            c (torch.Tensor): Initial cell state from the encoder.
            teacher (torch.Tensor | None): Ground truth sequence for teacher forcing, shape (B, horizon).
            tf_ratio (float): The probability of using the ground truth as the next input.

        Returns:
            torch.Tensor: The forecast sequence of shape (B, horizon).
        """
        B = y0.size(0)
        # Reshape to (B, 1, 1) to serve as the first input to the LSTM.
        y_prev = y0.view(B, 1, 1)

        outs = []
        for t in range(self.horizon):
            out, (h, c) = self.lstm(y_prev, (h, c))
            y_hat = self.fc(out[:, -1])  # Shape: (B, 1)
            outs.append(y_hat)

            use_teacher = (teacher is not None) and \
                          (torch.rand(1).item() < tf_ratio)
            next_in = teacher[:, t:t+1] if use_teacher else y_hat
            # Keep shape (B, 1, 1) for the next step
            y_prev = next_in.unsqueeze(1)

        return torch.cat(outs, dim=1)      # Shape: (B, horizon)


class Seq2SeqForecaster(nn.Module):
    """A standard Sequence-to-Sequence (Encoder-Decoder) model for forecasting."""

    def __init__(self, in_dim: int, horizon: int,
                 hid: int = 128, n_layers: int = 2,
                 drop: float = 0.1, tf_ratio: float = 0.7):
        """
        Args:
            in_dim (int): Number of input features.
            horizon (int): The forecast horizon length.
            hid (int): Number of hidden units in the LSTMs.
            n_layers (int): Number of LSTM layers for both encoder and decoder.
            drop (float): Dropout probability.
            tf_ratio (float): The probability of using teacher forcing during training.
        """
        super().__init__()
        self.encoder = Encoder(in_dim, hid, n_layers, drop)
        self.decoder = Decoder(hid, horizon, n_layers, drop)
        self.tf_ratio = tf_ratio

    def forward(self, x: torch.Tensor, y_future: torch.Tensor | None = None) -> torch.Tensor:
        h, c = self.encoder(x)
        # Assume the last feature of the last time step in x is the target value y_{t-1}.
        y0 = x[:, -1, -1]      # Shape: (B,)
        tf_r = self.tf_ratio if self.training else 0.0
        return self.decoder(y0, h, c,
                            teacher=y_future, tf_ratio=tf_r)
