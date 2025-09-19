import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128,
                 n_layers: int = 2, drop: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, n_layers,
                            dropout=drop, batch_first=True)

    def forward(self, x):                             # x: B × L × F
        _, (h, c) = self.lstm(x)
        return h, c                                   # n_layers × B × hid


class Decoder(nn.Module):
    def __init__(self, hid: int, horizon: int,
                 n_layers: int = 2, drop: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(1, hid, n_layers,
                            dropout=drop, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, y0, h, c,
                teacher=None, tf_ratio: float = 0.5):
        """
        y0       – last real y (B)
        teacher  – (B, horizon) ground truth sequence   (opt.)
        tf_ratio – prob. of using ground truth at step t
        """
        B = y0.size(0)
        y_prev = y0.view(B, 1, 1)       # ── FIX: make it 3-D ──

        outs = []
        for t in range(self.horizon):
            out, (h, c) = self.lstm(y_prev, (h, c))
            y_hat = self.fc(out[:, -1])  # B × 1
            outs.append(y_hat)

            use_teacher = (teacher is not None) and \
                          (torch.rand(1).item() < tf_ratio)
            next_in = teacher[:, t:t+1] if use_teacher else y_hat
            y_prev = next_in.unsqueeze(1)  # keep shape B × 1 × 1

        return torch.cat(outs, dim=1)      # B × horizon


class Seq2SeqForecaster(nn.Module):
    def __init__(self, in_dim: int, horizon: int,
                 hid: int = 128, n_layers: int = 2,
                 drop: float = 0.1, tf_ratio: float = 0.7):
        super().__init__()
        self.encoder   = Encoder(in_dim, hid, n_layers, drop)
        self.decoder   = Decoder(hid, horizon, n_layers, drop)
        self.tf_ratio  = tf_ratio

    def forward(self, x, y_future=None):
        self.decoder.horizon = self.decoder.horizon  # just in case
        h, c = self.encoder(x)
        # assume the LAST column in x is (scaled) target from t-1
        y0   = x[:, -1, -1]      # shape (B)
        tf_r = self.tf_ratio if self.training else 0.0
        return self.decoder(y0, h, c,
                            teacher=y_future, tf_ratio=tf_r)