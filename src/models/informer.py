import numpy as np
from torch import nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4000):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: (L, B, D)
        L = x.size(0)
        x = x + self.pe[:L].unsqueeze(1)
        return self.drop(x)
    
class InformerForecaster(nn.Module):

    """
    Lightweight Informer-style forecaster suitable for benchmarking.
    This is NOT a full ProbSparse-Informer research implementation — instead it:
      - projects inputs to a d_model,
      - optionally performs one Conv1d-based distillation/downsampling,
      - uses a TransformerEncoder (stacked) to produce sequence features,
      - pools the final encoder outputs (last token) and uses a small MLP head to predict the H-step horizon.
    This keeps the same input/output interface as other models in the codebase:
      ctor(input_dim, horizon, d_model=128, ...)
      forward(x) -> (B, H)
    """
    def __init__(self,
                 input_dim: int,
                 horizon: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 distill: bool = False):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.distill = distill

        # input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # optional distillation conv (downsample by factor 2)
        if distill:
            # conv expects (B, C, L) where C == d_model
            self.distill_conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        else:
            self.distill_conv = None

        # positional encoding (re-usable)
        self.pe = PositionalEncoding(d_model, dropout=dropout, max_len=4096)

        # encoder (stacked TransformerEncoder layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False, activation="relu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # head: MLP that maps pooled encoder features to horizon outputs
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )

    def forward(self, x):
        """
        x: (B, L, F)
        returns: (B, H)
        """
        # project -> (B, L, D)
        z = self.input_proj(x)

        # optionally distill / downsample in the time axis
        if self.distill_conv is not None:
            # conv expects (B, C, L)
            z = z.transpose(1, 2).contiguous()     # (B, D, L)
            z = self.distill_conv(z)                # (B, D, L//2)
            z = z.transpose(1, 2).contiguous()      # (B, L2, D)

        # transformer expects (L, B, D)
        z = z.transpose(0, 1)  # (L, B, D)
        z = self.pe(z)         # add pos enc + dropout
        z_enc = self.encoder(z)  # (L, B, D)

        # use last time-step representation (most recent encoded info)
        z_last = z_enc[-1]  # (B, D)
        out = self.head(z_last)  # (B, H)
        return out
