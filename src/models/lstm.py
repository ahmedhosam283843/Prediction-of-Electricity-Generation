import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, lookback, horizon):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=96, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=96, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        self.fc_out = nn.Linear(64, horizon)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc_out(x)